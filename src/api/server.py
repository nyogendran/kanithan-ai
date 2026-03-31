"""
FastAPI server for the NIE Tamil AI Tutor platform.

Usage:
  uvicorn src.api.server:app --reload --port 8000

Endpoints:
  POST /api/v1/query         — full teaching response (non-streaming)
  POST /api/v1/query/stream  — streaming teaching response (SSE)
  POST /api/v1/verify        — check student answer
  POST /api/v1/voice/converse — multi-turn voice conversation endpoint
  GET  /api/v1/student/{id}  — student profile + progress
  GET  /api/v1/hitl/queue    — pending HITL reviews
  POST /api/v1/hitl/resolve  — resolve a HITL flag
  GET  /health               — health check
  GET  /voice                — voice test UI
"""

from __future__ import annotations

import base64
import json
import logging
import pathlib
import re
import tempfile
import time
from dataclasses import asdict
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agents.orchestrator import OrchestratorAgent
from src.config import (
    GEMINI_API_KEY,
    GEMINI_TEACHING_MODEL,
    GEMINI_TRANSCRIBE_MODELS_RAW,
    LLM_BACKEND,
    llm_teaching_model,
)
from src.data.glossary import normalize_tamil_numbers
from src.models import Intent, Dialect

log = logging.getLogger("nie.api")

app = FastAPI(
    title="NIE Tamil AI Tutor",
    version="1.0.0",
    description="Multi-agent adaptive tutoring API for NIE Grade 7 Mathematics",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator: Optional[OrchestratorAgent] = None


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


def _safe_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _safe_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (Intent, Dialect)):
        return obj.value
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_dict(v) for v in obj]
    return obj


class QueryRequest(BaseModel):
    student_id: str = "SL_TM_2024_001"
    question: str
    district: str = "unknown"
    student_name: str = "மாணவர்"
    top_k: int = Field(default=6, ge=1, le=20)


class VerifyRequest(BaseModel):
    student_id: str = "SL_TM_2024_001"
    question: str
    student_answer: str
    exercise_topic: str = "factor_listing"
    district: str = "unknown"


class HITLResolveRequest(BaseModel):
    queue_id: int
    teacher_id: str
    status: str = "resolved"
    annotation: str = ""


_UI_DIR = pathlib.Path(__file__).resolve().parent.parent / "ui"


class ConversationState(str, Enum):
    LISTENING = "listening"
    CLARIFYING = "clarifying"
    CHECKING_FUNDAMENTALS = "checking_fundamentals"
    TEACHING_PREREQ = "teaching_prereq"
    RESPONDING = "responding"
    EXERCISING = "exercising"
    VERIFYING = "verifying"


_PREREQ_MAP = {
    "hcf": "factors",
    "lcm": "factors",
    "prime_factorization": "factors",
    "மீ.பொ.கா.": "factors",
    "மீ.பொ.ம.": "factors",
}

_voice_sessions: dict[str, dict] = {}


class TranscribeRequest(BaseModel):
    audio_base64: str
    mime_type: str = "audio/webm"


class VoiceRequest(BaseModel):
    student_id: str = "SL_TM_VOICE_001"
    transcript: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    district: str = "unknown"
    student_name: str = "குரல் மாணவர்"
    session_key: str = "default"


def _get_voice_session(key: str) -> dict:
    if key not in _voice_sessions:
        _voice_sessions[key] = {
            "state": ConversationState.LISTENING,
            "pending_question": None,
            "prereq_topic": None,
            "turns": 0,
            "exercise_count": 0,
        }
    return _voice_sessions[key]


@app.get("/health")
def health():
    """
    Includes LLM routing so clients can show whether teaching uses Gemini vs Ollama.
    Voice /voice/transcribe STT is still Gemini-only unless a local STT path is added.
    """
    return {
        "status": "ok",
        "service": "nie-tamil-tutor",
        "llm_backend": LLM_BACKEND,
        "teaching_model": llm_teaching_model(),
        "voice_transcribe_stt_backend": "gemini",
    }


@app.get("/voice")
def voice_ui():
    html_path = _UI_DIR / "voice_tutor.html"
    if not html_path.exists():
        raise HTTPException(404, "Voice UI not found")
    return FileResponse(html_path, media_type="text/html")


def _transcribe_model_list() -> list[str]:
    """Default: teaching model only (1 generate_content per voice clip)."""
    raw = (GEMINI_TRANSCRIBE_MODELS_RAW or "").strip()
    if not raw:
        return [GEMINI_TEACHING_MODEL]
    return [m.strip() for m in raw.split(",") if m.strip()]


def _parse_retry_after_seconds(err_str: str) -> Optional[float]:
    m = re.search(r"retry in ([\d.]+)\s*s", err_str, re.I)
    if m:
        return min(float(m.group(1)), 120.0)
    return None


def _gemini_transcribe(audio_bytes: bytes, mime_type: str) -> tuple[str, str]:
    """Try each configured model once; optional one retry if API suggests retry delay."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt_text = (
        "இது ஒரு தமிழ் மாணவரின் கணிதக் கேள்வி. "
        "ஒலிப்பதிவை துல்லியமாகத் தமிழில் எழுத்துருவாக்கம் செய்யுங்கள். "
        "எண்களை எப்போதும் இலக்கங்களாக எழுதுங்கள் (எ.கா. 24, 15, 30, 6) — "
        "தமிழ் சொற்களாக (இருபத்தி நான்கு) போல் அல்ல. "
        "கணிதக் குறியீடுகள் (× ÷ =) போல் அப்படியே. "
        "வேறு எதுவும் சேர்க்காதீர்கள் — transcript மட்டும் போதும்."
    )
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    text_part = types.Part(text=prompt_text)
    cfg = types.GenerateContentConfig(temperature=0.1, max_output_tokens=300)
    contents = [types.Content(parts=[audio_part, text_part])]

    models = _transcribe_model_list()
    last_err: Exception | None = None

    def _try_model(model: str) -> tuple[str, str]:
        response = client.models.generate_content(
            model=model, contents=contents, config=cfg,
        )
        return (response.text or "").strip(), model

    for model in models:
        try:
            return _try_model(model)
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if not is_quota:
                raise
            wait = _parse_retry_after_seconds(err_str)
            if wait is not None:
                log.warning("Quota on %s — retry once after %.1fs (API hint)", model, wait)
                time.sleep(wait)
                try:
                    return _try_model(model)
                except Exception as e2:
                    last_err = e2
                    log.warning("Retry failed on %s, trying next model...", model)
            else:
                log.warning("Quota on %s, trying next model...", model)

    raise last_err  # type: ignore[misc]


@app.post("/api/v1/voice/transcribe")
def voice_transcribe(req: TranscribeRequest):
    """Transcribe recorded audio to Tamil text using Gemini's audio understanding."""
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio data")

    if len(audio_bytes) < 100:
        raise HTTPException(400, "Audio too short")
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Audio exceeds 10MB limit")

    try:
        raw_transcript, model_used = _gemini_transcribe(audio_bytes, req.mime_type)
        transcript = normalize_tamil_numbers(raw_transcript)
        return {
            "transcript": transcript,
            "raw_transcript": raw_transcript,
            "source": "gemini",
            "model": model_used,
        }
    except Exception as e:
        err_str = str(e)
        is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
        if is_quota:
            log.error("All Gemini models quota-exhausted for transcription")
            wait = _parse_retry_after_seconds(err_str)
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "RESOURCE_EXHAUSTED",
                    "message": "Gemini API quota exhausted for voice transcription.",
                    "retry_after_seconds": wait,
                    "hint_ta": (
                        "சில நிமிடங்கள் காத்திருந்து மீண்டும் முயற்சிக்கவும்; அல்லது கேள்வியை உரையாக தட்டச்சு செய்யவும். "
                        "உலாவியின் உள்ளக குரல் அங்கீகாரத்தைப் பயன்படுத்தலும் முடியும் (Chrome)."
                    ),
                    "fallback": "browser_speech_or_text",
                },
            )
        log.error("Transcription failed: %s", e)
        raise HTTPException(502, f"Transcription failed: {e}")


@app.post("/api/v1/voice/converse")
def voice_converse(req: VoiceRequest):
    """Multi-turn voice conversation with clarification, prerequisite checks, and exercises."""
    orch = get_orchestrator()
    sess = _get_voice_session(req.session_key)
    sess["turns"] += 1
    actions: list[dict[str, Any]] = []

    if req.confidence < 0.6 and sess["state"] != ConversationState.CLARIFYING:
        sess["state"] = ConversationState.CLARIFYING
        sess["pending_question"] = req.transcript
        return {
            "state": sess["state"].value,
            "agent_says_ta": "கொஞ்சம் மீண்டும் சொல்ல முடியுமா? நான் சரியாகப் புரிந்துகொள்ள விரும்புகிறேன்.",
            "actions": [{"type": "request_repeat"}],
            "teaching": None,
            "diagram": None,
            "exercise": None,
        }

    if sess["state"] == ConversationState.CLARIFYING:
        sess["state"] = ConversationState.LISTENING

    question = req.transcript

    response = orch.handle(
        student_id=req.student_id,
        raw_query=question,
        district=req.district,
        student_name=req.student_name,
        n_retrieve=6,
    )
    resp_dict = _safe_dict(response)
    topic = resp_dict.get("teaching", {}).get("next_suggested_topic", "") or ""

    prereq_topic = _PREREQ_MAP.get(topic, None)
    if prereq_topic and sess.get("prereq_topic") != prereq_topic:
        student = orch.db.get_or_create_student(req.student_id)
        skill_level = student.skills.get(prereq_topic, 0.0)
        if skill_level < 0.4:
            sess["state"] = ConversationState.CHECKING_FUNDAMENTALS
            sess["pending_question"] = question
            sess["prereq_topic"] = prereq_topic
            actions.append({"type": "check_fundamentals", "topic": prereq_topic})
            return {
                "state": sess["state"].value,
                "agent_says_ta": f"முதலில் ஒரு கேள்வி — {prereq_topic} பற்றி உங்களுக்குத் தெரியுமா? எடுத்துக்காட்டாக, 12-இன் காரணிகள் என்ன?",
                "actions": actions,
                "teaching": resp_dict.get("teaching"),
                "diagram": resp_dict.get("diagram"),
                "exercise": None,
            }

    sess["state"] = ConversationState.RESPONDING
    sess["prereq_topic"] = None

    should_exercise = (
        resp_dict.get("intent") in ("EXPLAIN", "SHOW_METHOD")
        and sess["exercise_count"] < 3
    )
    if should_exercise:
        sess["state"] = ConversationState.EXERCISING
        sess["exercise_count"] += 1
        actions.append({"type": "offer_exercise"})

    return {
        "state": sess["state"].value,
        "agent_says_ta": resp_dict.get("teaching", {}).get("explanation_ta", ""),
        "actions": actions,
        "teaching": resp_dict.get("teaching"),
        "diagram": resp_dict.get("diagram"),
        "exercise": resp_dict.get("exercise"),
        "full_response": resp_dict,
    }


@app.post("/api/v1/query")
def query(req: QueryRequest):
    orch = get_orchestrator()
    response = orch.handle(
        student_id=req.student_id,
        raw_query=req.question,
        district=req.district,
        student_name=req.student_name,
        n_retrieve=req.top_k,
    )
    return _safe_dict(response)


@app.post("/api/v1/query/stream")
def query_stream(req: QueryRequest):
    orch = get_orchestrator()

    def event_generator():
        gen = orch.handle_streaming(
            student_id=req.student_id,
            raw_query=req.question,
            district=req.district,
            student_name=req.student_name,
            n_retrieve=req.top_k,
        )
        try:
            for chunk in gen:
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        except StopIteration as ex:
            final = ex.value
            if final:
                yield f"data: {json.dumps({'done': True, 'response': _safe_dict(final)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/verify")
def verify(req: VerifyRequest):
    orch = get_orchestrator()
    response = orch.handle(
        student_id=req.student_id,
        raw_query=req.question,
        district=req.district,
        student_answer=req.student_answer,
        exercise_topic=req.exercise_topic,
    )
    return _safe_dict(response)


@app.get("/api/v1/student/{student_id}")
def get_student(student_id: str):
    orch = get_orchestrator()
    student = orch.db.get_or_create_student(student_id)
    return {
        "student": asdict(student) if hasattr(student, "__dataclass_fields__") else student.__dict__,
        "progress": {
            "mastered": [k for k, v in student.skills.items() if v >= 0.75],
            "in_progress": [k for k, v in student.skills.items() if 0.3 <= v < 0.75],
            "not_started": [k for k, v in student.skills.items() if v < 0.3],
            "accuracy": student.accuracy() if hasattr(student, "accuracy") else 0.0,
            "total_exercises": student.total_exercises_attempted,
        },
    }


@app.get("/api/v1/hitl/queue")
def hitl_queue():
    orch = get_orchestrator()
    return {"queue": orch.db.get_hitl_queue(status="pending")}


@app.post("/api/v1/hitl/resolve")
def hitl_resolve(req: HITLResolveRequest):
    orch = get_orchestrator()
    orch.db.update_hitl_status(
        req.queue_id, req.status, req.teacher_id, req.annotation,
    )
    return {"status": "resolved", "queue_id": req.queue_id}
