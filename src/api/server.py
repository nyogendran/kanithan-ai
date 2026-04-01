"""
FastAPI server for the Kanithan Tamil AI Tutor platform.

Usage:
  uvicorn src.api.server:app --reload --port 8000

Endpoints:
  POST /api/v1/query         — full teaching response (non-streaming)
  POST /api/v1/query/stream  — streaming teaching response (SSE)
  POST /api/v1/verify        — check student answer
  POST /api/v1/voice/converse — multi-turn voice conversation endpoint
  WS   /ws/voice             — real-time voice via WebSocket (VAD + STT + TTS)
  GET  /api/v1/student/{id}  — student profile + progress
  GET  /api/v1/hitl/queue    — pending HITL reviews
  POST /api/v1/hitl/resolve  — resolve a HITL flag
  GET  /health               — health check
  GET  /voice                — voice test UI
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pathlib
import re
import tempfile
import time
from dataclasses import asdict
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.orchestrator import OrchestratorAgent
from src.chapters.registry import get_chapter_plugin
from src.config import (
    GEMINI_API_KEY,
    GEMINI_TEACHING_MODEL,
    GEMINI_TRANSCRIBE_MODELS_RAW,
    LLM_BACKEND,
    llm_teaching_model,
)
from src.data.glossary import normalize_tamil_numbers
from src.models import Intent, Dialect

log = logging.getLogger("kanithan.api")

app = FastAPI(
    title="Kanithan Tamil AI Tutor",
    version="1.0.0",
    description="Multi-agent adaptive tutoring API for Kanithan Grade 7 Mathematics",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator: Optional[OrchestratorAgent] = None
_diagnostic: Optional[DiagnosticAgent] = None


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


def get_diagnostic() -> DiagnosticAgent:
    global _diagnostic
    if _diagnostic is None:
        plugin = get_chapter_plugin(get_orchestrator().chapter)
        pack = plugin.topic_pack
        _diagnostic = DiagnosticAgent(
            chapter=plugin.chapter,
            prerequisite_graph=pack.prerequisite_graph,
            topic_to_skill=pack.topic_to_skill,
            skill_to_graph_entry=pack.skill_to_graph_entry,
            skill_labels_ta=pack.skill_labels_ta,
        )
    return _diagnostic


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
    DIAGNOSING = "diagnosing"
    RESPONDING = "responding"
    EXERCISING = "exercising"
    VERIFYING = "verifying"


_VOICE_LOG_DIR = pathlib.Path(
    os.getenv("VOICE_TRANSCRIPT_LOG_DIR", str(pathlib.Path("logs") / "voice_transcripts"))
)


class TranscribeRequest(BaseModel):
    audio_base64: str
    mime_type: str = "audio/webm"
    student_id: str = "SL_TM_VOICE_001"
    session_key: str = "default"


class VoiceRequest(BaseModel):
    student_id: str = "SL_TM_VOICE_001"
    transcript: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    district: str = "unknown"
    student_name: str = "குரல் மாணவர்"
    session_key: str = "default"
    tts: bool = Field(default=False, description="If true, include base64 TTS audio in response")


def _append_voice_transcript_log(
    *,
    student_id: str,
    session_key: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Append one transcript event line to logs/voice_transcripts/<student>/<session>.jsonl."""
    try:
        safe_student = re.sub(r"[^A-Za-z0-9_.-]", "_", student_id or "unknown")
        safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", session_key or "default")
        target_dir = _VOICE_LOG_DIR / safe_student
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{safe_session}.jsonl"
        row = {
            "ts": int(time.time() * 1000),
            "event": event_type,
            "student_id": student_id,
            "session_key": session_key,
            **payload,
        }
        with target_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        log.exception("Failed to write voice transcript log")


def _get_voice_session(key: str, student_id: str) -> dict:
    """Load or create a voice session from the database."""
    orch = get_orchestrator()
    row = orch.db.get_or_create_voice_session(key, student_id)
    return {
        "state": ConversationState(row["state"]),
        "pending_question": row["pending_question"],
        "prereq_topic": row["prereq_topic"],
        "turns": row["turns"],
        "exercise_count": row["exercise_count"],
        "diagnostic_queue": json.loads(row["diagnostic_queue"]) if row.get("diagnostic_queue") else [],
        "diagnostic_results": json.loads(row["diagnostic_results"]) if row.get("diagnostic_results") else {},
        "original_question": row.get("original_question"),
        "current_probe": json.loads(row["current_probe"]) if row.get("current_probe") else None,
    }


def _save_voice_session(key: str, sess: dict) -> None:
    """Persist voice session state back to the database."""
    orch = get_orchestrator()
    dq = sess.get("diagnostic_queue")
    dr = sess.get("diagnostic_results")
    cp = sess.get("current_probe")
    orch.db.save_voice_session(key, {
        "state": sess["state"].value if isinstance(sess["state"], ConversationState) else sess["state"],
        "pending_question": sess.get("pending_question"),
        "prereq_topic": sess.get("prereq_topic"),
        "turns": sess.get("turns", 0),
        "exercise_count": sess.get("exercise_count", 0),
        "diagnostic_queue": json.dumps(dq, ensure_ascii=False) if dq else None,
        "diagnostic_results": json.dumps(dr, ensure_ascii=False) if dr else None,
        "original_question": sess.get("original_question"),
        "current_probe": json.dumps(cp, ensure_ascii=False) if cp else None,
    })


@app.get("/health")
def health():
    """
    Includes LLM routing so clients can show whether teaching uses Gemini vs Ollama.
    Voice /voice/transcribe STT is still Gemini-only unless a local STT path is added.
    """
    plugin = get_chapter_plugin(get_orchestrator().chapter)
    ident = plugin.topic_pack.identity
    return {
        "status": "ok",
        "service": "kanithan-ai-tutor",
        "llm_backend": LLM_BACKEND,
        "teaching_model": llm_teaching_model(),
        "voice_transcribe_stt_backend": "gemini",
        "voice_stt_backend": os.environ.get("STT_BACKEND", "gemini"),
        "voice_tts_backend": os.environ.get("TTS_BACKEND", "gemini"),
        "ws_voice_endpoint": "/ws/voice",
        "chapter": {
            "grade": ident.grade,
            "subject": ident.subject,
            "part": ident.part,
            "chapter_number": ident.chapter_number,
            "chapter_code": ident.chapter_code,
            "chapter_name": ident.chapter_name,
            "canonical_path": ident.canonical_path,
        },
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
        _append_voice_transcript_log(
            student_id=req.student_id,
            session_key=req.session_key,
            event_type="voice_transcribe",
            payload={
                "mime_type": req.mime_type,
                "audio_size_bytes": len(audio_bytes),
                "raw_transcript": raw_transcript,
                "normalized_transcript": transcript,
                "model": model_used,
                "source": "gemini",
            },
        )
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


def _detect_target_skill(question: str) -> Optional[str]:
    """Detect the target skill from the student's question using keyword matching."""
    q = question.lower()
    pack = get_chapter_plugin(get_orchestrator().chapter).topic_pack
    for keyword, skill in pack.topic_detect_keywords.items():
        if keyword in q:
            return skill
    return None


def _build_converse_response(
    state: str,
    agent_says_ta: str,
    actions: list | None = None,
    **extra: Any,
) -> dict[str, Any]:
    base = {
        "state": state,
        "agent_says_ta": agent_says_ta,
        "actions": actions or [],
        "teaching": None,
        "diagram": None,
        "exercise": None,
    }
    base.update(extra)
    return base


def _maybe_add_tts(response: dict, req: VoiceRequest) -> dict:
    """If req.tts is True, synthesize agent_says_ta and attach base64 audio."""
    if not req.tts:
        return response
    agent_text = response.get("agent_says_ta", "")
    if not agent_text:
        return response
    try:
        from src.voice.tts import TamilTTSPipeline
        pipeline = TamilTTSPipeline()
        audio = asyncio.run(pipeline.synthesize(agent_text, dialect=req.district))
        if audio:
            response["tts_audio_base64"] = base64.b64encode(audio).decode("ascii")
            response["tts_mime_type"] = "audio/mp3"
    except Exception as e:
        log.warning("TTS synthesis failed for HTTP response: %s", e)
    return response


@app.post("/api/v1/voice/converse")
def voice_converse(req: VoiceRequest):
    """Multi-turn voice conversation with diagnostic probing, clarification, and exercises."""
    result = _voice_converse_impl(req)
    return _maybe_add_tts(result, req)


def _voice_converse_impl(req: VoiceRequest) -> dict[str, Any]:
    """Core conversational FSM logic (separated for TTS wrapping and WS reuse)."""
    orch = get_orchestrator()
    diag = get_diagnostic()
    sess = _get_voice_session(req.session_key, req.student_id)
    sess["turns"] += 1
    actions: list[dict[str, Any]] = []

    # ── Low confidence → ask to repeat ──
    if req.confidence < 0.6 and sess["state"] not in (
        ConversationState.CLARIFYING, ConversationState.DIAGNOSING,
    ):
        sess["state"] = ConversationState.CLARIFYING
        sess["pending_question"] = req.transcript
        _save_voice_session(req.session_key, sess)
        return _build_converse_response(
            state=sess["state"].value,
            agent_says_ta="கொஞ்சம் மீண்டும் சொல்ல முடியுமா? நான் சரியாகப் புரிந்துகொள்ள விரும்புகிறேன்.",
            actions=[{"type": "request_repeat"}],
        )

    if sess["state"] == ConversationState.CLARIFYING:
        sess["state"] = ConversationState.LISTENING

    question = req.transcript
    _append_voice_transcript_log(
        student_id=req.student_id,
        session_key=req.session_key,
        event_type="voice_converse_input",
        payload={
            "confidence": req.confidence,
            "district": req.district,
            "transcript": question,
            "conv_state": sess["state"].value,
        },
    )

    # ── DIAGNOSING state: evaluate diagnostic answer ──
    if sess["state"] == ConversationState.DIAGNOSING:
        return _handle_diagnostic_answer(req, sess, diag, orch)

    # ── Normal flow: process the student's question ──
    response = orch.handle(
        student_id=req.student_id,
        raw_query=question,
        district=req.district,
        student_name=req.student_name,
        n_retrieve=6,
    )
    resp_dict = _safe_dict(response)

    target_skill = _detect_target_skill(question)

    # ── Check if diagnostic is needed ──
    if target_skill:
        student = orch.db.get_or_create_student(
            req.student_id, req.student_name, req.district,
        )
        diag_queue = diag.build_diagnostic_queue(target_skill, student)

        if diag_queue:
            probe = diag.generate_probe_question(diag_queue[0])
            sess["state"] = ConversationState.DIAGNOSING
            sess["original_question"] = question
            sess["diagnostic_queue"] = diag_queue
            sess["diagnostic_results"] = {}
            sess["current_probe"] = probe

            total = len(diag_queue)
            skill_label = diag.skill_label_ta(diag_queue[0])
            intro = (
                f"நல்ல கேள்வி! முதலில் உங்கள் அடிப்படையை சோதிக்கிறேன் "
                f"({total} கேள்வி{'கள்' if total > 1 else ''}).\n\n"
                f"அடிப்படை 1/{total} — {skill_label}:\n"
                f"{probe['question_ta']}"
            )

            actions.append({
                "type": "diagnostic_start",
                "total": total,
                "current": 1,
                "skill": diag_queue[0],
            })
            _save_voice_session(req.session_key, sess)
            return _build_converse_response(
                state=sess["state"].value,
                agent_says_ta=intro,
                actions=actions,
                diagnostic={
                    "total": total,
                    "current": 1,
                    "skill": diag_queue[0],
                    "skill_label_ta": skill_label,
                },
            )

    # ── No diagnostic needed: respond directly ──
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

    _save_voice_session(req.session_key, sess)

    return {
        "state": sess["state"].value,
        "agent_says_ta": resp_dict.get("teaching", {}).get("explanation_ta", ""),
        "actions": actions,
        "teaching": resp_dict.get("teaching"),
        "diagram": resp_dict.get("diagram"),
        "exercise": resp_dict.get("exercise"),
        "full_response": resp_dict,
    }


def _handle_diagnostic_answer(
    req: VoiceRequest,
    sess: dict,
    diag: DiagnosticAgent,
    orch: OrchestratorAgent,
) -> dict[str, Any]:
    """Process the student's answer to a diagnostic probe question."""
    probe = sess.get("current_probe")
    queue = sess.get("diagnostic_queue", [])
    results = sess.get("diagnostic_results", {})
    student_answer = req.transcript

    if not probe or not queue:
        sess["state"] = ConversationState.LISTENING
        _save_voice_session(req.session_key, sess)
        return _build_converse_response(
            state="listening",
            agent_says_ta="ஏதோ பிழை ஏற்பட்டது. உங்கள் கேள்வியை மீண்டும் கேளுங்கள்.",
        )

    current_skill = queue[0]
    was_retry = current_skill in results and not results[current_skill].get("correct", False)

    evaluation = diag.evaluate_probe_answer(probe, student_answer)
    correct = evaluation["correct"]
    hint_ta = evaluation.get("hint_ta", "")

    results[current_skill] = {
        "correct": correct,
        "retried": was_retry,
        "question_ta": probe["question_ta"],
        "student_answer": student_answer,
    }
    sess["diagnostic_results"] = results

    delta = diag.compute_skill_delta(correct, was_retry)
    student = orch.db.get_or_create_student(
        req.student_id, req.student_name, req.district,
    )
    skill_key = current_skill
    old_val = student.skills.get(skill_key, 0.0)
    student.skills[skill_key] = max(0.0, min(1.0, old_val + delta))
    orch.db.save_student(student)

    _append_voice_transcript_log(
        student_id=req.student_id,
        session_key=req.session_key,
        event_type="diagnostic_eval",
        payload={
            "skill": current_skill,
            "correct": correct,
            "was_retry": was_retry,
            "delta": delta,
            "student_answer": student_answer,
        },
    )

    total_diag = len(sess.get("diagnostic_queue", [])) + len(
        [r for r in results.values() if r.get("correct")]
    )

    if correct:
        queue.pop(0)
        sess["diagnostic_queue"] = queue

        if not queue:
            return _finish_diagnostic(req, sess, orch, results)

        next_probe = diag.generate_probe_question(queue[0])
        sess["current_probe"] = next_probe
        done_count = len(results)
        total = done_count + len(queue)
        next_skill_label = diag.skill_label_ta(queue[0])

        feedback = (
            f"சரி! "
            f"அடிப்படை {done_count + 1}/{total} — {next_skill_label}:\n"
            f"{next_probe['question_ta']}"
        )
        _save_voice_session(req.session_key, sess)
        return _build_converse_response(
            state=ConversationState.DIAGNOSING.value,
            agent_says_ta=feedback,
            actions=[{
                "type": "diagnostic_next",
                "current": done_count + 1,
                "total": total,
                "skill": queue[0],
                "prev_correct": True,
            }],
            diagnostic={
                "total": total,
                "current": done_count + 1,
                "skill": queue[0],
                "skill_label_ta": next_skill_label,
                "prev_correct": True,
            },
        )

    # Incorrect answer
    if not was_retry:
        sess["diagnostic_results"] = results
        _save_voice_session(req.session_key, sess)
        skill_label = diag.skill_label_ta(current_skill)
        feedback = (
            f"கிட்டத்தட்ட! {hint_ta}\n\n"
            f"மீண்டும் முயற்சிக்கவும்:\n{probe['question_ta']}"
        )
        return _build_converse_response(
            state=ConversationState.DIAGNOSING.value,
            agent_says_ta=feedback,
            actions=[{
                "type": "diagnostic_retry",
                "skill": current_skill,
                "hint_ta": hint_ta,
            }],
            diagnostic={
                "total": len(queue) + len([r for r in results.values() if r.get("correct")]),
                "current": len(results),
                "skill": current_skill,
                "skill_label_ta": skill_label,
                "prev_correct": False,
                "retry": True,
            },
        )

    # Failed on retry — give the answer, move on
    queue.pop(0)
    sess["diagnostic_queue"] = queue
    teach_hint = probe.get("hint_on_fail_ta", "")

    if not queue:
        return _finish_diagnostic(req, sess, orch, results, preamble=teach_hint)

    next_probe = diag.generate_probe_question(queue[0])
    sess["current_probe"] = next_probe
    done_count = len(results)
    total = done_count + len(queue)
    next_skill_label = diag.skill_label_ta(queue[0])

    feedback = (
        f"பரவாயில்லை. {teach_hint}\n\n"
        f"அடுத்த கேள்விக்கு வருவோம். "
        f"அடிப்படை {done_count + 1}/{total} — {next_skill_label}:\n"
        f"{next_probe['question_ta']}"
    )
    _save_voice_session(req.session_key, sess)
    return _build_converse_response(
        state=ConversationState.DIAGNOSING.value,
        agent_says_ta=feedback,
        actions=[{
            "type": "diagnostic_next",
            "current": done_count + 1,
            "total": total,
            "skill": queue[0],
            "prev_correct": False,
        }],
        diagnostic={
            "total": total,
            "current": done_count + 1,
            "skill": queue[0],
            "skill_label_ta": next_skill_label,
            "prev_correct": False,
        },
    )


def _finish_diagnostic(
    req: VoiceRequest,
    sess: dict,
    orch: OrchestratorAgent,
    results: dict,
    preamble: str = "",
) -> dict[str, Any]:
    """Finish the diagnostic phase and answer the original question."""
    original = sess.get("original_question", "")

    correct_count = sum(1 for r in results.values() if r.get("correct"))
    total_count = len(results)

    sess["state"] = ConversationState.RESPONDING
    sess["diagnostic_queue"] = []
    sess["current_probe"] = None
    sess["original_question"] = None

    if not original:
        _save_voice_session(req.session_key, sess)
        return _build_converse_response(
            state=ConversationState.LISTENING.value,
            agent_says_ta="அடிப்படை சோதனை முடிந்தது. உங்கள் கேள்வியை மீண்டும் கேளுங்கள்.",
        )

    response = orch.handle(
        student_id=req.student_id,
        raw_query=original,
        district=req.district,
        student_name=req.student_name,
        n_retrieve=6,
    )
    resp_dict = _safe_dict(response)
    explanation = resp_dict.get("teaching", {}).get("explanation_ta", "")

    summary_parts = []
    if preamble:
        summary_parts.append(preamble)
    if correct_count == total_count:
        summary_parts.append(
            f"மிக நன்று! அனைத்து {total_count} அடிப்படைக் கேள்விகளும் சரி."
        )
    else:
        summary_parts.append(
            f"அடிப்படை சோதனை: {correct_count}/{total_count} சரி."
        )
    summary_parts.append("இப்போது உங்கள் கேள்விக்கு வருவோம்.\n")

    full_text = "\n".join(summary_parts) + "\n" + explanation

    _save_voice_session(req.session_key, sess)
    return {
        "state": ConversationState.RESPONDING.value,
        "agent_says_ta": full_text,
        "actions": [{"type": "diagnostic_complete", "results": results}],
        "teaching": resp_dict.get("teaching"),
        "diagram": resp_dict.get("diagram"),
        "exercise": resp_dict.get("exercise"),
        "full_response": resp_dict,
        "diagnostic_summary": {
            "correct": correct_count,
            "total": total_count,
            "results": results,
        },
    }


# ---------------------------------------------------------------------------
# WebSocket real-time voice endpoint  (Steps 4, 5, 6)
# ---------------------------------------------------------------------------

async def _ws_send_json(ws: WebSocket, data: dict) -> None:
    """Send a JSON message over the WebSocket, silently ignoring closed connections."""
    try:
        await ws.send_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


async def _ws_send_audio(ws: WebSocket, audio_bytes: bytes) -> None:
    """Send binary audio over the WebSocket."""
    try:
        await ws.send_bytes(audio_bytes)
    except Exception:
        pass


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    """
    Real-time voice tutoring via WebSocket.

    Protocol (browser -> server):
      - JSON: {"type":"start","student_id":"...","district":"...","session_key":"..."}
      - Binary: raw PCM 16 kHz mono 16-bit audio frames (960 bytes = 30 ms)
      - JSON: {"type":"stop"} — end the session
      - JSON: {"type":"text_input","text":"..."} — typed text fallback

    Protocol (server -> browser):
      - JSON: {"type":"listening_start"} — VAD is listening
      - JSON: {"type":"speech_detected"} — speech started
      - JSON: {"type":"utterance_ready","duration_ms":...}
      - JSON: {"type":"stt_result","text":"...","confidence":...}
      - JSON: {"type":"response",...} — same shape as /api/v1/voice/converse
      - JSON: {"type":"tts_start"}
      - Binary: TTS audio chunks (MP3)
      - JSON: {"type":"tts_end"}
      - JSON: {"type":"error","message":"..."}
    """
    await ws.accept()

    student_id = "SL_TM_VOICE_001"
    district = "unknown"
    student_name = "குரல் மாணவர்"
    session_key = "ws_default"
    started = False

    try:
        from src.voice.vad import VADConfig, VoiceActivityDetector
        from src.voice.stt import TamilSTTPipeline
        from src.voice.tts import TamilTTSPipeline

        stt_pipeline = TamilSTTPipeline()
        tts_pipeline = TamilTTSPipeline()

        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        vad: VoiceActivityDetector | None = None

        async def _process_utterances():
            """Background task: consume utterances from VAD and process through FSM."""
            nonlocal vad
            if vad is None:
                return

            async def audio_source():
                while True:
                    frame = await audio_queue.get()
                    if frame is None:
                        return
                    yield frame

            async for utterance in vad.listen(audio_source=audio_source()):
                await _ws_send_json(ws, {
                    "type": "utterance_ready",
                    "duration_ms": utterance.duration_ms,
                    "speech_ratio": utterance.speech_ratio,
                })

                stt_result = await stt_pipeline.transcribe(utterance)

                if not stt_result.raw_text:
                    await _ws_send_json(ws, {
                        "type": "error",
                        "message": "வாக்கு அறியப்படவில்லை. மீண்டும் முயற்சிக்கவும்.",
                    })
                    await _ws_send_json(ws, {"type": "listening_start"})
                    continue

                await _ws_send_json(ws, {
                    "type": "stt_result",
                    "text": stt_result.normalized_text,
                    "raw_text": stt_result.raw_text,
                    "confidence": stt_result.stt_confidence,
                    "dialect": stt_result.dialect.value,
                    "is_complete": stt_result.is_math_complete,
                })

                fake_req = VoiceRequest(
                            student_id=student_id,
                            transcript=stt_result.normalized_text,
                            confidence=stt_result.stt_confidence,
                            district=district,
                            student_name=student_name,
                            session_key=session_key,
                        )
                response = _voice_converse_impl(fake_req)

                await _ws_send_json(ws, {"type": "response", **response})

                agent_text = response.get("agent_says_ta", "")
                if agent_text:
                    await _ws_send_json(ws, {"type": "tts_start"})
                    tts_queue = await tts_pipeline.synthesize_streaming(
                        agent_text, dialect=district)
                    while True:
                        chunk = await tts_queue.get()
                        if chunk is None:
                            break
                        await _ws_send_audio(ws, chunk)
                    await _ws_send_json(ws, {"type": "tts_end"})

                await _ws_send_json(ws, {"type": "listening_start"})

        processor_task: asyncio.Task | None = None

        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "start":
                    student_id = data.get("student_id", student_id)
                    district = data.get("district", district)
                    student_name = data.get("student_name", student_name)
                    session_key = data.get("session_key", f"ws_{student_id}")
                    started = True

                    vad_config = VADConfig(district=district)
                    vad = VoiceActivityDetector(config=vad_config)

                    if processor_task and not processor_task.done():
                        processor_task.cancel()
                    processor_task = asyncio.create_task(_process_utterances())

                    await _ws_send_json(ws, {
                        "type": "connected",
                        "student_id": student_id,
                        "session_key": session_key,
                    })
                    await _ws_send_json(ws, {"type": "listening_start"})

                elif msg_type == "stop":
                    await audio_queue.put(None)
                    break

                elif msg_type == "text_input" and started:
                    text = data.get("text", "").strip()
                    if text:
                        fake_req = VoiceRequest(
                            student_id=student_id,
                            transcript=text,
                            confidence=1.0,
                            district=district,
                            student_name=student_name,
                            session_key=session_key,
                        )
                        response = _voice_converse_impl(fake_req)
                        await _ws_send_json(ws, {"type": "response", **response})

                        agent_text = response.get("agent_says_ta", "")
                        if agent_text:
                            await _ws_send_json(ws, {"type": "tts_start"})
                            tts_audio = await tts_pipeline.synthesize(agent_text, dialect=district)
                            if tts_audio:
                                await _ws_send_audio(ws, tts_audio)
                            await _ws_send_json(ws, {"type": "tts_end"})

            elif "bytes" in message and started:
                raw_audio = message["bytes"]
                if raw_audio:
                    await audio_queue.put(raw_audio)

    except WebSocketDisconnect:
        log.info("WebSocket voice client disconnected (student=%s)", student_id)
    except Exception as e:
        log.error("WebSocket voice error: %s", e, exc_info=True)
        try:
            await _ws_send_json(ws, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await audio_queue.put(None)
        if processor_task is not None and not processor_task.done():
            processor_task.cancel()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

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
    student = orch.db.get_or_create_student(student_id, "மாணவர்", "unknown")
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
