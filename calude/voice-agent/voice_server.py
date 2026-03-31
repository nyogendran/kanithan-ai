"""
voice_server.py — WebSocket Voice Server
=========================================
Bridges the Flutter app ↔ VoiceIOManager ↔ OrchestratorAgent.

Each WebSocket connection represents one student's voice session.
Audio flows in as binary WebSocket frames (PCM 16kHz mono 16-bit).
Events flow out as JSON + binary audio frames.

Architecture:
  Flutter (voice_widget.dart)
    ↕ WebSocket (binary audio in, JSON+binary out)
  voice_server.py (this file)
    ↕ VoiceIOManager
    ↕ TamilSTTPipeline + VoiceActivityDetector
    ↕ OrchestratorAgent (agent_orchestrator.py)

Usage:
  python voice_server.py --host 0.0.0.0 --port 8765
  python voice_server.py --host 0.0.0.0 --port 8765 --ssl-cert cert.pem --ssl-key key.pem

Dependencies:
  pip install websockets
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("nie.voice_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class VoiceSession:
    """
    One WebSocket connection = one student voice session.
    Manages audio buffering, VAD state, and event dispatch.
    """

    def __init__(self, websocket, student_id: str, district: str,
                 grade: int, chapter: int, subject: str):
        self.ws = websocket
        self.student_id = student_id
        self.district = district
        self.grade = grade
        self.chapter = chapter
        self.subject = subject
        self.session_id = f"{student_id}_{int(time.time())}"
        self.audio_buffer: list[bytes] = []
        self.partial_transcript: str = ""
        self._voice_manager = None
        self._orchestrator = None

    async def initialize(self):
        """Set up the voice manager and orchestrator for this session."""
        from voice_stt_tts import (STTConfig, TTSConfig, TamilSTTPipeline,
                                    TamilTTSPipeline, VoiceIOManager)
        from voice_vad import VADConfig, VoiceActivityDetector
        from agent_orchestrator import OrchestratorAgent

        # Initialize orchestrator
        self._orchestrator = OrchestratorAgent(
            grade=self.grade,
            chapter=self.chapter,
            subject=self.subject,
        )

        # Initialize voice I/O
        self._voice_manager = VoiceIOManager(
            district=self.district,
            grade=self.grade,
        )
        self._voice_manager.attach_orchestrator(
            self._orchestrator, self.student_id)
        await self._voice_manager.initialize()

        log.info(f"[{self.session_id}] Session initialized for "
                 f"student={self.student_id} district={self.district}")

    async def send_event(self, event_type: str, data: dict = None):
        """Send JSON event to Flutter client."""
        payload = {"type": event_type, **(data or {})}
        await self.ws.send(json.dumps(payload, ensure_ascii=False))

    async def send_audio(self, audio_bytes: bytes, meta: dict = None):
        """Send audio bytes to Flutter client (binary frame)."""
        if meta:
            # Send metadata first
            await self.send_event("response_audio", meta)
        # Send raw audio as binary frame
        await self.ws.send(audio_bytes)

    def update_partial_transcript(self, text: str):
        """Called by streaming STT to update partial transcript."""
        self.partial_transcript = text
        if self._voice_manager:
            self._voice_manager.vad.update_partial_transcript(text)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def handle_connection(websocket):
    """Handle one WebSocket connection from Flutter."""
    # Parse query params: ws://host:port?student_id=X&district=Y&grade=7
    path = websocket.path if hasattr(websocket, 'path') else '/'
    parsed = urlparse(path)
    params = parse_qs(parsed.query)

    student_id = params.get("student_id", ["unknown"])[0]
    district   = params.get("district", ["unknown"])[0]
    grade      = int(params.get("grade", ["7"])[0])
    chapter    = int(params.get("chapter", ["4"])[0])
    subject    = params.get("subject", ["mathematics"])[0]

    log.info(f"New connection: student={student_id} district={district}")

    session = VoiceSession(
        websocket=websocket,
        student_id=student_id,
        district=district,
        grade=grade,
        chapter=chapter,
        subject=subject,
    )

    try:
        await session.initialize()
        await session.send_event("connected", {
            "student_id": student_id,
            "district": district,
            "session_id": session.session_id,
        })

        # Start the voice interaction loop
        await _run_voice_loop(session)

    except Exception as e:
        log.error(f"[{session.session_id}] Session error: {e}", exc_info=True)
        try:
            await session.send_event("error", {"message": str(e)})
        except Exception:
            pass
    finally:
        log.info(f"[{session.session_id}] Connection closed")


async def _run_voice_loop(session: VoiceSession):
    """
    Core voice loop: receive audio from Flutter, process, respond.

    Protocol:
    1. Flutter sends binary audio frames (PCM 16kHz, 30ms chunks)
    2. Flutter sends JSON control messages:
       {"type": "start_listening"}
       {"type": "stop_listening"}
       {"type": "student_answer", "text": "...", "topic": "hcf"}
    3. Server sends JSON events + binary audio chunks back
    """
    from voice_stt_tts import STTConfig, TamilSTTPipeline, Dialect
    from voice_vad import VADConfig, VoiceActivityDetector, AudioUtterance

    vad_config = VADConfig(district=session.district, grade=session.grade)
    stt_config = STTConfig(district=session.district, grade=session.grade)
    stt_pipeline = TamilSTTPipeline(stt_config)

    # Create an async queue for audio frames coming from Flutter
    audio_queue: asyncio.Queue = asyncio.Queue()
    control_queue: asyncio.Queue = asyncio.Queue()

    # Start receiver task
    receiver_task = asyncio.create_task(
        _receive_from_flutter(session.ws, audio_queue, control_queue))

    # Start VAD processing
    vad = VoiceActivityDetector(vad_config)

    await session.send_event("listening_start")

    try:
        # Process incoming audio frames
        async def flutter_audio_source():
            while True:
                frame = await audio_queue.get()
                if frame is None:
                    return
                yield frame

        async for utterance in vad.listen(audio_source=flutter_audio_source()):
            await session.send_event("utterance_ready", {
                "duration_ms": utterance.duration_ms,
                "speech_ratio": utterance.speech_ratio,
            })

            # STT
            await session.send_event("processing", {"stage": "transcribing"})
            stt_result = await stt_pipeline.transcribe(utterance)

            if not stt_result.raw_text:
                await session.send_event("error",
                    {"message": "வாக்கு அறியப்படவில்லை. மீண்டும் முயற்சிக்கவும்."})
                await session.send_event("listening_start")
                continue

            await session.send_event("stt_result", {
                "text": stt_result.normalized_text,
                "raw_text": stt_result.raw_text,
                "dialect": stt_result.dialect.value,
                "dialect_confidence": stt_result.dialect_confidence,
                "confidence": stt_result.stt_confidence,
                "is_complete": stt_result.is_math_complete,
                "numbers": stt_result.numbers_extracted,
                "used_offline": stt_result.used_offline,
            })

            if not stt_result.is_math_complete and \
               stt_result.stt_confidence < 0.5:
                clarify_audio = await session._voice_manager.tts.synthesize(
                    "கொஞ்சம் முழுமையாக கேள்வி கேளுங்கள்.")
                if clarify_audio:
                    await session.send_audio(clarify_audio,
                                              meta={"is_reprompt": True})
                await session.send_event("listening_start")
                continue

            # Multi-agent orchestrator
            await session.send_event("processing", {"stage": "thinking"})

            response = await session._orchestrator.handle(
                student_id=session.student_id,
                raw_query=stt_result.normalized_text,
                district=session.district,
                n_retrieve=6,
            )

            # Send teaching response
            if response.teaching:
                await session.send_event("response_text", {
                    "text": response.teaching.explanation_ta,
                    "dialect": stt_result.dialect.value,
                    "key_concepts": response.teaching.key_concepts,
                    "next_topic": response.teaching.next_suggested_topic,
                })

                # Stream TTS audio
                audio_queue_tts = await session._voice_manager.tts.synthesize_streaming(
                    response.teaching.explanation_ta,
                    dialect=stt_result.dialect)

                chunk_idx = 0
                while True:
                    audio_chunk = await audio_queue_tts.get()
                    if audio_chunk is None:
                        break
                    await session.send_audio(audio_chunk, meta={
                        "chunk_index": chunk_idx,
                        "is_first": chunk_idx == 0,
                    })
                    chunk_idx += 1

            # Send diagram
            if response.diagram:
                await session.send_event("diagram_ready", {
                    "diagram_type": response.diagram.diagram_type,
                    "spec": response.diagram.spec,
                    "caption_ta": response.diagram.caption_ta,
                })
                # Read caption aloud
                caption_audio = await session._voice_manager.tts.synthesize(
                    response.diagram.caption_ta, stt_result.dialect)
                if caption_audio:
                    await session.send_audio(caption_audio,
                                              meta={"is_caption": True})

            # Send exercise
            if response.exercise:
                await session.send_event("exercise_ready", {
                    "question_ta": response.exercise.question_ta,
                    "difficulty": response.exercise.difficulty,
                    "topic": response.exercise.topic,
                    "hint_ta": response.exercise.hint_ta,
                })
                # Read exercise aloud
                ex_audio = await session._voice_manager.tts.synthesize(
                    response.exercise.question_ta, stt_result.dialect)
                if ex_audio:
                    await session.send_audio(ex_audio,
                                              meta={"is_exercise": True})

            # Back to listening
            await session.send_event("listening_start")

    except Exception as e:
        log.error(f"Voice loop error: {e}", exc_info=True)
    finally:
        receiver_task.cancel()


async def _receive_from_flutter(websocket, audio_queue: asyncio.Queue,
                                  control_queue: asyncio.Queue):
    """Receive audio frames and control messages from Flutter."""
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary audio frame — 30ms PCM chunk
                await audio_queue.put(message)
            elif isinstance(message, str):
                # Control message
                try:
                    event = json.loads(message)
                    event_type = event.get("type", "")
                    if event_type == "stop_listening":
                        await audio_queue.put(None)  # sentinel
                    elif event_type == "student_answer":
                        await control_queue.put(event)
                    log.debug(f"Control: {event_type}")
                except json.JSONDecodeError:
                    pass
    except Exception:
        await audio_queue.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# SERVER ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def run_server(host: str = "0.0.0.0", port: int = 8765,
                      ssl_cert: str = None, ssl_key: str = None):
    """Start the WebSocket voice server."""
    try:
        import websockets
    except ImportError:
        raise ImportError("Install: pip install websockets")

    ssl_context = None
    if ssl_cert and ssl_key:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(ssl_cert, ssl_key)
        log.info(f"TLS enabled (cert={ssl_cert})")

    log.info(f"Starting Tamil Math Tutor Voice Server on {host}:{port}")
    log.info("Waiting for Flutter connections...")

    async with websockets.serve(
        handle_connection,
        host, port,
        ssl=ssl_context,
        max_size=10 * 1024 * 1024,   # 10MB max message (large audio chunks)
        ping_interval=20,
        ping_timeout=30,
    ):
        await asyncio.Future()   # run forever


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="NIE Tamil Math Tutor — Voice WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ssl-cert", default=None)
    parser.add_argument("--ssl-key", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(run_server(args.host, args.port, args.ssl_cert, args.ssl_key))


if __name__ == "__main__":
    main()
