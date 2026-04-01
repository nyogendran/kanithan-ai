"""
Voice Activity Detection with math-aware pause handling for Tamil students.

Silero VAD (92kB ONNX model) + adaptive timeout + partial transcript analysis.
Students say "72 உம் 108 உம்..." then pause while thinking — a naive timeout
cuts them off. This module extends the silence window when numbers or
continuation signals are detected in the partial transcript.

Adapted from the prototype in _obsolete/claude/voice-agent/voice_vad.py
for proper integration with the src/ package structure.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import numpy as np

log = logging.getLogger("kanithan.vad")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    channels: int = 1

    speech_threshold: float = 0.50
    silence_threshold: float = 0.35

    pause_short_ms: int = 1500
    pause_long_ms: int = 3000
    pause_math_ms: int = 4500
    max_utterance_ms: int = 30000

    min_words_to_process: int = 3
    confidence_threshold: float = 0.55

    district: str = "unknown"
    grade: int = 7

    def __post_init__(self):
        if self.district in ("estate", "nuwara_eliya", "hatton", "badulla"):
            self.pause_long_ms = 4000
            self.pause_math_ms = 6000
        elif self.district == "batticaloa":
            self.pause_long_ms = 3500


# ---------------------------------------------------------------------------
# Tamil math completeness detection
# ---------------------------------------------------------------------------

TAMIL_NUMBER_WORDS = {
    "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5,
    "ஆறு": 6, "ஏழு": 7, "எட்டு": 8, "ஒன்பது": 9, "பத்து": 10,
    "பதினொன்று": 11, "பன்னிரண்டு": 12, "இருபது": 20, "முப்பது": 30,
    "நாற்பது": 40, "ஐம்பது": 50, "அறுபது": 60, "எழுபது": 70,
    "எண்பது": 80, "தொண்ணூறு": 90, "நூறு": 100, "ஆயிரம்": 1000,
}

QUESTION_TERMINATORS_TA = [
    "காண்க", "காண்போம்", "காணுங்கள்", "கண்டுபிடி", "கண்டறி",
    "கணக்கிடு", "தீர்மானி", "விடையளி", "சொல்லுங்கள்", "விளக்கு",
    "விளக்குங்கள்", "காட்டு", "காட்டுங்கள்", "எழுதுக", "வரை",
    "?", "என்ன", "ஏன்", "எப்படி", "எவ்வளவு",
]

CONTINUATION_SIGNALS_TA = [
    "உம்", "ஆகிய", "ஆன", "மற்றும்", "கொண்ட",
    "என்று", "என்னும்", "ஐ", "இன்",
]

MATH_OPERATORS_TA = [
    "பெருக்கல்", "வகுத்தல்", "கூட்டல்", "கழித்தல்",
    "காரணி", "மடங்கு", "பொ.கா.பெ.", "பொ.ம.சி.",
    "×", "÷", "+", "-", "=",
]


class MathCompletenessChecker:
    """Decides whether a partial Tamil math transcript is complete enough to process."""

    def check(self, transcript: str, pause_duration_ms: float) -> dict:
        if not transcript or not transcript.strip():
            return self._incomplete(confidence=0.0, wait=2000)

        words = transcript.strip().split()
        if len(words) < 2:
            return self._incomplete(confidence=0.1, wait=2000)

        text = transcript.lower()

        has_numbers = bool(re.search(r'\b\d+\b', transcript)) or \
                      any(w in text for w in TAMIL_NUMBER_WORDS)
        has_operator = any(op in transcript for op in MATH_OPERATORS_TA)
        has_terminator = any(t in transcript for t in QUESTION_TERMINATORS_TA)
        has_continuation = any(s in transcript for s in CONTINUATION_SIGNALS_TA)

        score = 0.0
        reasons: list[str] = []

        if has_terminator:
            score += 0.55
            reasons.append("terminator_found")
        if has_numbers and has_operator:
            score += 0.2
            reasons.append("math_expression_complete")
        elif has_numbers:
            score += 0.1
            reasons.append("has_numbers")
        if len(words) >= 5:
            score += 0.1
            reasons.append("sufficient_length")
        if has_continuation and not has_terminator:
            score -= 0.3
            reasons.append("continuation_signal")
        if pause_duration_ms >= 3000:
            score += 0.2
            reasons.append("long_pause")
        elif pause_duration_ms >= 1500 and has_terminator:
            score += 0.1
            reasons.append("moderate_pause_with_terminator")

        score = max(0.0, min(1.0, score))
        is_complete = score >= 0.55

        if not is_complete and has_numbers and not has_terminator:
            wait = 3000
        elif not is_complete and has_continuation:
            wait = 2500
        else:
            wait = 0

        return {
            "is_complete": is_complete,
            "confidence": score,
            "has_numbers": has_numbers,
            "has_operator": has_operator,
            "has_terminator": has_terminator,
            "has_continuation": has_continuation,
            "recommended_wait_ms": wait,
            "word_count": len(words),
            "reasons": reasons,
        }

    def _incomplete(self, confidence: float, wait: int) -> dict:
        return {
            "is_complete": False, "confidence": confidence,
            "has_numbers": False, "has_operator": False,
            "has_terminator": False, "has_continuation": False,
            "recommended_wait_ms": wait, "word_count": 0, "reasons": [],
        }


# ---------------------------------------------------------------------------
# Silero VAD ONNX wrapper
# ---------------------------------------------------------------------------

class SileroVAD:
    """
    92kB ONNX model for frame-level speech probability.
    Auto-downloads on first use. Falls back to energy-based stub
    if onnxruntime is not installed.
    """

    MODEL_URL = (
        "https://github.com/snakers4/silero-vad/raw/master/"
        "src/silero_vad/data/silero_vad.onnx"
    )

    def __init__(self, model_path: Optional[str] = None,
                 sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.session = None
        self._h = None
        self._c = None
        self._model_path = model_path or "models/silero_vad.onnx"
        self._load()

    def _load(self):
        from pathlib import Path
        model_path = Path(self._model_path)
        if not model_path.exists():
            log.info("Downloading Silero VAD model to %s ...", model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            log.info("Silero VAD model downloaded.")

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(model_path), sess_options=opts,
                providers=["CPUExecutionProvider"])
            self._reset_state()
            log.info("Silero VAD initialized (ONNX).")
        except ImportError:
            log.warning("onnxruntime not installed — using energy-based stub VAD.")
            self.session = None

    def _reset_state(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def predict_frame(self, audio_frame: np.ndarray) -> float:
        """Speech probability for a single 30ms frame (480 samples at 16kHz)."""
        if self.session is None:
            rms = np.sqrt(np.mean(audio_frame ** 2))
            return min(1.0, rms * 20)

        x = audio_frame.reshape(1, -1).astype(np.float32)
        sr = np.array(self.sample_rate, dtype=np.int64)
        out, self._h, self._c = self.session.run(
            None, {"input": x, "sr": sr, "h": self._h, "c": self._c})
        return float(out[0][0])

    def reset(self):
        self._reset_state()


# ---------------------------------------------------------------------------
# Utterance data
# ---------------------------------------------------------------------------

class VADState(str, Enum):
    IDLE = "idle"
    SPEECH = "speech"
    PAUSE = "pause"
    ENDED = "ended"


@dataclass
class AudioUtterance:
    """Complete utterance ready for STT."""
    audio_bytes: bytes
    start_time: float
    end_time: float
    duration_ms: float
    speech_ratio: float
    pause_count: int
    sample_rate: int = 16000
    district: str = "unknown"
    partial_transcript: str = ""
    completeness: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Voice activity detector (main class)
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """
    Async generator that consumes PCM audio frames and yields complete
    AudioUtterance objects when the student finishes speaking.
    """

    def __init__(self, config: VADConfig | None = None):
        self.cfg = config or VADConfig()
        self.vad = SileroVAD(sample_rate=self.cfg.sample_rate)
        self.completeness = MathCompletenessChecker()
        self._frame_bytes = self.cfg.frame_ms * self.cfg.sample_rate * 2 // 1000

    async def listen(self, audio_source=None) -> AsyncIterator[AudioUtterance]:
        """Yield complete utterances from *audio_source* (async iterable of bytes)."""
        if audio_source is None:
            audio_source = self._mic_source()

        buffer: list[bytes] = []
        state = VADState.IDLE
        speech_frames = 0
        pause_count = 0
        pause_start: float | None = None
        utterance_start: float | None = None
        partial_transcript = ""
        extra_wait_applied = False

        async for frame_bytes in audio_source:
            frame = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            prob = self.vad.predict_frame(frame)
            now = time.time()

            if prob >= self.cfg.speech_threshold:
                if state == VADState.IDLE:
                    state = VADState.SPEECH
                    utterance_start = now
                elif state == VADState.PAUSE:
                    state = VADState.SPEECH
                    pause_count += 1
                    extra_wait_applied = False
                buffer.append(frame_bytes)
                speech_frames += 1

            elif state in (VADState.SPEECH, VADState.PAUSE):
                buffer.append(frame_bytes)
                state = VADState.PAUSE
                if pause_start is None:
                    pause_start = now

                elapsed_pause_ms = (now - pause_start) * 1000
                timeout_ms = self._adaptive_timeout(
                    partial_transcript, pause_count, elapsed_pause_ms)

                if not extra_wait_applied:
                    check = self.completeness.check(partial_transcript, elapsed_pause_ms)
                    if not check["is_complete"] and check["recommended_wait_ms"] > 0:
                        timeout_ms = max(timeout_ms, check["recommended_wait_ms"])
                        extra_wait_applied = True

                if elapsed_pause_ms >= timeout_ms:
                    total_frames = len(buffer)
                    ratio = speech_frames / max(total_frames, 1)
                    duration_ms = total_frames * self.cfg.frame_ms

                    if duration_ms >= 300 and speech_frames >= 5:
                        check = self.completeness.check(partial_transcript, elapsed_pause_ms)
                        yield AudioUtterance(
                            audio_bytes=b"".join(buffer),
                            start_time=utterance_start or now,
                            end_time=now,
                            duration_ms=duration_ms,
                            speech_ratio=ratio,
                            pause_count=pause_count,
                            district=self.cfg.district,
                            partial_transcript=partial_transcript,
                            completeness=check,
                        )

                    buffer, speech_frames, pause_count = [], 0, 0
                    pause_start = None
                    utterance_start = None
                    partial_transcript = ""
                    extra_wait_applied = False
                    state = VADState.IDLE
                    self.vad.reset()

            if utterance_start and (time.time() - utterance_start) * 1000 > self.cfg.max_utterance_ms:
                if buffer:
                    yield AudioUtterance(
                        audio_bytes=b"".join(buffer),
                        start_time=utterance_start,
                        end_time=time.time(),
                        duration_ms=len(buffer) * self.cfg.frame_ms,
                        speech_ratio=speech_frames / max(len(buffer), 1),
                        pause_count=pause_count,
                        district=self.cfg.district,
                        partial_transcript=partial_transcript,
                        completeness={"is_complete": True, "confidence": 0.5,
                                       "reasons": ["max_duration_reached"]},
                    )
                buffer, speech_frames, pause_count = [], 0, 0
                pause_start = None
                utterance_start = None
                state = VADState.IDLE
                self.vad.reset()

    def _adaptive_timeout(self, partial_transcript: str,
                           pause_count: int, _current_pause_ms: float) -> int:
        base = self.cfg.pause_long_ms
        if partial_transcript:
            if re.search(r'\b\d+\b', partial_transcript) or \
               any(w in partial_transcript for w in TAMIL_NUMBER_WORDS):
                base = self.cfg.pause_math_ms
        if any(t in partial_transcript for t in QUESTION_TERMINATORS_TA):
            base = min(base, self.cfg.pause_long_ms)
        if pause_count >= 2:
            base = int(base * 1.3)
        return base

    async def _mic_source(self):
        """Fallback: read from local mic via pyaudio (testing only)."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.cfg.channels,
                rate=self.cfg.sample_rate,
                input=True,
                frames_per_buffer=self._frame_bytes // 2,
            )
            try:
                while True:
                    data = stream.read(self._frame_bytes // 2,
                                        exception_on_overflow=False)
                    yield data
                    await asyncio.sleep(0)
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
        except ImportError:
            log.error("pyaudio not installed. Pass audio_source explicitly.")
            return


# ---------------------------------------------------------------------------
# Tamil number normalizer
# ---------------------------------------------------------------------------

class TamilNumberNormalizer:
    """Spoken Tamil number words -> digits, operator words -> symbols."""

    NUMBER_MAP = [
        ("ஆயிரம்", 1000), ("நூறு", 100), ("தொண்ணூறு", 90),
        ("எண்பது", 80), ("எழுபது", 70), ("அறுபது", 60),
        ("ஐம்பது", 50), ("நாற்பது", 40), ("முப்பது", 30),
        ("இருபத்தி ஒன்பது", 29), ("இருபத்தி எட்டு", 28),
        ("இருபத்தி ஏழு", 27), ("இருபத்தி ஆறு", 26),
        ("இருபத்தி ஐந்து", 25), ("இருபத்தி நான்கு", 24),
        ("இருபத்தி மூன்று", 23), ("இருபத்தி இரண்டு", 22),
        ("இருபத்தி ஒன்று", 21), ("இருபது", 20),
        ("பத்தொன்பது", 19), ("பதினெட்டு", 18), ("பதினேழு", 17),
        ("பதினாறு", 16), ("பதினைந்து", 15), ("பதினான்கு", 14),
        ("பதிமூன்று", 13), ("பன்னிரண்டு", 12), ("பதினொன்று", 11),
        ("பத்து", 10), ("ஒன்பது", 9), ("எட்டு", 8), ("ஏழு", 7),
        ("ஆறு", 6), ("ஐந்து", 5), ("நான்கு", 4), ("மூன்று", 3),
        ("இரண்டு", 2), ("ஒன்று", 1),
    ]

    OPERATOR_MAP = {
        "பெருக்கல்": "×", "வகுத்தல்": "÷", "கூட்டல்": "+",
        "கழித்தல்": "-", "சமம்": "=", "சதவீதம்": "%",
    }

    def normalize(self, text: str) -> str:
        result = text
        for word, symbol in self.OPERATOR_MAP.items():
            result = result.replace(word, symbol)
        for word, digit in self.NUMBER_MAP:
            if word in result:
                result = result.replace(word, str(digit))
        return result

    def extract_numbers(self, text: str) -> list[int]:
        normalized = self.normalize(text)
        return [int(n) for n in re.findall(r'\b\d+\b', normalized)
                if int(n) < 1_000_000]
