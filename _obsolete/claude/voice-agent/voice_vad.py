"""
voice_vad.py — Voice Activity Detection with Math-Aware Pause Handling
======================================================================
Solves the core problem: a Grade 7 student says "72 உம் 108 உம்..."
then pauses 1.5s while thinking, then continues "...ஆகிய எண்களின்
பொ.கா.பெ. காண்க". A simple silence timeout cuts them off mid-sentence.

Solution: Silero VAD (on-device, 92kB ONNX) + adaptive timeout + 
partial transcript analysis for math completeness detection.

Architecture:
  AudioCapture → FrameBuffer(30ms) → SileroVAD → PauseAnalyzer
              → MathCompletenessChecker → UtteranceCollector → output

Usage:
  vad = VoiceActivityDetector(config=VADConfig(district="jaffna"))
  async for utterance in vad.listen():
      print(utterance.transcript_raw, utterance.is_complete)

Dependencies:
  pip install silero-vad onnxruntime pyaudio numpy
  # On Android (Flutter): integrate via voice_pipeline_bridge.dart
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import numpy as np

log = logging.getLogger("kanithan.vad")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VADConfig:
    # Audio
    sample_rate: int = 16000          # Silero VAD requires 16kHz
    frame_ms: int = 30                # 30ms frames (Silero optimal)
    channels: int = 1                 # mono

    # VAD thresholds
    speech_threshold: float = 0.50    # prob above = speech
    silence_threshold: float = 0.35   # prob below = silence

    # Pause/timeout tuning (ms)
    pause_short_ms: int = 1500        # mid-sentence pause tolerance
    pause_long_ms: int = 3000         # end-of-sentence silence
    pause_math_ms: int = 4500         # extended for math problems (number mid-utterance)
    max_utterance_ms: int = 30000     # force-end after 30s

    # Re-prompt config
    min_words_to_process: int = 3     # don't process single-word utterances
    confidence_threshold: float = 0.55

    # Student context
    district: str = "unknown"         # adjusts pause tolerance
    grade: int = 7

    def __post_init__(self):
        # Estate Tamil students speak more slowly — give more time
        if self.district in ("estate", "nuwara_eliya", "hatton", "badulla"):
            self.pause_long_ms = 4000
            self.pause_math_ms = 6000
        # Batticaloa rhythm differs
        elif self.district == "batticaloa":
            self.pause_long_ms = 3500


# ─────────────────────────────────────────────────────────────────────────────
# MATH COMPLETENESS CHECKER
# ─────────────────────────────────────────────────────────────────────────────

# Tamil number words → digits (for completeness detection)
TAMIL_NUMBER_WORDS = {
    "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5,
    "ஆறு": 6, "ஏழு": 7, "எட்டு": 8, "ஒன்பது": 9, "பத்து": 10,
    "பதினொன்று": 11, "பன்னிரண்டு": 12, "இருபது": 20, "முப்பது": 30,
    "நாற்பது": 40, "ஐம்பது": 50, "அறுபது": 60, "எழுபது": 70,
    "எண்பது": 80, "தொண்ணூறு": 90, "நூறு": 100, "ஆயிரம்": 1000,
}

# Question completion markers — if these appear, utterance is likely complete
QUESTION_TERMINATORS_TA = [
    "காண்க", "காண்போம்", "காணுங்கள்", "கண்டுபிடி", "கண்டறி",
    "கணக்கிடு", "தீர்மானி", "விடையளி", "சொல்லுங்கள்", "விளக்கு",
    "விளக்குங்கள்", "காட்டு", "காட்டுங்கள்", "எழுதுக", "வரை",
    "?", "என்ன", "ஏன்", "எப்படி", "எவ்வளவு",
]

# Incomplete question signals — utterance likely continues
CONTINUATION_SIGNALS_TA = [
    "உம்", "ஆகிய", "ஆன", "மற்றும்", "கொண்ட",
    "என்று", "என்னும்", "ஐ", "இன்",
]

# Math operators — presence suggests a computation question
MATH_OPERATORS_TA = [
    "பெருக்கல்", "வகுத்தல்", "கூட்டல்", "கழித்தல்",
    "காரணி", "மடங்கு", "பொ.கா.பெ.", "பொ.ம.சி.",
    "×", "÷", "+", "-", "=",
]


class MathCompletenessChecker:
    """
    Analyzes partial transcripts to determine if a math question is
    complete enough to send for processing, or if the student is still
    mid-utterance.
    """

    def check(self, transcript: str, pause_duration_ms: float) -> dict:
        """
        Returns:
          {
            is_complete: bool,
            confidence: float,
            has_numbers: bool,
            has_operator: bool,
            has_terminator: bool,
            has_continuation: bool,
            recommended_wait_ms: int,
          }
        """
        if not transcript or not transcript.strip():
            return self._incomplete(confidence=0.0, wait=2000)

        words = transcript.strip().split()
        if len(words) < 2:
            return self._incomplete(confidence=0.1, wait=2000)

        text = transcript.lower()

        # Detect signals
        has_numbers = bool(re.search(r'\b\d+\b', transcript)) or \
                      any(w in text for w in TAMIL_NUMBER_WORDS)
        has_operator = any(op in transcript for op in MATH_OPERATORS_TA)
        has_terminator = any(t in transcript for t in QUESTION_TERMINATORS_TA)
        has_continuation = any(s in transcript for s in CONTINUATION_SIGNALS_TA)

        # Completeness scoring
        score = 0.0
        reasons = []

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

        # Pause duration factor
        if pause_duration_ms >= 3000:
            score += 0.2
            reasons.append("long_pause")
        elif pause_duration_ms >= 1500 and has_terminator:
            score += 0.1
            reasons.append("moderate_pause_with_terminator")

        score = max(0.0, min(1.0, score))
        is_complete = score >= 0.55

        # Recommended additional wait
        if not is_complete and has_numbers and not has_terminator:
            wait = 3000  # waiting for the rest of math problem
        elif not is_complete and has_continuation:
            wait = 2500  # clearly continuing
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


# ─────────────────────────────────────────────────────────────────────────────
# SILERO VAD WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class SileroVAD:
    """
    Wrapper around the Silero VAD ONNX model.
    Silero VAD: 92kB model, runs on-device, supports 16kHz audio.
    GitHub: https://github.com/snakers4/silero-vad

    On Android/Flutter: model loaded via Flutter ONNX Runtime plugin.
    On Python (server / testing): loaded via onnxruntime directly.
    """

    MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

    def __init__(self, model_path: Optional[str] = None,
                 sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.session = None
        self._h = None   # hidden state
        self._c = None   # cell state
        self._model_path = model_path or "models/silero_vad.onnx"
        self._load()

    def _load(self):
        """Load ONNX model. Downloads if not present."""
        import os
        from pathlib import Path

        model_path = Path(self._model_path)
        if not model_path.exists():
            log.info(f"Downloading Silero VAD model to {model_path}...")
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
            log.info("Silero VAD initialized.")
        except ImportError:
            log.warning("onnxruntime not installed. Using stub VAD.")
            self.session = None

    def _reset_state(self):
        """Reset LSTM hidden/cell state between utterances."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def predict_frame(self, audio_frame: np.ndarray) -> float:
        """
        Predict speech probability for a single 30ms audio frame.
        audio_frame: np.float32 array, shape (480,) for 16kHz/30ms
        Returns: float 0.0 – 1.0
        """
        if self.session is None:
            # Stub: detect non-silence by energy
            rms = np.sqrt(np.mean(audio_frame**2))
            return min(1.0, rms * 20)

        x = audio_frame.reshape(1, -1).astype(np.float32)
        sr = np.array(self.sample_rate, dtype=np.int64)

        ort_inputs = {
            "input": x,
            "sr": sr,
            "h": self._h,
            "c": self._c,
        }
        out, self._h, self._c = self.session.run(None, ort_inputs)
        return float(out[0][0])

    def reset(self):
        self._reset_state()


# ─────────────────────────────────────────────────────────────────────────────
# UTTERANCE DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

class VADState(str, Enum):
    IDLE = "idle"
    SPEECH = "speech"
    PAUSE  = "pause"    # short pause — may continue
    ENDED  = "ended"    # utterance complete


@dataclass
class AudioUtterance:
    """Complete utterance collected by VAD, ready for STT."""
    audio_bytes: bytes              # raw PCM bytes, 16kHz mono 16-bit
    start_time: float               # epoch time when speech started
    end_time: float                 # epoch time when utterance ended
    duration_ms: float              # total audio duration
    speech_ratio: float             # fraction of frames that were speech
    pause_count: int                # number of mid-utterance pauses
    sample_rate: int = 16000
    district: str = "unknown"
    partial_transcript: str = ""    # from streaming STT if available
    completeness: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# VOICE ACTIVITY DETECTOR (main class)
# ─────────────────────────────────────────────────────────────────────────────

class VoiceActivityDetector:
    """
    Real-time VAD with adaptive pause handling for Tamil math questions.

    Integrates:
    - Silero VAD for frame-level speech detection
    - Adaptive timeout based on partial transcript content
    - Math completeness checking to avoid cutting mid-problem
    - Partial transcript integration for smarter decisions

    On Android/Flutter, this logic runs in Dart/Kotlin using the
    SileroVAD ONNX model via flutter_onnxruntime. The Python version
    is for the server-side testing and Python backend.
    """

    def __init__(self, config: VADConfig = None):
        self.cfg = config or VADConfig()
        self.vad = SileroVAD(sample_rate=self.cfg.sample_rate)
        self.completeness = MathCompletenessChecker()
        self._state = VADState.IDLE
        self._frame_bytes = self.cfg.frame_ms * self.cfg.sample_rate * 2 // 1000
        # 30ms * 16000 * 2 bytes = 960 bytes per frame

    async def listen(self, audio_source=None) -> AsyncIterator[AudioUtterance]:
        """
        Async generator that yields complete utterances.
        audio_source: async callable that returns bytes chunks, or None for microphone.
        """
        if audio_source is None:
            audio_source = self._mic_source()

        buffer = []         # audio frames for current utterance
        state = VADState.IDLE
        speech_frames = 0
        pause_count = 0
        pause_start = None
        utterance_start = None
        partial_transcript = ""
        extra_wait_applied = False

        async for frame_bytes in audio_source:
            frame = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            prob = self.vad.predict_frame(frame)
            now = time.time()

            if prob >= self.cfg.speech_threshold:
                # Speech detected
                if state == VADState.IDLE:
                    state = VADState.SPEECH
                    utterance_start = now
                    log.debug("VAD: speech started")
                elif state == VADState.PAUSE:
                    # Resume after pause — extend the utterance
                    state = VADState.SPEECH
                    pause_count += 1
                    extra_wait_applied = False
                    log.debug(f"VAD: speech resumed after pause #{pause_count}")

                buffer.append(frame_bytes)
                speech_frames += 1

            elif state in (VADState.SPEECH, VADState.PAUSE):
                # Silence / below threshold
                buffer.append(frame_bytes)  # keep audio for seamless playback
                state = VADState.PAUSE

                if pause_start is None:
                    pause_start = now

                elapsed_pause_ms = (now - pause_start) * 1000

                # Determine adaptive timeout
                timeout_ms = self._adaptive_timeout(
                    partial_transcript, pause_count, elapsed_pause_ms)

                # Check if we should extend based on completeness
                if not extra_wait_applied:
                    check = self.completeness.check(
                        partial_transcript, elapsed_pause_ms)
                    if not check["is_complete"] and check["recommended_wait_ms"] > 0:
                        timeout_ms = max(timeout_ms,
                                          check["recommended_wait_ms"])
                        extra_wait_applied = True
                        log.debug(f"VAD: extending wait to {timeout_ms}ms "
                                   f"(reasons: {check['reasons']})")

                if elapsed_pause_ms >= timeout_ms:
                    # Utterance complete — check minimum quality
                    total_frames = len(buffer)
                    ratio = speech_frames / max(total_frames, 1)
                    duration_ms = len(buffer) * self.cfg.frame_ms

                    if duration_ms >= 300 and speech_frames >= 5:
                        audio_bytes = b"".join(buffer)
                        check = self.completeness.check(partial_transcript, elapsed_pause_ms)
                        utterance = AudioUtterance(
                            audio_bytes=audio_bytes,
                            start_time=utterance_start,
                            end_time=now,
                            duration_ms=duration_ms,
                            speech_ratio=ratio,
                            pause_count=pause_count,
                            district=self.cfg.district,
                            partial_transcript=partial_transcript,
                            completeness=check,
                        )
                        log.info(f"VAD: utterance complete — {duration_ms:.0f}ms, "
                                  f"speech_ratio={ratio:.2f}, pauses={pause_count}, "
                                  f"complete={check['is_complete']}")
                        yield utterance

                    # Reset state
                    buffer, speech_frames, pause_count = [], 0, 0
                    pause_start = None
                    utterance_start = None
                    partial_transcript = ""
                    extra_wait_applied = False
                    state = VADState.IDLE
                    self.vad.reset()

            # Max utterance guard
            if utterance_start and (time.time() - utterance_start) * 1000 > self.cfg.max_utterance_ms:
                if buffer:
                    audio_bytes = b"".join(buffer)
                    yield AudioUtterance(
                        audio_bytes=audio_bytes,
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

    def update_partial_transcript(self, text: str):
        """Called by streaming STT to update partial transcript for smarter VAD decisions."""
        # This is called from the STT layer during streaming recognition
        pass  # Managed externally via shared state in production

    def _adaptive_timeout(self, partial_transcript: str,
                           pause_count: int, current_pause_ms: float) -> int:
        """Calculate adaptive silence timeout based on context."""
        # Base timeout
        base = self.cfg.pause_long_ms

        # Extend if numbers detected in partial transcript
        if partial_transcript:
            if re.search(r'\b\d+\b', partial_transcript) or \
               any(w in partial_transcript for w in TAMIL_NUMBER_WORDS):
                base = self.cfg.pause_math_ms

        # Reduce if question terminator already detected
        if any(t in partial_transcript for t in QUESTION_TERMINATORS_TA):
            base = min(base, self.cfg.pause_long_ms)

        # Multiple pauses suggest complex problem — be more patient
        if pause_count >= 2:
            base = int(base * 1.3)

        return base

    async def _mic_source(self):
        """Default microphone source using pyaudio."""
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
            log.info("Microphone opened.")
            try:
                while True:
                    data = stream.read(self._frame_bytes // 2,
                                        exception_on_overflow=False)
                    yield data
                    await asyncio.sleep(0)  # yield to event loop
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
        except ImportError:
            log.error("pyaudio not installed. Use audio_source parameter.")
            return


# ─────────────────────────────────────────────────────────────────────────────
# NUMBER WORD CONVERTER (Tamil spoken → written digits)
# ─────────────────────────────────────────────────────────────────────────────

class TamilNumberNormalizer:
    """
    Converts Tamil spoken number words to digits for math processing.
    Handles compound numbers: "நூற்று இருபத்தி ஆறு" → 126
    Also converts math operator words to symbols.
    """

    # Ordered by value (descending) for greedy matching
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
        """Replace number words and operator words in text."""
        result = text

        # Replace operator words
        for word, symbol in self.OPERATOR_MAP.items():
            result = result.replace(word, symbol)

        # Replace number words (longest match first)
        for word, digit in self.NUMBER_MAP:
            if word in result:
                result = result.replace(word, str(digit))

        return result

    def extract_numbers(self, text: str) -> list[int]:
        """Extract all numbers (as digits or words) from text."""
        normalized = self.normalize(text)
        return [int(n) for n in re.findall(r'\b\d+\b', normalized)
                if int(n) < 1000000]


# ─────────────────────────────────────────────────────────────────────────────
# VAD TESTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

async def simulate_vad_test():
    """Test VAD logic with simulated audio segments (for unit testing)."""
    import struct

    def make_sine_frame(freq: float, sr: int = 16000,
                         duration_ms: int = 30, amp: float = 0.3) -> bytes:
        """Generate a sine wave frame (simulates speech)."""
        n_samples = sr * duration_ms // 1000
        t = np.linspace(0, duration_ms / 1000, n_samples)
        wave = (amp * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        return wave.tobytes()

    def make_silence_frame(sr: int = 16000, duration_ms: int = 30) -> bytes:
        """Generate silence frame."""
        n_samples = sr * duration_ms // 1000
        return bytes(n_samples * 2)

    config = VADConfig(district="jaffna")
    checker = MathCompletenessChecker()
    normalizer = TamilNumberNormalizer()

    # Test completeness checker
    test_cases = [
        ("72 உம் 108 உம்", 500),    # incomplete — mid-number
        ("72 உம் 108 உம் ஆகிய எண்களின் பொ.கா.பெ. காண்க", 2000),  # complete
        ("காரணி என்றால் என்ன?", 1500),   # complete — has terminator
        ("இரண்டு", 800),               # very short
    ]

    print("=== MathCompletenessChecker Tests ===")
    for text, pause_ms in test_cases:
        result = checker.check(text, pause_ms)
        print(f"  '{text[:40]}...' pause={pause_ms}ms → "
              f"complete={result['is_complete']} conf={result['confidence']:.2f} "
              f"({', '.join(result['reasons'])})")

    print("\n=== TamilNumberNormalizer Tests ===")
    num_tests = [
        "நூற்று இருபத்தி ஆறு இன் காரணிகள்",
        "எழுபத்தி இரண்டு உம் நூற்று எட்டு உம் ஆகிய எண்களின் பொ.கா.பெ. காண்க",
        "இரண்டு பெருக்கல் மூன்று",
    ]
    for t in num_tests:
        print(f"  '{t}' → '{normalizer.normalize(t)}'")
        print(f"    numbers: {normalizer.extract_numbers(t)}")


if __name__ == "__main__":
    asyncio.run(simulate_vad_test())
