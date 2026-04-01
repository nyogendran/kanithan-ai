"""
Pluggable Speech-to-Text pipeline for Tamil math tutoring.

Backends (selected via STT_BACKEND env var):
  - gemini   (default) — uses existing GEMINI_API_KEY, no extra credentials
  - google_cloud       — Google Cloud Speech v2 chirp_2 model
  - whisper            — OpenAI Whisper offline fallback

Includes dialect detection, math normalization, and completeness checking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .vad import AudioUtterance, MathCompletenessChecker

log = logging.getLogger("kanithan.stt")

STT_BACKEND: str = os.environ.get("STT_BACKEND", "gemini").strip().lower()

# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

class Dialect(str, Enum):
    JAFFNA = "jaffna"
    BATTICALOA = "batticaloa"
    ESTATE = "estate"
    COLOMBO = "colombo"
    VANNI = "vanni"
    UNKNOWN = "unknown"


DIALECT_LEXICAL_MARKERS: dict[Dialect, list[str]] = {
    Dialect.JAFFNA: ["விளக்கு", "காண்போம்", "சொல்லுங்கள்", "என்பது", "ஆகும்"],
    Dialect.BATTICALOA: ["வகுத்தல்னா", "என்னன்னு", "போடு", "இருக்கு"],
    Dialect.ESTATE: [
        "பண்ணுவது", "பண்ற", "சொல்லுங்க", "இதுக்கு", "அதுக்கு",
        "எப்படி பண்றது", "வகுத்தல்க்கு",
    ],
    Dialect.COLOMBO: [
        "factor", "HCF", "LCM", "find", "calculate", "method",
        "answer", "how to", "what is",
    ],
    Dialect.VANNI: ["கண்டுபிடிப்பது", "எப்படி", "காண்பது"],
}

DIALECT_TO_NIE_MAP: dict[str, str] = {
    "பண்ணுவது": "செய்வது",
    "பண்ற": "செய்கின்ற",
    "சொல்லுங்க": "சொல்லுங்கள்",
    "இதுக்கு": "இதற்கு",
    "அதுக்கு": "அதற்கு",
    "எப்படி பண்றது": "எப்படி செய்வது",
    "வகுத்தல்க்கு": "வகுத்தல் மூலம்",
    "factor காண்க": "காரணி காண்க",
    "HCF காண்க": "பொ.கா.பெ. காண்க",
    "LCM காண்க": "பொ.ம.சி. காண்க",
    "find பண்க": "காண்க",
    "calculate பண்க": "கணக்கிடு",
    "factor ஆனது": "காரணி ஆகும்",
    "method காட்டு": "முறை காட்டு",
    "answer என்ன": "விடை என்ன",
    "வகுத்தல்னா": "வகுத்தல் என்றால்",
    "என்னன்னு": "என்னவென்று",
    "போடு": "எழுதுக",
}


class DialectDetector:
    """District-based + lexical dialect detection and normalization."""

    _DISTRICT_MAP: dict[str, Dialect] = {
        "jaffna": Dialect.JAFFNA, "kilinochchi": Dialect.JAFFNA,
        "mullaitivu": Dialect.JAFFNA, "mannar": Dialect.VANNI,
        "vavuniya": Dialect.VANNI,
        "batticaloa": Dialect.BATTICALOA, "ampara": Dialect.BATTICALOA,
        "trincomalee": Dialect.BATTICALOA,
        "nuwara_eliya": Dialect.ESTATE, "hatton": Dialect.ESTATE,
        "badulla": Dialect.ESTATE, "kandy": Dialect.ESTATE,
        "matale": Dialect.ESTATE,
        "colombo": Dialect.COLOMBO, "gampaha": Dialect.COLOMBO,
        "kalutara": Dialect.COLOMBO,
    }

    def detect(self, text: str, district: str = "unknown",
                alternatives: list[str] | None = None) -> tuple[Dialect, float]:
        if district.lower() in self._DISTRICT_MAP:
            return self._DISTRICT_MAP[district.lower()], 0.90

        all_texts = [text] + (alternatives or [])
        combined = " ".join(all_texts).lower()

        scores: dict[Dialect, int] = {d: 0 for d in Dialect if d != Dialect.UNKNOWN}
        for dialect, markers in DIALECT_LEXICAL_MARKERS.items():
            for marker in markers:
                if marker.lower() in combined:
                    scores[dialect] += 1

        if not any(scores.values()):
            return Dialect.UNKNOWN, 0.3

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.3
        return best, min(confidence, 0.85)

    def normalize(self, text: str, _dialect: Dialect) -> str:
        result = text
        sorted_replacements = sorted(
            DIALECT_TO_NIE_MAP.items(), key=lambda x: len(x[0]), reverse=True)
        for variant, standard in sorted_replacements:
            result = result.replace(variant, standard)
        return result


# ---------------------------------------------------------------------------
# Math text normalizer (ASR error correction)
# ---------------------------------------------------------------------------

NIE_MATH_ADAPTATION_PHRASES = [
    "காரணி", "காரணிகள்", "மடங்கு", "மடங்குகள்",
    "இலக்கச் சுட்டி", "முதன்மை எண்", "முதன்மைக் காரணி",
    "பொ.கா.பெ.", "பொதுக் காரணிகளுட் பெரியது",
    "பொ.ம.சி.", "பொது மடங்குகளுட் சிறியது",
    "வகுபடும்", "வகுபடாது", "மீதி",
    "காரணி மரம்", "வகுத்தல் ஏணி",
]


class MathTextNormalizer:
    """Fix ASR mistakes on math terms, numbers, and operators."""

    COMPOUND_NUMBERS = [
        ("நூற்று இருபத்தி ஆறு", "126"),
        ("நூற்று இருபத்தி", "120"),
        ("நூற்று எட்டு", "108"),
        ("எழுபத்தி இரண்டு", "72"),
        ("எட்டு பத்து நான்கு", "84"),
    ]

    NUMBER_WORDS = [
        ("ஆயிரம்", "1000"), ("நூறு", "100"), ("தொண்ணூறு", "90"),
        ("எண்பது", "80"), ("எழுபது", "70"), ("அறுபது", "60"),
        ("ஐம்பது", "50"), ("நாற்பது", "40"), ("முப்பது", "30"),
        ("இருபது", "20"), ("பத்து", "10"), ("ஒன்பது", "9"),
        ("எட்டு", "8"), ("ஏழு", "7"), ("ஆறு", "6"), ("ஐந்து", "5"),
        ("நான்கு", "4"), ("மூன்று", "3"), ("இரண்டு", "2"), ("ஒன்று", "1"),
    ]

    OPERATOR_WORDS = {
        "பெருக்கல்": "×", "வகுத்தல்": "÷",
        "கூட்டல்": "+", "கழித்தல்": "-",
    }

    ABBREVIATION_FIXES = {
        "HCF": "பொ.கா.பெ.", "GCD": "பொ.கா.பெ.",
        "LCM": "பொ.ம.சி.", "LCF": "பொ.ம.சி.",
        "factor": "காரணி", "factors": "காரணிகள்",
        "multiple": "மடங்கு", "prime": "முதன்மை",
    }

    def normalize(self, text: str) -> tuple[str, list[int]]:
        result = text
        for wrong, right in self.ABBREVIATION_FIXES.items():
            result = re.sub(r'\b' + wrong + r'\b', right, result, flags=re.IGNORECASE)
        for phrase, digit in self.COMPOUND_NUMBERS:
            result = result.replace(phrase, digit)
        for word, symbol in self.OPERATOR_WORDS.items():
            result = result.replace(word, symbol)
        for word, digit in self.NUMBER_WORDS:
            result = result.replace(word, digit)
        numbers = [int(n) for n in re.findall(r'\b\d+\b', result) if int(n) < 1_000_000]
        return result, numbers


# ---------------------------------------------------------------------------
# STT result
# ---------------------------------------------------------------------------

@dataclass
class STTResult:
    raw_text: str
    normalized_text: str
    dialect: Dialect
    dialect_confidence: float
    stt_confidence: float
    language_detected: str
    numbers_extracted: list[int]
    math_operators: list[str]
    is_math_complete: bool
    alternative_texts: list[str]
    processing_ms: float
    used_offline: bool = False


# ---------------------------------------------------------------------------
# STT backend: Gemini (default)
# ---------------------------------------------------------------------------

@dataclass
class STTConfig:
    primary_language: str = "ta-IN"
    alternative_languages: list = field(
        default_factory=lambda: ["si-LK", "en-IN"])
    model: str = "chirp_2"
    enable_automatic_punctuation: bool = True
    enable_word_time_offsets: bool = True
    max_alternatives: int = 3
    whisper_model: str = "small"
    use_offline_threshold_ms: float = 3000.0
    enable_speech_adaptation: bool = True
    district: str = "unknown"
    grade: int = 7


class GeminiSTT:
    """Transcribe audio via Gemini generate_content (uses existing API key)."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def transcribe(self, audio_bytes: bytes,
                          mime_type: str = "audio/wav") -> dict:
        t0 = time.perf_counter()
        try:
            from google.genai import types
            client = self._get_client()
            model = os.environ.get("GEMINI_TEACHING_MODEL", "gemini-2.5-flash")

            prompt = (
                "இது ஒரு தமிழ் மாணவரின் கணிதக் கேள்வி. "
                "ஒலிப்பதிவை துல்லியமாகத் தமிழில் எழுத்துருவாக்கம் செய்யுங்கள். "
                "எண்களை எப்போதும் இலக்கங்களாக எழுதுங்கள் (எ.கா. 24, 15, 30, 6). "
                "கணிதக் குறியீடுகள் (× ÷ =) போல் அப்படியே. "
                "வேறு எதுவும் சேர்க்காதீர்கள் — transcript மட்டும் போதும்."
            )
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            text_part = types.Part(text=prompt)
            cfg = types.GenerateContentConfig(temperature=0.1, max_output_tokens=300)
            contents = [types.Content(parts=[audio_part, text_part])]

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model, contents=contents, config=cfg),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            text = (response.text or "").strip()
            return {
                "success": bool(text),
                "transcript": text,
                "confidence": 0.85 if text else 0.0,
                "alternatives": [],
                "language": "ta-IN",
                "processing_ms": elapsed,
            }
        except Exception as e:
            log.error("Gemini STT error: %s", e)
            return {
                "success": False, "transcript": "", "confidence": 0.0,
                "alternatives": [], "language": "ta-IN",
                "processing_ms": (time.perf_counter() - t0) * 1000,
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# STT backend: Google Cloud Speech v2
# ---------------------------------------------------------------------------

class GoogleCloudSTT:
    """Google Cloud Speech-to-Text v2 with chirp_2 + math adaptation."""

    def __init__(self, config: STTConfig | None = None):
        self.cfg = config or STTConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import speech_v2 as speech
                self._client = speech.SpeechClient()
            except ImportError:
                raise ImportError("Install: pip install google-cloud-speech")
        return self._client

    async def transcribe(self, audio_bytes: bytes,
                          context_topic: str | None = None) -> dict:
        t0 = time.perf_counter()
        try:
            from google.cloud import speech_v2 as speech
            from google.cloud.speech_v2.types import cloud_speech

            client = self._get_client()
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")

            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=[self.cfg.primary_language] + self.cfg.alternative_languages,
                model=self.cfg.model,
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=True,
                    max_alternatives=self.cfg.max_alternatives,
                ),
                adaptation=cloud_speech.SpeechAdaptation(
                    phrase_sets=[
                        cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                            inline_phrase_set=cloud_speech.PhraseSet(
                                phrases=[
                                    cloud_speech.PhraseSet.Phrase(value=p, boost=15.0)
                                    for p in NIE_MATH_ADAPTATION_PHRASES[:200]
                                ]
                            )
                        )
                    ]
                ) if self.cfg.enable_speech_adaptation else None,
            )

            recognizer = f"projects/{project_id}/locations/global/recognizers/_"
            request = cloud_speech.RecognizeRequest(
                recognizer=recognizer, config=config, content=audio_bytes)

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.recognize(request=request))

            results = []
            for r in response.results:
                if r.alternatives:
                    best = r.alternatives[0]
                    alts = [a.transcript for a in r.alternatives[1:]]
                    results.append({
                        "transcript": best.transcript,
                        "confidence": best.confidence,
                        "alternatives": alts,
                        "language": r.language_code,
                    })

            elapsed = (time.perf_counter() - t0) * 1000
            if results:
                return {
                    "success": True,
                    "transcript": " ".join(r["transcript"] for r in results),
                    "confidence": results[0]["confidence"],
                    "alternatives": results[0]["alternatives"],
                    "language": results[0].get("language", "ta-IN"),
                    "processing_ms": elapsed,
                }
            return {"success": False, "transcript": "", "confidence": 0.0,
                    "alternatives": [], "language": "ta-IN", "processing_ms": elapsed}
        except Exception as e:
            log.error("Google Cloud STT error: %s", e)
            return {"success": False, "transcript": "", "confidence": 0.0,
                    "alternatives": [], "language": "ta-IN",
                    "processing_ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------------------
# STT backend: Whisper (offline)
# ---------------------------------------------------------------------------

class WhisperOfflineSTT:
    """OpenAI Whisper for offline Tamil STT."""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import whisper
                log.info("Loading Whisper '%s' model ...", self.model_size)
                self._model = whisper.load_model(self.model_size)
            except ImportError:
                raise ImportError("Install: pip install openai-whisper")
        return self._model

    async def transcribe(self, audio_bytes: bytes,
                          language: str = "ta") -> dict:
        import tempfile
        import wave
        t0 = time.perf_counter()
        try:
            model = self._load()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wf = wave.open(f, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
                wf.close()
                tmp_path = f.name

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.transcribe(
                    tmp_path, language=language, task="transcribe",
                    initial_prompt="காரணி மடங்கு இலக்கச் சுட்டி பொ.கா.பெ. பொ.ம.சி.",
                    fp16=False,
                ),
            )
            os.unlink(tmp_path)

            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "success": True,
                "transcript": result["text"].strip(),
                "confidence": 0.75,
                "alternatives": [],
                "language": result.get("language", "ta"),
                "processing_ms": elapsed,
                "used_offline": True,
            }
        except Exception as e:
            log.error("Whisper STT error: %s", e)
            return {"success": False, "transcript": "", "confidence": 0.0,
                    "alternatives": [], "language": "ta",
                    "processing_ms": (time.perf_counter() - t0) * 1000,
                    "used_offline": True}


# ---------------------------------------------------------------------------
# Main STT pipeline
# ---------------------------------------------------------------------------

class TamilSTTPipeline:
    """
    Full pipeline: audio -> STT backend -> dialect detection ->
    math normalization -> completeness check -> STTResult.
    """

    def __init__(self, config: STTConfig | None = None, backend: str | None = None):
        self.cfg = config or STTConfig()
        self._backend_name = backend or STT_BACKEND

        if self._backend_name == "google_cloud":
            self._backend = GoogleCloudSTT(self.cfg)
        elif self._backend_name == "whisper":
            self._backend = WhisperOfflineSTT(self.cfg.whisper_model)
        else:
            self._backend = GeminiSTT()

        self.dialect_detector = DialectDetector()
        self.math_normalizer = MathTextNormalizer()
        self.completeness_checker = MathCompletenessChecker()

    async def transcribe(self, utterance: AudioUtterance,
                          context_topic: str | None = None) -> STTResult:
        t0 = time.perf_counter()

        stt_raw = await self._backend.transcribe(utterance.audio_bytes)

        if not stt_raw.get("success") or not stt_raw.get("transcript"):
            return STTResult(
                raw_text="", normalized_text="",
                dialect=Dialect.UNKNOWN, dialect_confidence=0.0,
                stt_confidence=0.0, language_detected="ta-IN",
                numbers_extracted=[], math_operators=[],
                is_math_complete=False, alternative_texts=[],
                processing_ms=(time.perf_counter() - t0) * 1000,
                used_offline=stt_raw.get("used_offline", False),
            )

        raw_text = stt_raw["transcript"]
        alternatives = stt_raw.get("alternatives", [])

        dialect, dialect_conf = self.dialect_detector.detect(
            raw_text, self.cfg.district, alternatives)

        normalized = self.dialect_detector.normalize(raw_text, dialect)
        normalized, numbers = self.math_normalizer.normalize(normalized)

        operators = [sym for sym in ["×", "÷", "+", "-", "="] if sym in normalized]
        completeness = self.completeness_checker.check(
            normalized, pause_duration_ms=utterance.duration_ms)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("STT pipeline: %.0fms — '%s' complete=%s",
                 elapsed, normalized[:60], completeness["is_complete"])

        return STTResult(
            raw_text=raw_text,
            normalized_text=normalized,
            dialect=dialect,
            dialect_confidence=dialect_conf,
            stt_confidence=stt_raw.get("confidence", 0.7),
            language_detected=stt_raw.get("language", "ta-IN"),
            numbers_extracted=numbers,
            math_operators=operators,
            is_math_complete=completeness["is_complete"],
            alternative_texts=alternatives,
            processing_ms=elapsed,
            used_offline=stt_raw.get("used_offline", False),
        )

    async def transcribe_raw(self, audio_bytes: bytes,
                              duration_ms: float = 0.0) -> STTResult:
        """Convenience: transcribe raw audio bytes without a full AudioUtterance."""
        utterance = AudioUtterance(
            audio_bytes=audio_bytes,
            start_time=time.time(),
            end_time=time.time(),
            duration_ms=duration_ms,
            speech_ratio=1.0,
            pause_count=0,
        )
        return await self.transcribe(utterance)
