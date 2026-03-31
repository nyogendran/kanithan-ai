"""
voice_stt_tts.py — Speech-to-Text + Dialect Detection + Text-to-Speech
=======================================================================
Covers:
  1. STT: Google Cloud Speech-to-Text v2 (cloud) + Whisper.cpp (offline)
  2. Dialect detection from audio features + lexical signals
  3. Math term normalization: "factor காண்க" → "காரணி காண்க"
  4. TTS: Google Cloud TTS with SSML math pronunciation
  5. Offline TTS fallback via Android TTS API

Dependencies:
  pip install google-cloud-speech google-cloud-texttospeech
  pip install openai-whisper  # for offline fallback
  pip install langdetect

Usage:
  stt = TamilSTTPipeline(config=STTConfig(district="batticaloa"))
  result = await stt.transcribe(utterance)
  print(result.normalized_text, result.dialect)

  tts = TamilTTSPipeline()
  audio = await tts.synthesize("72 இன் காரணி மரம் காண்போம்")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("nie.stt_tts")

# ─────────────────────────────────────────────────────────────────────────────
# STT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class STTConfig:
    # Language
    primary_language: str = "ta-IN"       # Tamil (India/Sri Lanka)
    alternative_languages: list = field(
        default_factory=lambda: ["si-LK", "en-IN"])  # Sinhala, English fallback

    # Google Cloud STT v2
    model: str = "chirp_2"                # Best multilingual model (2024+)
    enable_automatic_punctuation: bool = True
    enable_word_time_offsets: bool = True  # for partial transcript sync
    max_alternatives: int = 3             # top-3 hypotheses for dialect detection

    # Offline fallback
    whisper_model: str = "small"          # ~300MB, good Tamil accuracy
    use_offline_threshold_ms: float = 3000.0  # network RTT threshold

    # Math-specific
    enable_speech_adaptation: bool = True  # inject NIE math vocab

    # Student context
    district: str = "unknown"
    grade: int = 7


# ─────────────────────────────────────────────────────────────────────────────
# SPEECH ADAPTATION PHRASES (NIE math vocabulary)
# ─────────────────────────────────────────────────────────────────────────────
# These are injected into Google STT as "hints" so the ASR model
# recognises math terms correctly instead of transcribing them as
# similar-sounding common Tamil words.

NIE_MATH_ADAPTATION_PHRASES = [
    # Core Chapter 4 terms
    "காரணி", "காரணிகள்", "மடங்கு", "மடங்குகள்",
    "இலக்கச் சுட்டி", "முதன்மை எண்", "முதன்மைக் காரணி",
    "போ.கா.பெ.", "பொதுக் காரணிகளுட் பெரியது",
    "போ.ம.சி.", "பொது மடங்குகளுட் சிறியது",
    "வகுபடும்", "வகுபடாது", "மீதி",
    "காரணி மரம்", "வகுத்தல் ஏணி",
    # Numbers
    "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து",
    "ஆறு", "ஏழு", "எட்டு", "ஒன்பது", "பத்து",
    "நூறு", "ஆயிரம்",
    # Operators
    "பெருக்கல்", "வகுத்தல்", "கூட்டல்", "கழித்தல்",
    # Question words
    "காண்க", "காண்போம்", "கண்டுபிடி", "விளக்கு", "வரை",
    # NIE section names
    "பயிற்சி", "உதாரணம்", "செயற்பாடு", "சிந்தனைக்கு",
]

# Phrase boosts by context
CONTEXT_PHRASE_BOOSTS = {
    "hcf": ["போ.கா.பெ.", "பொதுக் காரணி", "வகுத்தல் முறை"],
    "lcm": ["போ.ம.சி.", "பொது மடங்கு", "மடங்கு கோடு"],
    "prime": ["முதன்மை எண்", "முதன்மைக் காரணி", "காரணி மரம்"],
    "divisibility": ["இலக்கச் சுட்டி", "வகுபடும்", "மீதி"],
}


# ─────────────────────────────────────────────────────────────────────────────
# DIALECT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class Dialect(str, Enum):
    JAFFNA     = "jaffna"
    BATTICALOA = "batticaloa"
    ESTATE     = "estate"
    COLOMBO    = "colombo"
    VANNI      = "vanni"
    UNKNOWN    = "unknown"


# Lexical markers unique to each dialect (from audio transcript)
DIALECT_LEXICAL_MARKERS = {
    Dialect.JAFFNA: [
        "விளக்கு", "காண்போம்", "சொல்லுங்கள்", "என்பது", "ஆகும்",
    ],
    Dialect.BATTICALOA: [
        "வகுத்தல்னா", "என்னன்னு", "போடு", "இருக்கு",
    ],
    Dialect.ESTATE: [
        "பண்ணுவது", "பண்ற", "சொல்லுங்க", "இதுக்கு", "அதுக்கு",
        "எப்படி பண்றது", "வகுத்தல்க்கு",
    ],
    Dialect.COLOMBO: [
        "factor", "HCF", "LCM", "find", "calculate", "method",
        "answer", "how to", "what is",
    ],
    Dialect.VANNI: [
        "கண்டுபிடிப்பது", "எப்படி", "காண்பது",
    ],
}

# Vocabulary normalization: dialect word → NIE standard
DIALECT_TO_NIE_MAP = {
    # Estate Tamil
    "பண்ணுவது": "செய்வது",
    "பண்ற": "செய்கின்ற",
    "சொல்லுங்க": "சொல்லுங்கள்",
    "இதுக்கு": "இதற்கு",
    "அதுக்கு": "அதற்கு",
    "எப்படி பண்றது": "எப்படி செய்வது",
    "வகுத்தல்க்கு": "வகுத்தல் மூலம்",
    # Colombo code-switching
    "factor காண்க": "காரணி காண்க",
    "HCF காண்க": "போ.கா.பெ. காண்க",
    "LCM காண்க": "போ.ம.சி. காண்க",
    "find பண்க": "காண்க",
    "calculate பண்க": "கணக்கிடு",
    "factor ஆனது": "காரணி ஆகும்",
    "method காட்டு": "முறை காட்டு",
    "answer என்ன": "விடை என்ன",
    # Batticaloa
    "வகுத்தல்னா": "வகுத்தல் என்றால்",
    "என்னன்னு": "என்னவென்று",
    "போடு": "எழுதுக",
}


# ─────────────────────────────────────────────────────────────────────────────
# STT RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class STTResult:
    raw_text: str               # direct ASR output
    normalized_text: str        # after dialect + math normalization
    dialect: Dialect
    dialect_confidence: float
    stt_confidence: float
    language_detected: str      # "ta-IN", "si-LK", etc.
    numbers_extracted: list[int]
    math_operators: list[str]
    is_math_complete: bool
    alternative_texts: list[str]  # other ASR hypotheses
    processing_ms: float
    used_offline: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# DIALECT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class DialectDetector:
    """
    Two-stage dialect detection:
    Stage 1: District lookup (fast, from student profile)
    Stage 2: Lexical marker analysis on transcript
    Stage 3: (Production) Audio feature analysis — pitch, formants, rhythm
    """

    def detect(self, text: str, district: str = "unknown",
                alternatives: list[str] = None) -> tuple[Dialect, float]:
        """Returns (dialect, confidence)."""

        # Stage 1: District override (most reliable signal)
        district_map = {
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
        if district.lower() in district_map:
            return district_map[district.lower()], 0.90

        # Stage 2: Lexical marker analysis
        all_texts = [text] + (alternatives or [])
        combined = " ".join(all_texts).lower()

        scores = {d: 0 for d in Dialect if d != Dialect.UNKNOWN}
        for dialect, markers in DIALECT_LEXICAL_MARKERS.items():
            for marker in markers:
                if marker.lower() in combined:
                    scores[dialect] += 1

        if not any(scores.values()):
            return Dialect.UNKNOWN, 0.3

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.3
        return best, min(confidence, 0.85)

    def normalize(self, text: str, dialect: Dialect) -> str:
        """Apply dialect → NIE standard normalization."""
        result = text
        # Apply longest match first
        sorted_replacements = sorted(
            DIALECT_TO_NIE_MAP.items(),
            key=lambda x: len(x[0]), reverse=True)
        for variant, standard in sorted_replacements:
            result = result.replace(variant, standard)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# MATH TEXT NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

class MathTextNormalizer:
    """
    Converts math expressions to standardized text after STT.
    Handles:
    - Number word → digit: "நூற்று இருபத்தி ஆறு" → "126"
    - ASR mistakes on numbers: "72 உம் ஒன்று ஆறு" → "72 உம் 16"
    - Operator normalization: "பெருக்கல்" → "×"
    - NIE abbreviation expansion: "போ.கா.பெ." preserved, "HCF" → "போ.கா.பெ."
    """

    # In order of length (longest first for greedy match)
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
        "HCF": "போ.கா.பெ.", "GCD": "போ.கா.பெ.",
        "LCM": "போ.ம.சி.", "LCF": "போ.ம.சி.",  # common ASR mishearing
        "factor": "காரணி", "factors": "காரணிகள்",
        "multiple": "மடங்கு",
        "prime": "முதன்மை",
    }

    def normalize(self, text: str) -> tuple[str, list[int]]:
        """
        Returns (normalized_text, extracted_numbers).
        """
        result = text

        # Fix ASR abbreviation errors
        for wrong, right in self.ABBREVIATION_FIXES.items():
            result = re.sub(r'\b' + wrong + r'\b', right, result, flags=re.IGNORECASE)

        # Replace compound numbers (longest first)
        for phrase, digit in self.COMPOUND_NUMBERS:
            result = result.replace(phrase, digit)

        # Replace operator words
        for word, symbol in self.OPERATOR_WORDS.items():
            result = result.replace(word, symbol)

        # Replace simple number words
        for word, digit in self.NUMBER_WORDS:
            result = result.replace(word, digit)

        # Extract all numbers
        numbers = [int(n) for n in re.findall(r'\b\d+\b', result)
                   if int(n) < 1000000]

        return result, numbers


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE CLOUD STT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class GoogleSTTClient:
    """
    Google Cloud Speech-to-Text v2 client.
    Uses chirp_2 model for best Tamil accuracy.
    Injects math vocabulary via Speech Adaptation.
    """

    def __init__(self, config: STTConfig):
        self.cfg = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import speech_v2 as speech
                self._client = speech.SpeechClient()
            except ImportError:
                raise ImportError(
                    "Install: pip install google-cloud-speech\n"
                    "Also set GOOGLE_APPLICATION_CREDENTIALS env var")
        return self._client

    def _build_adaptation(self, context_topic: str = None) -> dict:
        """Build speech adaptation with NIE math phrases."""
        phrases = list(NIE_MATH_ADAPTATION_PHRASES)
        if context_topic and context_topic in CONTEXT_PHRASE_BOOSTS:
            # Boost topic-specific phrases
            phrases = CONTEXT_PHRASE_BOOSTS[context_topic] + phrases

        return {
            "phrase_sets": [{
                "phrases": [{"value": p, "boost": 15.0} for p in phrases[:500]],
            }]
        }

    async def transcribe(self, audio_bytes: bytes,
                          context_topic: str = None) -> dict:
        """
        Transcribe audio using Google STT v2.
        Returns dict with transcript, confidence, alternatives.
        """
        t0 = time.perf_counter()

        try:
            from google.cloud import speech_v2 as speech
            from google.cloud.speech_v2.types import cloud_speech

            client = self._get_client()
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")

            recognizer = (
                f"projects/{project_id}/locations/global/"
                f"recognizers/_"
            )

            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=[self.cfg.primary_language] +
                                self.cfg.alternative_languages,
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
                                    cloud_speech.PhraseSet.Phrase(
                                        value=p, boost=15.0)
                                    for p in NIE_MATH_ADAPTATION_PHRASES[:200]
                                ]
                            )
                        )
                    ]
                ) if self.cfg.enable_speech_adaptation else None,
            )

            request = cloud_speech.RecognizeRequest(
                recognizer=recognizer,
                config=config,
                content=audio_bytes,
            )

            response = client.recognize(request=request)

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
                    "alternatives": [], "language": "ta-IN",
                    "processing_ms": elapsed}

        except Exception as e:
            log.error(f"Google STT error: {e}")
            return {"success": False, "error": str(e), "transcript": "",
                    "confidence": 0.0, "alternatives": [],
                    "language": "ta-IN", "processing_ms": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER OFFLINE STT
# ─────────────────────────────────────────────────────────────────────────────

class WhisperOfflineSTT:
    """
    OpenAI Whisper for offline STT. ~300MB 'small' model.
    Handles Tamil and Sinhala. Good enough for math problems.
    Download: happens automatically on first use.
    """

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import whisper
                log.info(f"Loading Whisper '{self.model_size}' model...")
                self._model = whisper.load_model(self.model_size)
                log.info("Whisper model loaded.")
            except ImportError:
                raise ImportError("Install: pip install openai-whisper")
        return self._model

    async def transcribe(self, audio_bytes: bytes,
                          language: str = "ta") -> dict:
        """Transcribe audio bytes using Whisper."""
        import io
        import tempfile
        import numpy as np
        t0 = time.perf_counter()

        try:
            model = self._load()

            # Write to temp WAV file (Whisper needs file input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import wave
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
                    tmp_path,
                    language=language,
                    task="transcribe",
                    initial_prompt="காரணி மடங்கு இலக்கச் சுட்டி போ.கா.பெ. போ.ம.சி.",
                    fp16=False,  # CPU mode
                )
            )

            import os
            os.unlink(tmp_path)

            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "success": True,
                "transcript": result["text"].strip(),
                "confidence": 0.75,   # Whisper doesn't give confidence scores
                "alternatives": [],
                "language": result.get("language", "ta"),
                "processing_ms": elapsed,
                "used_offline": True,
            }
        except Exception as e:
            log.error(f"Whisper STT error: {e}")
            return {"success": False, "error": str(e), "transcript": "",
                    "confidence": 0.0, "alternatives": [],
                    "language": "ta", "processing_ms": 0.0,
                    "used_offline": True}


# ─────────────────────────────────────────────────────────────────────────────
# TAMIL STT PIPELINE (main)
# ─────────────────────────────────────────────────────────────────────────────

class TamilSTTPipeline:
    """
    Full STT pipeline:
    1. Try Google Cloud STT (best accuracy, math adaptation)
    2. Fall back to Whisper if offline or Google fails
    3. Dialect detection + normalization
    4. Math text normalization
    5. Completeness check
    """

    def __init__(self, config: STTConfig = None):
        self.cfg = config or STTConfig()
        self.google_stt = GoogleSTTClient(self.cfg)
        self.whisper_stt = WhisperOfflineSTT(self.cfg.whisper_model)
        self.dialect_detector = DialectDetector()
        self.math_normalizer = MathTextNormalizer()

        from voice_vad import MathCompletenessChecker
        self.completeness_checker = MathCompletenessChecker()

    async def transcribe(self, utterance,  # AudioUtterance
                          context_topic: str = None) -> STTResult:
        """
        Full pipeline: audio → normalized NIE Tamil text + dialect.
        """
        t0 = time.perf_counter()
        used_offline = False

        # Step 1: Measure network latency to decide path
        online = await self._check_network()

        if online:
            stt_raw = await self.google_stt.transcribe(
                utterance.audio_bytes, context_topic)
        else:
            log.info("Network unavailable — using Whisper offline STT")
            stt_raw = await self.whisper_stt.transcribe(utterance.audio_bytes)
            used_offline = True

        if not stt_raw.get("success") or not stt_raw.get("transcript"):
            # Both failed — return empty result
            return STTResult(
                raw_text="", normalized_text="",
                dialect=Dialect.UNKNOWN, dialect_confidence=0.0,
                stt_confidence=0.0, language_detected="ta-IN",
                numbers_extracted=[], math_operators=[],
                is_math_complete=False, alternative_texts=[],
                processing_ms=(time.perf_counter() - t0) * 1000,
                used_offline=used_offline,
            )

        raw_text = stt_raw["transcript"]
        alternatives = stt_raw.get("alternatives", [])

        # Step 2: Dialect detection
        dialect, dialect_conf = self.dialect_detector.detect(
            raw_text, self.cfg.district, alternatives)
        log.info(f"STT: dialect={dialect.value} conf={dialect_conf:.2f}")

        # Step 3: Dialect → NIE normalization
        normalized = self.dialect_detector.normalize(raw_text, dialect)

        # Step 4: Math normalization
        normalized, numbers = self.math_normalizer.normalize(normalized)

        # Step 5: Extract math operators
        operators = [sym for sym in ["×", "÷", "+", "-", "="]
                     if sym in normalized]

        # Step 6: Completeness check
        completeness = self.completeness_checker.check(
            normalized, pause_duration_ms=utterance.duration_ms)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(f"STT pipeline: {elapsed:.0f}ms — '{normalized[:60]}...' "
                 f"complete={completeness['is_complete']}")

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
            used_offline=used_offline,
        )

    async def _check_network(self, timeout: float = 2.0) -> bool:
        """Quick network check to decide cloud vs offline path."""
        try:
            import asyncio
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("speech.googleapis.com", 443),
                timeout=timeout)
            writer.close()
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# TTS CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTSConfig:
    language_code: str = "ta-IN"
    voice_name: str = "ta-IN-Neural2-A"   # Best Tamil female voice (2024)
    # Alternatives:
    # ta-IN-Neural2-B (male), ta-IN-Wavenet-A (female), ta-IN-Wavenet-B (male)
    # si-LK-Standard-A (Sinhala)
    speaking_rate: float = 0.90            # Slightly slower for students
    pitch: float = 0.0
    volume_gain_db: float = 2.0

    cache_dir: Path = Path("data/tts_cache")
    cache_common_phrases: bool = True
    streaming: bool = True                 # Stream first sentence while rest synthesises


# ─────────────────────────────────────────────────────────────────────────────
# MATH SSML BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class MathSSMLBuilder:
    """
    Converts math text to SSML for natural Tamil pronunciation.

    The key challenge: Google TTS will mispronounce math abbreviations
    and symbols when reading Tamil text. SSML substitutions fix this.

    Examples:
    "72" → "எழுபத்தி இரண்டு"
    "36 = 2 × 2 × 3 × 3" → "முப்பத்தி ஆறு = இரண்டு × இரண்டு × மூன்று × மூன்று"
    "போ.கா.பெ." → "பொதுக் காரணிகளுட் பெரியது"
    "84 இன் காரணிகள்" → reads "எண்பத்தி நான்கு இன் காரணிகள்"
    """

    # Number → Tamil pronunciation
    NUMBER_PRONUNCIATION = {
        1: "ஒன்று", 2: "இரண்டு", 3: "மூன்று", 4: "நான்கு",
        5: "ஐந்து", 6: "ஆறு", 7: "ஏழு", 8: "எட்டு",
        9: "ஒன்பது", 10: "பத்து", 11: "பதினொன்று",
        12: "பன்னிரண்டு", 13: "பதிமூன்று", 14: "பதினான்கு",
        15: "பதினைந்து", 16: "பதினாறு", 17: "பதினேழு",
        18: "பதினெட்டு", 19: "பத்தொன்பது", 20: "இருபது",
        21: "இருபத்தி ஒன்று", 22: "இருபத்தி இரண்டு",
        24: "இருபத்தி நான்கு", 25: "இருபத்தி ஐந்து",
        30: "முப்பது", 36: "முப்பத்தி ஆறு", 42: "நாற்பத்தி இரண்டு",
        48: "நாற்பத்தி எட்டு", 54: "ஐம்பத்தி நான்கு",
        60: "அறுபது", 63: "அறுபத்தி மூன்று",
        72: "எழுபத்தி இரண்டு", 75: "எழுபத்தி ஐந்து",
        84: "எண்பத்தி நான்கு", 90: "தொண்ணூறு",
        96: "தொண்ணூற்றி ஆறு", 100: "நூறு",
        108: "நூற்று எட்டு", 126: "நூற்று இருபத்தி ஆறு",
        150: "நூற்று ஐம்பது", 180: "நூற்று எண்பது",
        204: "இருநூற்று நான்கு",
    }

    # Text replacements for natural reading
    TEXT_SUBSTITUTIONS = {
        "போ.கா.பெ.": "பொதுக் காரணிகளுட் பெரியது",
        "போ.ம.சி.": "பொது மடங்குகளுட் சிறியது",
        "÷": "வகுத்தல்",
        "×": "பெருக்கல்",
        "=": "சமம்",
        "+": "கூட்டல்",
        "-": "கழித்தல்",
        "∴": "ஆகவே",
        "∵": "ஏனெனில்",
        "√": "வர்க்கமூலம்",
        "%": "சதவீதம்",
    }

    def to_ssml(self, text: str) -> str:
        """Convert math text to SSML for Google TTS."""
        processed = text

        # Replace abbreviations and symbols
        for original, replacement in self.TEXT_SUBSTITUTIONS.items():
            processed = processed.replace(original, replacement)

        # Replace known numbers with Tamil words
        def replace_number(match):
            n = int(match.group())
            return self.NUMBER_PRONUNCIATION.get(n, match.group())

        # Replace numbers ≤ 999 with Tamil pronunciation
        processed = re.sub(r'\b(\d{1,3})\b', replace_number, processed)

        # Wrap in SSML with rate control
        ssml = f"""<speak>
  <prosody rate="90%" pitch="+0st">
    {processed}
  </prosody>
</speak>"""
        return ssml

    def chunk_for_streaming(self, text: str,
                             max_sentence_chars: int = 150) -> list[str]:
        """
        Split response text into chunks for streaming TTS.
        First chunk is synthesised and played while rest is queued.
        Split on sentence boundaries to avoid mid-sentence audio cuts.
        """
        # Split on Tamil sentence boundaries
        sentences = re.split(r'(?<=[.!?।\n])\s+|(?<=காண்க)\s+|(?<=ஆகும்\.)\s+', text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) > max_sentence_chars and current:
                chunks.append(current.strip())
                current = sent
            else:
                current += " " + sent if current else sent

        if current.strip():
            chunks.append(current.strip())

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# TAMIL TTS PIPELINE (main)
# ─────────────────────────────────────────────────────────────────────────────

class TamilTTSPipeline:
    """
    Full TTS pipeline with streaming and caching.

    Features:
    - SSML builder for correct math pronunciation
    - Streaming: first sentence plays while rest synthesises
    - Cache: common NIE phrases pre-synthesised on first run
    - Offline fallback: Android TTS API (via platform channel)
    """

    # Common NIE phrases to pre-cache at startup
    PHRASES_TO_PRECACHE = [
        "மிகவும் சரி! தொடர்ந்து முயற்சி செய்யுங்கள்.",
        "கொஞ்சம் மீண்டும் சிந்தியுங்கள்.",
        "நல்ல முயற்சி! ஒரு படி மட்டும் மீண்டும் பார்ப்போம்.",
        "இலக்கச் சுட்டியை மீண்டும் கணக்கிடுங்கள்.",
        "காரணி மரம் வரைவோம்.",
        "வகுத்தல் ஏணி முறையில் காண்போம்.",
        "அடுத்த கேள்விக்கு தயாரா?",
    ]

    def __init__(self, config: TTSConfig = None):
        self.cfg = config or TTSConfig()
        self.ssml_builder = MathSSMLBuilder()
        self._client = None
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._phrase_cache: dict[str, bytes] = {}

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import texttospeech_v1 as tts
                self._client = tts.TextToSpeechClient()
            except ImportError:
                raise ImportError(
                    "Install: pip install google-cloud-texttospeech")
        return self._client

    async def preload_cache(self):
        """Pre-synthesise common phrases at app startup."""
        log.info("Pre-loading TTS cache...")
        tasks = [self.synthesize(phrase) for phrase in self.PHRASES_TO_PRECACHE]
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info(f"TTS cache loaded: {len(self._phrase_cache)} phrases")

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    async def synthesize(self, text: str,
                          dialect: Dialect = Dialect.JAFFNA) -> bytes:
        """
        Synthesise full text to audio bytes (MP3).
        Uses cache if available.
        """
        key = self._cache_key(text)

        # Check in-memory cache
        if key in self._phrase_cache:
            return self._phrase_cache[key]

        # Check disk cache
        cache_file = self.cfg.cache_dir / f"{key}.mp3"
        if cache_file.exists():
            audio = cache_file.read_bytes()
            self._phrase_cache[key] = audio
            return audio

        # Synthesise
        ssml = self.ssml_builder.to_ssml(text)
        audio = await self._call_tts(ssml, dialect)

        # Cache
        if audio:
            cache_file.write_bytes(audio)
            self._phrase_cache[key] = audio

        return audio

    async def synthesize_streaming(
            self, text: str,
            dialect: Dialect = Dialect.JAFFNA) -> asyncio.Queue:
        """
        Streaming TTS: splits text into sentences, synthesises and
        queues audio chunks. Caller plays first chunk while rest loads.

        Returns an asyncio.Queue that yields bytes chunks (or None for end).
        """
        queue: asyncio.Queue = asyncio.Queue()
        chunks = self.ssml_builder.chunk_for_streaming(text)

        async def _synthesise_chunks():
            for chunk in chunks:
                audio = await self.synthesize(chunk, dialect)
                if audio:
                    await queue.put(audio)
            await queue.put(None)  # sentinel

        asyncio.create_task(_synthesise_chunks())
        return queue

    async def _call_tts(self, ssml: str, dialect: Dialect) -> Optional[bytes]:
        """Call Google Cloud TTS."""
        try:
            from google.cloud import texttospeech_v1 as tts

            client = self._get_client()

            # Adjust voice by dialect
            voice_name = self._dialect_voice(dialect)

            synthesis_input = tts.SynthesisInput(ssml=ssml)
            voice = tts.VoiceSelectionParams(
                language_code=self.cfg.language_code,
                name=voice_name,
            )
            audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3,
                speaking_rate=self.cfg.speaking_rate,
                pitch=self.cfg.pitch,
                volume_gain_db=self.cfg.volume_gain_db,
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                )
            )
            return response.audio_content

        except Exception as e:
            log.error(f"Google TTS error: {e}")
            return self._android_tts_fallback(ssml)

    def _dialect_voice(self, dialect: Dialect) -> str:
        """Select appropriate TTS voice by dialect.
        Currently all dialects use same Tamil voice since Google TTS
        doesn't differentiate SL dialects. Estate/Batticaloa students
        still understand standard Tamil TTS.
        """
        voices = {
            Dialect.JAFFNA:     "ta-IN-Neural2-A",   # Female
            Dialect.BATTICALOA: "ta-IN-Neural2-B",   # Male (slight variation)
            Dialect.ESTATE:     "ta-IN-Neural2-A",   # Female, slower rate
            Dialect.COLOMBO:    "ta-IN-Neural2-A",
            Dialect.VANNI:      "ta-IN-Neural2-A",
        }
        return voices.get(dialect, self.cfg.voice_name)

    def _android_tts_fallback(self, text: str) -> Optional[bytes]:
        """
        Android TTS fallback via platform channel.
        On mobile, this is handled by the Flutter VoiceOutputWidget
        which calls Android TextToSpeech API directly.
        Returns None to signal Flutter to handle TTS natively.
        """
        log.info("Falling back to Android TTS (platform channel)")
        return None   # Flutter side handles this case


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE VOICE I/O MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class VoiceIOManager:
    """
    Coordinates the full voice loop:
    listen → STT → dialect → normalize → orchestrator → TTS → play

    This is the single entry point for voice interaction.
    Used by both the Flutter platform channel handler and the Python CLI.
    """

    def __init__(self, district: str = "unknown", grade: int = 7):
        from voice_vad import VADConfig, VoiceActivityDetector

        self.cfg_vad = VADConfig(district=district, grade=grade)
        self.cfg_stt = STTConfig(district=district, grade=grade)
        self.cfg_tts = TTSConfig()

        self.vad = VoiceActivityDetector(self.cfg_vad)
        self.stt = TamilSTTPipeline(self.cfg_stt)
        self.tts = TamilTTSPipeline(self.cfg_tts)

        self._orchestrator = None  # injected by caller
        self._student_id = None
        self._re_prompt_audio: Optional[bytes] = None

    def attach_orchestrator(self, orchestrator, student_id: str):
        """Connect to multi-agent orchestrator from agent_orchestrator.py."""
        self._orchestrator = orchestrator
        self._student_id = student_id

    async def initialize(self):
        """Pre-load TTS cache and VAD model at startup."""
        await self.tts.preload_cache()
        # Pre-load re-prompt audio
        self._re_prompt_audio = await self.tts.synthesize(
            "மன்னிக்கவும், கொஞ்சம் மீண்டும் சொல்லுங்கள்?")

    async def voice_session(self) -> AsyncIterator[dict]:
        """
        Full voice interaction loop.
        Yields events: {type, data} for the Flutter UI to handle.

        Event types:
          listening_start  — VAD started listening
          speech_detected  — student is speaking
          pause_detected   — mid-utterance pause (show visual indicator)
          utterance_ready  — complete utterance captured
          stt_result       — transcript ready
          processing       — orchestrator thinking
          response_text    — teaching response text
          response_audio   — audio chunk bytes
          diagram_ready    — diagram JSON spec
          exercise_ready   — exercise bundle
          error            — error occurred
        """
        if not self._orchestrator:
            raise RuntimeError("Attach orchestrator before starting session")

        yield {"type": "listening_start"}

        async for utterance in self.vad.listen():
            yield {"type": "utterance_ready",
                   "duration_ms": utterance.duration_ms}

            # STT
            yield {"type": "processing", "stage": "transcribing"}
            stt_result = await self.stt.transcribe(utterance)

            if not stt_result.raw_text:
                yield {"type": "error", "message": "வாக்கு அறியப்படவில்லை"}
                if self._re_prompt_audio:
                    yield {"type": "response_audio",
                           "audio": self._re_prompt_audio,
                           "is_reprompt": True}
                continue

            yield {"type": "stt_result",
                   "text": stt_result.normalized_text,
                   "dialect": stt_result.dialect.value,
                   "confidence": stt_result.stt_confidence,
                   "is_complete": stt_result.is_math_complete}

            # If incomplete, prompt for more
            if not stt_result.is_math_complete and \
               stt_result.stt_confidence < 0.5:
                clarify = await self.tts.synthesize(
                    "கொஞ்சம் முழுமையாக கேள்வி கேளுங்கள்.")
                yield {"type": "response_audio", "audio": clarify,
                       "is_reprompt": True}
                continue

            # Multi-agent orchestrator
            yield {"type": "processing", "stage": "thinking"}
            response = await self._orchestrator.handle(
                student_id=self._student_id,
                raw_query=stt_result.normalized_text,
                district=self.cfg_stt.district,
            )

            # Yield text response
            if response.teaching:
                yield {"type": "response_text",
                       "text": response.teaching.explanation_ta,
                       "dialect": stt_result.dialect.value}

                # Stream TTS audio
                audio_queue = await self.tts.synthesize_streaming(
                    response.teaching.explanation_ta,
                    dialect=stt_result.dialect)

                chunk_idx = 0
                while True:
                    audio_chunk = await audio_queue.get()
                    if audio_chunk is None:
                        break
                    yield {"type": "response_audio",
                           "audio": audio_chunk,
                           "chunk_index": chunk_idx,
                           "is_first": chunk_idx == 0}
                    chunk_idx += 1

            # Yield diagram if available
            if response.diagram:
                yield {"type": "diagram_ready",
                       "spec": response.diagram.spec,
                       "diagram_type": response.diagram.diagram_type,
                       "caption_ta": response.diagram.caption_ta}

                # Speak the caption
                caption_audio = await self.tts.synthesize(
                    response.diagram.caption_ta, stt_result.dialect)
                if caption_audio:
                    yield {"type": "response_audio", "audio": caption_audio,
                           "is_caption": True}

            # Yield exercise
            if response.exercise:
                yield {"type": "exercise_ready",
                       "question_ta": response.exercise.question_ta,
                       "difficulty": response.exercise.difficulty}

                # Read the exercise aloud
                exercise_audio = await self.tts.synthesize(
                    response.exercise.question_ta, stt_result.dialect)
                if exercise_audio:
                    yield {"type": "response_audio", "audio": exercise_audio,
                           "is_exercise": True}

            yield {"type": "listening_start"}  # ready for next input


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────────────────────────

async def cli_test():
    """Test the STT/TTS pipeline components."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-ssml", action="store_true")
    parser.add_argument("--test-normalizer", action="store_true")
    parser.add_argument("--test-dialect", action="store_true")
    parser.add_argument("--district", default="jaffna")
    args = parser.parse_args()

    if args.test_ssml:
        builder = MathSSMLBuilder()
        tests = [
            "72 உம் 108 உம் ஆகிய எண்களின் போ.கா.பெ. காண்க",
            "84 = 2 × 2 × 3 × 7 ஆகும்",
            "36 இன் காரணிகள்: 1, 2, 3, 4, 6, 9, 12, 18, 36",
        ]
        print("=== SSML Builder ===")
        for t in tests:
            ssml = builder.to_ssml(t)
            print(f"IN:  {t}")
            print(f"OUT: {ssml[:100]}...")
            print()

    if args.test_normalizer:
        norm = MathTextNormalizer()
        tests = [
            "factor காண்க",
            "HCF காண்க விடை என்ன",
            "எழுபத்தி இரண்டு இன் காரணி மரம்",
            "LCM of 6 and 8",
        ]
        print("=== Math Normalizer ===")
        for t in tests:
            result, nums = norm.normalize(t)
            print(f"IN:  {t}")
            print(f"OUT: {result}  numbers={nums}")
            print()

    if args.test_dialect:
        detector = DialectDetector()
        tests = [
            ("factor காண்க", "colombo"),
            ("போடு", "unknown"),
            ("பண்ணுவது", "unknown"),
            ("காண்போம்", "unknown"),
        ]
        print("=== Dialect Detector ===")
        for text, district in tests:
            dialect, conf = detector.detect(text, district)
            normalized = detector.normalize(text, dialect)
            print(f"'{text}' [{district}] → {dialect.value} ({conf:.2f})")
            print(f"  normalized: '{normalized}'")
            print()


if __name__ == "__main__":
    asyncio.run(cli_test())
