"""
Pluggable Text-to-Speech pipeline for Tamil math tutoring.

Backends (selected via TTS_BACKEND env var):
  - gemini       (default) — uses existing GEMINI_API_KEY
  - google_cloud           — Google Cloud TTS Neural2 voices

Includes SSML math pronunciation, disk+memory caching, and streaming.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("kanithan.tts")

TTS_BACKEND: str = os.environ.get("TTS_BACKEND", "gemini").strip().lower()

# ---------------------------------------------------------------------------
# Math SSML builder
# ---------------------------------------------------------------------------

class MathSSMLBuilder:
    """Convert math text to SSML for natural Tamil pronunciation."""

    NUMBER_PRONUNCIATION: dict[int, str] = {
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

    TEXT_SUBSTITUTIONS: dict[str, str] = {
        "பொ.கா.பெ.": "பொதுக் காரணிகளுட் பெரியது",
        "பொ.ம.சி.": "பொது மடங்குகளுட் சிறியது",
        "÷": "வகுத்தல்", "×": "பெருக்கல்", "=": "சமம்",
        "+": "கூட்டல்", "-": "கழித்தல்",
        "∴": "ஆகவே", "∵": "ஏனெனில்",
        "√": "வர்க்கமூலம்", "%": "சதவீதம்",
    }

    def to_ssml(self, text: str) -> str:
        processed = text
        for original, replacement in self.TEXT_SUBSTITUTIONS.items():
            processed = processed.replace(original, replacement)

        def replace_number(match):
            n = int(match.group())
            return self.NUMBER_PRONUNCIATION.get(n, match.group())

        processed = re.sub(r'\b(\d{1,3})\b', replace_number, processed)
        return (
            '<speak><prosody rate="90%" pitch="+0st">'
            f'{processed}'
            '</prosody></speak>'
        )

    def to_plain_speech(self, text: str) -> str:
        """Non-SSML text normalization for backends that don't support SSML."""
        processed = text
        for original, replacement in self.TEXT_SUBSTITUTIONS.items():
            processed = processed.replace(original, replacement)

        def replace_number(match):
            n = int(match.group())
            return self.NUMBER_PRONUNCIATION.get(n, match.group())

        return re.sub(r'\b(\d{1,3})\b', replace_number, processed)

    def chunk_for_streaming(self, text: str,
                             max_sentence_chars: int = 150) -> list[str]:
        sentences = re.split(
            r'(?<=[.!?।\n])\s+|(?<=காண்க)\s+|(?<=ஆகும்\.)\s+', text)
        chunks: list[str] = []
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


# ---------------------------------------------------------------------------
# TTS config
# ---------------------------------------------------------------------------

@dataclass
class TTSConfig:
    language_code: str = "ta-IN"
    voice_name: str = "ta-IN-Neural2-A"
    speaking_rate: float = 0.90
    pitch: float = 0.0
    volume_gain_db: float = 2.0
    cache_dir: Path = Path("data/tts_cache")
    cache_common_phrases: bool = True
    streaming: bool = True


# ---------------------------------------------------------------------------
# TTS backend: Gemini
# ---------------------------------------------------------------------------

class GeminiTTS:
    """Use Gemini generate_content for TTS via existing API key."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def synthesize(self, text: str, _dialect: str = "jaffna") -> Optional[bytes]:
        """
        Use Gemini to generate speech audio from Tamil text.
        Returns MP3 bytes or None on failure.
        """
        try:
            from google.genai import types
            client = self._get_client()
            model = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash")

            ssml_builder = MathSSMLBuilder()
            speech_text = ssml_builder.to_plain_speech(text)

            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore",
                        ),
                    ),
                ),
            )
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model,
                    contents=speech_text,
                    config=config,
                ),
            )

            if (response.candidates and response.candidates[0].content
                    and response.candidates[0].content.parts):
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        return part.inline_data.data

            log.warning("Gemini TTS returned no audio data")
            return None
        except Exception as e:
            log.error("Gemini TTS error: %s", e)
            return None


# ---------------------------------------------------------------------------
# TTS backend: Google Cloud
# ---------------------------------------------------------------------------

class GoogleCloudTTS:
    """Google Cloud TTS with Neural2 Tamil voices."""

    def __init__(self, config: TTSConfig | None = None):
        self.cfg = config or TTSConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import texttospeech_v1 as tts
                self._client = tts.TextToSpeechClient()
            except ImportError:
                raise ImportError("Install: pip install google-cloud-texttospeech")
        return self._client

    async def synthesize(self, text: str, dialect: str = "jaffna") -> Optional[bytes]:
        try:
            from google.cloud import texttospeech_v1 as tts

            client = self._get_client()
            ssml_builder = MathSSMLBuilder()
            ssml = ssml_builder.to_ssml(text)
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
                ),
            )
            return response.audio_content
        except Exception as e:
            log.error("Google Cloud TTS error: %s", e)
            return None

    def _dialect_voice(self, dialect: str) -> str:
        voices = {
            "jaffna": "ta-IN-Neural2-A",
            "batticaloa": "ta-IN-Neural2-B",
            "estate": "ta-IN-Neural2-A",
            "colombo": "ta-IN-Neural2-A",
            "vanni": "ta-IN-Neural2-A",
        }
        return voices.get(dialect, self.cfg.voice_name)


# ---------------------------------------------------------------------------
# Main TTS pipeline
# ---------------------------------------------------------------------------

PHRASES_TO_PRECACHE = [
    "மிகவும் சரி! தொடர்ந்து முயற்சி செய்யுங்கள்.",
    "கொஞ்சம் மீண்டும் சிந்தியுங்கள்.",
    "நல்ல முயற்சி! ஒரு படி மட்டும் மீண்டும் பார்ப்போம்.",
    "இலக்கச் சுட்டியை மீண்டும் கணக்கிடுங்கள்.",
    "காரணி மரம் வரைவோம்.",
    "வகுத்தல் ஏணி முறையில் காண்போம்.",
    "அடுத்த கேள்விக்கு தயாரா?",
    "கொஞ்சம் மீண்டும் சொல்ல முடியுமா? நான் சரியாகப் புரிந்துகொள்ள விரும்புகிறேன்.",
]


class TamilTTSPipeline:
    """
    Full TTS pipeline with caching and streaming.
    Backend is selected via TTS_BACKEND env var (gemini | google_cloud).
    """

    def __init__(self, config: TTSConfig | None = None, backend: str | None = None):
        self.cfg = config or TTSConfig()
        self._backend_name = backend or TTS_BACKEND
        self.ssml_builder = MathSSMLBuilder()

        if self._backend_name == "google_cloud":
            self._backend = GoogleCloudTTS(self.cfg)
        else:
            self._backend = GeminiTTS()

        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._phrase_cache: dict[str, bytes] = {}

    async def preload_cache(self):
        """Pre-synthesize common phrases at startup."""
        log.info("Pre-loading TTS cache ...")
        tasks = [self.synthesize(phrase) for phrase in PHRASES_TO_PRECACHE]
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("TTS cache loaded: %d phrases", len(self._phrase_cache))

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    async def synthesize(self, text: str, dialect: str = "jaffna") -> Optional[bytes]:
        key = self._cache_key(text)

        if key in self._phrase_cache:
            return self._phrase_cache[key]

        cache_file = self.cfg.cache_dir / f"{key}.mp3"
        if cache_file.exists():
            audio = cache_file.read_bytes()
            self._phrase_cache[key] = audio
            return audio

        audio = await self._backend.synthesize(text, dialect)

        if audio:
            try:
                cache_file.write_bytes(audio)
            except OSError:
                pass
            self._phrase_cache[key] = audio

        return audio

    async def synthesize_streaming(self, text: str,
                                    dialect: str = "jaffna") -> asyncio.Queue:
        """
        Split text into sentences, synthesize each, push to queue.
        Caller plays first chunk while rest is synthesized.
        Returns a queue that yields bytes chunks (None = end).
        """
        queue: asyncio.Queue = asyncio.Queue()
        chunks = self.ssml_builder.chunk_for_streaming(text)

        async def _synthesize_chunks():
            for chunk in chunks:
                audio = await self.synthesize(chunk, dialect)
                if audio:
                    await queue.put(audio)
            await queue.put(None)

        asyncio.create_task(_synthesize_chunks())
        return queue
