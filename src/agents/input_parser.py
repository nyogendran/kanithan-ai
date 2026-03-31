"""Multimodal input parsing — text now; voice, handwriting, drawing stubs."""

from __future__ import annotations

from src.data.glossary import normalize_tamil_numbers


class InputParserAgent:
    """Normalize multimodal learner input into text or structured drawing data."""

    def parse_text(self, text: str) -> str:
        return normalize_tamil_numbers(text)

    def parse_voice(self, audio_bytes: bytes, language: str = "ta-IN") -> str:
        raise NotImplementedError(
            "Voice input requires Google Cloud Speech-to-Text — Phase 5"
        )

    def parse_handwriting(self, image_bytes: bytes) -> str:
        raise NotImplementedError()

    def parse_drawing(self, canvas_data: dict) -> dict:
        raise NotImplementedError()
