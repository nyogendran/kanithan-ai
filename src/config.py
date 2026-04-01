"""
Centralized configuration for the Kanithan AI Tutor platform.
All settings are loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_TEACHING_MODEL: str = os.environ.get("GEMINI_TEACHING_MODEL", "gemini-2.5-flash")
GEMINI_FAST_MODEL: str = os.environ.get("GEMINI_FAST_MODEL", "gemini-2.0-flash")

# LLM backend for OrchestratorAgent (teaching + answer verification): "gemini" | "ollama"
_raw_backend = (os.environ.get("LLM_BACKEND") or "gemini").strip().lower()
LLM_BACKEND: str = _raw_backend if _raw_backend in ("gemini", "ollama") else "gemini"
# Comma-separated models for /voice/transcribe. If unset or empty, server uses teaching model only.
# Example: GEMINI_TRANSCRIBE_MODELS=gemini-2.0-flash,gemini-2.5-flash
GEMINI_TRANSCRIBE_MODELS_RAW: str = os.environ.get("GEMINI_TRANSCRIBE_MODELS", "")
GEMINI_MAX_OUTPUT_TOKENS: int = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
GEMINI_TEMPERATURE: float = float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_LLM", "llama3:latest")
# Optional second Ollama tag for JSON / fast tasks; defaults to OLLAMA_MODEL
OLLAMA_FAST_MODEL: str = os.environ.get("OLLAMA_FAST_MODEL", OLLAMA_MODEL)


def llm_teaching_model() -> str:
    """Model id passed to LLMClient.generate* when LLM_BACKEND is active."""
    if LLM_BACKEND == "ollama":
        return OLLAMA_MODEL
    return GEMINI_TEACHING_MODEL


def llm_fast_model() -> str:
    """Model id for AnswerVerifierAgent.generate_json."""
    if LLM_BACKEND == "ollama":
        return OLLAMA_FAST_MODEL
    return GEMINI_FAST_MODEL

CHROMA_PATH: Path = Path(os.environ.get("VECTOR_DB_PATH", str(PROJECT_ROOT / "data" / "vector_db")))
COLLECTION_PREFIX: str = os.environ.get("COLLECTION_PREFIX", "kanithan_curriculum")
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
CHUNK_SIZE_TOKENS: int = 400
CHUNK_OVERLAP_TOKENS: int = 80
MIN_CHUNK_CHARS: int = 60

DB_PATH: Path = Path(os.environ.get("LEARNING_PROFILE_DB", str(PROJECT_ROOT / "data" / "learning_profile.db")))

TUTOR_DISTRICT: str = (os.environ.get("TUTOR_DISTRICT") or "unknown").strip()

DEFAULT_GRADE: int = 7
DEFAULT_CHAPTER: int = 4
DEFAULT_SUBJECT: str = "mathematics"
