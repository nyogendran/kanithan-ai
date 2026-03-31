"""Typed dataclass messages for inter-agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .intents import Intent, Dialect


@dataclass
class QueryContext:
    raw_query: str
    normalized_query: str
    intent: Intent
    topic: str
    section: str
    numbers: list[int]
    method_requested: Optional[str] = None
    is_word_problem: bool = False
    dialect: Dialect = Dialect.UNKNOWN
    confidence: float = 0.0
    student_answer: Optional[str] = None
    exercise_topic: Optional[str] = None
    region_hints: str = ""


@dataclass
class RetrievedContext:
    chunks: list[dict] = field(default_factory=list)
    answer_scheme_chunks: list[dict] = field(default_factory=list)
    total_retrieved: int = 0
    query_embedding: list[float] = field(default_factory=list)
    retrieval_time_ms: float = 0.0


@dataclass
class TeachingResponse:
    explanation_ta: str = ""
    key_concepts: list[str] = field(default_factory=list)
    next_suggested_topic: Optional[str] = None
    verification_block: str = ""


@dataclass
class DiagramSpec:
    diagram_type: str = ""
    spec: dict = field(default_factory=dict)
    caption_ta: str = ""
    animate: bool = True


@dataclass
class ExerciseBundle:
    question_ta: str = ""
    numbers: list[int] = field(default_factory=list)
    difficulty: int = 1
    topic: str = ""
    hint_ta: Optional[str] = None
    expected_steps: list[str] = field(default_factory=list)
    answer: Any = None
    method_expected: Optional[str] = None


@dataclass
class VerificationResult:
    is_correct: bool = False
    first_wrong_step: Optional[str] = None
    socratic_hint_ta: str = ""
    error_type: Optional[str] = None
    skill_delta: float = 0.0
    method_used: Optional[str] = None
    method_expected: Optional[str] = None


@dataclass
class SentimentSignal:
    engagement_score: float = 0.5
    confidence_level: float = 0.5
    frustration_detected: bool = False
    encourage: bool = False


@dataclass
class AgentResponse:
    session_id: str = ""
    student_id: str = ""
    intent: Intent = Intent.UNKNOWN
    teaching: Optional[TeachingResponse] = None
    diagram: Optional[DiagramSpec] = None
    exercise: Optional[ExerciseBundle] = None
    verification: Optional[VerificationResult] = None
    sentiment: Optional[SentimentSignal] = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    dialect_detected: Dialect = Dialect.UNKNOWN
    processing_time_ms: float = 0.0
    hitl_flagged: bool = False
    hitl_reason: Optional[str] = None
    error: Optional[str] = None
    quota_exhausted: bool = False
    retry_after_seconds: Optional[float] = None
