"""Lightweight engagement and frustration signals from query text and behaviour."""

from __future__ import annotations

import random
from typing import Optional

from ..models.messages import SentimentSignal
from ..models.student import StudentProfile

FRUSTRATION_KEYWORDS = [
    "புரியவில்லை",
    "தெரியவில்லை",
    "கஷ்டமாக",
    "முடியவில்லை",
    "help",
    "உதவி",
]

_ENCOURAGEMENT_TA = [
    "நன்றாக முயற்சிக்கிறீர்கள்!",
    "தொடர்ந்து முயற்சி செய்யுங்கள் — ஒவ்வொரு படியும் முக்கியம்.",
    "சிறிது நேரம் எடுத்துக் கொள்ளுங்கள்; நீங்கள் முன்னேறுகிறீர்கள்.",
    "தவறுகள் கற்றலின் பகுதி — மீண்டும் முயற்சி செய்வோம்.",
    "உங்கள் கேள்விகள் நல்லவை; அதே உற்சாகத்துடன் தொடருங்கள்.",
    "ஒவ்வொரு முறையும் நீங்கள் வலுப்படுகிறீர்கள்!",
]


class SentimentAgent:
    def analyze(
        self,
        student: StudentProfile,
        query: str,
        response_time_ms: int,
        is_retry: bool,
        exercise_correct: Optional[bool] = None,
    ) -> SentimentSignal:
        q = query or ""
        nq = student.total_questions_asked
        attempted = student.total_exercises_attempted
        engagement_score = min(
            1.0,
            0.25 + 0.08 * min(nq, 8) + 0.04 * min(attempted, 12),
        )
        if is_retry:
            engagement_score = max(0.2, engagement_score - 0.1)

        acc = student.accuracy()
        confidence_level = acc if attempted > 0 else 0.5
        if exercise_correct is True:
            confidence_level = min(1.0, confidence_level + 0.15)
        elif exercise_correct is False:
            confidence_level = max(0.0, confidence_level - 0.15)

        kw_hit = any(kw.lower() in q.lower() for kw in FRUSTRATION_KEYWORDS)
        long_pause = response_time_ms > 60_000
        streak = getattr(student, "consecutive_wrong_answers", 0)
        frustration_detected = kw_hit or long_pause or streak >= 3

        encourage = confidence_level < 0.3 or frustration_detected

        return SentimentSignal(
            engagement_score=engagement_score,
            confidence_level=confidence_level,
            frustration_detected=frustration_detected,
            encourage=encourage,
        )

    def get_encouragement_phrase(self) -> str:
        return random.choice(_ENCOURAGEMENT_TA)
