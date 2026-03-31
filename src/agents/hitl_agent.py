"""Human-in-the-loop flagging rules and queue helpers."""

from __future__ import annotations

import re
from typing import Optional

from ..models.messages import QueryContext, SentimentSignal
from ..models.student import StudentProfile
from ..storage import DatabaseManager

_ENGLISH_ALLOW = {
    "lcm",
    "hcf",
    "gcd",
    "math",
    "true",
    "false",
    "null",
    "step",
    "even",
    "odd",
    "prime",
}


class HITLAgent:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def should_flag(
        self,
        student: StudentProfile,
        ctx: QueryContext,
        sentiment: SentimentSignal,
        response_text: str,
        interaction_id: int,
    ) -> tuple[bool, Optional[str]]:
        _ = interaction_id
        history = list(getattr(student, "topic_history", []) or [])
        if (
            len(history) >= 3
            and history[-1] == history[-2] == history[-3]
            and history[-1] not in ("", "unknown")
        ):
            return True, "same_topic_streak"

        if sentiment.frustration_detected:
            return True, "frustration_signal"

        if "ஆசிரியர்" in (ctx.raw_query or ""):
            return True, "explicit_teacher_request"

        if response_text and len(response_text.strip()) < 50:
            return True, "response_too_short"

        if self._has_unwanted_english(response_text or ""):
            return True, "unexpected_english"

        return False, None

    def _has_unwanted_english(self, text: str) -> bool:
        words = re.findall(r"\b[A-Za-z]{4,}\b", text)
        for w in words:
            low = w.lower()
            if low in _ENGLISH_ALLOW:
                continue
            return True
        return False

    def flag_for_review(self, interaction_id: int, reason: str) -> int:
        return self.db.add_hitl_flag(interaction_id, reason)

    def get_pending_reviews(self) -> list[dict]:
        return self.db.get_hitl_queue(status="pending")

    def resolve_review(
        self,
        queue_id: int,
        teacher_id: str,
        status: str,
        annotation: str | None,
    ) -> None:
        self.db.update_hitl_status(
            queue_id=queue_id,
            status=status,
            teacher_id=teacher_id,
            annotation=annotation,
        )
