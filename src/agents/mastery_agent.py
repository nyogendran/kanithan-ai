"""Skill updates and topic progression using the prerequisite graph."""

from __future__ import annotations

from typing import Optional

from ..data.prerequisite_graph import topic_to_skill
from ..models.student import StudentProfile

TOPIC_PROGRESSION = [
    "divisibility_rules",
    "digit_sum",
    "factor_listing",
    "prime_factorization",
    "factors_via_prime",
    "hcf",
    "lcm",
    "word_problems",
]


class MasteryAgent:
    """Maps exercises to skill keys and suggests next topics from mastery."""

    def update_skill(
        self,
        student: StudentProfile,
        topic: str,
        correct: bool,
        difficulty: int,
        error_type: str = "",
    ) -> StudentProfile:
        student.update_skill(topic, correct, difficulty)
        if error_type:
            student.last_error_type = error_type
            if not correct:
                student.error_patterns[error_type] = student.error_patterns.get(error_type, 0) + 1
        return student

    def suggest_next_topic(self, student: StudentProfile, current_topic: str) -> Optional[str]:
        cur_skill = topic_to_skill(current_topic)
        idx = self._progression_index(current_topic)
        if idx < 0:
            return None
        if student.skills.get(cur_skill, 0.0) < 0.6:
            return None
        if idx + 1 >= len(TOPIC_PROGRESSION):
            return None
        return TOPIC_PROGRESSION[idx + 1]

    @staticmethod
    def _progression_index(topic: str) -> int:
        sk = topic_to_skill(topic)
        for i, p in enumerate(TOPIC_PROGRESSION):
            if p == topic:
                return i
        for i, p in enumerate(TOPIC_PROGRESSION):
            if topic_to_skill(p) == sk:
                return i
        return -1

    def should_review(self, student: StudentProfile, topic: str) -> bool:
        skill_key = topic_to_skill(topic)
        score = student.skills.get(skill_key, 0.0)
        if score >= 0.4:
            return False
        if student.total_exercises_attempted <= 0:
            return False
        last_sk = topic_to_skill(student.last_topic) if student.last_topic else ""
        return last_sk == skill_key or score > 0.0

    def get_mastery_summary(self, student: StudentProfile) -> dict:
        topics = list(student.skills.keys())
        mastered = [t for t in topics if student.skills.get(t, 0) >= 0.75]
        weak = [t for t in topics if student.skills.get(t, 0) < 0.4]
        next_topic = None
        for p in TOPIC_PROGRESSION:
            sk = topic_to_skill(p)
            if student.skills.get(sk, 0) < 0.6:
                next_topic = p
                break
        return {
            "skills": {t: float(student.skills.get(t, 0)) for t in topics},
            "mastered": mastered,
            "weak": weak,
            "next": next_topic or "",
        }

    def record_session_context(
        self,
        student: StudentProfile,
        query: str,
        intent: str,
        topic: str,
    ) -> StudentProfile:
        student.total_questions_asked += 1
        if topic and topic != "unknown":
            student.last_topic = topic
        student.session_count += 1
        return student


__all__ = ["MasteryAgent", "TOPIC_PROGRESSION"]
