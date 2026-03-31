"""Student profile dataclass — the canonical student model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from typing import Optional


@dataclass
class StudentProfile:
    student_id: str
    name: str
    grade: int = 7
    school_type: str = "tamil_medium"
    district: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    skills: dict = field(default_factory=lambda: {
        "divisibility_rules": 0.0,
        "digit_sum": 0.0,
        "factor_listing": 0.0,
        "prime_factorization": 0.0,
        "hcf": 0.0,
        "lcm": 0.0,
        "word_problems": 0.0,
    })

    total_questions_asked: int = 0
    total_exercises_attempted: int = 0
    total_exercises_correct: int = 0
    preferred_method: str = "none"
    last_topic: str = ""
    last_error_type: str = ""
    session_count: int = 0
    error_patterns: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> StudentProfile:
        """Safe deserialization — ignores unknown keys from old DB rows."""
        known = {f.name for f in dataclass_fields(cls)}
        safe = {k: v for k, v in data.items() if k in known}
        obj = cls.__new__(cls)
        defaults = cls(student_id="__default__", name="__default__")
        for f in dataclass_fields(cls):
            setattr(obj, f.name, safe.get(f.name, getattr(defaults, f.name)))
        base_skills = dict(defaults.skills)
        base_skills.update(safe.get("skills", {}))
        obj.skills = base_skills
        return obj

    def get_difficulty_ceiling(self) -> int:
        avg_skill = sum(self.skills.values()) / len(self.skills)
        if avg_skill < 0.3:
            return 1
        elif avg_skill < 0.6:
            return 2
        return 3

    def get_unlocked_topics(self) -> set[str]:
        from ..data.prerequisite_graph import PREREQUISITE_GRAPH, topic_to_skill

        FOUNDATION = {
            "factor_definition", "divisibility_rules", "digit_sum",
            "divisibility_2", "divisibility_3", "divisibility_9",
            "divisibility_6", "divisibility_4", "divisibility_rules_all",
        }
        unlocked: set[str] = set(FOUNDATION)
        changed = True
        while changed:
            changed = False
            for topic, prereq_topics in PREREQUISITE_GRAPH.items():
                if topic in unlocked:
                    continue
                if all(
                    self.skills.get(topic_to_skill(pt), 0.0) >= 0.5
                    for pt in prereq_topics
                ):
                    unlocked.add(topic)
                    changed = True
        return unlocked

    def update_skill(self, topic: str, correct: bool, difficulty: int):
        from ..data.prerequisite_graph import topic_to_skill
        skill_key = topic_to_skill(topic)
        if skill_key in self.skills:
            delta = 0.1 * difficulty if correct else -0.05
            self.skills[skill_key] = max(0.0, min(1.0, self.skills[skill_key] + delta))
        self.total_exercises_attempted += 1
        if correct:
            self.total_exercises_correct += 1

    def accuracy(self) -> float:
        if self.total_exercises_attempted == 0:
            return 0.0
        return self.total_exercises_correct / self.total_exercises_attempted

    def mastered_topics(self) -> list[str]:
        return [k for k, v in self.skills.items() if v >= 0.75]

    def weak_topics(self) -> list[str]:
        return [k for k, v in self.skills.items() if v < 0.4]
