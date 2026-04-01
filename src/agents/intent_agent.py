"""Intent classification, topic detection, and QueryContext assembly."""

from __future__ import annotations

import re
from typing import Optional

from src.chapters.base import ChapterTopicPack
from src.chapters.registry import get_chapter_plugin
from src.models.intents import Dialect, Intent
from src.models.messages import QueryContext
from src.models.student import StudentProfile

INTENT_PRIORITY: list[str] = [
    "CHECK_ANSWER",
    "DIAGRAM_REQUEST",
    "SHOW_METHOD",
    "EXERCISE_REQUEST",
    "WORD_PROBLEM",
    "EXPLAIN",
]

INTENT_KEYWORDS: dict[str, list[str]] = {
    "EXPLAIN": [
        "என்றால் என்ன",
        "வரையறை",
        "விளக்கு",
        "புரியவில்லை",
        "கூறு",
        "எப்படி",
        "கூறுங்கள்",
        "விளக்குங்கள்",
        "விளக்கவும்",
        "கூறவும்",
        "என்ன",
        "கற்றுக்கொடு",
        "சொல்லுங்கள்",
        "what is",
        "explain",
        "teach",
        "define",
    ],
    "SHOW_METHOD": [
        "முறை",
        "எப்படி காண்பது",
        "எப்படி கணக்கிடுவது",
        "காட்டு",
        "காட்டுங்கள்",
        "காட்டவும்",
        "steps",
        "படிகள்",
        "method",
        "வகுத்தல் முறை",
        "ஏணி முறை",
        "காரணி மரம் முறை",
        "show method",
        "step by step",
        "எப்படி",
    ],
    "EXERCISE_REQUEST": [
        "பயிற்சி",
        "கேள்வி கொடு",
        "கணக்கு கொடு",
        "சோதனை",
        "practice",
        "exercise",
        "question",
        "problem",
        "கொடு",
        "கொடுங்கள்",
        "கொடுக்கவும்",
        "தரவும்",
        "தாருங்கள்",
    ],
    "CHECK_ANSWER": [
        "சரியா",
        "இது சரியா",
        "என் பதில்",
        "விடை",
        "விடை சரிதானா",
        "என் பதில் சரியா",
        "நான் கண்டேன்",
        "check",
        "correct",
        "answer",
        "= ",
    ],
    "DIAGRAM_REQUEST": [
        "வரை",
        "படம்",
        "draw",
        "diagram",
        "காரணி மரம்",
        "factor tree",
        "வகுத்தல் ஏணி",
        "division ladder",
        "number line",
        "மடங்கு கோடு",
        "காட்டு",
        "chart",
    ],
    "WORD_PROBLEM": [
        "கதை கணக்கு",
        "சிந்தனைக்கு",
        "பென்சில்",
        "மணி",
        "பழம்",
        "பொதி",
        "பகிர்",
        "word problem",
        "real life",
        "நிமிடம்",
        "நேரம்",
        "பூக்கள்",
        "மரம்",
    ],
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "divisibility_rules": ["வகுபடும்", "வகுபடாது", "÷2", "÷3", "÷9", "÷6", "÷4", "÷5"],
    "digit_sum": ["இலக்கச் சுட்டி", "digit sum"],
    "factor_listing": ["காரணி", "காரணிகள்", "factor"],
    "prime_factorization": ["முதன்மை", "prime", "காரணி மரம்", "ஏணி"],
    "hcf": ["பொ.கா.பெ.", "பொதுக் காரணி", "HCF", "GCD", "பெரியது"],
    "lcm": ["பொ.ம.சி.", "பொது மடங்கு", "LCM", "சிறியது", "மடங்கு"],
    "factors_via_prime": ["முதன்மைக் காரணி", "prime factor", "காரணிப்படுத்தல்"],
    "word_problem": ["பொதி", "மணி", "பகிர்", "சம", "பழம்", "கதை கணக்கு"],
}

SECTION_TOPIC_MAP: dict[str, str] = {
    "divisibility_rules": "4.1",
    "digit_sum": "4.1",
    "factor_listing": "4.2",
    "prime_factorization": "4.3",
    "factors_via_prime": "4.4",
    "hcf": "4.5",
    "lcm": "4.6",
    "word_problem": "4.6",
}

_INTENT_ENUM: dict[str, Intent] = {i.name: i for i in Intent}


class IntentAgent:
    """Keyword-based intent and topic parsing aligned with curriculum chapter-4 scaffolding."""
    def __init__(self, topic_pack: ChapterTopicPack | None = None, chapter: int = 4):
        pack = topic_pack or get_chapter_plugin(chapter).topic_pack
        self.intent_priority = pack.intent_priority
        self.intent_keywords = pack.intent_keywords
        self.topic_keywords = pack.topic_keywords
        self.section_topic_map = pack.section_topic_map
        self.default_topic = pack.default_topic

    def _intent_scores(self, query: str) -> dict[str, int]:
        query_lower = query.lower()
        scores: dict[str, int] = {k: 0 for k in self.intent_keywords}
        for intent, keywords in self.intent_keywords.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    scores[intent] += 1
        return scores

    def classify(self, query: str) -> Intent:
        scores = self._intent_scores(query)
        for name in self.intent_priority:
            if scores.get(name, 0) > 0:
                return _INTENT_ENUM[name]
        return Intent.EXPLAIN

    def detect_topic(self, query: str, last_topic: str) -> str:
        text = query.lower()
        scores = {t: 0 for t in self.topic_keywords}
        for topic, kws in self.topic_keywords.items():
            for kw in kws:
                if kw.lower() in text:
                    scores[topic] += 1
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return last_topic or self.default_topic
        return best

    def extract_numbers(self, query: str) -> list[int]:
        return [int(x) for x in re.findall(r"\b\d+\b", query)]

    def parse(
        self,
        raw_query: str,
        normalized_query: str,
        dialect: Dialect,
        student: StudentProfile,
        student_answer: Optional[str] = None,
        exercise_topic: Optional[str] = None,
    ) -> QueryContext:
        text = normalized_query.lower()
        scores = self._intent_scores(normalized_query)
        if student_answer:
            scores["CHECK_ANSWER"] = scores.get("CHECK_ANSWER", 0) + 5

        chosen_name = Intent.EXPLAIN.name
        for name in self.intent_priority:
            if scores.get(name, 0) > 0:
                chosen_name = name
                break
        best_intent = _INTENT_ENUM[chosen_name]
        confidence = min(scores.get(chosen_name, 0) / 5.0, 1.0)

        best_topic = self.detect_topic(normalized_query, student.last_topic or "")
        numbers = [
            n for n in self.extract_numbers(raw_query) if n < 100_000
        ]

        method = None
        if "வகுத்தல் முறை" in normalized_query or "division" in text:
            method = "division"
        elif "காரணி மரம்" in normalized_query or "factor tree" in text:
            method = "factor_tree"
        elif "முறை I" in normalized_query or "பட்டியல்" in normalized_query:
            method = "list"

        return QueryContext(
            raw_query=raw_query,
            normalized_query=normalized_query,
            intent=best_intent,
            topic=best_topic,
            section=self.section_topic_map.get(best_topic, "4.1"),
            numbers=numbers,
            method_requested=method,
            is_word_problem=(
                best_intent == Intent.WORD_PROBLEM or best_topic == "word_problem"
            ),
            dialect=dialect,
            confidence=confidence,
            student_answer=student_answer,
            exercise_topic=exercise_topic,
        )
