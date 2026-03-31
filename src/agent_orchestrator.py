#!/usr/bin/env python3
"""
agent_orchestrator.py — Multi-Agent Tamil Math Tutor System
============================================================
Enterprise-grade multi-agent architecture replacing the monolithic
AdaptiveRAGEngine. Each agent has a single responsibility and
communicates via a structured message bus (AgentMessage).

Agents:
  OrchestratorAgent   — coordinates all agents, manages conversation
  QueryAgent          — intent + topic + number extraction
  DialectAgent        — detects SL Tamil dialect, normalizes to NIE standard
  RetrievalAgent      — dynamic hybrid search against ChromaDB
  TeachingAgent       — Socratic NIE-register Tamil explanation
  DrawingAgent        — generates diagram JSON specs for Flutter canvas
  ExerciseAgent       — generates NIE-style exercises at adaptive difficulty
  VerificationAgent   — checks answers against NIE marking scheme
  StudentProfileAgent — tracks mastery, updates skill scores

Usage:
  python agent_orchestrator.py \
    --student-id SL_TM_2024_001 \
    --district jaffna \
    --grade 7 --chapter 4 \
    -q "72 உம் 108 உம் ஆகிய எண்களின் பொ.கா.பெ. காண்க"

  # With student answer for verification
  python agent_orchestrator.py \
    --student-id SL_TM_2024_001 -q "36 இன் காரணிகள் காண்க" \
    --student-answer "1, 2, 3, 4, 6, 9, 18, 36" \
    --exercise-topic factor_listing

Environment:
  GEMINI_API_KEY   — required
  NIE_VECTOR_DB    — path to ChromaDB (default: data/vectordb)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("nie.orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE BUS — typed inter-agent communication
# ─────────────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    EXPLAIN       = "EXPLAIN"
    SHOW_METHOD   = "SHOW_METHOD"
    EXERCISE      = "EXERCISE"
    CHECK_ANSWER  = "CHECK_ANSWER"
    DRAW_DIAGRAM  = "DRAW_DIAGRAM"
    WORD_PROBLEM  = "WORD_PROBLEM"
    UNKNOWN       = "UNKNOWN"


class Dialect(str, Enum):
    JAFFNA      = "jaffna"
    BATTICALOA  = "batticaloa"
    ESTATE      = "estate"
    COLOMBO     = "colombo"
    VANNI       = "vanni"
    STANDARD    = "standard"
    UNKNOWN     = "unknown"


@dataclass
class QueryContext:
    """Structured output from QueryAgent — shared across all agents."""
    raw_query: str
    normalized_query: str           # dialect-normalized NIE Tamil
    intent: Intent
    topic: str                      # hcf, lcm, factor_listing, etc.
    section: str                    # 4.5, 4.6 etc
    numbers: list[int]              # numbers mentioned in query
    method_requested: Optional[str] # "வகுத்தல் முறை", "காரணி மரம்" etc
    is_word_problem: bool
    dialect: Dialect
    confidence: float               # 0-1 intent confidence
    student_answer: Optional[str]   # if CHECK_ANSWER intent
    exercise_topic: Optional[str]   # if CHECK_ANSWER intent


@dataclass
class RetrievedContext:
    """Output from RetrievalAgent."""
    chunks: list[dict]              # {id, text, metadata, score}
    answer_scheme_chunks: list[dict]
    total_retrieved: int
    query_embedding: list[float]
    retrieval_time_ms: float


@dataclass
class TeachingResponse:
    """Output from TeachingAgent."""
    explanation_ta: str             # Tamil explanation
    explanation_en: Optional[str]   # optional English gloss
    key_concepts: list[str]
    next_suggested_topic: Optional[str]


@dataclass
class DiagramSpec:
    """Output from DrawingAgent — sent to Flutter canvas."""
    diagram_type: str               # factor_tree, division_ladder, etc.
    spec: dict                      # diagram-specific JSON
    caption_ta: str
    animate: bool = True


@dataclass
class ExerciseBundle:
    """Output from ExerciseAgent."""
    question_ta: str
    numbers: list[int]
    difficulty: int
    topic: str
    hint_ta: Optional[str]
    expected_steps: list[str]       # for Verification agent
    answer: Any


@dataclass
class VerificationResult:
    """Output from VerificationAgent."""
    is_correct: bool
    first_wrong_step: Optional[str]
    socratic_hint_ta: str           # guiding question — never reveals answer
    error_type: Optional[str]
    skill_delta: float              # +ve if correct, -ve if wrong


@dataclass
class AgentResponse:
    """Final aggregated response from OrchestratorAgent."""
    session_id: str
    student_id: str
    intent: Intent
    # Core response
    teaching: Optional[TeachingResponse] = None
    diagram: Optional[DiagramSpec] = None
    exercise: Optional[ExerciseBundle] = None
    verification: Optional[VerificationResult] = None
    # Meta
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    dialect_detected: Dialect = Dialect.UNKNOWN
    processing_time_ms: float = 0.0
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT PROFILE (now owned by StudentProfileAgent, persisted in DB)
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3

@dataclass
class StudentProfile:
    student_id: str
    name: str = "மாணவர்"
    grade: int = 7
    district: str = "unknown"
    dialect: str = "unknown"
    skills: dict = field(default_factory=lambda: {
        "divisibility_rules": 0.0, "digit_sum": 0.0,
        "factor_listing": 0.0, "prime_factorization": 0.0,
        "hcf": 0.0, "lcm": 0.0, "word_problems": 0.0,
    })
    total_questions: int = 0
    total_exercises: int = 0
    correct_exercises: int = 0
    preferred_method: str = ""
    last_topic: str = ""
    last_error_type: str = ""
    session_count: int = 0
    error_patterns: dict = field(default_factory=dict)

    def skill_level(self) -> int:
        avg = sum(self.skills.values()) / len(self.skills)
        return 1 if avg < 0.35 else (2 if avg < 0.65 else 3)

    def mastered_topics(self) -> list[str]:
        return [k for k, v in self.skills.items() if v >= 0.75]

    def weak_topics(self) -> list[str]:
        return [k for k, v in self.skills.items() if v < 0.4]

    def accuracy(self) -> float:
        if self.total_exercises == 0:
            return 0.0
        return self.correct_exercises / self.total_exercises


class StudentProfileAgent:
    """Manages student profiles in SQLite. Thread-safe."""

    DB_PATH = Path("data/student_profiles.db")

    def __init__(self):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    student_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT, intent TEXT, topic TEXT,
                    correct INTEGER, dialect TEXT, timestamp TEXT,
                    response_summary TEXT
                )""")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_student
                ON sessions(student_id)""")

    def get_or_create(self, student_id: str, name: str = "மாணவர்",
                       district: str = "unknown") -> StudentProfile:
        with sqlite3.connect(self.DB_PATH) as conn:
            row = conn.execute(
                "SELECT profile_json FROM profiles WHERE student_id=?",
                (student_id,)).fetchone()
        if row:
            return StudentProfile(**json.loads(row[0]))
        profile = StudentProfile(student_id=student_id, name=name,
                                  district=district)
        self.save(profile)
        return profile

    def save(self, profile: StudentProfile):
        from datetime import datetime
        data = json.dumps(asdict(profile))
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO profiles (student_id, profile_json, updated_at)
                VALUES (?, ?, ?)""",
                (profile.student_id, data, datetime.now().isoformat()))

    def record_outcome(self, profile: StudentProfile, topic: str,
                        correct: bool, difficulty: int,
                        error_type: str = "") -> StudentProfile:
        """Update skill scores based on exercise outcome."""
        skill_topic_map = {
            "divisibility_rules": ["divisibility_2", "divisibility_3",
                                   "divisibility_9", "divisibility_6", "divisibility_4"],
            "digit_sum": ["digit_sum"],
            "factor_listing": ["factor_listing", "factor_definition", "factor_pairs"],
            "prime_factorization": ["prime_factorization_tree",
                                    "prime_factorization_division"],
            "hcf": ["hcf", "hcf_method_1_list", "hcf_method_2_prime",
                    "hcf_method_3_division", "hcf_word_problem"],
            "lcm": ["lcm", "lcm_prime_method", "lcm_division_method", "lcm_word_problem"],
            "word_problems": ["hcf_word_problem", "lcm_word_problem"],
        }
        for skill, topics in skill_topic_map.items():
            if topic in topics or topic == skill:
                delta = 0.1 * difficulty if correct else -0.05
                profile.skills[skill] = max(0.0, min(1.0,
                                            profile.skills[skill] + delta))
        profile.total_exercises += 1
        if correct:
            profile.correct_exercises += 1
        if error_type:
            profile.error_patterns[error_type] = \
                profile.error_patterns.get(error_type, 0) + 1
        profile.last_error_type = error_type
        self.save(profile)
        return profile


# ─────────────────────────────────────────────────────────────────────────────
# DIALECT AGENT
# ─────────────────────────────────────────────────────────────────────────────

# Dialect vocabulary signatures — unique words/patterns per region
DIALECT_SIGNATURES = {
    Dialect.JAFFNA: [
        "வகுதல்", "ஆகும்", "என்பது", "காண்போம்", "செய்க",
        "விடை", "கேட்கப்படுகிறது"
    ],
    Dialect.BATTICALOA: [
        "வகுத்தல்னா", "எவ்வளவு", "போடு", "போடுவது",
        "இது என்னன்னு"
    ],
    Dialect.ESTATE: [
        "வகுத்தல்க்கு", "பண்ணுவது", "சொல்லுங்க",
        "இதுக்கு", "அதுக்கு", "எப்படி பண்றது"
    ],
    Dialect.COLOMBO: [
        "factor", "HCF", "LCM", "find பண்றது", "calculate",
        "எப்படி find", "answer என்ன", "method காட்டு"
    ],
    Dialect.VANNI: [
        "வகுத்தல்", "காண்பது", "எப்படி கண்டுபிடிப்பது"
    ],
}

# Dialect vocabulary → NIE standard normalization
DIALECT_NORMALIZER = {
    # Estate Tamil → NIE standard
    "வகுத்தல்க்கு": "வகுத்தல் மூலம்",
    "பண்ணுவது": "செய்வது",
    "சொல்லுங்க": "சொல்லுங்கள்",
    "இதுக்கு": "இதற்கு",
    "அதுக்கு": "அதற்கு",
    "எப்படி பண்றது": "எப்படி செய்வது",
    # Colombo code-switching → NIE standard
    "factor காண்க": "காரணி காண்க",
    "HCF காண்க": "பொ.கா.பெ. காண்க",
    "LCM காண்க": "பொ.ம.சி. காண்க",
    "find பண்றது": "காண்பது",
    "calculate பண்க": "கணக்கிட்டு காண்க",
    "answer என்ன": "விடை என்ன",
    # Batticaloa → NIE standard
    "வகுத்தல்னா": "வகுத்தல் என்றால்",
    "போடு": "எழுதுக",
    "இது என்னன்னு": "இது என்னவென்று",
}


class DialectAgent:
    """
    Detects Sri Lankan Tamil regional dialect and normalizes to NIE standard.
    
    Two modes:
    1. Rule-based (fast, PoC): keyword signature matching
    2. LLM-based (production): dedicated Gemini call with dialect examples
    """

    def __init__(self, use_llm: bool = False, gemini_client=None):
        self.use_llm = use_llm
        self.gemini_client = gemini_client

    def detect_and_normalize(self, raw_query: str,
                              student_district: str = "unknown") -> tuple[Dialect, str]:
        """Returns (detected_dialect, normalized_query)."""
        dialect = self._detect_dialect(raw_query, student_district)
        normalized = self._normalize(raw_query, dialect)
        return dialect, normalized

    def _detect_dialect(self, text: str, district: str) -> Dialect:
        # Fast: district override
        district_map = {
            "jaffna": Dialect.JAFFNA, "kilinochchi": Dialect.JAFFNA,
            "mannar": Dialect.VANNI, "vavuniya": Dialect.VANNI,
            "batticaloa": Dialect.BATTICALOA, "ampara": Dialect.BATTICALOA,
            "trincomalee": Dialect.BATTICALOA,
            "nuwara_eliya": Dialect.ESTATE, "hatton": Dialect.ESTATE,
            "badulla": Dialect.ESTATE, "kandy": Dialect.ESTATE,
            "colombo": Dialect.COLOMBO, "gampaha": Dialect.COLOMBO,
        }
        if district.lower() in district_map:
            return district_map[district.lower()]

        # Keyword signature matching
        scores = {d: 0 for d in Dialect}
        for dialect, keywords in DIALECT_SIGNATURES.items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    scores[dialect] += 1

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return Dialect.UNKNOWN
        return best

    def _normalize(self, text: str, dialect: Dialect) -> str:
        """Map dialect vocabulary to NIE standard Tamil."""
        normalized = text
        for variant, standard in DIALECT_NORMALIZER.items():
            normalized = normalized.replace(variant, standard)
        return normalized

    async def detect_llm_async(self, query: str, client,
                                model: str = "gemini-2.0-flash") -> tuple[Dialect, str]:
        """Production mode: LLM-based dialect detection."""
        if not client:
            return Dialect.UNKNOWN, query

        system = """நீங்கள் இலங்கை தமிழ் மொழியியல் நிபுணர்.
கீழ்க்காணும் வாக்கியம் எந்த இலங்கை தமிழ் வகையிலிருந்து வருகிறது என்று கண்டறிக.
வகைகள்: jaffna, batticaloa, estate, colombo, vanni, unknown

பதிலை JSON ஆக மட்டுமே தரவும்:
{"dialect": "jaffna", "confidence": 0.85, "normalized": "<NIE standard Tamil version>"}"""

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system, temperature=0.1, max_output_tokens=200)
            response = client.models.generate_content(
                model=model, contents=query, config=config)
            data = json.loads(response.text.strip())
            dialect_str = data.get("dialect", "unknown")
            dialect = Dialect(dialect_str) if dialect_str in [d.value for d in Dialect] \
                     else Dialect.UNKNOWN
            return dialect, data.get("normalized", query)
        except Exception as e:
            log.warning(f"Dialect LLM failed: {e}")
            return Dialect.UNKNOWN, query


# ─────────────────────────────────────────────────────────────────────────────
# QUERY AGENT
# ─────────────────────────────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    Intent.EXPLAIN:      ["என்றால் என்ன", "என்ன", "விளக்கு", "புரியவில்லை",
                          "கற்றுக்கொடு", "சொல்லுங்கள்", "எப்படி", "what is"],
    Intent.SHOW_METHOD:  ["முறை காட்டு", "எப்படி காண்பது", "step", "படிகள்",
                          "முறை", "method", "வகுத்தல் முறை", "ஏணி"],
    Intent.EXERCISE:     ["பயிற்சி", "கேள்வி கொடு", "கணக்கு கொடு",
                          "practice", "exercise", "கொடு"],
    Intent.CHECK_ANSWER: ["சரியா", "விடை", "இது சரியா", "என் பதில்",
                          "check", "correct", "="],
    Intent.DRAW_DIAGRAM: ["வரை", "மரம் வரை", "ஏணி காட்டு", "படம்",
                          "draw", "diagram", "காரணி மரம்", "வகுத்தல் ஏணி"],
    Intent.WORD_PROBLEM: ["கதை கணக்கு", "பென்சில்", "மணி ஒலிக்கும்",
                          "பழம்", "பொதி", "பகிர்", "word problem"],
}

TOPIC_KEYWORDS = {
    "divisibility_rules": ["வகுபடும்", "வகுபடாது", "÷2", "÷3", "÷9", "÷6", "÷4", "÷5"],
    "digit_sum":          ["இலக்கச் சுட்டி", "digit sum"],
    "factor_listing":     ["காரணி", "காரணிகள்", "factor"],
    "prime_factorization":["முதன்மை", "prime", "காரணி மரம்", "ஏணி"],
    "hcf":                ["பொ.கா.பெ.", "பொதுக் காரணி", "HCF", "GCD", "பெரியது"],
    "lcm":                ["பொ.ம.சி.", "பொது மடங்கு", "LCM", "சிறியது", "மடங்கு"],
    "word_problem":       ["பொதி", "மணி", "பகிர்", "சம", "பழம்"],
}

SECTION_TOPIC_MAP = {
    "divisibility_rules": "4.1", "digit_sum": "4.1",
    "factor_listing": "4.2", "prime_factorization": "4.3",
    "factors_via_prime": "4.4", "hcf": "4.5", "lcm": "4.6",
    "word_problem": "4.6",
}


class QueryAgent:
    """
    Parses student query into structured QueryContext.
    Combines rule-based (fast) + optional LLM refinement (accurate).
    """

    def __init__(self, use_llm: bool = True, gemini_client=None,
                 gemini_model: str = "gemini-2.0-flash"):
        self.use_llm = use_llm
        self.gemini_client = gemini_client
        self.gemini_model = gemini_model

    async def parse(self, raw_query: str, normalized_query: str,
                    dialect: Dialect, student: StudentProfile,
                    student_answer: str = None,
                    exercise_topic: str = None) -> QueryContext:
        """Parse query into structured context. Uses LLM if enabled."""
        if self.use_llm and self.gemini_client:
            return await self._parse_llm(raw_query, normalized_query,
                                          dialect, student,
                                          student_answer, exercise_topic)
        return self._parse_rules(raw_query, normalized_query, dialect,
                                  student, student_answer, exercise_topic)

    def _parse_rules(self, raw: str, normalized: str, dialect: Dialect,
                      student: StudentProfile,
                      student_answer: str, exercise_topic: str) -> QueryContext:
        import re
        text = normalized.lower()

        # Intent
        intent_scores = {i: 0 for i in Intent}
        for intent, kws in INTENT_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in text:
                    intent_scores[intent] += 1
        if student_answer:
            intent_scores[Intent.CHECK_ANSWER] += 5
        best_intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[best_intent] == 0:
            best_intent = Intent.EXPLAIN
        confidence = min(intent_scores[best_intent] / 5.0, 1.0)

        # Topic
        topic_scores = {t: 0 for t in TOPIC_KEYWORDS}
        for topic, kws in TOPIC_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in text:
                    topic_scores[topic] += 1
        best_topic = max(topic_scores, key=topic_scores.get)
        if topic_scores[best_topic] == 0:
            best_topic = student.last_topic or "factor_listing"

        numbers = [int(n) for n in re.findall(r'\b\d+\b', raw) if int(n) < 100000]

        # Method requested
        method = None
        if "வகுத்தல் முறை" in normalized or "division" in text.lower():
            method = "division"
        elif "காரணி மரம்" in normalized or "factor tree" in text.lower():
            method = "factor_tree"
        elif "முறை I" in normalized or "பட்டியல்" in normalized:
            method = "list"

        return QueryContext(
            raw_query=raw,
            normalized_query=normalized,
            intent=best_intent,
            topic=best_topic,
            section=SECTION_TOPIC_MAP.get(best_topic, "4.1"),
            numbers=numbers,
            method_requested=method,
            is_word_problem=best_intent == Intent.WORD_PROBLEM
                            or best_topic == "word_problem",
            dialect=dialect,
            confidence=confidence,
            student_answer=student_answer,
            exercise_topic=exercise_topic,
        )

    async def _parse_llm(self, raw: str, normalized: str, dialect: Dialect,
                          student: StudentProfile,
                          student_answer: str, exercise_topic: str) -> QueryContext:
        """Use Gemini to parse query intent and topic accurately."""
        system = f"""நீங்கள் NIE Grade {student.grade} கணித கேள்வி பகுப்பாய்வாளர்.
கீழ்க்காணும் மாணவர் கேள்வியை பகுப்பாய்ந்து JSON மட்டும் தரவும்:

{{
  "intent": "EXPLAIN|SHOW_METHOD|EXERCISE|CHECK_ANSWER|DRAW_DIAGRAM|WORD_PROBLEM",
  "topic": "divisibility_rules|digit_sum|factor_listing|prime_factorization|hcf|lcm|word_problem",
  "section": "4.1|4.2|4.3|4.4|4.5|4.6",
  "numbers": [list of integers in question],
  "method_requested": null or "division|factor_tree|list|prime",
  "is_word_problem": true/false,
  "confidence": 0.0-1.0
}}

மாணவர் வட்டாரம்: {dialect.value}
மாணவர் திறன் நிலை: {student.skill_level()}/3
கடைசிப் படித்த தலைப்பு: {student.last_topic}
மாணவர் பதில் (ஏதேனும்): {student_answer or 'இல்லை'}"""

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system, temperature=0.1,
                max_output_tokens=300)
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model, contents=normalized, config=config)

            # Strip markdown code fences if present
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip().strip("```")

            data = json.loads(text)

            return QueryContext(
                raw_query=raw,
                normalized_query=normalized,
                intent=Intent(data.get("intent", "EXPLAIN")),
                topic=data.get("topic", "factor_listing"),
                section=data.get("section", "4.1"),
                numbers=data.get("numbers", []),
                method_requested=data.get("method_requested"),
                is_word_problem=data.get("is_word_problem", False),
                dialect=dialect,
                confidence=data.get("confidence", 0.7),
                student_answer=student_answer,
                exercise_topic=exercise_topic,
            )
        except Exception as e:
            log.warning(f"QueryAgent LLM failed: {e}. Falling back to rules.")
            return self._parse_rules(raw, normalized, dialect, student,
                                      student_answer, exercise_topic)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL AGENT
# ─────────────────────────────────────────────────────────────────────────────

PREREQUISITE_GRAPH = {
    "digit_sum": ["divisibility_rules"],
    "factor_listing": ["divisibility_rules"],
    "prime_factorization": ["factor_listing", "divisibility_rules"],
    "factors_via_prime": ["prime_factorization"],
    "hcf": ["prime_factorization", "factor_listing"],
    "lcm": ["prime_factorization", "factor_listing"],
    "word_problem": ["hcf", "lcm"],
}


class RetrievalAgent:
    """
    Dynamic hybrid retrieval from ChromaDB.
    Replaces the hardcoded NIE_CORPUS list entirely.
    
    Strategy:
    1. Pre-filter by metadata (student skill level, prerequisites)
    2. Vector similarity search on remaining chunks
    3. Re-rank by student profile (preferred method boost)
    4. Separately retrieve answer scheme chunks if CHECK_ANSWER
    """

    def __init__(self, vector_store, embedder):
        self.store = vector_store
        self.embedder = embedder

    async def retrieve(self, query_ctx: QueryContext, student: StudentProfile,
                       grade: int, chapter: int, subject: str,
                       n_results: int = 6) -> RetrievedContext:
        t0 = time.perf_counter()

        # Build metadata filter based on student level
        where_filter = self._build_filter(query_ctx, student)

        # Embed query
        query_emb = self.embedder.embed_query(query_ctx.normalized_query)

        # Main content retrieval
        chunks = self.store.hybrid_query(
            query_embedding=query_emb,
            grade=grade, chapter=chapter, subject=subject,
            n_results=n_results,
            where_filter=where_filter,
        )

        # Rerank by student profile
        chunks = self._rerank(chunks, student, query_ctx)

        # Auto-inject prerequisite chunks if student weak
        chunks = self._inject_prerequisites(chunks, student,
                                             query_emb, grade, chapter, subject)

        # Retrieve answer scheme if checking answer
        answer_chunks = []
        if query_ctx.intent == Intent.CHECK_ANSWER:
            answer_chunks = self.store.hybrid_query(
                query_embedding=query_emb,
                grade=grade, chapter=chapter, subject=subject,
                n_results=4,
                where_filter={"is_answer_scheme": True},
                collection_type="answers",
            )

        elapsed = (time.perf_counter() - t0) * 1000
        return RetrievedContext(
            chunks=chunks,
            answer_scheme_chunks=answer_chunks,
            total_retrieved=len(chunks),
            query_embedding=query_emb,
            retrieval_time_ms=elapsed,
        )

    def _build_filter(self, ctx: QueryContext, student: StudentProfile) -> dict:
        """ChromaDB where filter to pre-filter by metadata."""
        max_diff = student.skill_level() + 1  # allow one level above current

        # Build filter — ChromaDB supports $and, $or, $eq, $lte, $gte
        filters = {"difficulty": {"$lte": max_diff}}

        # If specific topic identified, boost with topic filter
        if ctx.topic and ctx.topic != "unknown":
            # Don't hard-filter by topic — let vector search handle it
            # Just use difficulty ceiling
            pass

        return filters

    def _rerank(self, chunks: list[dict], student: StudentProfile,
                ctx: QueryContext) -> list[dict]:
        """Rerank chunks by student-specific signals."""
        def score(chunk: dict) -> float:
            base = chunk.get("score", 0.5)
            meta = chunk.get("metadata")
            if not meta:
                return base

            # Boost preferred method
            if (student.preferred_method and
                    student.preferred_method in (meta.diagram_types or [])):
                base += 0.1

            # Boost exact topic match
            if meta.topic == ctx.topic:
                base += 0.15

            # Boost chunks with numbers that appear in query
            if meta.has_numbers and ctx.numbers:
                base += 0.05

            # Slight penalty for already-mastered difficulty-1 chunks
            if meta.difficulty == 1 and student.skills.get(meta.topic, 0) > 0.8:
                base -= 0.1

            return base

        return sorted(chunks, key=score, reverse=True)

    def _inject_prerequisites(self, chunks: list[dict], student: StudentProfile,
                               query_emb: list[float],
                               grade: int, chapter: int, subject: str) -> list[dict]:
        """Inject prerequisite chunks when student skill < 0.4."""
        injected_topics = set()
        for chunk in chunks[:3]:  # check top 3 retrieved
            meta = chunk.get("metadata")
            if not meta:
                continue
            for prereq_topic in PREREQUISITE_GRAPH.get(meta.topic, []):
                if (student.skills.get(prereq_topic, 0) < 0.4
                        and prereq_topic not in injected_topics):
                    prereq_chunks = self.store.hybrid_query(
                        query_embedding=query_emb,
                        grade=grade, chapter=chapter, subject=subject,
                        n_results=1,
                        where_filter={"topic": prereq_topic},
                    )
                    if prereq_chunks:
                        chunks.insert(0, prereq_chunks[0])
                        injected_topics.add(prereq_topic)

        return chunks[:8]  # cap at 8 total


# ─────────────────────────────────────────────────────────────────────────────
# TEACHING AGENT
# ─────────────────────────────────────────────────────────────────────────────

class TeachingAgent:
    """
    Generates Socratic NIE-register Tamil explanations.
    Adapts depth and method to student skill level and dialect.
    """

    def __init__(self, gemini_client, model: str = "gemini-2.5-flash"):
        self.client = gemini_client
        self.model = model

    def build_system_prompt(self, ctx: QueryContext,
                             student: StudentProfile,
                             retrieved: RetrievedContext) -> str:
        skill_summary = ", ".join([
            f"{k}: {v:.1f}" for k, v in student.skills.items() if v > 0.05
        ]) or "தொடக்க நிலை"

        context_text = "\n\n---\n\n".join([
            f"[பகுதி {c['metadata'].section if c.get('metadata') else '?'} "
            f"— {c['metadata'].topic if c.get('metadata') else '?'}]\n{c['text']}"
            for c in retrieved.chunks
        ])

        dialect_note = {
            Dialect.ESTATE: "இந்த மாணவர் மலையக வட்டாரத்தைச் சேர்ந்தவர். மிகவும் எளிமையான மொழியில் விளக்கவும்.",
            Dialect.BATTICALOA: "இந்த மாணவர் மட்டக்களப்பு வட்டாரத்தைச் சேர்ந்தவர்.",
            Dialect.COLOMBO: "இந்த மாணவர் கொழும்பு வட்டாரத்தைச் சேர்ந்தவர். ஆங்கில சொற்களை தமிழாக்கி விளக்கவும்.",
            Dialect.JAFFNA: "யாழ்ப்பாணம் வட்டாரம் — NIE நிலையான தமிழ் பொருத்தமானது.",
        }.get(ctx.dialect, "")

        depth_instruction = {
            1: "மிகவும் எளிமையாக, படிப்படியாக, நிறைய உதாரணங்களுடன் விளக்கவும்.",
            2: "தெளிவாக விளக்கி NIE பயிற்சி கணக்குகளில் முயற்சிக்க ஊக்குவிக்கவும்.",
            3: "சுருக்கமாக விளக்கி சிந்தனைக்கு நிலை கேள்விகள் கேளுங்கள்.",
        }.get(student.skill_level(), "")

        return f"""நீங்கள் ஒரு அனுபவமிக்க கணித ஆசிரியர்.
தரம் {student.grade} NIE இலவசப் பாடநூல் படி தமிழ் வழியில் கற்பிக்கிறீர்கள்.

{dialect_note}
{depth_instruction}

கட்டாய மொழி விதிகள்:
• எப்போதும் NIE தமிழ் பாடநூலின் சொற்களையே பயன்படுத்தவும்
• காரணி, மடங்கு, இலக்கச் சுட்டி, பொ.கா.பெ., பொ.ம.சி. — இந்த NIE சொற்களையே பயன்படுத்தவும்
• ஆங்கிலம்-தமிழ் கலவை பயன்படுத்தாதீர்கள்
• மாணவர் தவறு செய்தால் நேரடியாக விடையை சொல்லாதீர்கள்
• ஒரே ஒரு வழிகாட்டும் கேள்வி மட்டும் கேளுங்கள்

மாணவர் தகவல்:
• திறன் நிலை: {student.skill_level()}/3
• திறன்கள்: {skill_summary}
• கடைசிப் பிழை: {student.last_error_type or 'இல்லை'}
• வட்டாரம்: {ctx.dialect.value}

NIE பாடநூல் உள்ளடக்கம் (இதை மட்டுமே பயன்படுத்தவும்):
{context_text}"""

    async def generate(self, ctx: QueryContext, student: StudentProfile,
                        retrieved: RetrievedContext,
                        temperature: float = 0.3,
                        max_tokens: int = 1500) -> TeachingResponse:
        from google.genai import types

        system = self.build_system_prompt(ctx, student, retrieved)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=ctx.normalized_query,
            config=config,
        )

        explanation = (response.text or "").strip()

        # Extract key concepts mentioned
        key_concepts = []
        for term in ["காரணி", "பொ.கா.பெ.", "பொ.ம.சி.", "இலக்கச் சுட்டி",
                     "முதன்மைக் காரணி", "வகுத்தல் முறை"]:
            if term in explanation:
                key_concepts.append(term)

        # Suggest next topic
        next_topic = self._suggest_next(ctx.topic, student)

        return TeachingResponse(
            explanation_ta=explanation,
            explanation_en=None,
            key_concepts=key_concepts,
            next_suggested_topic=next_topic,
        )

    def _suggest_next(self, current_topic: str,
                       student: StudentProfile) -> Optional[str]:
        progression = [
            "divisibility_rules", "digit_sum", "factor_listing",
            "prime_factorization", "factors_via_prime", "hcf", "lcm",
            "word_problem"
        ]
        try:
            idx = progression.index(current_topic)
            if idx < len(progression) - 1:
                next_t = progression[idx + 1]
                if student.skills.get(current_topic, 0) >= 0.6:
                    return next_t
        except ValueError:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DrawingAgent:
    """
    Generates diagram JSON specs for Flutter canvas rendering.
    Uses LLM to determine what to draw, then deterministic math for spec.
    
    Supported diagrams:
    - factor_tree: prime factorization tree
    - division_ladder: HCF/LCM via division
    - factor_pairs: all factor pairs of a number
    - multiples_line: common multiples visualization
    - venn_diagram: for showing common factors/multiples
    """

    def should_draw(self, ctx: QueryContext,
                     retrieved: RetrievedContext) -> bool:
        if ctx.intent == Intent.DRAW_DIAGRAM:
            return True
        if ctx.intent == Intent.SHOW_METHOD:
            return True
        # Auto-trigger for methods with natural diagram counterparts
        draw_topics = {"prime_factorization", "factor_listing",
                        "hcf", "lcm", "factors_via_prime"}
        if ctx.topic in draw_topics and ctx.numbers:
            return True
        return False

    def generate(self, ctx: QueryContext) -> Optional[DiagramSpec]:
        if not ctx.numbers:
            return None

        diagram_type = self._select_diagram_type(ctx)
        if not diagram_type:
            return None

        spec = self._build_spec(diagram_type, ctx)
        caption = self._caption(diagram_type, ctx)

        return DiagramSpec(
            diagram_type=diagram_type,
            spec=spec,
            caption_ta=caption,
            animate=True,
        )

    def _select_diagram_type(self, ctx: QueryContext) -> Optional[str]:
        query = ctx.normalized_query
        if "காரணி மரம்" in query or ctx.method_requested == "factor_tree":
            return "factor_tree"
        if "வகுத்தல் ஏணி" in query or ctx.method_requested == "division":
            return "division_ladder"
        if ctx.topic == "hcf" and len(ctx.numbers) >= 2:
            return "division_ladder"
        if ctx.topic == "prime_factorization" and ctx.numbers:
            return "factor_tree"
        if ctx.topic == "lcm" and ctx.numbers:
            return "multiples_line"
        if ctx.topic == "factor_listing" and ctx.numbers:
            return "factor_pairs"
        return None

    def _build_spec(self, diagram_type: str, ctx: QueryContext) -> dict:
        nums = ctx.numbers
        if diagram_type == "factor_tree":
            return self._factor_tree_spec(nums[0])
        elif diagram_type == "division_ladder":
            return self._division_ladder_spec(nums[:3])
        elif diagram_type == "factor_pairs":
            return self._factor_pairs_spec(nums[0])
        elif diagram_type == "multiples_line":
            return self._multiples_line_spec(nums[:3])
        return {}

    def _prime_factors(self, n: int) -> list[int]:
        factors, d = [], 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        return factors

    def _factor_tree_spec(self, n: int) -> dict:
        branches, remaining = [], n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            while remaining > 1 and remaining % p == 0:
                child = remaining // p
                branches.append({"divisor": p, "child": child})
                remaining = child
        primes = self._prime_factors(n)
        return {
            "diagram": "factor_tree",
            "root": n,
            "branches": branches,
            "prime_factors": primes,
            "result_ta": f"{n} = " + " × ".join(map(str, primes)),
            "highlight_primes": True,
            "animate_step_by_step": True,
        }

    def _division_ladder_spec(self, numbers: list[int]) -> dict:
        steps, remaining = [], list(numbers)
        primes_used = []
        for p in [2, 3, 5, 7, 11, 13]:
            if all(r == 1 for r in remaining):
                break
            divisible = [i for i, r in enumerate(remaining) if r % p == 0]
            if len(divisible) >= min(2, len(remaining)):
                new_rem = [r // p if r % p == 0 else r for r in remaining]
                steps.append({"divisor": p,
                               "before": list(remaining),
                               "after": new_rem})
                primes_used.append(p)
                remaining = new_rem
        from functools import reduce
        from math import gcd
        hcf_val = reduce(gcd, numbers)
        return {
            "diagram": "division_ladder",
            "numbers": numbers,
            "steps": steps,
            "hcf": hcf_val,
            "hcf_product_ta": " × ".join(map(str, primes_used))
                              + f" = {hcf_val}" if primes_used else str(hcf_val),
            "animate": True,
        }

    def _factor_pairs_spec(self, n: int) -> dict:
        pairs = [[i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0]
        all_factors = sorted({f for pair in pairs for f in pair})
        return {
            "diagram": "factor_pairs",
            "number": n,
            "pairs": pairs,
            "all_factors": all_factors,
            "count": len(all_factors),
            "label_ta": f"{n} இன் காரணிகள்: " + ", ".join(map(str, all_factors)),
            "animate": True,
        }

    def _multiples_line_spec(self, numbers: list[int]) -> dict:
        from math import lcm
        from functools import reduce
        lcm_val = reduce(lcm, numbers)
        show_to = min(lcm_val * 3, 200)
        colors = {"0": "#378ADD", "1": "#7F77DD", "2": "#1D9E75"}
        return {
            "diagram": "multiples_line",
            "numbers": numbers,
            "lcm": lcm_val,
            "show_to": show_to,
            "highlight_at": [lcm_val, lcm_val * 2],
            "color_map": {str(numbers[i]): colors[str(i % 3)]
                           for i in range(len(numbers))},
            "label_ta": f"பொ.ம.சி. = {lcm_val}",
        }

    def _caption(self, diagram_type: str, ctx: QueryContext) -> str:
        captions = {
            "factor_tree": f"{ctx.numbers[0]} இன் காரணி மரம்",
            "division_ladder": f"{', '.join(map(str, ctx.numbers[:3]))} ஆகிய எண்களின் வகுத்தல் ஏணி",
            "factor_pairs": f"{ctx.numbers[0]} இன் காரணி ஜோடிகள்",
            "multiples_line": f"{', '.join(map(str, ctx.numbers[:3]))} ஆகிய எண்களின் மடங்கு கோடு",
        }
        return captions.get(diagram_type, "கணித வரைபடம்")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ExerciseAgent:
    """
    Generates NIE-style exercises calibrated to student's skill level.
    Exercises follow the பயிற்சி format from the textbook.
    """

    def generate(self, ctx: QueryContext,
                  student: StudentProfile) -> Optional[ExerciseBundle]:
        if ctx.intent not in {Intent.EXERCISE, Intent.EXPLAIN}:
            return None

        import random
        topic = ctx.topic if ctx.topic != "unknown" else student.last_topic
        difficulty = student.skill_level()

        generators = {
            "divisibility_rules": self._gen_divisibility,
            "digit_sum": self._gen_digit_sum,
            "factor_listing": self._gen_factors,
            "prime_factorization": self._gen_prime_factors,
            "hcf": self._gen_hcf,
            "lcm": self._gen_lcm,
            "word_problem": self._gen_word_problem,
        }
        gen = generators.get(topic, self._gen_factors)
        return gen(difficulty)

    def _gen_divisibility(self, difficulty: int) -> ExerciseBundle:
        import random
        divisor = random.choice([2, 3, 6, 9, 4] if difficulty > 1 else [2, 3, 9])
        numbers = random.sample(range(100, 5000), 6)
        correct = [n for n in numbers if n % divisor == 0]
        return ExerciseBundle(
            question_ta=f"பின்வரும் எண்களில் {divisor} ஆல் மீதியின்றி வகுபடும் எண்களை வகுக்காமல் தெரிவு செய்க:\n{', '.join(map(str, numbers))}",
            numbers=numbers,
            difficulty=difficulty,
            topic="divisibility_rules",
            hint_ta=f"வகுபடும் விதி: {self._divisibility_rule(divisor)}",
            expected_steps=[f"ஒவ்வொரு எண்ணுக்கும் {divisor} ஆல் வகுபடும் விதியை பயன்படுத்துக"],
            answer=correct,
        )

    def _divisibility_rule(self, d: int) -> str:
        rules = {
            2: "ஒன்றினிட இலக்கம் இரட்டை எண் ஆயின் 2 ஆல் வகுபடும்",
            3: "இலக்கச் சுட்டி 3 ஆல் வகுபடும் ஆயின் 3 ஆல் வகுபடும்",
            4: "கடைசி இரண்டு இலக்கங்கள் 4 ஆல் வகுபடும் ஆயின் 4 ஆல் வகுபடும்",
            5: "ஒன்றினிட இலக்கம் 0 அல்லது 5 ஆயின் 5 ஆல் வகுபடும்",
            6: "2 ஆலும் 3 ஆலும் வகுபடும் ஆயின் 6 ஆல் வகுபடும்",
            9: "இலக்கச் சுட்டி 9 ஆயின் 9 ஆல் வகுபடும்",
        }
        return rules.get(d, "")

    def _gen_digit_sum(self, difficulty: int) -> ExerciseBundle:
        import random
        numbers = random.sample(range(10, 9999), 5)
        return ExerciseBundle(
            question_ta=f"பின்வரும் எண்களின் இலக்கச் சுட்டியைக் காண்க:\n{', '.join(map(str, numbers))}",
            numbers=numbers,
            difficulty=1,
            topic="digit_sum",
            hint_ta="ஒவ்வொரு எண்ணின் இலக்கங்களையும் கூட்டுக. தனி இலக்கம் வரும் வரை திரும்பவும் கூட்டவும்.",
            expected_steps=["ஒவ்வொரு இலக்கத்தையும் தனியாக எழுதுக",
                            "அனைத்தையும் கூட்டுக",
                            "விடை இரண்டு இலக்கமாக இருந்தால் மீண்டும் கூட்டுக"],
            answer={n: self._digit_sum(n) for n in numbers},
        )

    def _digit_sum(self, n: int) -> int:
        while n >= 10:
            n = sum(int(d) for d in str(n))
        return n

    def _gen_factors(self, difficulty: int) -> ExerciseBundle:
        import random
        pool = {1: [12, 18, 24, 36], 2: [48, 60, 72, 84, 96],
                3: [120, 150, 180, 204]}
        n = random.choice(pool.get(difficulty, pool[2]))
        factors = sorted([i for i in range(1, n+1) if n % i == 0])
        return ExerciseBundle(
            question_ta=f"{n} இன் அனைத்து காரணிகளையும் காண்க.",
            numbers=[n],
            difficulty=difficulty,
            topic="factor_listing",
            hint_ta="ஜோடி பெருக்க முறை: 1 × ?, 2 × ?, 3 × ? ... என்று காண்க",
            expected_steps=[f"{n} = 1 × {n}", f"ஒவ்வொரு ஜோடியும் எழுதுக",
                            "அனைத்து காரணிகளை ஏறுவரிசையில் எழுதுக"],
            answer=factors,
        )

    def _gen_prime_factors(self, difficulty: int) -> ExerciseBundle:
        import random
        pool = {1: [12, 18, 30], 2: [48, 60, 84, 90],
                3: [120, 168, 210, 252]}
        n = random.choice(pool.get(difficulty, pool[2]))
        from functools import reduce
        from math import gcd

        def prime_fact(x):
            factors, d = [], 2
            while x > 1:
                while x % d == 0:
                    factors.append(d)
                    x //= d
                d += 1
            return factors

        primes = prime_fact(n)
        return ExerciseBundle(
            question_ta=f"{n} ஐ முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுக.",
            numbers=[n],
            difficulty=difficulty,
            topic="prime_factorization",
            hint_ta="மிகச் சிறிய முதன்மை எண்ணான 2 இலிருந்து தொடங்கி வகுத்தல் ஏணி வரையுங்கள்",
            expected_steps=[f"{n} ÷ {primes[0]} = {n//primes[0]}",
                            "விடை 1 ஆகும் வரை தொடரவும்",
                            f"{n} = " + " × ".join(map(str, primes))],
            answer=" × ".join(map(str, primes)),
        )

    def _gen_hcf(self, difficulty: int) -> ExerciseBundle:
        import random
        from math import gcd
        pairs = {1: [(12,18), (24,36)], 2: [(48,72), (60,90), (84,108)],
                 3: [(72,108,144), (36,54,90)]}
        nums = random.choice(pairs.get(difficulty, pairs[2]))
        hcf_val = nums[0]
        for n in nums[1:]:
            hcf_val = gcd(hcf_val, n)
        return ExerciseBundle(
            question_ta=f"{', '.join(map(str, nums))} ஆகிய எண்களின் பொ.கா.பெ. காண்க.",
            numbers=list(nums),
            difficulty=difficulty,
            topic="hcf",
            hint_ta="வகுத்தல் முறை அல்லது முதன்மைக் காரணிகள் மூலம் காண்க",
            expected_steps=["ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளாக பகுக்கவும்",
                            "பொதுவான முதன்மைக் காரணிகளைக் காண்க",
                            "அவற்றை பெருக்குக"],
            answer=hcf_val,
        )

    def _gen_lcm(self, difficulty: int) -> ExerciseBundle:
        import random
        from math import lcm
        from functools import reduce
        pairs = {1: [(2,3), (4,6)], 2: [(6,8,12), (4,9,12)],
                 3: [(8,12,18), (6,10,15)]}
        nums = random.choice(pairs.get(difficulty, pairs[2]))
        lcm_val = reduce(lcm, nums)
        return ExerciseBundle(
            question_ta=f"{', '.join(map(str, nums))} ஆகிய எண்களின் பொ.ம.சி. காண்க.",
            numbers=list(nums),
            difficulty=difficulty,
            topic="lcm",
            hint_ta="முதன்மைக் காரணிகளின் உயர் வலுவைப் பெருக்குக",
            expected_steps=["ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளாக பகுக்கவும்",
                            "ஒவ்வொரு முதன்மை எண்ணின் உயர் வலுவைத் தேர்வு செய்க",
                            "அவற்றை பெருக்குக"],
            answer=lcm_val,
        )

    def _gen_word_problem(self, difficulty: int) -> ExerciseBundle:
        import random
        problems = [
            {
                "q": "ஒரு கூடையில் 96 அப்பிள்களும் 60 ஆரஞ்சு பழங்களும் உள்ளன. இரு வகைப் பழங்களும் சம எண்ணிக்கையில் இருக்கும் வகையில் பொதிகளில் இடப்பட்டால் பெறக்கூடிய அதிகூடிய பொதிகளின் எண்ணிக்கை யாது?",
                "nums": [96, 60], "answer": 12, "topic": "hcf",
                "hint": "சம பகிர்வு → பொ.கா.பெ. பயன்படுத்தவும்"
            },
            {
                "q": "இரண்டு மணிகள் முறையே 6 நிமிடங்கள், 8 நிமிடங்களுக்கு ஒரு முறை ஒலிக்கின்றன. காலை 8.00 மணிக்கு ஒருமித்து ஒலித்தால், அவை மீண்டும் எத்தனை மணிக்கு ஒருமித்து ஒலிக்கும்?",
                "nums": [6, 8], "answer": "8.24 மணி", "topic": "lcm",
                "hint": "முதல் சந்திப்பு → பொ.ம.சி. பயன்படுத்தவும்"
            },
        ]
        prob = random.choice(problems)
        return ExerciseBundle(
            question_ta=prob["q"],
            numbers=prob["nums"],
            difficulty=3,
            topic=prob["topic"],
            hint_ta=prob["hint"],
            expected_steps=["தேவையான தகவல்களை எழுதுக",
                            f"பொ.கா.பெ. அல்லது பொ.ம.சி. தீர்மானி",
                            "கணக்கிட்டு விடை எழுதுக"],
            answer=prob["answer"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION AGENT
# ─────────────────────────────────────────────────────────────────────────────

SOCRATIC_HINTS = {
    "used_lcm_for_hcf": "நீங்கள் மடங்குகளைக் காண்கிறீர்கள். 'அதிகூடிய பொதிகள்' என்றால் வகுபடுதல் தேவையா அல்லது மடங்கு தேவையா?",
    "used_hcf_for_lcm": "நீங்கள் காரணிகளைக் காண்கிறீர்கள். 'முதல் சந்திப்பு' என்றால் பொது மடங்கு வேண்டுமா?",
    "wrong_digit_sum": "இலக்கச் சுட்டி இரண்டு இலக்கமாக வந்தால் மீண்டும் கூட வேண்டும். உங்கள் விடையை மீண்டும் சோதியுங்கள்.",
    "incomplete_factors": "காரணிகளை ஜோடி முறையில் தேடுங்கள். 1 × ? = எண், 2 × ? = எண்... என்று ஒவ்வொரு ஜோடியையும் சோதியுங்கள்.",
    "prime_missed": "உங்கள் காரணி மரம் இன்னும் தொடர வேண்டும். கடைசி விடை 1 ஆகும் வரை வகுக்க வேண்டும்.",
    "computation_error": "உங்கள் கணக்கீட்டை மீண்டும் சோதியுங்கள். ஒரு படி பார்ப்போமா?",
    "generic": "உங்கள் முறை சரியாக உள்ளது. ஆனால் ஒரு படியில் சிறிய பிழை உள்ளது. மீண்டும் தொடக்கத்திலிருந்து படிப்படியாக முயற்சி செய்வீர்களா?",
}


class VerificationAgent:
    """
    Verifies student answers against NIE marking scheme.
    Identifies the specific error step and generates Socratic hints.
    Never reveals the correct answer directly.
    """

    def __init__(self, gemini_client, model: str = "gemini-2.5-flash"):
        self.client = gemini_client
        self.model = model

    async def verify(self, ctx: QueryContext, exercise: ExerciseBundle,
                      retrieved: RetrievedContext,
                      student: StudentProfile) -> VerificationResult:
        if not ctx.student_answer or not exercise:
            return VerificationResult(
                is_correct=False,
                first_wrong_step=None,
                socratic_hint_ta=SOCRATIC_HINTS["generic"],
                error_type="no_answer",
                skill_delta=0.0,
            )

        # Use LLM to evaluate step-by-step
        return await self._verify_llm(ctx, exercise, retrieved, student)

    async def _verify_llm(self, ctx: QueryContext, exercise: ExerciseBundle,
                           retrieved: RetrievedContext,
                           student: StudentProfile) -> VerificationResult:
        answer_context = "\n".join(
            c["text"] for c in retrieved.answer_scheme_chunks[:2]
        ) if retrieved.answer_scheme_chunks else "NIE marking scheme not available."

        system = f"""நீங்கள் NIE Grade {student.grade} கணித மதிப்பீட்டு நிபுணர்.
மாணவர் பதிலை NIE வினாவிடை திட்டத்துடன் ஒப்பிட்டு பகுப்பாய்க.

கட்டாய விதிகள்:
• நேரடியாக சரியான விடையை ஒருபோதும் சொல்லாதீர்கள்
• ஒரே ஒரு வழிகாட்டும் கேள்வி மட்டும் கேளுங்கள்
• பிழையான படியை மட்டும் சுட்டுங்கள்

NIE வினாவிடை திட்டம்:
{answer_context}

கேள்வி: {exercise.question_ta}
எதிர்பார்க்கப்படும் படிகள்: {json.dumps(exercise.expected_steps, ensure_ascii=False)}
சரியான விடை: (hidden from student)

JSON மட்டும் தரவும்:
{{
  "is_correct": true/false,
  "first_wrong_step": "குறிப்பு அல்லது null",
  "error_type": "used_lcm_for_hcf|used_hcf_for_lcm|wrong_digit_sum|incomplete_factors|prime_missed|computation_error|generic|none",
  "socratic_hint_ta": "வழிகாட்டும் கேள்வி (விடை சொல்லாமல்)",
  "skill_delta": 0.1 (if correct, scaled by difficulty) or -0.05 (if wrong)
}}"""

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system, temperature=0.1,
                max_output_tokens=400,
                thinking_config=types.ThinkingConfig(thinking_budget=0))
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"மாணவர் பதில்: {ctx.student_answer}",
                config=config,
            )
            text = response.text.strip().strip("```json").strip("```").strip()
            data = json.loads(text)

            error_type = data.get("error_type", "generic")
            hint = data.get("socratic_hint_ta") or SOCRATIC_HINTS.get(
                error_type, SOCRATIC_HINTS["generic"])

            return VerificationResult(
                is_correct=data.get("is_correct", False),
                first_wrong_step=data.get("first_wrong_step"),
                socratic_hint_ta=hint,
                error_type=error_type,
                skill_delta=data.get("skill_delta", -0.05),
            )
        except Exception as e:
            log.warning(f"Verification LLM failed: {e}")
            # Fallback: simple exact match
            is_correct = str(exercise.answer).strip() == ctx.student_answer.strip()
            return VerificationResult(
                is_correct=is_correct,
                first_wrong_step=None,
                socratic_hint_ta=SOCRATIC_HINTS["generic"],
                error_type="generic",
                skill_delta=0.1 * exercise.difficulty if is_correct else -0.05,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Central coordinator. Dispatches to specialist agents,
    aggregates results, manages session state.
    
    Agent execution order (with parallelism where safe):
    1. DialectAgent (normalize query) — always first
    2. QueryAgent (parse intent) — after dialect
    3. StudentProfileAgent (load profile) — parallel with dialect
    4. RetrievalAgent (fetch context) — after query parsed
    5. Parallel: TeachingAgent + DrawingAgent + ExerciseAgent
    6. VerificationAgent (only if CHECK_ANSWER)
    7. StudentProfileAgent.save (update skills)
    """

    def __init__(self, grade: int = 7, chapter: int = 4,
                 subject: str = "mathematics"):
        self.grade = grade
        self.chapter = chapter
        self.subject = subject

        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        from google import genai
        self.gemini_client = genai.Client(api_key=api_key)

        # Initialize agents
        self.dialect_agent = DialectAgent(use_llm=False)
        self.query_agent = QueryAgent(use_llm=True,
                                      gemini_client=self.gemini_client)
        self.profile_agent = StudentProfileAgent()
        self.drawing_agent = DrawingAgent()
        self.exercise_agent = ExerciseAgent()

        # Import and initialize vector store + retrieval
        try:
            from pipeline_ingestion import NIEVectorStore, TamilEmbedder
            self.vector_store = NIEVectorStore()
            self.embedder = TamilEmbedder()
            self.retrieval_agent = RetrievalAgent(self.vector_store,
                                                   self.embedder)
            self.use_vector_db = True
            log.info("Vector DB (ChromaDB) initialized.")
        except Exception as e:
            log.warning(f"Vector DB unavailable: {e}. Run pipeline_ingestion.py first.")
            self.use_vector_db = False
            self.retrieval_agent = None

        self.teaching_agent = TeachingAgent(self.gemini_client)
        self.verification_agent = VerificationAgent(self.gemini_client)

    async def handle(self, student_id: str, raw_query: str,
                      district: str = "unknown", student_name: str = "மாணவர்",
                      student_answer: str = None,
                      exercise_topic: str = None,
                      n_retrieve: int = 6) -> AgentResponse:
        """Main entry point. Async to allow parallel agent execution."""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        log.info(f"[{session_id}] query='{raw_query[:60]}' student={student_id}")

        # ── Step 1: Load profile (async-friendly, fast DB read) ──────────
        student = self.profile_agent.get_or_create(
            student_id, student_name, district)
        student.session_count += 1

        # ── Step 2: Dialect detection + normalization ─────────────────────
        dialect, normalized_query = self.dialect_agent.detect_and_normalize(
            raw_query, district)
        log.info(f"[{session_id}] dialect={dialect.value}")

        # ── Step 3: Query parsing ─────────────────────────────────────────
        ctx = await self.query_agent.parse(
            raw_query, normalized_query, dialect, student,
            student_answer, exercise_topic)
        log.info(f"[{session_id}] intent={ctx.intent} topic={ctx.topic} nums={ctx.numbers}")

        # ── Step 4: Retrieval ─────────────────────────────────────────────
        if self.use_vector_db and self.retrieval_agent:
            retrieved = await self.retrieval_agent.retrieve(
                ctx, student, self.grade, self.chapter, self.subject, n_retrieve)
        else:
            # Fallback: empty context (teaching agent uses only system knowledge)
            retrieved = RetrievedContext(
                chunks=[], answer_scheme_chunks=[], total_retrieved=0,
                query_embedding=[], retrieval_time_ms=0.0)
        log.info(f"[{session_id}] retrieved={retrieved.total_retrieved} chunks "
                 f"in {retrieved.retrieval_time_ms:.0f}ms")

        # ── Step 5: Parallel specialist agents ───────────────────────────
        # Drawing and Exercise can run without LLM calls (deterministic)
        diagram = None
        if self.drawing_agent.should_draw(ctx, retrieved):
            diagram = self.drawing_agent.generate(ctx)

        exercise = None
        if ctx.intent == Intent.EXERCISE:
            exercise = self.exercise_agent.generate(ctx, student)

        # Teaching agent — main LLM call
        teaching = await self.teaching_agent.generate(ctx, student, retrieved)

        # ── Step 6: Verification (only if checking answer) ────────────────
        verification = None
        if ctx.intent == Intent.CHECK_ANSWER and exercise_topic:
            # Reconstruct exercise bundle for verification
            ex_bundle = self.exercise_agent.generate(
                QueryContext(
                    raw_query="", normalized_query="",
                    intent=Intent.EXERCISE, topic=exercise_topic,
                    section="", numbers=ctx.numbers,
                    method_requested=None, is_word_problem=False,
                    dialect=dialect, confidence=1.0,
                    student_answer=None, exercise_topic=None
                ), student)
            if ex_bundle:
                verification = await self.verification_agent.verify(
                    ctx, ex_bundle, retrieved, student)

                # Update student skill based on verification
                if verification:
                    self.profile_agent.record_outcome(
                        student, exercise_topic,
                        correct=verification.is_correct,
                        difficulty=student.skill_level(),
                        error_type=verification.error_type or "")

        # ── Step 7: Update student profile ────────────────────────────────
        student.total_questions += 1
        student.last_topic = ctx.topic
        if ctx.dialect != Dialect.UNKNOWN:
            student.dialect = ctx.dialect.value
        self.profile_agent.save(student)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(f"[{session_id}] total={elapsed_ms:.0f}ms")

        return AgentResponse(
            session_id=session_id,
            student_id=student_id,
            intent=ctx.intent,
            teaching=teaching,
            diagram=diagram,
            exercise=exercise,
            verification=verification,
            retrieved_chunk_ids=[c["id"] for c in retrieved.chunks],
            dialect_detected=dialect,
            processing_time_ms=elapsed_ms,
        )

    def handle_sync(self, student_id: str, raw_query: str, **kwargs) -> AgentResponse:
        """Synchronous wrapper for CLI usage."""
        return asyncio.run(self.handle(student_id, raw_query, **kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NIE Tamil Math Tutor — Multi-Agent System")
    parser.add_argument("--student-id", default="SL_TM_2024_001")
    parser.add_argument("--student-name", default="மாணவர்")
    parser.add_argument("--district", default="unknown",
                        help="jaffna|batticaloa|estate|colombo|vanni|unknown")
    parser.add_argument("--grade", type=int, default=7)
    parser.add_argument("--chapter", type=int, default=4)
    parser.add_argument("--subject", default="mathematics")
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--student-answer",
                        help="Student's answer (for CHECK_ANSWER intent)")
    parser.add_argument("--exercise-topic",
                        help="Topic of the exercise being checked")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json-out", action="store_true",
                        help="Output full response as JSON")
    args = parser.parse_args()

    orchestrator = OrchestratorAgent(
        grade=args.grade, chapter=args.chapter, subject=args.subject)

    response = orchestrator.handle_sync(
        student_id=args.student_id,
        raw_query=args.question,
        district=args.district,
        student_name=args.student_name,
        student_answer=args.student_answer,
        exercise_topic=args.exercise_topic,
        n_retrieve=args.top_k,
    )

    if args.json_out:
        # Convert dataclasses to dict for JSON output
        def to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            if isinstance(obj, Enum):
                return obj.value
            return obj
        print(json.dumps(to_dict(response), ensure_ascii=False, indent=2))
        return

    # Human-readable output
    print("\n" + "="*70)
    print(f"Session: {response.session_id} | Intent: {response.intent.value}")
    print(f"Dialect: {response.dialect_detected.value} | "
          f"Chunks: {len(response.retrieved_chunk_ids)} | "
          f"Time: {response.processing_time_ms:.0f}ms")
    print("="*70)

    if response.teaching:
        print("\n📖 Tamil Explanation:")
        print(response.teaching.explanation_ta)
        if response.teaching.next_suggested_topic:
            print(f"\n→ அடுத்த பாடம்: {response.teaching.next_suggested_topic}")

    if response.diagram:
        print(f"\n🎨 Diagram: {response.diagram.diagram_type}")
        print(f"   {response.diagram.caption_ta}")
        print(f"   Spec: {json.dumps(response.diagram.spec, ensure_ascii=False)[:200]}...")

    if response.exercise:
        print(f"\n🏋️ Exercise (difficulty {response.exercise.difficulty}/3):")
        print(f"   {response.exercise.question_ta}")
        print(f"   Hint: {response.exercise.hint_ta}")

    if response.verification:
        v = response.verification
        status = "✅ சரி!" if v.is_correct else "❌ தவறு"
        print(f"\n{status}")
        print(f"   {v.socratic_hint_ta}")
        if v.error_type:
            print(f"   Error type: {v.error_type}")

    if response.error:
        print(f"\n⚠️ Error: {response.error}")

    print("="*70)


if __name__ == "__main__":
    main()
