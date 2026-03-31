"""
OrchestratorAgent — coordinates modular agents for the NIE Tamil math tutor.

Companion agents without standalone modules (Exercise, verification, mastery,
sentiment, progress, HITL) are defined in this file for a single import surface.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from functools import reduce
from math import gcd, lcm
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from src import config
from src.agents.dialect_agent import DialectAgent
from src.agents.drawing_agent import DrawingAgent
from src.agents.input_parser import InputParserAgent
from src.agents.intent_agent import IntentAgent
from src.agents.math_verifier import MathVerifierAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.teaching_agent import TeachingAgent
from src.llm_client import LLMClient
from src.llm_errors import format_llm_error_for_user
from src.models import (
    AgentResponse,
    Dialect,
    ExerciseBundle,
    Intent,
    QueryContext,
    RetrievedContext,
    SentimentSignal,
    StudentProfile,
    TeachingResponse,
    VerificationResult,
)
from src.storage import DatabaseManager

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

SOCRATIC_HINTS: dict[str, str] = {
    "used_lcm_for_hcf": "நீங்கள் மடங்குகளைக் காண்கிறீர்கள். 'அதிகூடிய பொதிகள்' என்றால் வகுபடுதல் தேவையா அல்லது மடங்கு தேவையா?",
    "used_hcf_for_lcm": "நீங்கள் காரணிகளைக் காண்கிறீர்கள். 'முதல் சந்திப்பு' என்றால் பொது மடங்கு வேண்டுமா?",
    "wrong_digit_sum": "இலக்கச் சுட்டி இரண்டு இலக்கமாக வந்தால் மீண்டும் கூட வேண்டும். உங்கள் விடையை மீண்டும் சோதியுங்கள்.",
    "incomplete_factors": "காரணிகளை ஜோடி முறையில் தேடுங்கள். 1 × ? = எண், 2 × ? = எண்... என்று ஒவ்வொரு ஜோடியையும் சோதியுங்கள்.",
    "prime_missed": "உங்கள் காரணி மரம் இன்னும் தொடர வேண்டும். கடைசி விடை 1 ஆகும் வரை வகுக்க வேண்டும்.",
    "computation_error": "உங்கள் கணக்கீட்டை மீண்டும் சோதியுங்கள். ஒரு படி பார்ப்போமா?",
    "no_answer": "பதிலை உள்ளிடவும்.",
    "generic": "உங்கள் முறை சரியாக உள்ளது. ஆனால் ஒரு படியில் சிறிய பிழை உள்ளது. மீண்டும் தொடக்கத்திலிருந்து படிப்படியாக முயற்சி செய்வீர்களா?",
}


class ExerciseAgent:
    """NIE-style exercise generation at adaptive difficulty."""

    def generate(
        self, ctx: QueryContext, student: StudentProfile
    ) -> ExerciseBundle | None:
        if ctx.intent != Intent.EXERCISE_REQUEST:
            return None
        topic = ctx.topic if ctx.topic else (student.last_topic or "factor_listing")
        difficulty = student.get_difficulty_ceiling()
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
        divisor = random.choice([2, 3, 6, 9, 4] if difficulty > 1 else [2, 3, 9])
        numbers = random.sample(range(100, 5000), 6)
        correct = [n for n in numbers if n % divisor == 0]
        return ExerciseBundle(
            question_ta=(
                f"பின்வரும் எண்களில் {divisor} ஆல் மீதியின்றி வகுபடும் எண்களை வகுக்காமல் தெரிவு செய்க:\n"
                f"{', '.join(map(str, numbers))}"
            ),
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
        numbers = random.sample(range(10, 9999), 5)

        def _digit_sum(n: int) -> int:
            while n >= 10:
                n = sum(int(d) for d in str(n))
            return n

        return ExerciseBundle(
            question_ta=f"பின்வரும் எண்களின் இலக்கச் சுட்டியைக் காண்க:\n{', '.join(map(str, numbers))}",
            numbers=numbers,
            difficulty=1,
            topic="digit_sum",
            hint_ta="ஒவ்வொரு எண்ணின் இலக்கங்களையும் கூட்டுக. தனி இலக்கம் வரும் வரை திரும்பவும் கூட்டவும்.",
            expected_steps=[
                "ஒவ்வொரு இலக்கத்தையும் தனியாக எழுதுக",
                "அனைத்தையும் கூட்டுக",
                "விடை இரண்டு இலக்கமாக இருந்தால் மீண்டும் கூட்டுக",
            ],
            answer={n: _digit_sum(n) for n in numbers},
        )

    def _gen_factors(self, difficulty: int) -> ExerciseBundle:
        pool = {1: [12, 18, 24, 36], 2: [48, 60, 72, 84, 96], 3: [120, 150, 180, 204]}
        n = random.choice(pool.get(difficulty, pool[2]))
        factors = sorted([i for i in range(1, n + 1) if n % i == 0])
        return ExerciseBundle(
            question_ta=f"{n} இன் அனைத்து காரணிகளையும் காண்க.",
            numbers=[n],
            difficulty=difficulty,
            topic="factor_listing",
            hint_ta="ஜோடி பெருக்க முறை: 1 × ?, 2 × ?, 3 × ? ... என்று காண்க",
            expected_steps=[
                f"{n} = 1 × {n}",
                "ஒவ்வொரு ஜோடியும் எழுதுக",
                "அனைத்து காரணிகளை ஏறுவரிசையில் எழுதுக",
            ],
            answer=factors,
        )

    def _gen_prime_factors(self, difficulty: int) -> ExerciseBundle:
        pool = {1: [12, 18, 30], 2: [48, 60, 84, 90], 3: [120, 168, 210, 252]}
        n = random.choice(pool.get(difficulty, pool[2]))

        def prime_fact(x: int) -> list[int]:
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
            expected_steps=[
                f"{n} ÷ {primes[0]} = {n // primes[0]}",
                "விடை 1 ஆகும் வரை தொடரவும்",
                f"{n} = " + " × ".join(map(str, primes)),
            ],
            answer=" × ".join(map(str, primes)),
        )

    def _gen_hcf(self, difficulty: int) -> ExerciseBundle:
        pairs = {
            1: [(12, 18), (24, 36)],
            2: [(48, 72), (60, 90), (84, 108)],
            3: [(72, 108, 144), (36, 54, 90)],
        }
        nums = list(random.choice(pairs.get(difficulty, pairs[2])))
        hcf_val = reduce(gcd, nums)
        return ExerciseBundle(
            question_ta=f"{', '.join(map(str, nums))} ஆகிய எண்களின் பொ.கா.பெ. காண்க.",
            numbers=nums,
            difficulty=difficulty,
            topic="hcf",
            hint_ta="வகுத்தல் முறை அல்லது முதன்மைக் காரணிகள் மூலம் காண்க",
            expected_steps=[
                "ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளாக பகுக்கவும்",
                "பொதுவான முதன்மைக் காரணிகளைக் காண்க",
                "அவற்றை பெருக்குக",
            ],
            answer=hcf_val,
        )

    def _gen_lcm(self, difficulty: int) -> ExerciseBundle:
        pairs = {1: [(2, 3), (4, 6)], 2: [(6, 8, 12), (4, 9, 12)], 3: [(8, 12, 18), (6, 10, 15)]}
        nums = list(random.choice(pairs.get(difficulty, pairs[2])))
        lcm_val = reduce(lcm, nums)
        return ExerciseBundle(
            question_ta=f"{', '.join(map(str, nums))} ஆகிய எண்களின் பொ.ம.சி. காண்க.",
            numbers=nums,
            difficulty=difficulty,
            topic="lcm",
            hint_ta="முதன்மைக் காரணிகளின் உயர் வலுவைப் பெருக்குக",
            expected_steps=[
                "ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளாக பகுக்கவும்",
                "ஒவ்வொரு முதன்மை எண்ணின் உயர் வலுவைத் தேர்வு செய்க",
                "அவற்றை பெருக்குக",
            ],
            answer=lcm_val,
        )

    def _gen_word_problem(self, _difficulty: int) -> ExerciseBundle:
        problems = [
            {
                "q": (
                    "ஒரு கூடையில் 96 அப்பிள்களும் 60 ஆரஞ்சு பழங்களும் உள்ளன. இரு வகைப் பழங்களும் சம எண்ணிக்கையில் "
                    "இருக்கும் வகையில் பொதிகளில் இடப்பட்டால் பெறக்கூடிய அதிகூடிய பொதிகளின் எண்ணிக்கை யாது?"
                ),
                "nums": [96, 60],
                "answer": 12,
                "topic": "hcf",
                "hint": "சம பகிர்வு → பொ.கா.பெ. பயன்படுத்தவும்",
            },
            {
                "q": (
                    "இரண்டு மணிகள் முறையே 6 நிமிடங்கள், 8 நிமிடங்களுக்கு ஒரு முறை ஒலிக்கின்றன. காலை 8.00 மணிக்கு "
                    "ஒருமித்து ஒலித்தால், அவை மீண்டும் எத்தனை மணிக்கு ஒருமித்து ஒலிக்கும்?"
                ),
                "nums": [6, 8],
                "answer": "8.24 மணி",
                "topic": "lcm",
                "hint": "முதல் சந்திப்பு → பொ.ம.சி. பயன்படுத்தவும்",
            },
        ]
        prob = random.choice(problems)
        return ExerciseBundle(
            question_ta=prob["q"],
            numbers=prob["nums"],
            difficulty=3,
            topic=prob["topic"],
            hint_ta=prob["hint"],
            expected_steps=[
                "தேவையான தகவல்களை எழுதுக",
                "பொ.கா.பெ. அல்லது பொ.ம.சி. தீர்மானி",
                "கணக்கிட்டு விடை எழுதுக",
            ],
            answer=prob["answer"],
        )


class AnswerVerifierAgent:
    """LLM-assisted answer checking with JSON protocol."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.llm_fast_model()

    def verify(
        self,
        llm: LLMClient,
        ctx: QueryContext,
        exercise: ExerciseBundle | None,
        retrieved: RetrievedContext,
        student: StudentProfile,
    ) -> VerificationResult:
        if ctx.intent != Intent.CHECK_ANSWER:
            return VerificationResult()
        ans = (ctx.student_answer or "").strip()
        if not ans:
            return VerificationResult(
                is_correct=False,
                socratic_hint_ta=SOCRATIC_HINTS["no_answer"],
                error_type="no_answer",
                skill_delta=0.0,
            )

        ctx_bits = "\n\n".join(
            (c.get("content_ta") or "")[:1200] for c in retrieved.chunks[:3]
        )
        if exercise:
            system = f"""நீங்கள் NIE Grade {student.grade} கணித மதிப்பீட்டு நிபுணர்.
மாணவர் பதிலை ஒப்பிட்டுப் பகுப்பாய்வு செய்க.
• நேரடியாக சரியான முழு விடையைச் சொல்லாதீர்கள்
• JSON மட்டும் தரவும்

கேள்வி: {exercise.question_ta}
எதிர்பார்க்கப்படும் படிகள்: {json.dumps(exercise.expected_steps, ensure_ascii=False)}
குறிப்பு விடை (உள்ளக மதிப்பீட்டுக்கு மட்டும்): {json.dumps(exercise.answer, ensure_ascii=False)}

NIE சூழல்:
{ctx_bits}

JSON வடிவம்:
{{
  "is_correct": true/false,
  "first_wrong_step": "string or null",
  "error_type": "used_lcm_for_hcf|used_hcf_for_lcm|wrong_digit_sum|incomplete_factors|prime_missed|computation_error|generic|none",
  "socratic_hint_ta": "வழிகாட்டும் கேள்வி (விடை சொல்லாமல்)",
  "skill_delta": 0.1
}}"""
            user = f"மாணவர் பதில்: {ans}"
        else:
            system = f"""நீங்கள் NIE Grade {student.grade} கணித ஆசிரியர்.
மாணவரின் பதில் கேள்விக்குச் சரியா என மதிப்பிடுக. JSON மட்டும்.
சூழல்:\n{ctx_bits}
JSON: {{"is_correct": bool, "error_type": "none|generic", "socratic_hint_ta": "...", "skill_delta": float}}"""
            user = f"கேள்வி: {ctx.normalized_query}\nமாணவர் பதில்: {ans}"

        data = llm.generate_json(self.model, system, user, temperature=0.1, max_tokens=400)
        if not data:
            is_ok = str(exercise.answer).strip() == ans if exercise and exercise.answer is not None else False
            return VerificationResult(
                is_correct=is_ok,
                socratic_hint_ta=SOCRATIC_HINTS["generic"],
                error_type="generic",
                skill_delta=0.1 * (exercise.difficulty if exercise else 1) if is_ok else -0.05,
            )

        err = data.get("error_type", "generic")
        hint = data.get("socratic_hint_ta") or SOCRATIC_HINTS.get(err, SOCRATIC_HINTS["generic"])
        return VerificationResult(
            is_correct=bool(data.get("is_correct", False)),
            first_wrong_step=data.get("first_wrong_step"),
            socratic_hint_ta=hint,
            error_type=err,
            skill_delta=float(data.get("skill_delta", -0.05)),
        )


class MasteryAgent:
    """Skill updates from verification outcomes."""

    def apply(
        self,
        student: StudentProfile,
        ctx: QueryContext,
        verification: VerificationResult | None,
        exercise: ExerciseBundle | None,
    ) -> None:
        if ctx.intent != Intent.CHECK_ANSWER or verification is None:
            return
        topic = (exercise.topic if exercise else None) or ctx.topic or student.last_topic
        if exercise:
            student.update_skill(topic, verification.is_correct, exercise.difficulty)
        if not verification.is_correct and verification.error_type:
            student.last_error_type = verification.error_type


class ProgressAgent:
    """Per-turn progression metadata on the student profile."""

    def record_turn(
        self, student: StudentProfile, retrieved: RetrievedContext, query_ctx: QueryContext
    ) -> None:
        student.total_questions_asked += 1
        if retrieved.chunks:
            student.last_topic = retrieved.chunks[0].get("topic", "") or query_ctx.topic
        elif query_ctx.topic:
            student.last_topic = query_ctx.topic


class SentimentAgent:
    """Lightweight engagement / frustration heuristics."""

    def analyze(self, raw_query: str, explanation_ta: str) -> SentimentSignal:
        frustration_kw = ["புரியவில்லை", "கடினம்", "முடியவில்லை", "தவறு", "சோர்வு"]
        encourage_kw = ["நன்றி", "புரிந்தது", "சரி", "thanks", "ok"]
        q = raw_query.lower()
        frustrated = any(k in raw_query for k in frustration_kw) or any(
            k.lower() in q for k in ("hard", "difficult")
        )
        encourage = any(k in raw_query for k in encourage_kw)
        engagement = 0.75 if explanation_ta else 0.5
        if frustrated:
            engagement = max(0.2, engagement - 0.35)
        if encourage:
            engagement = min(1.0, engagement + 0.15)
        return SentimentSignal(
            engagement_score=engagement,
            confidence_level=0.55,
            frustration_detected=frustrated,
            encourage=encourage,
        )


class HITLAgent:
    """Escalation rules for human-in-the-loop review."""

    def evaluate(
        self,
        sentiment: SentimentSignal | None,
        verification: VerificationResult | None,
        query_ctx: QueryContext,
    ) -> tuple[bool, str | None]:
        if sentiment and sentiment.frustration_detected and sentiment.engagement_score < 0.35:
            return True, "low_engagement_frustration"
        if verification and verification.error_type in ("used_lcm_for_hcf", "used_hcf_for_lcm"):
            return True, "systematic_concept_confusion"
        if query_ctx.confidence < 0.15 and query_ctx.intent == Intent.UNKNOWN:
            return True, "low_intent_confidence"
        return False, None


class OrchestratorAgent:
    def __init__(
        self,
        grade: int = 7,
        chapter: int = 4,
        subject: str = "mathematics",
    ) -> None:
        self.grade = grade
        self.chapter = chapter
        self.subject = subject

        self.llm = LLMClient(backend=config.LLM_BACKEND)
        self.db = DatabaseManager()
        self.dialect_agent = DialectAgent()
        self.intent_agent = IntentAgent()
        self.math_verifier = MathVerifierAgent()
        self.teaching = TeachingAgent(
            gemini_client=None,
            model=config.llm_teaching_model(),
        )
        self.drawing_agent = DrawingAgent()
        self.exercise_agent = ExerciseAgent()
        self.answer_verifier = AnswerVerifierAgent()
        self.mastery_agent = MasteryAgent()
        self.sentiment_agent = SentimentAgent()
        self.progress_agent = ProgressAgent()
        self.hitl_agent = HITLAgent()
        self.input_parser = InputParserAgent()

        self.use_vector_db = False
        self.vector_store: Any = None
        self.embedder: Any = None
        try:
            import sentence_transformers  # noqa: F401
            from src.ingestion.vector_store import NIEVectorStore, TamilEmbedder

            self.vector_store = NIEVectorStore()
            self.embedder = TamilEmbedder()
            self.use_vector_db = True
        except Exception:
            self.vector_store = None
            self.embedder = None

        self.retrieval = RetrievalAgent(
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

    def _derive_expected_method(
        self,
        query_ctx: QueryContext,
        retrieved: RetrievedContext,
    ) -> tuple[int, str]:
        """
        Decide which NIE method the TeachingAgent must follow.
        Priority:
        1) explicit method_requested from the student query
        2) method_number/topic mapping from retrieved NIE chunks
        3) conservative fallback based on query topic
        """
        # 1) explicit request in query
        if query_ctx.method_requested:
            req = query_ctx.method_requested.lower().strip()
            if req == "list":
                return 1, "முறை I (பட்டியல்/ஜோடி பெருக்கம்)"
            if req == "factor_tree":
                return 2, "முறை II (காரணி மரம்/முதன்மைக் காரணிகள்)"
            if req == "division":
                return 3, "முறை III (வகுத்தல் ஏணி)"

        # 2) derive from retrieved NIE chunks
        topic_to_method: dict[str, tuple[int, str]] = {
            # Factor methods (method_number not stored on these chunks)
            "factor_listing_pair_method": (1, "முறை I (ஜோடி பெருக்கம் / காரணிப் பட்டியல்)"),
            "prime_factorization_tree": (2, "முறை II (காரணி மரம் / முதன்மைக் காரணிகள்)"),
            "prime_factorization_division": (3, "முறை III (வகுத்தல் ஏணி)"),
            # HCF methods
            "hcf_method_1_list": (1, "முறை I (காரணிப் பட்டியல் மூலம் பொ.கா.பெ.)"),
            "hcf_method_2_prime": (2, "முறை II (முதன்மைக் காரணிகள் மூலம் பொ.கா.பெ.)"),
            "hcf_method_3_division": (3, "முறை III (வகுத்தல் முறை மூலம் பொ.கா.பெ.)"),
            # LCM methods
            "lcm_prime_method": (1, "முறை I (முதன்மைக் காரணிகள் மூலம் பொ.ம.சி.)"),
            "lcm_division_method": (2, "முறை II (வகுத்தல் ஏணி மூலம் பொ.ம.சி.)"),
        }

        for chunk in retrieved.chunks:
            mnum = chunk.get("method_number")
            if isinstance(mnum, int) and mnum in (1, 2, 3):
                label = "முறை I" if mnum == 1 else "முறை II" if mnum == 2 else "முறை III"
                return mnum, label
            topic = (chunk.get("topic") or "").strip()
            if topic in topic_to_method:
                return topic_to_method[topic]

        # 3) fallback
        t = (query_ctx.topic or "").lower()
        if t == "lcm":
            return 2, "முறை II (வகுத்தல் ஏணி மூலம் பொ.ம.சி.)"
        if t == "hcf":
            return 3, "முறை III (வகுத்தல் ஏணி மூலம் பொ.கா.பெ.)"
        return 1, "முறை I (ஜோடி பெருக்கம் / காரணிப் பட்டியல்)"

    def handle(
        self,
        student_id: str,
        raw_query: str,
        district: str = "unknown",
        student_name: str = "மாணவர்",
        student_answer: str | None = None,
        exercise_topic: str | None = None,
        n_retrieve: int = 6,
    ) -> AgentResponse:
        t0 = time.perf_counter()
        session_db_id = self.db.start_session(student_id)
        session_id = str(session_db_id)

        text = self.input_parser.parse_text(raw_query)
        student = self.db.get_or_create_student(student_id, student_name, district)
        student.grade = self.grade
        student.district = district

        dialect, normalized = self.dialect_agent.detect_and_normalize(text, student.district)
        register = self.dialect_agent.get_nie_register_guidance(student)
        query_ctx = self.intent_agent.parse(
            text, normalized, dialect, student,
            student_answer=student_answer, exercise_topic=exercise_topic,
        )
        query_ctx = replace(query_ctx, region_hints=register)

        retrieved = self.retrieval.retrieve(
            query_ctx, student, self.grade, self.chapter, self.subject, n_results=n_retrieve,
        )
        factor_note, hcf_note, lcm_note = self.math_verifier.get_verification_blocks(normalized)

        exercise: ExerciseBundle | None = None
        if query_ctx.intent == Intent.EXERCISE_REQUEST:
            exercise = self.exercise_agent.generate(query_ctx, student)

        expected_method_number, expected_method_label = self._derive_expected_method(
            query_ctx, retrieved
        )
        diagram = self.drawing_agent.generate(
            query_ctx,
            retrieved,
            expected_method_number=expected_method_number,
        )
        system_prompt = self.teaching.build_system_prompt(
            query_ctx,
            student,
            retrieved,
            factor_note,
            hcf_note,
            lcm_note,
            query_ctx.region_hints,
            expected_method_number=expected_method_number,
            expected_method_label=expected_method_label,
        )
        teaching: TeachingResponse | None = None
        error: str | None = None
        quota_exhausted = False
        retry_after_seconds: float | None = None
        try:
            explanation, _fr = self.llm.generate(
                config.llm_teaching_model(),
                system_prompt,
                text,
                temperature=config.GEMINI_TEMPERATURE,
                max_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
                disable_thinking=True,
            )
            teaching = TeachingResponse(explanation_ta=explanation, key_concepts=[])
        except Exception as e:
            error, quota_exhausted, retry_after_seconds = format_llm_error_for_user(e)

        verification: VerificationResult | None = None
        if query_ctx.intent == Intent.CHECK_ANSWER:
            verification = self.answer_verifier.verify(
                self.llm, query_ctx, exercise, retrieved, student,
            )

        self.progress_agent.record_turn(student, retrieved, query_ctx)
        if verification is not None:
            self.mastery_agent.apply(student, query_ctx, verification, exercise)
        self.db.save_student(student)

        sentiment = self.sentiment_agent.analyze(text, teaching.explanation_ta if teaching else "")
        hitl_flag, hitl_reason = self.hitl_agent.evaluate(sentiment, verification, query_ctx)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary = (teaching.explanation_ta[:500] if teaching else "") or (
            "quota_exhausted" if quota_exhausted else (error or "")
        )

        interaction_id = self.db.record_interaction(
            student_id=student_id,
            session_id=session_db_id,
            query=text,
            intent=query_ctx.intent.value,
            response_summary=summary,
            response_time_ms=int(elapsed_ms),
            diagram_shown=diagram is not None,
            exercise_given=exercise is not None,
        )
        self.db.record_sentiment(
            interaction_id,
            sentiment.engagement_score,
            sentiment.confidence_level,
            sentiment.frustration_detected,
        )
        if hitl_flag and hitl_reason:
            self.db.add_hitl_flag(interaction_id, hitl_reason)

        return AgentResponse(
            session_id=session_id,
            student_id=student_id,
            intent=query_ctx.intent,
            teaching=teaching,
            diagram=diagram,
            exercise=exercise,
            verification=verification,
            sentiment=sentiment,
            retrieved_chunk_ids=[c["id"] for c in retrieved.chunks if c.get("id")],
            dialect_detected=dialect,
            processing_time_ms=elapsed_ms,
            hitl_flagged=hitl_flag,
            hitl_reason=hitl_reason,
            error=error,
            quota_exhausted=quota_exhausted,
            retry_after_seconds=retry_after_seconds,
        )

    def handle_streaming(
        self,
        student_id: str,
        raw_query: str,
        district: str = "unknown",
        student_name: str = "மாணவர்",
        student_answer: str | None = None,
        exercise_topic: str | None = None,
        n_retrieve: int = 6,
    ) -> Iterator[str]:
        """
        Yields Tamil text chunks from the teaching model stream.
        The final AgentResponse is available as the generator's return value
        (StopIteration.value) when the iterator completes.
        """
        t0 = time.perf_counter()
        session_db_id = self.db.start_session(student_id)
        session_id = str(session_db_id)

        text = self.input_parser.parse_text(raw_query)
        student = self.db.get_or_create_student(student_id, student_name, district)
        student.grade = self.grade
        student.district = district

        dialect, normalized = self.dialect_agent.detect_and_normalize(text, student.district)
        register = self.dialect_agent.get_nie_register_guidance(student)
        query_ctx = self.intent_agent.parse(
            text, normalized, dialect, student,
            student_answer=student_answer, exercise_topic=exercise_topic,
        )
        query_ctx = replace(query_ctx, region_hints=register)

        retrieved = self.retrieval.retrieve(
            query_ctx, student, self.grade, self.chapter, self.subject, n_results=n_retrieve,
        )
        factor_note, hcf_note, lcm_note = self.math_verifier.get_verification_blocks(normalized)

        exercise: ExerciseBundle | None = None
        if query_ctx.intent == Intent.EXERCISE_REQUEST:
            exercise = self.exercise_agent.generate(query_ctx, student)

        expected_method_number, expected_method_label = self._derive_expected_method(
            query_ctx, retrieved
        )
        diagram = self.drawing_agent.generate(
            query_ctx,
            retrieved,
            expected_method_number=expected_method_number,
        )
        system_prompt = self.teaching.build_system_prompt(
            query_ctx,
            student,
            retrieved,
            factor_note,
            hcf_note,
            lcm_note,
            query_ctx.region_hints,
            expected_method_number=expected_method_number,
            expected_method_label=expected_method_label,
        )

        parts: list[str] = []
        error: str | None = None
        quota_exhausted = False
        retry_after_seconds: float | None = None
        try:
            for piece, _ in self.llm.generate_stream(
                config.llm_teaching_model(),
                system_prompt,
                text,
                temperature=config.GEMINI_TEMPERATURE,
                max_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
                disable_thinking=True,
            ):
                if piece:
                    parts.append(piece)
                    yield piece
        except Exception as e:
            error, quota_exhausted, retry_after_seconds = format_llm_error_for_user(e)

        explanation = "".join(parts).strip()
        teaching = TeachingResponse(explanation_ta=explanation, key_concepts=[]) if explanation else None

        verification: VerificationResult | None = None
        if query_ctx.intent == Intent.CHECK_ANSWER:
            verification = self.answer_verifier.verify(
                self.llm, query_ctx, exercise, retrieved, student,
            )

        self.progress_agent.record_turn(student, retrieved, query_ctx)
        if verification is not None:
            self.mastery_agent.apply(student, query_ctx, verification, exercise)
        self.db.save_student(student)

        sentiment = self.sentiment_agent.analyze(text, teaching.explanation_ta if teaching else "")
        hitl_flag, hitl_reason = self.hitl_agent.evaluate(sentiment, verification, query_ctx)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary = (teaching.explanation_ta[:500] if teaching else "") or (
            "quota_exhausted" if quota_exhausted else (error or "")
        )

        interaction_id = self.db.record_interaction(
            student_id=student_id,
            session_id=session_db_id,
            query=text,
            intent=query_ctx.intent.value,
            response_summary=summary,
            response_time_ms=int(elapsed_ms),
            diagram_shown=diagram is not None,
            exercise_given=exercise is not None,
        )
        self.db.record_sentiment(
            interaction_id,
            sentiment.engagement_score,
            sentiment.confidence_level,
            sentiment.frustration_detected,
        )
        if hitl_flag and hitl_reason:
            self.db.add_hitl_flag(interaction_id, hitl_reason)

        response = AgentResponse(
            session_id=session_id,
            student_id=student_id,
            intent=query_ctx.intent,
            teaching=teaching,
            diagram=diagram,
            exercise=exercise,
            verification=verification,
            sentiment=sentiment,
            retrieved_chunk_ids=[c["id"] for c in retrieved.chunks if c.get("id")],
            dialect_detected=dialect,
            processing_time_ms=elapsed_ms,
            hitl_flagged=hitl_flag,
            hitl_reason=hitl_reason,
            error=error,
            quota_exhausted=quota_exhausted,
            retry_after_seconds=retry_after_seconds,
        )
        return response


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (Intent, Dialect)):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NIE Tamil Math Tutor — modular OrchestratorAgent CLI",
    )
    parser.add_argument("--student-id", default="SL_TM_2024_001")
    parser.add_argument("--student-name", default="மாணவர்")
    parser.add_argument(
        "--district", default="unknown",
        help="jaffna|batticaloa|estate|colombo|vanni|unknown",
    )
    parser.add_argument("--grade", type=int, default=config.DEFAULT_GRADE)
    parser.add_argument("--chapter", type=int, default=config.DEFAULT_CHAPTER)
    parser.add_argument("--subject", default=config.DEFAULT_SUBJECT)
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--student-answer", help="Student answer for CHECK_ANSWER")
    parser.add_argument("--exercise-topic", help="Topic hint for verification")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json-out", action="store_true", help="Print full AgentResponse as JSON")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunk ids to stderr")
    args = parser.parse_args()

    orch = OrchestratorAgent(grade=args.grade, chapter=args.chapter, subject=args.subject)

    response = orch.handle(
        student_id=args.student_id,
        raw_query=args.question,
        district=args.district,
        student_name=args.student_name,
        student_answer=args.student_answer,
        exercise_topic=args.exercise_topic,
        n_retrieve=args.top_k,
    )

    if args.json_out:
        print(json.dumps(_json_safe(response), ensure_ascii=False, indent=2))
        return

    if args.show_context:
        print("--- Retrieved chunk ids ---", file=sys.stderr)
        for cid in response.retrieved_chunk_ids:
            print(f"  {cid}", file=sys.stderr)
        print(file=sys.stderr)

    print("\n" + "=" * 70)
    print(f"Session: {response.session_id} | Intent: {response.intent.value}")
    print(
        f"Dialect: {response.dialect_detected.value} | "
        f"Chunks: {len(response.retrieved_chunk_ids)} | "
        f"Time: {response.processing_time_ms:.0f}ms",
    )
    print("=" * 70)

    if response.teaching:
        print("\n📖 Tamil Explanation:")
        print(response.teaching.explanation_ta)
    if response.diagram:
        print(f"\n🎨 Diagram: {response.diagram.diagram_type}")
        print(f"   {response.diagram.caption_ta}")
    if response.exercise:
        print(f"\n🏋️ Exercise (difficulty {response.exercise.difficulty}/3):")
        print(f"   {response.exercise.question_ta}")
    if response.verification:
        v = response.verification
        status = "✅ சரி!" if v.is_correct else "❌ தவறு"
        print(f"\n{status}")
        print(f"   {v.socratic_hint_ta}")
    if response.hitl_flagged:
        print(f"\n⚑ HITL: {response.hitl_reason}")
    if response.error:
        print(f"\n⚠️ Error: {response.error}")
    print("=" * 70)


if __name__ == "__main__":
    main()
