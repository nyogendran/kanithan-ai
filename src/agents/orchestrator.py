"""
OrchestratorAgent — coordinates modular agents for the Kanithan Tamil math tutor.

All agents are imported from their standalone modules in src/agents/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from src import config
from src.chapters.registry import get_chapter_plugin
from src.agents.answer_verifier import AnswerVerifierAgent
from src.agents.dialect_agent import DialectAgent
from src.agents.drawing_agent import DrawingAgent
from src.agents.exercise_agent import ExerciseAgent
from src.agents.hitl_agent import HITLAgent
from src.agents.input_parser import InputParserAgent
from src.agents.intent_agent import IntentAgent
from src.agents.mastery_agent import MasteryAgent
from src.agents.math_verifier import MathVerifierAgent
from src.agents.progress_agent import ProgressAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.sentiment_agent import SentimentAgent
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
        self.chapter_plugin = get_chapter_plugin(chapter)
        self.topic_pack = self.chapter_plugin.topic_pack

        self.llm = LLMClient(backend=config.LLM_BACKEND)
        self.db = DatabaseManager()
        self.dialect_agent = DialectAgent()
        self.intent_agent = IntentAgent(topic_pack=self.topic_pack, chapter=self.chapter)
        self.math_verifier = MathVerifierAgent()
        self.teaching = TeachingAgent(
            gemini_client=None,
            model=config.llm_teaching_model(),
        )
        self.drawing_agent = DrawingAgent(
            topic_pack=self.topic_pack,
            diagram_adapter=self.chapter_plugin.diagram_adapter,
            chapter=self.chapter,
        )
        self.exercise_agent = ExerciseAgent(topic_pack=self.topic_pack, chapter=self.chapter)
        gemini_client = (
            self.llm.get_gemini_client()
            if config.LLM_BACKEND == "gemini"
            else None
        )
        self.answer_verifier = AnswerVerifierAgent(
            gemini_client=gemini_client,
            model=config.llm_fast_model(),
        )
        self.mastery_agent = MasteryAgent()
        self.sentiment_agent = SentimentAgent()
        self.progress_agent = ProgressAgent()
        self.hitl_agent = HITLAgent(db=self.db)
        self.input_parser = InputParserAgent()

        self.use_vector_db = False
        self.vector_store: Any = None
        self.embedder: Any = None
        try:
            import sentence_transformers  # noqa: F401
            from src.ingestion.vector_store import CurriculumVectorStore, TamilEmbedder

            self.vector_store = CurriculumVectorStore()
            self.embedder = TamilEmbedder()
            self.use_vector_db = True
        except Exception:
            self.vector_store = None
            self.embedder = None

        self.retrieval = RetrievalAgent(
            vector_store=self.vector_store,
            embedder=self.embedder,
            chapter=self.chapter,
            corpus=self.topic_pack.corpus,
            prerequisite_graph=self.topic_pack.prerequisite_graph,
            topic_to_skill=self.topic_pack.topic_to_skill,
        )

    def _derive_expected_method(
        self,
        query_ctx: QueryContext,
        retrieved: RetrievedContext,
    ) -> tuple[int, str]:
        """
        Decide which curriculum method the TeachingAgent must follow.
        Priority:
        1) explicit method_requested from the student query
        2) method_number/topic mapping from retrieved curriculum chunks
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

        # 2) derive from retrieved curriculum chunks
        topic_to_method = self.topic_pack.method_topic_map

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
        q = (query_ctx.normalized_query or query_ctx.raw_query or "").lower()
        if t == "word_problem":
            # Word-problem heuristic:
            # - "max equal groups / அதி கூடிய பொதி" usually maps to HCF style.
            # - "least/common cycle" usually maps to LCM style.
            hcf_hints = self.topic_pack.hcf_word_problem_hints
            lcm_hints = self.topic_pack.lcm_word_problem_hints
            if any(h in q for h in hcf_hints):
                return 3, "முறை III (வகுத்தல் ஏணி மூலம் பொ.கா.பெ.)"
            if any(h in q for h in lcm_hints):
                return 2, "முறை II (வகுத்தல் ஏணி மூலம் பொ.ம.சி.)"
            if len(query_ctx.numbers or []) >= 2:
                return 3, "முறை III (வகுத்தல் ஏணி மூலம் பொ.கா.பெ.)"
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
        register = self.dialect_agent.get_curriculum_register_guidance(student)
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
                query_ctx, exercise, retrieved, student,
            )

        self.mastery_agent.record_session_context(
            student, text, query_ctx.intent.value, query_ctx.topic or "",
        )
        if verification is not None:
            topic = (exercise.topic if exercise else None) or query_ctx.topic or student.last_topic
            self.mastery_agent.update_skill(
                student, topic or "unknown",
                verification.is_correct,
                exercise.difficulty if exercise else 1,
                error_type=verification.error_type or "",
            )
        self.db.save_student(student)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        sentiment = self.sentiment_agent.analyze(
            student, text, int(elapsed_ms),
            is_retry=False,
            exercise_correct=verification.is_correct if verification else None,
        )
        response_text = teaching.explanation_ta if teaching else ""
        hitl_flag, hitl_reason = self.hitl_agent.should_flag(
            student, query_ctx, sentiment, response_text, interaction_id=0,
        )

        summary = (response_text[:500]) or (
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
        register = self.dialect_agent.get_curriculum_register_guidance(student)
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
                query_ctx, exercise, retrieved, student,
            )

        self.mastery_agent.record_session_context(
            student, text, query_ctx.intent.value, query_ctx.topic or "",
        )
        if verification is not None:
            topic = (exercise.topic if exercise else None) or query_ctx.topic or student.last_topic
            self.mastery_agent.update_skill(
                student, topic or "unknown",
                verification.is_correct,
                exercise.difficulty if exercise else 1,
                error_type=verification.error_type or "",
            )
        self.db.save_student(student)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        sentiment = self.sentiment_agent.analyze(
            student, text, int(elapsed_ms),
            is_retry=False,
            exercise_correct=verification.is_correct if verification else None,
        )
        response_text = teaching.explanation_ta if teaching else ""
        hitl_flag, hitl_reason = self.hitl_agent.should_flag(
            student, query_ctx, sentiment, response_text, interaction_id=0,
        )

        summary = (response_text[:500]) or (
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
        description="Kanithan Tamil Math Tutor — modular OrchestratorAgent CLI",
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
