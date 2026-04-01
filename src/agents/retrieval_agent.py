"""Hybrid retrieval: optional vector store + curated curriculum keyword fallback."""

from __future__ import annotations

import logging
import json
import time
from typing import Any

from src.chapters.registry import get_chapter_plugin
from ..models import Intent, QueryContext, RetrievedContext, StudentProfile


class RetrievalAgent:
    """
    Vector search (when store + embedder are set) combined with AdaptiveRetriever-style
    keyword scoring over CURRICULUM_CORPUS, merge/dedupe, then prerequisite injection.
    """

    def __init__(
        self,
        vector_store: Any = None,
        embedder: Any = None,
        *,
        chapter: int = 4,
        corpus: list[dict] | None = None,
        prerequisite_graph: dict[str, list[str]] | None = None,
        topic_to_skill=None,
    ):
        plugin = get_chapter_plugin(chapter)
        pack = plugin.topic_pack
        self.vector_store = vector_store
        self.embedder = embedder
        self.corpus: list[dict] = list(corpus if corpus is not None else pack.corpus)
        self.prerequisite_graph = prerequisite_graph if prerequisite_graph is not None else pack.prerequisite_graph
        self.topic_to_skill = topic_to_skill if topic_to_skill is not None else pack.topic_to_skill
        self.log = logging.getLogger("kanithan.retrieval")

    def retrieve(
        self,
        query_ctx: QueryContext,
        student: StudentProfile,
        grade: int,
        chapter: int,
        subject: str,
        n_results: int = 6,
    ) -> RetrievedContext:
        t0 = time.perf_counter()
        ceiling = student.get_difficulty_ceiling()
        where_filter = {"difficulty": {"$lte": ceiling + 1}}

        vector_chunks: list[dict] = []
        query_emb: list[float] = []

        if self.vector_store is not None and self.embedder is not None:
            try:
                query_emb = list(self.embedder.embed_query(query_ctx.normalized_query))
                raw = self.vector_store.hybrid_query(
                    query_embedding=query_emb,
                    grade=grade,
                    chapter=chapter,
                    subject=subject,
                    n_results=n_results,
                    where_filter=where_filter,
                )
                vector_chunks = [self._vector_hit_to_chunk(h) for h in raw]
                vector_chunks = [c for c in vector_chunks if c.get("id")]
            except Exception as e:
                # Enterprise-grade requirement: never let retrieval crash the whole tutor.
                # If embeddings/model download fails (first run) or vector DB errors,
                # fall back to keyword retrieval only.
                self.log.warning("Vector retrieval failed; falling back to keyword-only: %s", e)
                query_emb = []
                vector_chunks = []

        intent_val = (
            query_ctx.intent.value
            if isinstance(query_ctx.intent, Intent)
            else str(query_ctx.intent)
        )
        keyword_chunks = self._keyword_retrieve(
            query_ctx.normalized_query,
            intent_val,
            student,
            top_k=n_results,
        )

        merged = self._merge_dedupe(vector_chunks, keyword_chunks)
        merged = self._inject_prerequisites(merged, student)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return RetrievedContext(
            chunks=merged[:n_results],
            total_retrieved=len(merged[:n_results]),
            query_embedding=query_emb,
            retrieval_time_ms=elapsed_ms,
        )

    def _vector_hit_to_chunk(self, hit: dict) -> dict:
        """Map CurriculumVectorStore hybrid_query row to corpus-shaped dict."""
        meta = hit.get("metadata")
        text = hit.get("text") or ""
        if meta is None:
            return {"id": hit.get("id"), "content_ta": text}

        if hasattr(meta, "chunk_id"):
            diagram_types = getattr(meta, "diagram_types", None) or []
            diagram_trigger = diagram_types[0] if diagram_types else None
            return {
                "id": meta.chunk_id,
                "type": meta.chunk_type,
                "topic": meta.topic,
                "section": meta.section,
                "page": getattr(meta, "page_start", 0),
                "difficulty": meta.difficulty,
                "content_ta": text,
                "diagram_trigger": diagram_trigger,
            }

        diagram_types = meta.get("diagram_types")
        if isinstance(diagram_types, str):
            try:
                diagram_types = json.loads(diagram_types)
            except json.JSONDecodeError:
                diagram_types = []
        diagram_types = diagram_types or []
        diagram_trigger = diagram_types[0] if diagram_types else None
        return {
            "id": meta.get("chunk_id", hit.get("id")),
            "type": meta.get("chunk_type", "concept"),
            "topic": meta.get("topic", ""),
            "section": meta.get("section", ""),
            "page": meta.get("page_start", 0),
            "difficulty": meta.get("difficulty", 1),
            "content_ta": text,
            "diagram_trigger": diagram_trigger,
        }

    def _merge_dedupe(self, a: list[dict], b: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for chunk in a + b:
            cid = chunk.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(chunk)
        return out

    def _inject_prerequisites(self, results: list[dict], student: StudentProfile) -> list[dict]:
        injected_ids = {c["id"] for c in results}
        out = list(results)
        for chunk in list(out):
            for prereq_topic in self.prerequisite_graph.get(chunk.get("topic", ""), []):
                skill_key = self.topic_to_skill(prereq_topic)
                if student.skills.get(skill_key, 0) < 0.4:
                    prereq_chunks = [
                        c
                        for c in self.corpus
                        if c.get("topic") == prereq_topic and c["id"] not in injected_ids
                    ]
                    if prereq_chunks:
                        out.insert(0, prereq_chunks[0])
                        injected_ids.add(prereq_chunks[0]["id"])
        return out

    def _keyword_retrieve(
        self,
        query: str,
        intent: str,
        student: StudentProfile,
        top_k: int = 4,
    ) -> list[dict]:
        """AdaptiveRetriever pipeline: pre-filter, score, rank (no prerequisite pass)."""
        filtered = self._pre_filter(intent, student)
        if not filtered:
            filtered = [c for c in self.corpus if c.get("difficulty", 1) == 1]

        scored = [(chunk, self._score_relevance(chunk, query, student)) for chunk in filtered]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored[:top_k]]

    def _pre_filter(self, intent: str, student: StudentProfile) -> list[dict]:
        """
        Filter corpus by student level and unlocked topics (fix A).

        Prerequisites are now resolved via TOPIC_TO_SKILL so that
        PREREQUISITE_GRAPH topic names correctly map to student.skills keys.
        """
        max_diff = student.get_difficulty_ceiling()
        unlocked = student.get_unlocked_topics()

        type_map = {
            "EXPLAIN": ["concept", "summary"],
            "SHOW_METHOD": ["method", "worked_example"],
            "EXERCISE_REQUEST": ["exercise", "worked_example"],
            "CHECK_ANSWER": ["exercise", "worked_example", "concept"],
            "DIAGRAM_REQUEST": ["method", "worked_example"],
            "WORD_PROBLEM": ["worked_example", "exercise", "concept"],
        }
        allowed_types = type_map.get(intent, ["concept", "method", "worked_example"])

        filtered = []
        for chunk in self.corpus:
            if chunk["type"] not in allowed_types:
                continue
            if chunk.get("difficulty", 1) > max_diff + 1:
                continue
            if chunk.get("difficulty", 1) == 1:
                filtered.append(chunk)
                continue
            chunk_topic = chunk.get("topic", "")
            if chunk_topic in unlocked:
                filtered.append(chunk)

        return filtered

    def _score_relevance(self, chunk: dict, query: str, student: StudentProfile) -> float:
        """Simple keyword overlap score — replace with embeddings in production."""
        query_words = set(query.lower().split())
        content = chunk.get("content_ta", "") + " " + chunk.get("topic", "")
        content_words = set(content.lower().split())
        overlap = len(query_words & content_words)
        score = overlap / max(len(query_words), 1)

        if chunk.get("method_number") and str(chunk.get("method_number")) in student.preferred_method:
            score += 0.2

        if student.last_topic and student.last_topic in chunk.get("topic", ""):
            score += 0.15

        chunk_diff = chunk.get("difficulty", 1)
        student_ceiling = student.get_difficulty_ceiling()
        if chunk_diff > student_ceiling + 1:
            score -= 0.3

        return score
