from __future__ import annotations

from pathlib import Path

from .pipeline import IngestionPipeline


def ingest_answer_scheme(pdf_path: str | Path, grade: int, chapter: int) -> int:
    """Convenience wrapper around IngestionPipeline.ingest_answer_scheme."""
    path = Path(pdf_path)
    return IngestionPipeline().ingest_answer_scheme(path, grade, chapter)
