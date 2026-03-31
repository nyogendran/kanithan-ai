"""SQLite persistence for student profiles, sessions, interactions, and HITL."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import DB_PATH
from ..models.student import StudentProfile


def _bool_to_sql(v: bool) -> int:
    return 1 if v else 0


class DatabaseManager:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = Path(db_path) if db_path is not None else DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync endpoints in worker threads; allow cross-thread usage.
        # For PoC load this is acceptable; production should use a proper DB pool.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._lock = threading.RLock()
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                student_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                query_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                query TEXT,
                intent TEXT,
                response_summary TEXT,
                response_time_ms INTEGER,
                diagram_shown INTEGER NOT NULL DEFAULT 0,
                exercise_given INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exercise_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                question TEXT,
                student_answer TEXT,
                correct_answer TEXT,
                is_correct INTEGER NOT NULL,
                method_used TEXT,
                method_expected TEXT,
                feedback TEXT
            );

            CREATE TABLE IF NOT EXISTS sentiment_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                engagement_score REAL,
                confidence_level REAL,
                frustration_flag INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS hitl_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                flag_reason TEXT,
                teacher_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                annotation TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def start_session(self, student_id: str) -> int:
        """Insert a new session row; returns integer session id for interactions."""
        now = datetime.now().isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO sessions (student_id, started_at, ended_at, query_count)
                VALUES (?, ?, NULL, 0)
                """,
                (student_id, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_or_create_student(self, student_id: str, name: str, district: str) -> StudentProfile:
        cur = self._conn.execute(
            "SELECT profile_json FROM profiles WHERE student_id = ?",
            (student_id,),
        )
        row = cur.fetchone()
        if row:
            profile = StudentProfile.from_dict(json.loads(row["profile_json"]))
            profile.name = name
            profile.district = district
            self.save_student(profile)
            return profile
        profile = StudentProfile(student_id=student_id, name=name, district=district)
        self.save_student(profile)
        return profile

    def save_student(self, profile: StudentProfile) -> None:
        payload = json.dumps(asdict(profile), ensure_ascii=False)
        now = datetime.now().isoformat()
        self._conn.execute(
            """
            INSERT INTO profiles (student_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (profile.student_id, payload, now),
        )
        self._conn.commit()

    def record_interaction(
        self,
        student_id: str,
        session_id: int,
        query: str,
        intent: str,
        response_summary: str,
        response_time_ms: int,
        diagram_shown: bool,
        exercise_given: bool,
    ) -> int:
        ts = datetime.now().isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO interactions (
                session_id, student_id, query, intent, response_summary,
                response_time_ms, diagram_shown, exercise_given, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                student_id,
                query,
                intent,
                response_summary,
                response_time_ms,
                _bool_to_sql(diagram_shown),
                _bool_to_sql(exercise_given),
                ts,
            ),
        )
        self._conn.execute(
            "UPDATE sessions SET query_count = query_count + 1 WHERE id = ?",
            (session_id,),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_exercise_outcome(
        self,
        interaction_id: int,
        question: str,
        student_answer: str,
        correct_answer: str,
        is_correct: bool,
        method_used: str,
        method_expected: str,
        feedback: str,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO exercise_outcomes (
                interaction_id, question, student_answer, correct_answer,
                is_correct, method_used, method_expected, feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                question,
                student_answer,
                correct_answer,
                _bool_to_sql(is_correct),
                method_used,
                method_expected,
                feedback,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_sentiment(
        self,
        interaction_id: int,
        engagement_score: float,
        confidence_level: float,
        frustration_flag: bool,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO sentiment_signals (
                interaction_id, engagement_score, confidence_level, frustration_flag
            ) VALUES (?, ?, ?, ?)
            """,
            (
                interaction_id,
                engagement_score,
                confidence_level,
                _bool_to_sql(frustration_flag),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_hitl_flag(self, interaction_id: int, flag_reason: str) -> int:
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO hitl_queue (
                interaction_id, flag_reason, teacher_id, status, annotation,
                created_at, updated_at
            ) VALUES (?, ?, NULL, 'pending', NULL, ?, ?)
            """,
            (interaction_id, flag_reason, now, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_hitl_queue(self, status: str = "pending") -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT id, interaction_id, flag_reason, teacher_id, status, annotation,
                   created_at, updated_at
            FROM hitl_queue
            WHERE status = ?
            ORDER BY created_at ASC
            """,
            (status,),
        )
        rows = cur.fetchall()
        return [{k: row[k] for k in row.keys()} for row in rows]

    def update_hitl_status(
        self,
        queue_id: int,
        status: str,
        teacher_id: str | None,
        annotation: str | None,
    ) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            """
            UPDATE hitl_queue
            SET status = ?, teacher_id = ?, annotation = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, teacher_id, annotation, now, queue_id),
        )
        self._conn.commit()
