"""Intent and Dialect enumerations used across all agents."""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    EXPLAIN = "EXPLAIN"
    SHOW_METHOD = "SHOW_METHOD"
    EXERCISE_REQUEST = "EXERCISE_REQUEST"
    CHECK_ANSWER = "CHECK_ANSWER"
    DIAGRAM_REQUEST = "DIAGRAM_REQUEST"
    WORD_PROBLEM = "WORD_PROBLEM"
    UNKNOWN = "UNKNOWN"


class Dialect(str, Enum):
    JAFFNA = "jaffna"
    BATTICALOA = "batticaloa"
    ESTATE = "estate"
    COLOMBO = "colombo"
    VANNI = "vanni"
    STANDARD = "standard"
    UNKNOWN = "unknown"
