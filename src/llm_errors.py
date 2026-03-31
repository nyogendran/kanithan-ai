"""Normalize Gemini / LLM errors for logs and API responses."""

from __future__ import annotations

import re
from typing import Any


def parse_retry_after_seconds(message: str) -> float | None:
    m = re.search(r"retry in ([\d.]+)\s*s", message, re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def is_quota_exhausted_error(exc_or_message: Any) -> bool:
    s = str(exc_or_message)
    return "429" in s or "RESOURCE_EXHAUSTED" in s.upper()


def format_llm_error_for_user(exc: BaseException) -> tuple[str, bool, float | None]:
    """
    Returns (user_message_ta, is_quota, retry_after_seconds).
    For quota errors, user_message is short Tamil + operational hint (no raw JSON).
    """
    raw = str(exc)
    retry = parse_retry_after_seconds(raw)
    if not is_quota_exhausted_error(exc):
        return raw, False, None

    wait = ""
    if retry is not None:
        wait = f"சுமார் {max(1, int(round(retry)))} விநாடிகள் கழித்து மீண்டும் முயற்சிக்கவும். "

    msg = (
        "Gemini API இலவச அளவு எல்லை எட்டிவிட்டது. "
        + wait
        + "இலவச நிலை: ஒரு மாதிரிக்கு நாள் தோறும் மிகக் குறைந்த கோரிக்கைகள். "
        "தொடர்ந்து பயன்படுத்த Google AI Studio / Cloud-இல் பில்லிங் அல்லது உயர் கட்டுப்பாட்டை இயக்கவும்."
    )
    return msg, True, retry
