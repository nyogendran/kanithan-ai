"""Verify student answers — deterministic math checks plus Gemini JSON verification."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from ..models import (
    ExerciseBundle,
    QueryContext,
    RetrievedContext,
    StudentProfile,
    VerificationResult,
)

log = logging.getLogger(__name__)

SOCRATIC_HINTS = {
    "used_lcm_for_hcf": "நீங்கள் மடங்குகளைக் காண்கிறீர்கள். 'அதிகூடிய பொதிகள்' என்றால் வகுபடுதல் தேவையா அல்லது மடங்கு தேவையா?",
    "used_hcf_for_lcm": "நீங்கள் காரணிகளைக் காண்கிறீர்கள். 'முதல் சந்திப்பு' என்றால் பொது மடங்கு வேண்டுமா?",
    "wrong_digit_sum": "இலக்கச் சுட்டி இரண்டு இலக்கமாக வந்தால் மீண்டும் கூட வேண்டும். உங்கள் விடையை மீண்டும் சோதியுங்கள்.",
    "incomplete_factors": "காரணிகளை ஜோடி முறையில் தேடுங்கள். 1 × ? = எண், 2 × ? = எண்... என்று ஒவ்வொரு ஜோடியையும் சோதியுங்கள்.",
    "prime_missed": "உங்கள் காரணி மரம் இன்னும் தொடர வேண்டும். கடைசி விடை 1 ஆகும் வரை வகுக்க வேண்டும்.",
    "computation_error": "உங்கள் கணக்கீட்டை மீண்டும் சோதியுங்கள். ஒரு படி பார்ப்போமா?",
    "generic": "உங்கள் முறை சரியாக உள்ளது. ஆனால் ஒரு படியில் சிறிய பிழை உள்ளது. மீண்டும் தொடக்கத்திலிருந்து படிப்படியாக முயற்சி செய்வீர்களா?",
}


class AnswerVerifierAgent:
    """Curriculum-aligned verification with deterministic shortcuts and LLM fallback."""

    def __init__(self, gemini_client: object | None = None, model: str = "gemini-2.5-flash") -> None:
        self.client = gemini_client
        self.model = model

    def verify(
        self,
        ctx: QueryContext,
        exercise: ExerciseBundle | None,
        retrieved: RetrievedContext,
        student: StudentProfile,
    ) -> VerificationResult:
        if not ctx.student_answer or not exercise:
            return VerificationResult(
                is_correct=False,
                first_wrong_step=None,
                socratic_hint_ta=SOCRATIC_HINTS["generic"],
                error_type="no_answer",
                skill_delta=0.0,
                method_used=None,
                method_expected=exercise.method_expected if exercise else None,
            )

        det = self._deterministic_check(exercise, ctx.student_answer)
        if det is True:
            return VerificationResult(
                is_correct=True,
                first_wrong_step=None,
                socratic_hint_ta="",
                error_type="none",
                skill_delta=0.1 * exercise.difficulty,
                method_used=None,
                method_expected=exercise.method_expected,
            )

        return self._verify_llm(ctx, exercise, retrieved, student)

    def _deterministic_check(self, exercise: ExerciseBundle, student_answer: str) -> Optional[bool]:
        ans: Any = exercise.answer
        raw = student_answer.strip()

        if isinstance(ans, int):
            nums = self._extract_ints(raw)
            if not nums:
                return None
            return nums[0] == ans

        if isinstance(ans, str):
            norm_s = self._normalize_prime_string(raw)
            norm_a = self._normalize_prime_string(ans)
            if norm_a and norm_s == norm_a:
                return True
            if raw.strip() == ans.strip():
                return True
            return None

        if isinstance(ans, list):
            if ans and isinstance(ans[0], int):
                got = sorted(set(self._extract_ints(raw)))
                exp = sorted(set(ans))
                if not got and raw:
                    return None
                return got == exp
            return None

        if isinstance(ans, dict):
            return self._check_digit_sum_dict(ans, raw)

        return None

    @staticmethod
    def _extract_ints(text: str) -> list[int]:
        return [int(x) for x in re.findall(r"-?\d+", text)]

    @staticmethod
    def _normalize_prime_string(s: str) -> str:
        s = s.lower().replace("x", "×").replace("*", "×")
        parts = re.findall(r"\d+", s)
        if not parts:
            return ""
        return "×".join(sorted(parts, key=int))

    @staticmethod
    def _check_digit_sum_dict(expected: dict[int, int], raw: str) -> Optional[bool]:
        pairs = re.findall(r"(\d+)\s*[:=]\s*(\d+)", raw)
        if not pairs:
            return None
        got = {int(a): int(b) for a, b in pairs}
        return got == expected

    def _verify_llm(
        self,
        ctx: QueryContext,
        exercise: ExerciseBundle,
        retrieved: RetrievedContext,
        student: StudentProfile,
    ) -> VerificationResult:
        answer_context = (
            "\n".join(c["text"] for c in retrieved.answer_scheme_chunks[:2])
            if retrieved.answer_scheme_chunks
            else "Curriculum marking scheme not available."
        )
        method_line = (
            f"எதிர்பார்க்கப்படும் முறை (scheme): {exercise.method_expected}\n"
            "JSON இல் method_used புலத்தில் மாணவர் பயன்படுத்திய முறையைக் குறிப்பிடுக "
            "(division_ladder, factor_pairs, three_methods, lcm, hcf, word_problem, digit_sum, generic)."
        )

        system = f"""நீங்கள் NIE Grade {student.grade} கணித மதிப்பீட்டு நிபுணர்.
மாணவர் பதிலை NIE வினாவிடை திட்டத்துடன் ஒப்பிட்டு பகுப்பாய்க.

கட்டாய விதிகள்:
• நேரடியாக சரியான விடையை ஒருபோதும் சொல்லாதீர்கள்
• ஒரே ஒரு வழிகாட்டும் கேள்வி மட்டும் கேளுங்கள்
• பிழையான படியை மட்டும் சுட்டுங்கள்

{method_line}

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
  "skill_delta": 0.1 (if correct, scaled by difficulty) or -0.05 (if wrong),
  "method_used": "string or null"
}}"""

        if not self.client:
            return self._string_fallback(ctx, exercise)

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
                max_output_tokens=400,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"மாணவர் பதில்: {ctx.student_answer}",
                config=config,
            )
            text = response.text.strip().strip("```json").strip("```").strip()
            data = json.loads(text)

            error_type = data.get("error_type", "generic")
            hint = data.get("socratic_hint_ta") or SOCRATIC_HINTS.get(error_type, SOCRATIC_HINTS["generic"])
            method_used = data.get("method_used")
            method_expected = exercise.method_expected
            is_correct = bool(data.get("is_correct", False))

            if method_expected and method_used and method_used != "generic":
                if method_used != method_expected and error_type in ("none", "generic"):
                    error_type = "computation_error"
                    if "lcm" in method_used.lower() and "hcf" in (method_expected or "").lower():
                        error_type = "used_lcm_for_hcf"
                    elif "hcf" in method_used.lower() and "lcm" in (method_expected or "").lower():
                        error_type = "used_hcf_for_lcm"
                    hint = SOCRATIC_HINTS.get(error_type, hint)
                    is_correct = False

            return VerificationResult(
                is_correct=is_correct,
                first_wrong_step=data.get("first_wrong_step"),
                socratic_hint_ta=hint,
                error_type=error_type,
                skill_delta=float(data.get("skill_delta", -0.05)),
                method_used=method_used,
                method_expected=method_expected,
            )
        except Exception as e:
            log.warning("Verification LLM failed: %s", e)
            return self._string_fallback(ctx, exercise)

    def _string_fallback(self, ctx: QueryContext, exercise: ExerciseBundle) -> VerificationResult:
        is_correct = str(exercise.answer).strip() == ctx.student_answer.strip()
        return VerificationResult(
            is_correct=is_correct,
            first_wrong_step=None,
            socratic_hint_ta=SOCRATIC_HINTS["generic"],
            error_type="generic",
            skill_delta=0.1 * exercise.difficulty if is_correct else -0.05,
            method_used=None,
            method_expected=exercise.method_expected,
        )
