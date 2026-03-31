from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METHOD_BY_DIAGRAM_TYPE: dict[str, int] = {
    "factor_pairs": 1,
    "factor_tree": 2,
    "division_ladder": 3,
}


EXPECTED_SPEC_KEYS: dict[int, list[str]] = {
    1: ["pairs", "all_factors", "number"],
    2: ["root", "tree", "prime_factors"],
    3: ["numbers", "steps", "hcf_value"],
}


BANNED_TEACHING_RE = re.compile(r"[A-Za-z]")
BANNED_TEACHING_WORDS = [
    "Let's",
    "We",
    "factor",
    "remainder",
    "whole number",
]


def _get(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _infer_method_from_question(question: str) -> int | None:
    q = (question or "").strip()
    if "பட்டியல்" in q or "ஜோடி" in q:
        return 1
    if "காரணி மர" in q or "முதன்மைக் காரண" in q:
        return 2
    if "வகுத்தல்" in q or "ஏணி" in q:
        return 3
    return None


def _infer_method_from_diagram(resp: dict[str, Any]) -> int | None:
    diagram_type = _get(resp, "diagram", "diagram_type")
    if not diagram_type:
        return None
    return METHOD_BY_DIAGRAM_TYPE.get(str(diagram_type))


def _extract_declared_result_number(explanation: str, key: str) -> int | None:
    """
    key examples:
      "பொ.கா.பெ."  (HCF)
      "பொ.ம.சி."  (LCM)
    """
    if not explanation:
        return None

    # We want the final numeric result even for expressions like:
    # "பொ.கா.பெ. = 2 × 3 = 6" (return 6, not 2)
    if key not in explanation:
        return None

    # Use the *last* occurrence, so we capture the final computed result
    # (e.g., "பொ.கா.பெ. = 2 × 3 = 6" should return 6, not 2/3).
    idx = explanation.rfind(key)
    window = explanation[idx:]
    nums = re.findall(r"\b(\d+)\b", window)
    if not nums:
        return None
    return int(nums[-1])


def _extract_numbers_from_question(question: str) -> list[int]:
    return [int(x) for x in re.findall(r"\b\d+\b", question or "")]


def _method_keywords_method1() -> list[str]:
    return ["ஜோடி", "பட்டியல்", "பட்டியலை", "எழுதுங்கள்", "பொதுக் காரணிகள"]


def _method_keywords_method2() -> list[str]:
    return ["முதன்மைக் காரண", "காரணி மரம்", "காரணி மர", "பெருக்கம்"]


def _method_keywords_method3() -> list[str]:
    return ["வகுத்தல்", "வகுப்போம்", "ஏணி", "|"]


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_method_alignment(
    resp: dict[str, Any],
    *,
    question: str | None = None,
    expected_method_number: int | None = None,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    # If Gemini quota/rate-limits happen, `teaching` might be missing.
    # For diagram alignment testing, we treat this as a warning (we can
    # still validate `diagram.diagram_type` + `diagram.spec`).
    if resp.get("error"):
        warnings.append(f"Response contains error (ignored for diagram-spec validation): {resp.get('error')}")

    teaching = resp.get("teaching") or {}
    explanation = teaching.get("explanation_ta") or ""
    diagram = resp.get("diagram") or None
    diagram_type = _get(resp, "diagram", "diagram_type")
    method_from_diagram = _infer_method_from_diagram(resp)

    # 1) Tamil-only guard for teaching text
    if BANNED_TEACHING_RE.search(explanation):
        errors.append("Teaching explanation contains English letters [A-Za-z].")

    for w in BANNED_TEACHING_WORDS:
        if w in explanation or w.lower() in explanation.lower():
            errors.append(f"Teaching explanation contains banned word: {w}")
            break

    # 2) Diagram spec shape validation
    if expected_method_number is not None:
        if method_from_diagram != expected_method_number:
            errors.append(
                f"Diagram method mismatch. expected_method_number={expected_method_number} diagram_method={method_from_diagram} diagram_type={diagram_type}"
            )
    elif expected_method_number is None and question:
        inferred = _infer_method_from_question(question)
        if inferred is not None and method_from_diagram != inferred:
            warnings.append(
                f"Question method cue inferred as {inferred} but diagram method inferred as {method_from_diagram}."
            )

    if diagram is None:
        errors.append("No diagram present in response.")
    else:
        spec = diagram.get("spec") or {}
        if expected_method_number is not None:
            need_keys = EXPECTED_SPEC_KEYS.get(expected_method_number, [])
        else:
            need_keys = EXPECTED_SPEC_KEYS.get(method_from_diagram or -1, [])

        for k in need_keys:
            if k not in spec:
                errors.append(f"Diagram spec missing key: {k}")

    # 3) Teaching content heuristics match expected method
    expected_method = expected_method_number or method_from_diagram
    if expected_method in (1, 2, 3) and explanation:
        if expected_method == 1:
            must = _method_keywords_method1()
            if not any(k in explanation for k in must if k != "|"):
                warnings.append("Method I heuristics: teaching text does not contain obvious pair/listing markers.")
        elif expected_method == 2:
            must = _method_keywords_method2()
            if not any(k in explanation for k in must):
                warnings.append("Method II heuristics: teaching text does not contain obvious prime-factor/tree markers.")
        elif expected_method == 3:
            must = _method_keywords_method3()
            if not any(k in explanation for k in must):
                warnings.append("Method III heuristics: teaching text does not contain obvious division-ladder markers.")

    # 4) Optional: validate declared HCF/LCM correctness (if parseable)
    numbers = _extract_numbers_from_question(question or "")
    if numbers:
        if "பொ.கா.பெ." in (question or "") or "பொ.கா.பெ." in explanation:
            import math
            hcf = 0
            for n in numbers:
                hcf = math.gcd(hcf, n)
            declared = _extract_declared_result_number(explanation, "பொ.கா.பெ.")
            if declared is not None and declared != hcf:
                errors.append(f"HCF mismatch: computed gcd={hcf} declared={declared}")
        if "பொ.ம.சி." in (question or "") or "பொ.ம.சி." in explanation:
            import math
            # lcm for multiple numbers
            def lcm(a: int, b: int) -> int:
                return abs(a * b) // math.gcd(a, b) if a and b else 0
            l = 1
            for n in numbers:
                l = lcm(l, n)
            declared = _extract_declared_result_number(explanation, "பொ.ம.சி.")
            if declared is not None and declared != l:
                errors.append(f"LCM mismatch: computed lcm={l} declared={declared}")

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate method alignment for NIE tutor response JSON")
    parser.add_argument("--json", required=True, help="Response JSON path")
    parser.add_argument("--question", help="Question text (optional, for HCF/LCM correctness)")
    parser.add_argument("--expected-method", type=int, choices=[1, 2, 3], help="Expected method number")
    args = parser.parse_args()

    json_path = Path(args.json).expanduser().resolve()
    resp = json.loads(json_path.read_text(encoding="utf-8"))

    res = validate_method_alignment(
        resp,
        question=args.question,
        expected_method_number=args.expected_method,
    )

    print(f"OK={res.ok}")
    if res.errors:
        print("Errors:")
        for e in res.errors:
            print(f"- {e}")
    if res.warnings:
        print("Warnings:")
        for w in res.warnings:
            print(f"- {w}")

    raise SystemExit(0 if res.ok else 1)


if __name__ == "__main__":
    main()

