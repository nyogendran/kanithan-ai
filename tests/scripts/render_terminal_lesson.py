from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


def _safe_get(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def render_terminal_lesson(resp: dict[str, Any]) -> None:
    student_id = resp.get("student_id") or "unknown"
    intent = resp.get("intent") or "unknown"
    dialect = resp.get("dialect_detected") or "unknown"
    diagram = resp.get("diagram")
    teaching = resp.get("teaching") or {}

    print("=" * 80)
    print(f"Lesson View | student_id={student_id} | intent={intent} | dialect={dialect}")

    if diagram:
        diagram_type = diagram.get("diagram_type") or "unknown"
        caption = diagram.get("caption_ta") or ""
        animate = diagram.get("animate")
        print(f"Diagram | type={diagram_type} | animate={animate}")
        if caption:
            print(f"Caption | {caption}")
        spec = diagram.get("spec")
        if spec is not None:
            spec_preview = json.dumps(spec, ensure_ascii=False, indent=2)
            print("Diagram Spec (JSON):")
            print(textwrap.indent(spec_preview, "  "))
    else:
        print("Diagram | None")

    if teaching:
        explanation = teaching.get("explanation_ta") or ""
        key_concepts = teaching.get("key_concepts") or []
        print("-" * 80)
        print(f"Teaching | key_concepts_count={len(key_concepts)}")
        if explanation:
            explanation_preview = explanation
            if len(explanation_preview) > 6000:
                explanation_preview = explanation_preview[:6000] + "\n...(truncated)"
            print(explanation_preview)
        else:
            print("Teaching | explanation_ta is empty")

    if resp.get("error"):
        print("-" * 80)
        print(f"ERROR: {resp['error']}")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render NIE tutor response in terminal")
    parser.add_argument("--json", required=True, help="Path to response JSON file")
    args = parser.parse_args()

    json_path = Path(args.json).expanduser().resolve()
    resp = json.loads(json_path.read_text(encoding="utf-8"))
    render_terminal_lesson(resp)


if __name__ == "__main__":
    main()

