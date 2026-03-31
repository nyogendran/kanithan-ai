from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Allow running this file directly (python3 tests/scripts/run_method_matrix.py)
# by adding repo root to sys.path.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.scripts.render_html_lesson import render_html
from tests.scripts.render_terminal_lesson import render_terminal_lesson
from tests.scripts.validate_method_alignment import validate_method_alignment


@dataclass
class Scenario:
    name: str
    question: str
    district: str
    expected_method_number: int | None
    # If expected_method_number is None, we still validate diagram/spec/tamil guard.


def _default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="T02_method3_division_ladder",
            question="6, 12, 18 இன் பொ.கா.பெ. காண்க (வகுத்தல் முறை)",
            district="batticaloa",
            expected_method_number=3,
        ),
        Scenario(
            name="T03_method2_factor_tree",
            question="6, 12, 18 இன் பொ.கா.பெ. காண்க (காரணி மரம் முறை)",
            district="colombo",
            expected_method_number=2,
        ),
        Scenario(
            name="T04_method1_factor_pairs",
            question="6, 12, 18 இன் பொ.கா.பெ. காண்க (பட்டியல் முறை)",
            district="estate",
            expected_method_number=1,
        ),
        # Auto-trigger: expected method is NOT fixed; we infer diagram method in validator.
        Scenario(
            name="T01_auto_diagram_method_any",
            question="6, 12, 18 இன் பொ.கா.பெ. காண்க",
            district="jaffna",
            expected_method_number=None,
        ),
    ]


def _out_file(out_dir: Path, scenario: Scenario) -> Path:
    return out_dir / f"{scenario.name}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run method-alignment E2E matrix against local server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="FastAPI base URL")
    parser.add_argument("--out-dir", default="tests/results/matrix", help="Output directory for JSON + previews")
    parser.add_argument("--student-prefix", default="matrix", help="student_id prefix")
    parser.add_argument("--timeout-sec", type=int, default=120, help="HTTP timeout")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=4,
        help="Run only the first N scenarios (useful to avoid Gemini rate limits).",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated scenario names to run (e.g. --only T04_method1_factor_pairs).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _default_scenarios()
    if args.only:
        allowed = {s.strip() for s in args.only.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s.name in allowed]
    if args.max_scenarios is not None:
        scenarios = scenarios[: max(0, args.max_scenarios)]

    with httpx.Client(timeout=args.timeout_sec) as client:
        for i, sc in enumerate(scenarios, start=1):
            student_id = f"{args.student_prefix}_{sc.name}_{int(time.time())}_{i}"
            payload: dict[str, Any] = {
                "student_id": student_id,
                "question": sc.question,
                "district": sc.district,
            }

            print(f"\n[{i}/{len(scenarios)}] Running {sc.name}...")
            try:
                resp = client.post(f"{args.base_url}/api/v1/query", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except httpx.TransportError as e:
                print(f"Request failed (transport error): {e}")
                print(
                    "Check: is the FastAPI server running and listening on the same host/port?\n"
                    f"  base-url={args.base_url}\n"
                    "Also run: curl -s " + args.base_url + "/health"
                )
                break

            out_json = _out_file(out_dir, sc)
            out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved JSON: {out_json}")

            # Validate method alignment + diagram spec.
            q_for_validator = sc.question
            validation = validate_method_alignment(
                data,
                question=q_for_validator,
                expected_method_number=sc.expected_method_number,
            )
            if validation.ok:
                print("Validation: OK")
            else:
                print("Validation: FAIL")
                for e in validation.errors:
                    print(f"  - {e}")
                for w in validation.warnings:
                    print(f"  ! {w}")

            # Render terminal + HTML preview
            print("Terminal lesson view:")
            render_terminal_lesson(data)

            out_html = out_dir / f"{sc.name}.html"
            # render_html writes to disk.
            render_html(data, out_html)
            print(f"HTML preview: {out_html}")


if __name__ == "__main__":
    main()

