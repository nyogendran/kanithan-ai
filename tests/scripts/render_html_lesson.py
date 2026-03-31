from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


HTML_TEMPLATE = """<!doctype html>
<html lang="ta">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NIE Tutor Lesson Preview</title>
    <style>
      :root {{ --bg: #0b1020; --fg: #e9eefc; --muted: #a9b3d8; --accent: #4cc9f0; --card: #121a34; }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        background: radial-gradient(1200px 600px at 20% 0%, rgba(76,201,240,0.25), transparent 55%),
                    radial-gradient(900px 500px at 90% 20%, rgba(160,122,255,0.18), transparent 60%),
                    var(--bg);
        color: var(--fg);
      }}
      .wrap {{ max-width: 980px; margin: 0 auto; padding: 20px; }}
      .header {{
        display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
        padding: 16px 18px; border-radius: 14px; background: rgba(18,26,52,0.85); border: 1px solid rgba(255,255,255,0.08);
        position: sticky; top: 12px;
        backdrop-filter: blur(6px);
      }}
      .header h1 {{ margin: 0; font-size: 18px; letter-spacing: 0.2px; }}
      .meta {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }}
      .card {{
        background: rgba(18,26,52,0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        overflow: hidden;
      }}
      .card .title {{
        padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08);
        font-weight: 700;
      }}
      .content {{ padding: 14px 16px; }}
      pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        background: rgba(0,0,0,0.18);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 12px 14px;
        border-radius: 12px;
        color: var(--fg);
        line-height: 1.45;
      }}
      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(76,201,240,0.12);
        border: 1px solid rgba(76,201,240,0.35);
        color: var(--accent);
        font-weight: 700;
        font-size: 12px;
      }}
      .small {{ color: var(--muted); font-size: 12px; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="header">
        <div>
          <h1>NIE Tutor Lesson Preview</h1>
          <div class="small">Local file preview (no external assets)</div>
        </div>
        <div class="meta">
          <div><span class="pill">{student_id}</span></div>
          <div>intent: <b>{intent}</b> | dialect: <b>{dialect}</b></div>
          <div>diagram: <b>{diagram_type}</b></div>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <div class="title">Teaching Explanation (Tamil)</div>
          <div class="content">
            <pre>{explanation_ta}</pre>
          </div>
        </div>

        <div class="card">
          <div class="title">Diagram (Spec for Flutter renderer)</div>
          <div class="content">
            <div class="small" style="margin-bottom:10px;">
              caption_ta: <b>{caption_ta}</b>
            </div>
            <pre>{diagram_spec_json}</pre>
          </div>
        </div>
      </div>
    </div>
  </body>
</html>
"""


def render_html(resp: dict[str, Any], out_path: Path) -> None:
    student_id = escape(str(resp.get("student_id") or "unknown"))
    intent = escape(str(resp.get("intent") or "unknown"))
    dialect = escape(str(resp.get("dialect_detected") or "unknown"))

    diagram = resp.get("diagram") or None
    diagram_type = escape(str(diagram.get("diagram_type") if diagram else "None"))
    caption_ta = escape(str(diagram.get("caption_ta") if diagram else ""))

    teaching = resp.get("teaching") or {}
    explanation_ta = escape(str(teaching.get("explanation_ta") or ""))

    diagram_spec_json = ""
    if diagram and diagram.get("spec") is not None:
        diagram_spec_json = escape(json.dumps(diagram.get("spec"), ensure_ascii=False, indent=2))

    html = HTML_TEMPLATE.format(
        student_id=student_id,
        intent=intent,
        dialect=dialect,
        diagram_type=diagram_type,
        explanation_ta=explanation_ta,
        caption_ta=caption_ta,
        diagram_spec_json=diagram_spec_json,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render NIE tutor response as HTML preview")
    parser.add_argument("--json", required=True, help="Path to response JSON")
    parser.add_argument("--out", required=True, help="Output HTML file path")
    args = parser.parse_args()

    json_path = Path(args.json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    resp = json.loads(json_path.read_text(encoding="utf-8"))
    render_html(resp, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

