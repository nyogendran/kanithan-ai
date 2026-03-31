#!/usr/bin/env python3
"""
Wire AdaptiveRAGEngine → LLM (Gemini API or Ollama fallback)

Usage (Gemini — default, recommended):
  export GEMINI_API_KEY="..."          # or put it in .env
  python tutor_llm.py -q "காரணி என்றால் என்ன?"
  python tutor_llm.py -q "..." --gemini-model gemini-2.5-flash --max-output-tokens 1024

Usage (Ollama — local fallback):
  python tutor_llm.py -q "..." --backend ollama --model llama3:latest

Environment:
  GEMINI_API_KEY   required for --backend gemini (free: https://aistudio.google.com/apikey)
  OLLAMA_HOST      default http://127.0.0.1:11434  (only for --backend ollama)
  TUTOR_DISTRICT   optional region for NIE register hints (same values as --district)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from adaptive_rag_chapter4 import AdaptiveRAGEngine

load_dotenv(Path(__file__).resolve().parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini backend
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_build_config(
    types: object,
    *,
    system: str,
    temperature: float,
    max_output_tokens: int,
    disable_thinking: bool,
) -> object:
    """
    Gemini 2.5+ may use a separate 'thinking' token budget; visible answer can look
    truncated in streaming if thinking consumes the budget. thinking_budget=0
    disables that (see ThinkingConfig in google-genai).
    """
    kwargs: dict = {
        "system_instruction": system,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if disable_thinking:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def gemini_chat_stream(
    model: str,
    system: str,
    user: str,
    *,
    api_key: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    disable_thinking: bool = True,
    quiet: bool = False,
) -> tuple[str, float, float, int, str | None]:
    """Stream from Gemini API. Returns (text, secs_to_first_token, total_secs, chunk_count, finish_reason)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = _gemini_build_config(
        types,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        disable_thinking=disable_thinking,
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    n_chunks = 0
    finish_reason: str | None = None

    _log("[tutor] calling Gemini (streaming)…", quiet=quiet)

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=user,
        config=config,
    ):
        n_chunks += 1
        piece = chunk.text or ""
        if piece:
            if t_first is None:
                t_first = time.perf_counter() - t0
                _log(
                    f"[tutor] first token after {t_first:.2f}s",
                    quiet=quiet,
                )
            parts.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        if chunk.candidates:
            fr = chunk.candidates[0].finish_reason
            if fr is not None:
                finish_reason = str(fr.name) if hasattr(fr, "name") else str(fr)

    total = time.perf_counter() - t0
    if t_first is None:
        t_first = total
    full = "".join(parts).strip()
    if not quiet:
        print(file=sys.stdout)
    return full, t_first, total, n_chunks, finish_reason


def gemini_chat_non_stream(
    model: str,
    system: str,
    user: str,
    *,
    api_key: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    disable_thinking: bool = True,
) -> tuple[str, float, str | None]:
    """Non-streaming Gemini call. Returns (text, total_secs, finish_reason)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = _gemini_build_config(
        types,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        disable_thinking=disable_thinking,
    )

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=config,
    )
    t1 = time.perf_counter()
    fr = None
    if response.candidates:
        r = response.candidates[0].finish_reason
        if r is not None:
            fr = str(r.name) if hasattr(r, "name") else str(r)
    return (response.text or "").strip(), t1 - t0, fr


# ─────────────────────────────────────────────────────────────────────────────
# Ollama backend (kept for local GPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

def ollama_chat_stream(
    model: str,
    system: str,
    user: str,
    *,
    base_url: str | None = None,
    timeout_sec: float = 1800.0,
    num_ctx: int = 4096,
    num_predict: int | None = 512,
    num_thread: int | None = None,
    quiet: bool = False,
) -> tuple[str, float, float, int]:
    """Stream from Ollama. Returns (text, secs_to_first_token, total_secs, chunk_count)."""
    import httpx
    from ollama import Client as OllamaClient

    host = (base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    timeout = httpx.Timeout(connect=30.0, read=timeout_sec, write=60.0, pool=30.0)
    client = OllamaClient(host=host, timeout=timeout)

    opts: dict = {"temperature": 0.2, "num_ctx": num_ctx}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    if num_thread is not None:
        opts["num_thread"] = num_thread

    t0 = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    n_chunks = 0

    _log("[tutor] calling Ollama (streaming)…", quiet=quiet)

    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options=opts,
        stream=True,
    )

    for chunk in stream:
        n_chunks += 1
        msg = getattr(chunk, "message", None)
        if msg is None:
            continue
        piece = getattr(msg, "content", None) or ""
        if piece:
            if t_first is None:
                t_first = time.perf_counter() - t0
                _log(
                    f"[tutor] first token after {t_first:.2f}s (prompt eval + KV fill)",
                    quiet=quiet,
                )
            parts.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()

    total = time.perf_counter() - t0
    if t_first is None:
        t_first = total
    full = "".join(parts).strip()
    if not quiet:
        print(file=sys.stdout)
    return full, t_first, total, n_chunks


def ollama_chat_non_stream(
    model: str,
    system: str,
    user: str,
    *,
    base_url: str | None = None,
    timeout_sec: float = 1800.0,
    num_ctx: int = 4096,
    num_predict: int | None = 512,
    num_thread: int | None = None,
) -> tuple[str, float]:
    import httpx
    from ollama import Client as OllamaClient

    host = (base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    timeout = httpx.Timeout(connect=30.0, read=timeout_sec, write=60.0, pool=30.0)
    client = OllamaClient(host=host, timeout=timeout)
    opts: dict = {"temperature": 0.2, "num_ctx": num_ctx}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    if num_thread is not None:
        opts["num_thread"] = num_thread

    t0 = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options=opts,
        stream=False,
    )
    t1 = time.perf_counter()

    msg = getattr(response, "message", None)
    if msg is not None:
        text = getattr(msg, "content", None) or ""
    elif isinstance(response, dict):
        m = response.get("message") or {}
        text = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
    else:
        text = ""
    return text.strip(), t1 - t0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="NIE Chapter 4 Tamil Math Tutor — AdaptiveRAG + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared args
    p.add_argument("--student-id", default="SL_TM_2024_001")
    p.add_argument(
        "--district",
        default=(os.environ.get("TUTOR_DISTRICT") or "").strip() or None,
        metavar="REGION",
        help=(
            "Student region for prompt register (NIE vs spoken synonyms). "
            "Examples: jaffna, estate, batticaloa, colombo, unknown. "
            "Also set env TUTOR_DISTRICT."
        ),
    )
    p.add_argument("--question", "-q", required=True, help="Tamil student question")
    p.add_argument(
        "--top-k", type=int, default=4,
        help="Retrieved NIE chunks (lower = faster, less context); default 4",
    )
    p.add_argument(
        "--backend", choices=("gemini", "ollama"), default="gemini",
        help="LLM backend (default: gemini)",
    )
    p.add_argument("--no-stream", action="store_true", help="Buffered response")
    p.add_argument("--quiet", action="store_true", help="Suppress progress lines on stderr")
    p.add_argument("--no-llm", action="store_true", help="Only print prompt_package JSON")
    p.add_argument("--show-context", action="store_true", help="Print retrieved chunk ids")

    # Gemini-specific
    g = p.add_argument_group("Gemini options")
    g.add_argument(
        "--gemini-model", default="gemini-2.5-flash",
        help="Gemini model (default: gemini-2.5-flash)",
    )
    g.add_argument(
        "--max-output-tokens", type=int, default=2048,
        help="Max output tokens for Gemini (default: 2048)",
    )
    g.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    g.add_argument(
        "--gemini-allow-thinking",
        action="store_true",
        help=(
            "Allow Gemini 2.5+ internal 'thinking' tokens (default: off). "
            "When off, thinking is disabled so the full answer streams visibly."
        ),
    )

    # Ollama-specific
    o = p.add_argument_group("Ollama options (only with --backend ollama)")
    o.add_argument("--model", default=os.environ.get("OLLAMA_LLM", "llama3:latest"))
    o.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST"))
    o.add_argument(
        "--timeout", type=float,
        default=float(os.environ.get("OLLAMA_CHAT_TIMEOUT", "1800")),
    )
    o.add_argument(
        "--num-ctx", type=int,
        default=int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
    )
    o.add_argument(
        "--num-predict", type=int,
        default=int(os.environ.get("OLLAMA_NUM_PREDICT", "512")),
    )
    o.add_argument("--num-thread", type=int, default=None)

    args = p.parse_args()

    # ── Engine ───────────────────────────────────────────────────────────
    engine = AdaptiveRAGEngine()
    engine.get_or_create_student(student_id=args.student_id, name="மாணவர்")
    if args.district:
        stu = engine.students[args.student_id]
        stu.district = args.district
        engine._save_student(stu)

    t_eng0 = time.perf_counter()
    result = engine.process_query(args.student_id, args.question, top_k=args.top_k)
    t_eng = time.perf_counter() - t_eng0
    pkg = result["prompt_package"]

    _log(f"[tutor] engine (intent + retrieval + prompt): {t_eng:.3f}s", quiet=args.quiet)
    _log(
        f"[tutor] prompt chars: system={len(pkg['system_prompt'])}  "
        f"user={len(pkg['user_message'])}",
        quiet=args.quiet,
    )

    if args.show_context:
        print("--- Retrieved chunks ---")
        for c in pkg.get("retrieved_chunks") or []:
            print(f"  {c.get('id')}  section={c.get('section','')}  "
                  f"page={c.get('page','')}  topic={c.get('topic')}")
        print()

    if args.no_llm:
        out = {
            "intent": result["intent"],
            "retrieved_chunk_ids": result["retrieved_chunk_ids"],
            "system_prompt": pkg["system_prompt"],
            "user_message": pkg["user_message"],
            "diagram_spec": pkg.get("diagram_spec") or {},
            "exercise": pkg.get("exercise"),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # ── LLM call ─────────────────────────────────────────────────────────
    _log(f"--- Tamil answer ({args.backend}) ---", quiet=args.quiet)

    try:
        if args.backend == "gemini":
            _run_gemini(args, pkg, t_eng)
        else:
            _run_ollama(args, pkg, t_eng)
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(130)

    # ── Diagram spec ─────────────────────────────────────────────────────
    if pkg.get("diagram_spec"):
        print("\n--- diagram_spec (for Flutter / canvas) ---", file=sys.stderr)
        print(json.dumps(pkg["diagram_spec"], ensure_ascii=False, indent=2))


def _run_gemini(args: argparse.Namespace, pkg: dict, t_eng: float) -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY not set.\n"
            "  1. Get a free key at https://aistudio.google.com/apikey\n"
            "  2. export GEMINI_API_KEY='your-key'   (or add to .env file)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    disable_thinking = not args.gemini_allow_thinking

    try:
        if args.no_stream:
            text, gen_s, fin = gemini_chat_non_stream(
                args.gemini_model,
                pkg["system_prompt"],
                pkg["user_message"],
                api_key=api_key,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                disable_thinking=disable_thinking,
            )
            print(text)
            _log(f"[tutor] timing: generation {gen_s:.2f}s total", quiet=args.quiet)
            if fin:
                _log(f"[tutor] finish_reason={fin}", quiet=args.quiet)
        else:
            _text, t_first, t_total, n_chunks, fin = gemini_chat_stream(
                args.gemini_model,
                pkg["system_prompt"],
                pkg["user_message"],
                api_key=api_key,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                disable_thinking=disable_thinking,
                quiet=args.quiet,
            )
            _log(
                f"[tutor] timing: first_token={t_first:.2f}s  "
                f"total={t_total:.2f}s  chunks={n_chunks}  "
                f"~out_chars={len(_text)}",
                quiet=args.quiet,
            )
            if fin:
                _log(f"[tutor] finish_reason={fin}", quiet=args.quiet)

        _log(
            f"[tutor] timing: engine={t_eng:.3f}s + llm above = end-to-end",
            quiet=args.quiet,
        )
    except Exception as e:
        print(f"Gemini API error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_ollama(args: argparse.Namespace, pkg: dict, t_eng: float) -> None:
    import httpx

    np_predict = None if args.num_predict == 0 else args.num_predict

    try:
        if args.no_stream:
            text, gen_s = ollama_chat_non_stream(
                args.model,
                pkg["system_prompt"],
                pkg["user_message"],
                base_url=args.ollama_host,
                timeout_sec=args.timeout,
                num_ctx=args.num_ctx,
                num_predict=np_predict,
                num_thread=args.num_thread,
            )
            print(text)
            _log(f"[tutor] timing: generation {gen_s:.2f}s total", quiet=args.quiet)
        else:
            _text, t_first, t_total, n_chunks = ollama_chat_stream(
                args.model,
                pkg["system_prompt"],
                pkg["user_message"],
                base_url=args.ollama_host,
                timeout_sec=args.timeout,
                num_ctx=args.num_ctx,
                num_predict=np_predict,
                num_thread=args.num_thread,
                quiet=args.quiet,
            )
            _log(
                f"[tutor] timing: first_token={t_first:.2f}s  "
                f"total={t_total:.2f}s  chunks={n_chunks}  "
                f"~out_chars={len(_text)}",
                quiet=args.quiet,
            )

        _log(
            f"[tutor] timing: engine={t_eng:.3f}s + llm above = end-to-end",
            quiet=args.quiet,
        )
    except OSError as e:
        print(f"Cannot reach Ollama: {e}", file=sys.stderr)
        print("Start: ollama serve   (or open the Ollama app)", file=sys.stderr)
        sys.exit(1)
    except httpx.TimeoutException as e:
        print(f"Timed out after {args.timeout}s: {e}", file=sys.stderr)
        print("Try: --timeout 3600  or a smaller model.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
