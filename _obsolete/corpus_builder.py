#!/usr/bin/env python3
"""
corpus_builder.py — Build NIE_CORPUS JSON from a Tamil NIE PDF.

The NIE Tamil-medium Grade 7 textbook (and many Sri Lankan government PDFs)
are published with TSCII encoding — an 8-bit encoding that maps Tamil glyphs
to Latin codepoints. PyMuPDF extracts bytes faithfully but the resulting
string contains mojibake if decoded as UTF-8.

This script:
  1. Reads each PDF page via PyMuPDF.
  2. Converts TSCII → Unicode Tamil using the TSCIIToUnicode map.
  3. Segments text by section/topic headings heuristically.
  4. Writes a JSON file of corpus chunks compatible with adaptive_rag_chapter4.py.

After running:
  python corpus_builder.py --pdf data/mathematics_gr7_part1_factors.pdf \
                           --out data/nie_corpus_ch4.json \
                           --chapter 4 --start-page 33 --end-page 58

Then import the result:
  import json
  with open("data/nie_corpus_ch4.json") as f:
      NIE_CORPUS = json.load(f)

IMPORTANT — Manual review step:
  The TSCII table below is incomplete for some ligature/conjunct characters.
  After generating, open nie_corpus_ch4.json in any editor and scan for
  remaining garbled characters (typically the ones still in Latin-range).
  Fix them in the JSON, then lock the JSON into version control.
  This one-time review is much faster than manual corpus authoring.

TSCII Reference: https://en.wikipedia.org/wiki/Tamil_script_computing#TSCII
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# TSCII → Unicode mapping
#
# TSCII uses codepoints 0x80–0xFF for Tamil characters.
# The Unicode Tamil block is U+0B80–U+0BFF.
# This table is adapted from the TSCII 1.7 specification.
# ─────────────────────────────────────────────────────────────────────────────

TSCII_TABLE: dict[int, str] = {
    # Key = TSCII byte (decimal)  Value = Unicode character(s)
    # ── vowels ──
    0x82: "\u0B85",   # அ
    0x83: "\u0B86",   # ஆ
    0x84: "\u0B87",   # இ
    0x85: "\u0B88",   # ஈ
    0x86: "\u0B89",   # உ
    0x87: "\u0B8A",   # ஊ
    0x88: "\u0B8E",   # எ
    0x89: "\u0B8F",   # ஏ
    0x8A: "\u0B90",   # ஐ
    0x8B: "\u0B92",   # ஒ
    0x8C: "\u0B93",   # ஓ
    0x8D: "\u0B94",   # ஔ
    # ── consonants ──
    0x8E: "\u0B95",   # க
    0x90: "\u0B99",   # ங
    0x91: "\u0B9A",   # ச
    0x93: "\u0B9C",   # ஜ
    0x94: "\u0B9E",   # ஞ
    0x95: "\u0B9F",   # ட
    0x97: "\u0BA3",   # ண
    0x99: "\u0BA4",   # த
    0x9C: "\u0BA8",   # ந
    0x9D: "\u0BA9",   # ன
    0x9E: "\u0BAA",   # ப
    0xA1: "\u0BAE",   # ம
    0xA2: "\u0BAF",   # ய
    0xA3: "\u0BB0",   # ர
    0xA4: "\u0BB1",   # ற
    0xA5: "\u0BB2",   # ல
    0xA6: "\u0BB3",   # ள
    0xA7: "\u0BB4",   # ழ
    0xA8: "\u0BB5",   # வ
    0xA9: "\u0BB6",   # ஶ (sha — rare in maths)
    0xAA: "\u0BB7",   # ஷ
    0xAB: "\u0BB8",   # ஸ
    0xAC: "\u0BB9",   # ஹ
    # ── vowel marks (matras) ──
    0xAE: "\u0BBE",   # ா
    0xAF: "\u0BBF",   # ி
    0xB0: "\u0BC0",   # ீ
    0xB1: "\u0BC1",   # ு
    0xB2: "\u0BC2",   # ூ
    0xB4: "\u0BC6",   # ெ
    0xB5: "\u0BC7",   # ே
    0xB6: "\u0BC8",   # ை
    0xB7: "\u0BCA",   # ொ
    0xB8: "\u0BCB",   # ோ
    0xB9: "\u0BCC",   # ௌ
    0xBA: "\u0BCD",   # ் (pulli / virama)
    # ── digits ──
    0xBB: "\u0BE6",   # ௦
    0xBC: "\u0BE7",   # ௧
    0xBD: "\u0BE8",   # ௨
    0xBE: "\u0BE9",   # ௩
    0xBF: "\u0BEA",   # ௪
    0xC0: "\u0BEB",   # ௫
    0xC1: "\u0BEC",   # ௬
    0xC2: "\u0BED",   # ௭
    0xC3: "\u0BEE",   # ௮
    0xC4: "\u0BEF",   # ௯
    # ── special / punctuation ──
    0xC5: "\u0BF3",   # ௳ (day sign — used for 'therefore' ∴ in some prints)
    0xC6: "\u0BD0",   # ௐ (om)
    0x80: "\u0B83",   # ஃ (aytham)
    # ── common ligatures: க + virama combos ──
    0x8F: "\u0B95\u0BCD",  # க்
    0x92: "\u0B9A\u0BCD",  # ச்
    0x96: "\u0B9F\u0BCD",  # ட்
    0x98: "\u0BA3\u0BCD",  # ண்
    0x9A: "\u0BA4\u0BCD",  # த்
    0x9B: "\u0BA8\u0BCD",  # ந்
    0x9F: "\u0BAA\u0BCD",  # ப்
    0xA0: "\u0BAE\u0BCD",  # ம் (alternate)
    # ── ஆ sign as separate mark ──
    0xAD: "\u0BBE",   # ா (duplicate slot in some fonts)
}


def tscii_to_unicode(raw: str) -> str:
    """
    Convert a TSCII-encoded string (read as latin-1 bytes) to Unicode Tamil.

    PyMuPDF returns page.get_text() as a Python str; for TSCII PDFs the
    Tamil bytes land in the Latin-supplement area (0x80–0xFF).
    We iterate codepoints and map anything in that range via TSCII_TABLE;
    ordinary ASCII passes through unchanged.
    """
    out: list[str] = []
    for ch in raw:
        cp = ord(ch)
        if cp > 0xFF:
            # Already Unicode (unlikely but safe to pass through)
            out.append(ch)
        elif cp in TSCII_TABLE:
            out.append(TSCII_TABLE[cp])
        else:
            out.append(ch)
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Page extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path, start_page: int, end_page: int) -> list[dict]:
    """
    Returns list of {page_num, text} for the given page range.
    page_num is 1-indexed (matching NIE textbook pages).
    start_page / end_page are NIE textbook page numbers; this function
    finds them by searching for the page number in the extracted text.
    """
    try:
        import fitz
    except ImportError:
        sys.exit("pymupdf not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages: list[dict] = []

    for pdf_idx in range(len(doc)):
        raw = doc[pdf_idx].get_text()
        unicode_text = tscii_to_unicode(raw)
        pages.append({
            "pdf_page_index": pdf_idx,
            "text": unicode_text,
            "raw_preview": raw[:120],   # for debugging encoding issues
        })

    doc.close()

    # Filter by textbook page number heuristic: NIE pages print their number
    # at the top of the page, sometimes as a standalone line.
    # We keep pages whose text contains the NIE page number in the range.
    def contains_page_number(text: str, n: int) -> bool:
        return bool(re.search(rf"(?:^|\n)\s*{n}\s*(?:\n|$)", text))

    # If page range detection is unreliable, fall back to PDF page index slice
    filtered = [
        p for p in pages
        if any(contains_page_number(p["text"], n)
               for n in range(start_page, end_page + 1))
    ]

    # Fallback: if heuristic finds nothing, use raw index-based slice
    if not filtered:
        # Estimate offset: NIE PDFs typically have ~2-5 front-matter pages
        offset = 1
        lo = max(0, start_page - offset)
        hi = min(len(pages), end_page - offset + 1)
        filtered = pages[lo:hi]

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic segmentation
# ─────────────────────────────────────────────────────────────────────────────

# Section-heading patterns (Unicode Tamil) — extend as needed
SECTION_PATTERNS = [
    (r"4\s*\.\s*1\b", "4.1", "divisibility_rules"),
    (r"4\s*\.\s*2\b", "4.2", "factor_listing"),
    (r"4\s*\.\s*3\b", "4.3", "prime_factorization"),
    (r"4\s*\.\s*4\b", "4.4", "factor_pairs"),
    (r"4\s*\.\s*5\b", "4.5", "hcf"),
    (r"4\s*\.\s*6\b", "4.6", "lcm"),
    (r"பயிற்சி|பயிற்\s*சி", None, "exercise"),
    (r"உதாரணம்|உதாரண\s*ம்", None, "worked_example"),
    (r"குறிப்பு|சுருக்கம்", None, "summary"),
    (r"செயற்பாடு|செயற்\s*பாடு", None, "activity"),
]

DIFFICULTY_MAP: dict[str, int] = {
    "divisibility_rules": 1,
    "factor_definition":  1,
    "digit_sum":          1,
    "factor_listing":     2,
    "prime_factorization": 2,
    "factor_pairs":       2,
    "hcf":                3,
    "lcm":                3,
    "exercise":           2,
    "worked_example":     3,
    "summary":            1,
    "activity":           1,
}

TYPE_MAP: dict[str, str] = {
    "divisibility_rules": "rule",
    "factor_listing":     "method",
    "prime_factorization": "method",
    "hcf":                "method",
    "lcm":                "method",
    "exercise":           "exercise",
    "worked_example":     "worked_example",
    "summary":            "summary",
    "activity":           "concept",
}


def segment_pages(pages: list[dict], chapter: int) -> list[dict]:
    """
    Heuristically split pages into corpus chunks.

    Strategy:
    - Concatenate all page text.
    - Split on section headings (4.1, 4.2 …) and known Tamil heading patterns.
    - Each segment becomes one chunk.
    """
    full_text = "\n".join(p["text"] for p in pages)

    # Build split pattern
    combined_pattern = "|".join(
        f"(?P<sec{i}>{pat})" for i, (pat, _, _) in enumerate(SECTION_PATTERNS)
    )

    segments: list[dict] = []
    current_section = f"{chapter}.0"
    current_topic = "general"
    current_type = "concept"
    buffer_lines: list[str] = []
    chunk_id_counter = 1

    lines = full_text.splitlines()
    for line in lines:
        matched = False
        for pat, sec, topic in SECTION_PATTERNS:
            if re.search(pat, line):
                # Flush current buffer
                if buffer_lines:
                    segments.append(_make_chunk(
                        chunk_id=f"AUTO_{chapter}_{chunk_id_counter:03d}",
                        content="\n".join(buffer_lines).strip(),
                        section=current_section,
                        topic=current_topic,
                        chunk_type=current_type,
                        chapter=chapter,
                    ))
                    chunk_id_counter += 1
                    buffer_lines = []
                if sec:
                    current_section = sec
                if topic:
                    current_topic = topic
                current_type = TYPE_MAP.get(topic or "concept", "concept")
                matched = True
                break
        buffer_lines.append(line)

    # Flush final buffer
    if buffer_lines:
        segments.append(_make_chunk(
            chunk_id=f"AUTO_{chapter}_{chunk_id_counter:03d}",
            content="\n".join(buffer_lines).strip(),
            section=current_section,
            topic=current_topic,
            chunk_type=current_type,
            chapter=chapter,
        ))

    return [s for s in segments if len(s["content_ta"].strip()) > 30]


def _make_chunk(chunk_id: str, content: str, section: str,
                topic: str, chunk_type: str, chapter: int) -> dict:
    return {
        "id": chunk_id,
        "type": chunk_type,
        "topic": topic,
        "section": section,
        "page": None,   # fill in manually after review
        "difficulty": DIFFICULTY_MAP.get(topic, 1),
        "prerequisites": [],  # fill in via PREREQUISITE_GRAPH after review
        "content_ta": content,
        "key_terms": {},
        "diagram_trigger": None,
        "exercise_follow_up": None,
        "_review_needed": True,   # flag for human review pass
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quality check
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(chunks: list[dict]) -> None:
    """Print diagnostics to help the human reviewer."""
    print(f"\n{'='*60}")
    print(f"Corpus build report — {len(chunks)} chunks generated")
    print(f"{'='*60}")

    garbled = 0
    for chunk in chunks:
        text = chunk["content_ta"]
        # Heuristic: if >30% chars are in Latin supplement (0x80–0xFF) after
        # conversion, conversion is incomplete
        high_bytes = sum(1 for c in text if 0x80 <= ord(c) <= 0xFF)
        ratio = high_bytes / max(len(text), 1)
        if ratio > 0.15:
            garbled += 1

    print(f"  Chunks with possible encoding issues (>15% non-Unicode): {garbled}")
    if garbled > 0:
        print("  → Review TSCII_TABLE in corpus_builder.py and add missing entries.")
        print("  → Or obtain the NIE PDF in UTF-8/Unicode (preferred).")

    tamil_chunks = [c for c in chunks
                    if any("\u0B80" <= ch <= "\u0BFF" for ch in c["content_ta"])]
    print(f"  Chunks with Tamil Unicode characters: {len(tamil_chunks)}")
    print(f"  Empty / too-short chunks filtered: (already removed)")
    print(f"\nNext steps:")
    print("  1. Open nie_corpus_ch4.json and scan each chunk['content_ta'].")
    print("  2. Fix remaining garbled characters.")
    print("  3. Set 'page', 'prerequisites', 'key_terms' for each chunk.")
    print("  4. Set 'diagram_trigger' for factor_tree / division_ladder chunks.")
    print("  5. Remove '_review_needed' key when done.")
    print("  6. Replace NIE_CORPUS list in adaptive_rag_chapter4.py with:")
    print("       with open('data/nie_corpus_ch4.json') as f:")
    print("           NIE_CORPUS = json.load(f)")


# ─────────────────────────────────────────────────────────────────────────────
# Unicode detection — tells you if the PDF is already Unicode
# ─────────────────────────────────────────────────────────────────────────────

def detect_encoding(pdf_path: Path, sample_pages: int = 5) -> str:
    """
    Detect whether the PDF text is TSCII, Unicode Tamil, or unrecognised.
    Prints a report and returns 'unicode', 'tscii', or 'unknown'.
    """
    try:
        import fitz
    except ImportError:
        return "unknown"

    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, len(doc))
    unicode_tamil = 0
    high_latin = 0
    total = 0

    for i in range(pages_to_check):
        text = doc[i].get_text()
        for ch in text:
            cp = ord(ch)
            total += 1
            if 0x0B80 <= cp <= 0x0BFF:
                unicode_tamil += 1
            elif 0x80 <= cp <= 0xFF:
                high_latin += 1

    doc.close()
    print(f"\nEncoding detection ({pages_to_check} pages, {total} chars):")
    print(f"  Unicode Tamil (U+0B80–0BFF): {unicode_tamil}")
    print(f"  Latin-supplement (0x80–0xFF): {high_latin}")

    if unicode_tamil > high_latin:
        print("  → PDF appears to be ALREADY Unicode Tamil. No TSCII conversion needed.")
        print("    You can use rag_poc.py directly with this PDF.")
        return "unicode"
    elif high_latin > unicode_tamil:
        print("  → PDF appears TSCII-encoded. corpus_builder.py TSCII conversion required.")
        return "tscii"
    else:
        print("  → Encoding uncertain — may be image-only (needs OCR) or mixed.")
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NIE_CORPUS JSON chunks from a Tamil NIE PDF."
    )
    parser.add_argument("--pdf", type=Path, required=True,
                        help="Path to NIE Tamil mathematics PDF")
    parser.add_argument("--out", type=Path,
                        default=Path("data/nie_corpus_ch4.json"),
                        help="Output JSON path")
    parser.add_argument("--chapter", type=int, default=4,
                        help="Chapter number (for chunk ID prefix)")
    parser.add_argument("--start-page", type=int, default=33,
                        help="First NIE textbook page to include")
    parser.add_argument("--end-page", type=int, default=58,
                        help="Last NIE textbook page to include")
    parser.add_argument("--detect-only", action="store_true",
                        help="Only detect PDF encoding; do not generate corpus")
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    encoding = detect_encoding(args.pdf)
    if args.detect_only:
        return

    print(f"\nExtracting pages {args.start_page}–{args.end_page}…")
    pages = extract_pages(args.pdf, args.start_page, args.end_page)
    print(f"  Found {len(pages)} PDF page(s) in range.")

    if not pages:
        print("  No pages found — check --start-page / --end-page range.")
        return

    print("Segmenting into chunks…")
    chunks = segment_pages(pages, args.chapter)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(chunks)} chunks → {args.out}")
    quality_report(chunks)


if __name__ == "__main__":
    main()
