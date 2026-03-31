from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

from src.config import (
    CHROMA_PATH,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBED_MODEL,
    MIN_CHUNK_CHARS,
)
from .vector_store import ChunkMetadata, NIEVectorStore, TamilEmbedder

log = logging.getLogger("nie.ingestion")

OCR_LANG = "tam+eng"                        # Tesseract: Tamil + English digits

TSCII_TABLE: dict[int, str] = {
    0x82: "\u0B85", 0x83: "\u0B86", 0x84: "\u0B87", 0x85: "\u0B88",
    0x86: "\u0B89", 0x87: "\u0B8A", 0x88: "\u0B8E", 0x89: "\u0B8F",
    0x8A: "\u0B90", 0x8B: "\u0B92", 0x8C: "\u0B93", 0x8D: "\u0B94",
    0x8E: "\u0B95", 0x8F: "\u0B95\u0BCD", 0x90: "\u0B99",
    0x91: "\u0B9A", 0x92: "\u0B9A\u0BCD", 0x93: "\u0B9C",
    0x94: "\u0B9E", 0x95: "\u0B9F", 0x96: "\u0B9F\u0BCD",
    0x97: "\u0BA3", 0x98: "\u0BA3\u0BCD", 0x99: "\u0BA4",
    0x9A: "\u0BA4\u0BCD", 0x9B: "\u0BA8\u0BCD", 0x9C: "\u0BA8",
    0x9D: "\u0BA9", 0x9E: "\u0BAA", 0x9F: "\u0BAA\u0BCD",
    0xA0: "\u0BAE\u0BCD", 0xA1: "\u0BAE", 0xA2: "\u0BAF",
    0xA3: "\u0BB0", 0xA4: "\u0BB1", 0xA5: "\u0BB2",
    0xA6: "\u0BB3", 0xA7: "\u0BB4", 0xA8: "\u0BB5",
    0xA9: "\u0BB6", 0xAA: "\u0BB7", 0xAB: "\u0BB8", 0xAC: "\u0BB9",
    0xAD: "\u0BBE", 0xAE: "\u0BBE", 0xAF: "\u0BBF",
    0xB0: "\u0BC0", 0xB1: "\u0BC1", 0xB2: "\u0BC2",
    0xB4: "\u0BC6", 0xB5: "\u0BC7", 0xB6: "\u0BC8",
    0xB7: "\u0BCA", 0xB8: "\u0BCB", 0xB9: "\u0BCC",
    0xBA: "\u0BCD", 0xBB: "\u0BE6", 0xBC: "\u0BE7",
    0xBD: "\u0BE8", 0xBE: "\u0BE9", 0xBF: "\u0BEA",
    0xC0: "\u0BEB", 0xC1: "\u0BEC", 0xC2: "\u0BED",
    0xC3: "\u0BEE", 0xC4: "\u0BEF", 0x80: "\u0B83",
}

SL_TAMIL_MATH_NORMALIZATION = {
    "வகுதல்": "வகுத்தல்",
    "பெருக்கல்": "பெருக்கல்",
    "கூட்டல்": "கூட்டல்",
    "factor ஆனது": "காரணி ஆகும்",
    "HCF காண்க": "பொ.கா.பெ. காண்க",
    "LCM காண்க": "பொ.ம.சி. காண்க",
    "கூட்டுறவு காரணி": "பொதுக் காரணி",
}


def tscii_to_unicode(raw: str) -> str:
    out = []
    for ch in raw:
        cp = ord(ch)
        if cp > 0xFF:
            out.append(ch)
        elif cp in TSCII_TABLE:
            out.append(TSCII_TABLE[cp])
        else:
            out.append(ch)
    return "".join(out)


def normalize_tamil(text: str) -> str:
    """Apply SL Tamil → NIE standard normalization."""
    for variant, standard in SL_TAMIL_MATH_NORMALIZATION.items():
        text = text.replace(variant, standard)
    return text


NIE_SECTION_PATTERNS = {
    r"4\s*\.\s*1\b": ("4.1", "divisibility_rules", 1),
    r"4\s*\.\s*2\b": ("4.2", "factor_listing", 2),
    r"4\s*\.\s*3\b": ("4.3", "prime_factorization", 2),
    r"4\s*\.\s*4\b": ("4.4", "factors_via_prime", 2),
    r"4\s*\.\s*5\b": ("4.5", "hcf", 3),
    r"4\s*\.\s*6\b": ("4.6", "lcm", 3),
    r"5\s*\.\s*1\b": ("5.1", "fractions_basic", 1),
    r"பயிற்சி\s+\d": ("exercise", "exercise", 2),
    r"உதாரணம்\s*\d": ("example", "worked_example", 2),
    r"செயற்பாடு\s*\d": ("activity", "activity", 1),
    r"குறிப்பு|சுருக்கம்|பொழிப்பு": ("summary", "summary", 1),
    r"சிந்தனைக்கு": ("challenge", "word_problem", 3),
}

NIE_DIAGRAM_KEYWORDS = {
    "factor_tree": ["காரணி மரம்", "மரம்", "factor tree"],
    "division_ladder": ["வகுத்தல் ஏணி", "ஏணி", "வகுத்தல் முறை", "division"],
    "cell_diagram": ["கலத்தின்", "செல்", "cell diagram", "உயிரணு"],
    "circuit_diagram": ["சர்க்கீட்", "மின்சுற்று", "circuit"],
    "ray_diagram": ["கதிர் வரைபடம்", "ஒளிக்கதிர்", "ray diagram"],
    "number_line": ["எண்கோடு", "மடங்கு கோடு", "number line"],
    "factor_pairs": ["ஜோடி", "pair", "காரணி ஜோடி"],
}

NIE_TERM_GLOSSARY = {
    "காரணி": "factor",
    "மடங்கு": "multiple",
    "இலக்கச் சுட்டி": "digit_sum",
    "முதன்மை எண்": "prime_number",
    "முதன்மைக் காரணி": "prime_factor",
    "பொ.கா.பெ.": "hcf",
    "பொதுக் காரணிகளுட் பெரியது": "hcf",
    "பொ.ம.சி.": "lcm",
    "பொது மடங்குகளுட் சிறியது": "lcm",
    "வகுபடும்": "divisible",
    "மீதி": "remainder",
    "பொதுக் காரணி": "common_factor",
    "முழுவெண்": "whole_number",
    "உயர் வலு": "highest_power",
    "பெருக்கல்": "multiplication",
}


class PDFExtractor:
    """
    Extracts text from NIE Tamil PDFs.
    Handles three cases:
    1. Unicode Tamil PDF (modern NIE PDFs) — direct extract
    2. TSCII-encoded PDF — decode via TSCII_TABLE
    3. Image-based PDF — OCR via Tesseract with Tamil language pack
    """

    def __init__(self, ocr_lang: str = OCR_LANG):
        self.ocr_lang = ocr_lang
        self._fitz = None
        self._pytesseract = None

    def _get_fitz(self):
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                sys.exit("Install: pip install pymupdf")
        return self._fitz

    def _get_tess(self):
        if self._pytesseract is None:
            try:
                import pytesseract
                self._pytesseract = pytesseract
            except ImportError:
                sys.exit("Install: pip install pytesseract pillow")
        return self._pytesseract

    def detect_encoding(self, pdf_path: Path, sample_pages: int = 5) -> str:
        fitz = self._get_fitz()
        doc = fitz.open(pdf_path)
        unicode_tamil = tscii = image_only = 0
        for i in range(min(sample_pages, len(doc))):
            text = doc[i].get_text()
            if len(text.strip()) < 20:
                image_only += 1
            for ch in text:
                cp = ord(ch)
                if 0x0B80 <= cp <= 0x0BFF:
                    unicode_tamil += 1
                elif 0x80 <= cp <= 0xFF:
                    tscii += 1
        doc.close()
        if image_only >= sample_pages - 1:
            return "image"
        if unicode_tamil > tscii:
            return "unicode"
        if tscii > unicode_tamil:
            return "tscii"
        return "unknown"

    def extract_page_text(self, page, encoding: str) -> str:
        """Extract and decode text from a single PDF page."""
        raw = page.get_text()

        if not raw.strip() or encoding == "image":
            # Fall back to OCR
            return self._ocr_page(page)

        if encoding == "tscii":
            text = tscii_to_unicode(raw)
        else:
            text = raw  # Already Unicode

        return normalize_tamil(text)

    def _ocr_page(self, page) -> str:
        """OCR a page image using Tesseract with Tamil language pack."""
        try:
            import io
            from PIL import Image
            tess = self._get_tess()
            fitz = self._get_fitz()

            # Render at 300 DPI for accurate OCR
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = tess.image_to_string(img, lang=self.ocr_lang,
                                         config="--psm 6")
            return normalize_tamil(text)
        except Exception as e:
            log.warning(f"OCR failed: {e}")
            return ""

    def extract_pdf(self, pdf_path: Path,
                    start_page: int = 1, end_page: int = 9999) -> list[dict]:
        """
        Extract all pages in range. Returns list of:
        {page_num, text, encoding, pdf_page_index}
        """
        fitz = self._get_fitz()
        encoding = self.detect_encoding(pdf_path)
        log.info(f"Detected encoding: {encoding} for {pdf_path.name}")

        doc = fitz.open(pdf_path)
        results = []

        for pdf_idx in range(len(doc)):
            page = doc[pdf_idx]
            text = self.extract_page_text(page, encoding)

            # Detect NIE page number from text
            nie_page = self._detect_nie_page_number(text, pdf_idx)

            if nie_page and (start_page <= nie_page <= end_page):
                results.append({
                    "page_num": nie_page,
                    "pdf_page_index": pdf_idx,
                    "text": text,
                    "encoding": encoding,
                    "char_count": len(text),
                })
            elif not nie_page and pdf_idx >= start_page - 2:
                # Fallback: include by index estimate
                if pdf_idx <= end_page + 2:
                    results.append({
                        "page_num": pdf_idx + 1,
                        "pdf_page_index": pdf_idx,
                        "text": text,
                        "encoding": encoding,
                        "char_count": len(text),
                    })

        doc.close()
        log.info(f"Extracted {len(results)} pages from {pdf_path.name}")
        return results

    def _detect_nie_page_number(self, text: str, fallback_idx: int) -> Optional[int]:
        """Find NIE textbook page number embedded in text."""
        patterns = [
            r"(?:^|\n)\s*(\d{1,3})\s*(?:\n|$)",    # standalone line
            r"இலவசப் பாடநூல்\s*(\d+)",               # footer pattern
            r"(\d+)\s*இலவசப் பாடநூல்",               # alternate footer
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 300:
                    return n
        return None


class SemanticChunker:
    """
    Splits extracted text into semantically meaningful chunks.

    Strategy:
    1. Detect NIE structural boundaries (section headings, exercises, examples)
    2. Split at boundaries first (structural chunking)
    3. Sub-split large structural chunks by token count
    4. Apply overlap between consecutive sub-chunks
    5. Enrich each chunk with metadata (topic, difficulty, NIE terms, diagrams)
    """

    def __init__(self, grade: int, chapter: int, subject: str):
        self.grade = grade
        self.chapter = chapter
        self.subject = subject

    def chunk(self, pages: list[dict], source_file: str) -> list[tuple[str, ChunkMetadata]]:
        """Returns list of (text, metadata) pairs ready for embedding."""
        full_text = "\n".join(p["text"] for p in pages)
        page_map = {p["page_num"]: p for p in pages}

        structural_chunks = self._structural_split(full_text, pages)
        final_chunks = []

        for sc in structural_chunks:
            sub_chunks = self._token_split(sc["text"],
                                           CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
            for i, sub_text in enumerate(sub_chunks):
                if len(sub_text.strip()) < MIN_CHUNK_CHARS:
                    continue

                chunk_id = self._make_chunk_id(sub_text, sc, i)
                meta = self._enrich_metadata(sub_text, sc, chunk_id,
                                             source_file, i, len(sub_chunks))
                final_chunks.append((sub_text.strip(), meta))

        log.info(f"Chunked into {len(final_chunks)} semantic chunks")
        return final_chunks

    def _structural_split(self, text: str, pages: list[dict]) -> list[dict]:
        """Split text at NIE section boundaries."""
        chunks = []
        current = {
            "text": "", "section": f"{self.chapter}.0",
            "topic": "general", "difficulty": 1,
            "page_start": pages[0]["page_num"] if pages else 1,
            "page_end": pages[0]["page_num"] if pages else 1,
            "chunk_type": "concept"
        }

        type_map = {
            "exercise": "exercise", "worked_example": "worked_example",
            "activity": "concept", "summary": "summary",
            "word_problem": "exercise", "divisibility_rules": "rule",
            "factor_listing": "concept", "prime_factorization": "method",
            "factors_via_prime": "method", "hcf": "method", "lcm": "method",
        }

        lines = text.splitlines()
        for line in lines:
            matched = False
            for pattern, (sec, topic, diff) in NIE_SECTION_PATTERNS.items():
                if re.search(pattern, line):
                    if len(current["text"].strip()) > MIN_CHUNK_CHARS:
                        chunks.append(dict(current))
                    current = {
                        "text": line + "\n",
                        "section": sec if sec != "exercise" else current["section"],
                        "topic": topic,
                        "difficulty": diff,
                        "page_start": current["page_end"],
                        "page_end": current["page_end"],
                        "chunk_type": type_map.get(topic, "concept"),
                    }
                    matched = True
                    break
            if not matched:
                current["text"] += line + "\n"

        if len(current["text"].strip()) > MIN_CHUNK_CHARS:
            chunks.append(current)

        return chunks

    def _token_split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Approximate token split (4 chars ≈ 1 token for Tamil)."""
        max_chars = max_tokens * 4
        overlap_chars = overlap * 4

        if len(text) <= max_chars:
            return [text]

        # Split at sentence boundaries first
        sentences = re.split(r'(?<=[.!?।\n])\s+', text)
        chunks, current, current_len = [], [], 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > max_chars and current:
                chunks.append(" ".join(current))
                # Overlap: keep last N chars worth of sentences
                overlap_sents = []
                ol = 0
                for s in reversed(current):
                    if ol + len(s) > overlap_chars:
                        break
                    overlap_sents.insert(0, s)
                    ol += len(s)
                current = overlap_sents
                current_len = ol
            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _make_chunk_id(self, text: str, sc: dict, sub_idx: int) -> str:
        checksum = hashlib.sha256(text.encode()).hexdigest()[:8]
        return f"G{self.grade}_CH{self.chapter}_{sc['section']}_{sc['topic'][:8]}_{sub_idx}_{checksum}"

    def _enrich_metadata(self, text: str, sc: dict, chunk_id: str,
                          source_file: str, sub_idx: int,
                          total_subs: int) -> ChunkMetadata:
        """Extract rich metadata from chunk content."""
        nie_terms = [term for term in NIE_TERM_GLOSSARY
                     if term in text]
        diagrams = [dtype for dtype, keywords in NIE_DIAGRAM_KEYWORDS.items()
                    if any(kw in text for kw in keywords)]
        has_numbers = bool(re.search(r'\b\d+\b', text))
        checksum = hashlib.sha256(text.encode()).hexdigest()

        # Infer prerequisites from topic
        prereq_map = {
            "factor_listing": ["divisibility_rules"],
            "prime_factorization": ["factor_listing", "divisibility_rules"],
            "factors_via_prime": ["prime_factorization"],
            "hcf": ["prime_factorization", "factor_listing"],
            "lcm": ["prime_factorization", "factor_listing"],
            "word_problem": ["hcf", "lcm"],
        }

        return ChunkMetadata(
            chunk_id=chunk_id,
            grade=self.grade,
            chapter=self.chapter,
            subject=self.subject,
            section=sc["section"],
            topic=sc["topic"],
            chunk_type=sc["chunk_type"],
            difficulty=sc["difficulty"],
            page_start=sc.get("page_start", 0),
            page_end=sc.get("page_end", 0),
            prerequisites=prereq_map.get(sc["topic"], []),
            diagram_types=diagrams,
            nie_terms=nie_terms,
            has_numbers=has_numbers,
            is_answer_scheme=False,
            language="tamil",
            source_file=source_file,
            checksum=checksum,
        )


class IngestionPipeline:
    """
    Full pipeline: PDF → extract → chunk → embed → store.

    Run once per textbook. Subsequent queries hit ChromaDB.
    Re-run with --force to rebuild (dedup via checksum prevents duplicates).
    """

    def __init__(self):
        self.extractor = PDFExtractor()
        self.embedder = TamilEmbedder()
        self.store = NIEVectorStore(persist_path=CHROMA_PATH)

    def ingest_textbook(self, pdf_path: Path, grade: int, chapter: int,
                        subject: str, start_page: int = 1,
                        end_page: int = 9999) -> int:
        log.info(f"=== Ingesting {pdf_path.name} G{grade} Ch{chapter} ===")
        t0 = time.perf_counter()

        # Step 1: Extract
        pages = self.extractor.extract_pdf(pdf_path, start_page, end_page)
        if not pages:
            log.error("No pages extracted — check start/end page range")
            return 0

        # Step 2: Chunk
        chunker = SemanticChunker(grade, chapter, subject)
        chunks = chunker.chunk(pages, source_file=pdf_path.name)
        if not chunks:
            log.error("No chunks generated")
            return 0

        # Step 3: Embed
        texts = [text for text, _ in chunks]
        log.info(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
        # Add "passage: " prefix for BGE-M3 document embedding
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.embedder.embed_batch(prefixed)

        # Step 4: Store
        n = self.store.upsert_chunks(chunks, embeddings, grade, chapter, subject)

        elapsed = time.perf_counter() - t0
        log.info(f"=== Ingested {n} chunks in {elapsed:.1f}s ===")
        return n

    def ingest_answer_scheme(self, pdf_path: Path, grade: int,
                              chapter: int) -> int:
        """Ingest NIE marking scheme / answer scheme separately."""
        log.info(f"=== Ingesting answer scheme: {pdf_path.name} ===")

        pages = self.extractor.extract_pdf(pdf_path)
        if not pages:
            return 0

        # Answer scheme chunks get special metadata
        chunks = []
        for page in pages:
            text = page["text"].strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue
            chunk_id = f"ANS_G{grade}_CH{chapter}_P{page['page_num']}"
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                grade=grade, chapter=chapter, subject="mathematics",
                section=str(chapter), topic="answer_scheme",
                chunk_type="answer_scheme", difficulty=3,
                page_start=page["page_num"], page_end=page["page_num"],
                prerequisites=[], diagram_types=[], nie_terms=[],
                has_numbers=bool(re.search(r'\d', text)),
                is_answer_scheme=True, language="tamil",
                source_file=pdf_path.name,
                checksum=hashlib.sha256(text.encode()).hexdigest(),
            )
            chunks.append((text, meta))

        if not chunks:
            return 0

        texts = [f"passage: {t}" for t, _ in chunks]
        embeddings = self.embedder.embed_batch(texts)
        return self.store.upsert_chunks(
            chunks, embeddings, grade, chapter, "mathematics",
            collection_type="answers")


def main():
    parser = argparse.ArgumentParser(
        description="NIE Curriculum Ingestion Pipeline")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest textbook PDF")
    p_ingest.add_argument("--pdf", type=Path, required=True)
    p_ingest.add_argument("--grade", type=int, required=True)
    p_ingest.add_argument("--chapter", type=int, required=True)
    p_ingest.add_argument("--subject", default="mathematics")
    p_ingest.add_argument("--start-page", type=int, default=1)
    p_ingest.add_argument("--end-page", type=int, default=9999)

    p_ans = sub.add_parser("ingest-answers", help="Ingest marking scheme PDF")
    p_ans.add_argument("--pdf", type=Path, required=True)
    p_ans.add_argument("--grade", type=int, required=True)
    p_ans.add_argument("--chapter", type=int, required=True)

    p_ins = sub.add_parser("inspect", help="Inspect stored collection")
    p_ins.add_argument("--grade", type=int, required=True)
    p_ins.add_argument("--chapter", type=int, required=True)
    p_ins.add_argument("--subject", default="mathematics")

    p_clr = sub.add_parser("clear", help="Delete and rebuild collection")
    p_clr.add_argument("--grade", type=int, required=True)
    p_clr.add_argument("--chapter", type=int, required=True)
    p_clr.add_argument("--subject", default="mathematics")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    pipeline = IngestionPipeline()

    if args.command == "ingest":
        n = pipeline.ingest_textbook(
            args.pdf, args.grade, args.chapter, args.subject,
            args.start_page, args.end_page)
        print(f"\nIngestion complete: {n} chunks stored.")
        print("Next: run agent_orchestrator.py to start the tutor.")

    elif args.command == "ingest-answers":
        n = pipeline.ingest_answer_scheme(args.pdf, args.grade, args.chapter)
        print(f"Answer scheme ingested: {n} chunks.")

    elif args.command == "inspect":
        pipeline.store.inspect(args.grade, args.chapter, args.subject)

    elif args.command == "clear":
        pipeline.store.delete_collection(args.grade, args.chapter, args.subject)
        print("Collection cleared. Re-run ingest to rebuild.")


if __name__ == "__main__":
    main()
