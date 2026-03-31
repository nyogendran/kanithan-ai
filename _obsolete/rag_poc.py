#!/usr/bin/env python3
"""
Week 1 PoC: chunk an NIE Tamil-medium PDF, index with Chroma + Ollama embeddings,
query with Ollama LLM in Tamil.

Prerequisites:
  1. ollama serve running locally.
  2. Pull LLM:   ollama pull llama3        # or gemma3 when your registry has it
  3. Pull embed: ollama pull nomic-embed-text

NIE Grade 7 Mathematics (Tamil): download PDF from https://nie.lk/ (e-thaksalawa /
publications) and pass --pdf path. Do not commit copyrighted PDFs to the repo.

Example:
  python week1_rag_poc.py --pdf ./data/nie_g7_math_tamil.pdf \\
    --question "காரணி என்றால் என்ன?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb


def load_pdf_documents(pdf_path: Path, backend: str) -> list[Document]:
    """Load PDF text. pymupdf usually preserves Tamil Unicode better than pypdf."""
    if backend == "pymupdf":
        import fitz

        doc = fitz.open(pdf_path)
        try:
            out: list[Document] = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    out.append(
                        Document(
                            text=text,
                            metadata={"page_label": str(page.number + 1)},
                        )
                    )
            return out
        finally:
            doc.close()
    if backend == "pypdf":
        reader = PDFReader()
        return reader.load_data(file=pdf_path)
    raise ValueError(f"Unknown --pdf-backend: {backend}")


def build_index(
    pdf_path: Path,
    persist_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
    reset: bool,
    pdf_backend: str,
) -> VectorStoreIndex:
    documents = load_pdf_documents(pdf_path, pdf_backend)

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
    )

    if reset and persist_dir.exists():
        import shutil

        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NIE PDF RAG PoC with Ollama")
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to NIE Tamil mathematics PDF",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="காரணி என்றால் என்ன?",
        help="Tamil query to the tutor",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target tokens per chunk (LlamaIndex tokenizer approximation)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Token overlap between chunks",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("OLLAMA_LLM", "llama3:latest"),
        help="Ollama model name for generation",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("OLLAMA_EMBED", "nomic-embed-text"),
        help="Ollama embedding model (pull with: ollama pull <name>)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("./chroma_nie_week1"),
        help="Chroma persistence directory",
    )
    parser.add_argument(
        "--collection",
        default="nie_g7_math_tamil",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete persisted Chroma data and rebuild from PDF",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks",
    )
    parser.add_argument(
        "--pdf-backend",
        choices=("pymupdf", "pypdf"),
        default="pymupdf",
        help="PDF text extractor; pymupdf is better for Tamil. Re-index with --reset-index if you change this.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=600.0,
        help="Ollama HTTP timeout in seconds (generation can be slow on CPU)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    Settings.llm = Ollama(
        model=args.llm_model,
        request_timeout=args.llm_timeout,
        temperature=0.2,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=args.embed_model,
        request_timeout=120.0,
    )

    chroma_path = args.persist_dir
    has_existing = chroma_path.exists() and any(chroma_path.iterdir())

    if args.reset_index or not has_existing:
        index = build_index(
            pdf_path=args.pdf,
            persist_dir=chroma_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            collection_name=args.collection,
            reset=args.reset_index or not has_existing,
            pdf_backend=args.pdf_backend,
        )
    else:
        chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        chroma_collection = chroma_client.get_collection(args.collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)

    tamil_qa_tmpl = (
        "நீங்கள் இலங்கை NIE பாடப்புத்தக வழிகாட்டுதலில் பயிற்றுவிக்கும் கணித ஆசிரியர்.\n"
        "கீழே உள்ள பாடப்புத்தகப் பகுதிகளை மட்டுமே அடிப்படையாகக் கொண்டு பதிலளிக்கவும்; "
        "அவற்றில் இல்லாததைக் கற்பனை செய்ய வேண்டாம்.\n"
        "முழு பதிலும் தமிழில், மாணவர் புரிந்துகொள்ளும் எளிய மொழியில் இருக்க வேண்டும்; "
        "கணிதச் சொற்கள் NIE பாடப்புத்தகப் பயன்பாட்டிற்கேற்ப இருக்க வேண்டும்.\n\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: {query_str}\n"
        "Answer (Tamil): "
    )
    query_engine = index.as_query_engine(
        similarity_top_k=args.top_k,
        text_qa_template=PromptTemplate(tamil_qa_tmpl),
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
    )

    print("--- Question ---")
    print(args.question)
    print("--- Answer ---")
    response = query_engine.query(args.question)
    print(str(response))


if __name__ == "__main__":
    main()
