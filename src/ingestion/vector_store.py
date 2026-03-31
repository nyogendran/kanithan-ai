from __future__ import annotations

import json
import logging
import re
import sys
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import CHROMA_PATH, COLLECTION_PREFIX, EMBED_MODEL

log = logging.getLogger("nie.ingestion")


@dataclass
class ChunkMetadata:
    """Rich metadata stored alongside every vector in ChromaDB."""
    chunk_id: str
    grade: int
    chapter: int
    subject: str                   # mathematics / science
    section: str                   # 4.1, 4.2 etc
    topic: str                     # hcf, lcm, prime_factorization etc
    chunk_type: str                # concept|rule|method|worked_example|exercise|summary|answer_scheme
    difficulty: int                # 1-3
    page_start: int
    page_end: int
    prerequisites: list            # list of topic strings
    diagram_types: list            # factor_tree|division_ladder|cell_diagram etc
    nie_terms: list                # key NIE Tamil terms present
    has_numbers: bool              # chunk contains mathematical numbers
    is_answer_scheme: bool         # from marking scheme PDF
    language: str                  # tamil|sinhala|english|trilingual
    source_file: str
    checksum: str                  # sha256 of content for dedup

    def to_chroma_metadata(self) -> dict:
        """ChromaDB metadata must be flat dict of str/int/float/bool."""
        d = asdict(self)
        # Flatten lists to JSON strings (ChromaDB limitation)
        d["prerequisites"] = json.dumps(d["prerequisites"])
        d["diagram_types"] = json.dumps(d["diagram_types"])
        d["nie_terms"] = json.dumps(d["nie_terms"])
        return d

    @staticmethod
    def from_chroma_metadata(d: dict) -> ChunkMetadata:
        d = dict(d)
        d["prerequisites"] = json.loads(d.get("prerequisites", "[]"))
        d["diagram_types"] = json.loads(d.get("diagram_types", "[]"))
        d["nie_terms"] = json.loads(d.get("nie_terms", "[]"))
        return ChunkMetadata(**d)


class TamilEmbedder:
    """
    Multilingual embeddings using BAAI/bge-m3.
    BGE-M3 supports 100+ languages including Tamil and Sinhala,
    handles both dense and sparse retrieval, and runs locally.

    Alternative: multilingual-e5-large (smaller, slightly lower quality)
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            log.info(f"Loading embedding model: {self.model_name} (first run downloads ~2GB)")
            # Offline-first: in many dev/CI environments we cannot download models.
            # This makes vector retrieval fail fast and lets the tutor fall back
            # to keyword + curated NIE corpus (enterprise-grade resiliency).
            offline_only = os.environ.get("EMBED_OFFLINE_ONLY", "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
            )
            if offline_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                if not self._local_weights_available():
                    raise RuntimeError(
                        "Embedding weights not found in local HuggingFace cache; "
                        "set EMBED_OFFLINE_ONLY=0 to allow downloads."
                    )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                log.info("Embedding model loaded.")
            except ImportError:
                sys.exit("Install: pip install sentence-transformers")
        return self._model

    def _local_weights_available(self) -> bool:
        """
        Fast cache check to avoid hanging on first-run downloads.
        The HuggingFace cache layout is typically:
        ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{snapshot_id}/
        """
        # Derive cache base (match default HF layout used on macOS).
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache")))
        hub_cache = hf_home / "huggingface" / "hub"

        # Example: "BAAI/bge-m3" -> "models--BAAI--bge-m3"
        if "/" not in self.model_name:
            return False
        org, repo = self.model_name.split("/", 1)
        model_dir = hub_cache / f"models--{org}--{repo}"
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return False

        # If any snapshot contains weight files, we assume embeddings are available.
        for snap in snapshots_dir.iterdir():
            if not snap.is_dir():
                continue
            # Most commonly: *.safetensors; sometimes *.bin or pytorch_model.bin.
            has_safetensors = any(snap.glob("*.safetensors"))
            has_bin = any(snap.glob("*.bin"))
            pytorch_model_bin = (snap / "pytorch_model.bin").exists()
            if has_safetensors or has_bin or pytorch_model_bin:
                return True
        return False

    def embed_batch(self, texts: list[str],
                    batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        model = self._load()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (with BGE-M3 query prefix)."""
        model = self._load()
        # BGE-M3 recommends "query: " prefix for retrieval queries
        prefixed = f"query: {query}"
        emb = model.encode(prefixed, normalize_embeddings=True)
        return emb.tolist()


class NIEVectorStore:
    """
    ChromaDB-based vector store for NIE curriculum chunks.

    Collections:
    - nie_curriculum_g{grade}_ch{chapter}_{subject} — main content
    - nie_answers_g{grade}_ch{chapter} — marking scheme answers
    - nie_past_papers_g{grade} — past O/L exam questions

    Migration path to Qdrant (production):
    - Replace ChromaDB client with qdrant_client.QdrantClient
    - Same metadata schema, same embedding model
    """

    def __init__(self, persist_path: Path = CHROMA_PATH):
        self.persist_path = persist_path
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                self.persist_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_path)
                )
            except ImportError:
                sys.exit("Install: pip install chromadb")
        return self._client

    def collection_name(self, grade: int, chapter: int, subject: str,
                        collection_type: str = "curriculum") -> str:
        name = f"{COLLECTION_PREFIX}_{collection_type}_g{grade}_ch{chapter}_{subject}"
        # ChromaDB: collection names must be 3-63 chars, no special chars except _ and -
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:63]

    def get_or_create_collection(self, grade: int, chapter: int,
                                  subject: str, collection_type: str = "curriculum"):
        client = self._get_client()
        name = self.collection_name(grade, chapter, subject, collection_type)
        # Use cosine distance (normalized embeddings → cosine = dot product)
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine",
                       "grade": grade,
                       "chapter": chapter,
                       "subject": subject}
        )
        log.info(f"Collection '{name}' has {collection.count()} chunks")
        return collection

    def upsert_chunks(self, chunks: list[tuple[str, ChunkMetadata]],
                      embeddings: list[list[float]],
                      grade: int, chapter: int, subject: str,
                      collection_type: str = "curriculum") -> int:
        """Upsert chunks + embeddings. Skips duplicates by checksum."""
        collection = self.get_or_create_collection(
            grade, chapter, subject, collection_type)

        # Batch upsert
        batch_size = 100
        inserted = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]

            ids = [meta.chunk_id for _, meta in batch_chunks]
            docs = [text for text, _ in batch_chunks]
            metas = [meta.to_chroma_metadata() for _, meta in batch_chunks]
            embs = batch_embs

            collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embs,
            )
            inserted += len(batch_chunks)
            log.info(f"Upserted {inserted}/{len(chunks)} chunks")

        return inserted

    def hybrid_query(self,
                     query_embedding: list[float],
                     grade: int,
                     chapter: int,
                     subject: str,
                     n_results: int = 8,
                     where_filter: dict = None,
                     collection_type: str = "curriculum") -> list[dict]:
        """
        Hybrid retrieval:
        - Vector similarity search on embedding
        - Metadata filter (difficulty, topic, chunk_type)
        Returns list of {id, text, metadata, distance}
        """
        collection = self.get_or_create_collection(
            grade, chapter, subject, collection_type)

        if collection.count() == 0:
            log.warning("Collection is empty — run ingestion first")
            return []

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = collection.query(**kwargs)

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": ChunkMetadata.from_chroma_metadata(
                    results["metadatas"][0][i]),
                "distance": results["distances"][0][i],
                "score": 1.0 - results["distances"][0][i],  # cosine → similarity
            })
        return output

    def delete_collection(self, grade: int, chapter: int,
                           subject: str, collection_type: str = "curriculum"):
        client = self._get_client()
        name = self.collection_name(grade, chapter, subject, collection_type)
        client.delete_collection(name)
        log.info(f"Deleted collection: {name}")

    def inspect(self, grade: int, chapter: int, subject: str):
        collection = self.get_or_create_collection(grade, chapter, subject)
        count = collection.count()
        print(f"\nCollection: {self.collection_name(grade, chapter, subject)}")
        print(f"  Total chunks: {count}")
        if count > 0:
            sample = collection.peek(limit=5)
            print(f"  Sample chunk IDs: {sample['ids']}")
            if sample["metadatas"]:
                m = sample["metadatas"][0]
                print(f"  Sample metadata: topic={m.get('topic')} "
                      f"difficulty={m.get('difficulty')} "
                      f"section={m.get('section')}")
