"""Simple in-memory FAISS storage with JSON-backed chunk persistence."""

import json
from pathlib import Path

import faiss
import numpy as np
from pydantic import TypeAdapter, ValidationError

from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk
from enterprise_multi_agent_rag.retrieval.models import SearchResult


class FAISSVectorStore:
    """Store embedded chunks in an exact inner-product FAISS index."""

    INDEX_FILENAME = "index.faiss"
    CHUNKS_FILENAME = "chunks.json"

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[EmbeddedChunk] = []
        self._chunk_ids: set[str] = set()
        self._dimension: int | None = None

    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        return len(self._chunks)

    @property
    def dimension(self) -> int | None:
        """Return the established vector dimension, if any."""
        return self._dimension

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Validate and add embedded chunks without partially changing the store."""
        if not chunks:
            return

        expected_dimension = self._dimension or len(chunks[0].embedding)
        if expected_dimension <= 0:
            raise ValueError("Chunk embeddings must be non-empty.")

        batch_ids: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in self._chunk_ids or chunk.chunk_id in batch_ids:
                raise ValueError(f"Duplicate chunk ID: '{chunk.chunk_id}'.")
            if len(chunk.embedding) != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch for chunk '{chunk.chunk_id}': "
                    f"expected {expected_dimension}, received {len(chunk.embedding)}."
                )
            batch_ids.add(chunk.chunk_id)

        vectors = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Chunk embeddings must contain only finite numeric values.")

        if self._index is None:
            self._index = faiss.IndexFlatIP(expected_dimension)
            self._dimension = expected_dimension
        self._index.add(vectors)
        self._chunks.extend(chunks)
        self._chunk_ids.update(batch_ids)

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        """Return up to ``k`` chunks in descending inner-product order."""
        if k <= 0:
            raise ValueError("k must be greater than zero.")
        if self._index is None:
            return []
        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self._dimension}, "
                f"received {len(query_embedding)}."
            )

        query = np.asarray([query_embedding], dtype=np.float32)
        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite numeric values.")
        result_count = min(k, self.size)
        scores, row_ids = self._index.search(query, result_count)
        return [
            SearchResult(
                chunk=self._chunks[int(row_id)],
                score=float(score),
                rank=rank,
            )
            for rank, (score, row_id) in enumerate(
                zip(scores[0], row_ids[0], strict=True), start=1
            )
        ]

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and its position-to-chunk mapping."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        index = self._index or faiss.IndexFlatIP(0)
        faiss.write_index(index, str(target / self.INDEX_FILENAME))
        serialized = [chunk.model_dump(mode="json") for chunk in self._chunks]
        (target / self.CHUNKS_FILENAME).write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """Load persisted vectors and chunks without regenerating embeddings."""
        source = Path(directory)
        index_path = source / cls.INDEX_FILENAME
        chunks_path = source / cls.CHUNKS_FILENAME
        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                f"Vector store requires '{cls.INDEX_FILENAME}' and "
                f"'{cls.CHUNKS_FILENAME}' in '{source}'."
            )

        index = faiss.read_index(str(index_path))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Persisted FAISS index is not an IndexFlatIP.")
        try:
            raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = TypeAdapter(list[EmbeddedChunk]).validate_python(raw_chunks)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Could not load persisted chunks: {exc}") from exc

        if index.ntotal != len(chunks):
            raise ValueError(
                "Persisted FAISS row count does not match the chunk mapping: "
                f"{index.ntotal} rows and {len(chunks)} chunks."
            )
        if chunks and any(len(chunk.embedding) != index.d for chunk in chunks):
            raise ValueError("Persisted chunk dimensions do not match the FAISS index.")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Persisted chunk mapping contains duplicate chunk IDs.")

        store = cls()
        store._index = index if index.d > 0 else None
        store._dimension = index.d if index.d > 0 else None
        store._chunks = chunks
        store._chunk_ids = set(chunk_ids)
        return store
