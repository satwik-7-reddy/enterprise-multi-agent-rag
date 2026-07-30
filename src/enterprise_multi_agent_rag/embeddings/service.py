"""Provider-independent orchestration for embedding document chunks."""

from copy import deepcopy
from numbers import Real

from enterprise_multi_agent_rag.chunking.models import DocumentChunk
from enterprise_multi_agent_rag.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    EmptyEmbeddingInputError,
)
from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk
from enterprise_multi_agent_rag.embeddings.providers import BaseEmbeddingProvider


class EmbeddingService:
    """Generate and validate embeddings while preserving chunk order."""

    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        self.provider = provider

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        """Embed chunks; an empty collection is a valid no-op returning an empty list."""
        if not chunks:
            return []
        for chunk in chunks:
            if not chunk.content or not chunk.content.strip():
                raise EmptyEmbeddingInputError(
                    f"Chunk '{chunk.chunk_id}' contains no embeddable text."
                )

        vectors = self.provider.embed_texts([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingProviderError(
                f"Provider returned {len(vectors)} vectors for {len(chunks)} chunks."
            )

        expected_dimensions: int | None = None
        for index, vector in enumerate(vectors):
            if not vector:
                raise EmptyEmbeddingInputError(
                    f"Provider returned an empty vector for chunk index {index}."
                )
            if any(
                isinstance(value, bool) or not isinstance(value, Real) for value in vector
            ):
                raise EmbeddingProviderError(
                    f"Provider returned a non-numeric vector value for chunk index {index}."
                )
            if expected_dimensions is None:
                expected_dimensions = len(vector)
            elif len(vector) != expected_dimensions:
                raise EmbeddingDimensionMismatchError(
                    "Provider returned inconsistent embedding dimensions: "
                    f"expected {expected_dimensions}, received {len(vector)} at index {index}."
                )

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=[float(value) for value in vector],
                metadata=deepcopy(chunk.metadata),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
