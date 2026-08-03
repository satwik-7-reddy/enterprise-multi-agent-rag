"""Tests for the document indexing workflow."""

from pathlib import Path
from unittest.mock import patch

import pytest

from enterprise_multi_agent_rag.chunking import DocumentChunker
from enterprise_multi_agent_rag.embeddings import EmbeddingService
from enterprise_multi_agent_rag.embeddings.providers import BaseEmbeddingProvider
from enterprise_multi_agent_rag.ingestion import DocumentLoader
from enterprise_multi_agent_rag.ingestion.exceptions import (
    DocumentNotFoundError,
    UnsupportedDocumentTypeError,
)
from enterprise_multi_agent_rag.ingestion.indexing_service import DocumentIndexingService
from enterprise_multi_agent_rag.retrieval import FAISSVectorStore


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Return deterministic local embeddings without external calls."""

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]


@pytest.fixture
def vector_store() -> FAISSVectorStore:
    return FAISSVectorStore()


@pytest.fixture
def indexing_service(vector_store: FAISSVectorStore) -> DocumentIndexingService:
    return DocumentIndexingService(
        document_loader=DocumentLoader(),
        document_chunker=DocumentChunker(chunk_size=20, chunk_overlap=5),
        embedding_service=EmbeddingService(FakeEmbeddingProvider()),
        vector_store=vector_store,
    )


def test_indexes_one_document_and_returns_statistics(
    indexing_service: DocumentIndexingService,
    vector_store: FAISSVectorStore,
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "handbook.txt"
    document_path.write_text("Vacation policy. " * 8, encoding="utf-8")
    output_directory = tmp_path / "vector_store"

    result = indexing_service.index_document(str(document_path), str(output_directory))

    assert result.document_id == DocumentLoader().load(document_path).document_id
    assert result.chunks_created > 1
    assert result.embeddings_stored == result.chunks_created
    assert vector_store.size == result.embeddings_stored
    assert (output_directory / "index.faiss").is_file()
    assert (output_directory / "chunks.json").is_file()


def test_indexes_multiple_documents_into_one_store_and_saves_once(
    indexing_service: DocumentIndexingService,
    vector_store: FAISSVectorStore,
    tmp_path: Path,
) -> None:
    first = tmp_path / "handbook.txt"
    second = tmp_path / "policy.md"
    first.write_text("Vacation policy. " * 5, encoding="utf-8")
    second.write_text("Health-policy-details." * 5, encoding="utf-8")
    output_directory = tmp_path / "vector_store"

    with patch.object(vector_store, "save", wraps=vector_store.save) as save:
        results = indexing_service.index_documents(
            [str(first), str(second)], str(output_directory)
        )

    assert len(results) == 2
    assert all(result.chunks_created > 0 for result in results)
    assert all(
        result.embeddings_stored == result.chunks_created for result in results
    )
    assert vector_store.size == sum(result.embeddings_stored for result in results)
    save.assert_called_once_with(str(output_directory))


def test_rejects_empty_document_list(
    indexing_service: DocumentIndexingService,
) -> None:
    with pytest.raises(ValueError, match="at least one document"):
        indexing_service.index_documents([])


def test_reuses_loader_validation_for_missing_document(
    indexing_service: DocumentIndexingService,
    tmp_path: Path,
) -> None:
    with pytest.raises(DocumentNotFoundError, match="Document not found"):
        indexing_service.index_document(str(tmp_path / "missing.txt"))


def test_reuses_loader_validation_for_unsupported_extension(
    indexing_service: DocumentIndexingService,
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "records.csv"
    document_path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="\\.csv"):
        indexing_service.index_document(str(document_path))
