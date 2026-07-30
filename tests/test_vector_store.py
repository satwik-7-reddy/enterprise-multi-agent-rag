"""Deterministic tests for FAISS vector storage and search."""

from pathlib import Path

import pytest

from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk
from enterprise_multi_agent_rag.retrieval import FAISSVectorStore


def _chunk(chunk_id: str, embedding: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id="generated-document",
        chunk_index=0,
        content=f"Generated {chunk_id} content",
        embedding=embedding,
        metadata={"topic": chunk_id},
    )


@pytest.fixture
def topic_chunks() -> list[EmbeddedChunk]:
    """Return generated chunks with simple orthogonal vectors."""
    return [
        _chunk("vacation", [1.0, 0.0, 0.0]),
        _chunk("insurance", [0.0, 1.0, 0.0]),
        _chunk("remote-work", [0.0, 0.0, 1.0]),
    ]


def test_add_chunks_and_initialize_dimension(topic_chunks: list[EmbeddedChunk]) -> None:
    store = FAISSVectorStore()
    store.add_chunks(topic_chunks)

    assert store.size == 3
    assert store.dimension == 3


def test_empty_add_is_no_op() -> None:
    store = FAISSVectorStore()
    store.add_chunks([])

    assert store.size == 0
    assert store.dimension is None


def test_rejects_dimension_mismatch_atomically() -> None:
    store = FAISSVectorStore()
    store.add_chunks([_chunk("first", [1.0, 0.0, 0.0])])

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add_chunks(
            [
                _chunk("valid", [0.0, 1.0, 0.0]),
                _chunk("invalid", [1.0, 0.0]),
            ]
        )

    assert store.size == 1


def test_rejects_duplicate_chunk_id() -> None:
    store = FAISSVectorStore()
    store.add_chunks([_chunk("duplicate", [1.0, 0.0])])

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        store.add_chunks([_chunk("duplicate", [0.0, 1.0])])

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        FAISSVectorStore().add_chunks(
            [
                _chunk("same-batch", [1.0, 0.0]),
                _chunk("same-batch", [0.0, 1.0]),
            ]
        )


def test_search_returns_correct_ranking(topic_chunks: list[EmbeddedChunk]) -> None:
    store = FAISSVectorStore()
    store.add_chunks(topic_chunks)

    results = store.search([0.9, 0.1, 0.0])

    assert [result.chunk.chunk_id for result in results] == [
        "vacation",
        "insurance",
        "remote-work",
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.1)


def test_search_respects_top_k(topic_chunks: list[EmbeddedChunk]) -> None:
    store = FAISSVectorStore()
    store.add_chunks(topic_chunks)

    results = store.search([0.9, 0.1, 0.0], k=2)

    assert len(results) == 2
    assert [result.chunk.chunk_id for result in results] == ["vacation", "insurance"]


def test_search_empty_store_returns_empty_list() -> None:
    assert FAISSVectorStore().search([1.0, 0.0]) == []


def test_rejects_query_dimension_mismatch(topic_chunks: list[EmbeddedChunk]) -> None:
    store = FAISSVectorStore()
    store.add_chunks(topic_chunks)

    with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
        store.search([1.0, 0.0])


def test_rejects_non_positive_k(topic_chunks: list[EmbeddedChunk]) -> None:
    store = FAISSVectorStore()
    store.add_chunks(topic_chunks)

    with pytest.raises(ValueError, match="greater than zero"):
        store.search([1.0, 0.0, 0.0], k=0)


def test_save_and_load_preserve_search_results(
    topic_chunks: list[EmbeddedChunk], tmp_path: Path
) -> None:
    store = FAISSVectorStore()
    store.add_chunks(topic_chunks)
    directory = tmp_path / "vector_store"
    expected = store.search([0.9, 0.1, 0.0])

    store.save(directory)
    loaded = FAISSVectorStore.load(directory)
    actual = loaded.search([0.9, 0.1, 0.0])

    assert (directory / "index.faiss").is_file()
    assert (directory / "chunks.json").is_file()
    assert loaded.size == store.size
    assert loaded.dimension == store.dimension
    assert actual == expected


def test_save_and_load_empty_store(tmp_path: Path) -> None:
    directory = tmp_path / "empty_store"

    FAISSVectorStore().save(directory)
    loaded = FAISSVectorStore.load(directory)

    assert loaded.size == 0
    assert loaded.dimension is None
    assert loaded.search([1.0]) == []
