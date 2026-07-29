"""Unit tests for document chunking."""

from typing import Any

import pytest

from enterprise_multi_agent_rag.chunking import DocumentChunker
from enterprise_multi_agent_rag.chunking.exceptions import (
    EmptyDocumentContentError,
    InvalidChunkingConfigurationError,
)
from enterprise_multi_agent_rag.ingestion.models import IngestedDocument


def _document(content: str, metadata: dict[str, Any] | None = None) -> IngestedDocument:
    return IngestedDocument(
        document_id="doc-123",
        filename="generated.txt",
        source_path="/generated/generated.txt",
        file_type="txt",
        content=content,
        metadata=metadata or {"department": "testing"},
    )


def test_chunks_short_document_without_rewriting_content() -> None:
    document = _document("  A short document.\n")

    chunks = DocumentChunker(chunk_size=100, chunk_overlap=10).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content
    assert chunks[0].start_character == 0
    assert chunks[0].end_character == len(document.content)


def test_chunks_long_document_into_multiple_chunks() -> None:
    document = _document("abcdefghij" * 8)

    chunks = DocumentChunker(chunk_size=20, chunk_overlap=5).chunk(document)

    assert len(chunks) > 1
    assert all(0 < len(chunk.content) <= 20 for chunk in chunks)


def test_chunk_overlap_is_preserved() -> None:
    document = _document("abcdefghijklmnopqrstuvwxyz")

    chunks = DocumentChunker(chunk_size=10, chunk_overlap=3).chunk(document)

    assert len(chunks) == 4
    assert chunks[0].content[-3:] == chunks[1].content[:3]
    assert chunks[1].content[-3:] == chunks[2].content[:3]
    assert chunks[1].start_character == chunks[0].end_character - 3


def test_chunk_ids_are_deterministic() -> None:
    document = _document("A deterministic document. " * 10)
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    first_ids = [chunk.chunk_id for chunk in chunker.chunk(document)]
    second_ids = [chunk.chunk_id for chunk in chunker.chunk(document)]

    assert first_ids == second_ids
    assert all(len(chunk_id) == 64 for chunk_id in first_ids)


def test_chunks_inherit_and_extend_metadata() -> None:
    document = _document("metadata content", {"classification": "generated"})

    chunk = DocumentChunker().chunk(document)[0]

    assert chunk.metadata == {
        "classification": "generated",
        "document_id": "doc-123",
        "chunk_index": 0,
        "total_chunks": 1,
        "filename": "generated.txt",
        "file_type": "txt",
        "source_path": "/generated/generated.txt",
    }


def test_indexes_are_sequential_and_total_count_is_correct() -> None:
    chunks = DocumentChunker(chunk_size=12, chunk_overlap=2).chunk(
        _document("0123456789" * 5)
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.metadata["total_chunks"] for chunk in chunks} == {len(chunks)}


@pytest.mark.parametrize("content", ["", " \n\t "])
def test_rejects_empty_or_whitespace_only_content(content: str) -> None:
    with pytest.raises(EmptyDocumentContentError, match="no chunkable content"):
        DocumentChunker().chunk(_document(content))


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(InvalidChunkingConfigurationError, match="greater than zero"):
        DocumentChunker(chunk_size=0)


def test_rejects_negative_overlap() -> None:
    with pytest.raises(InvalidChunkingConfigurationError, match="zero or greater"):
        DocumentChunker(chunk_overlap=-1)


@pytest.mark.parametrize("overlap", [10, 11])
def test_rejects_overlap_equal_to_or_larger_than_chunk_size(overlap: int) -> None:
    with pytest.raises(InvalidChunkingConfigurationError, match="smaller than"):
        DocumentChunker(chunk_size=10, chunk_overlap=overlap)


def test_content_smaller_than_chunk_size_has_one_chunk() -> None:
    chunks = DocumentChunker(chunk_size=50, chunk_overlap=5).chunk(_document("small"))

    assert len(chunks) == 1
    assert chunks[0].content == "small"


def test_repeated_text_offsets_are_monotonic_and_exact() -> None:
    document = _document("abcabcabcabcabcabc")

    chunks = DocumentChunker(chunk_size=6, chunk_overlap=3).chunk(document)

    starts = [chunk.start_character for chunk in chunks]
    assert starts == sorted(starts)
    for chunk in chunks:
        assert chunk.start_character is not None
        assert chunk.end_character is not None
        assert (
            document.content[chunk.start_character : chunk.end_character] == chunk.content
        )


def test_unicode_content_and_offsets() -> None:
    document = _document("你好世界🌍 café — résumé " * 4)

    chunks = DocumentChunker(chunk_size=18, chunk_overlap=4).chunk(document)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.start_character is not None
        assert chunk.end_character is not None
        assert (
            document.content[chunk.start_character : chunk.end_character] == chunk.content
        )
