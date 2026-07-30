"""Focused tests for final prompt formatting."""

import pytest

from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk
from enterprise_multi_agent_rag.generation import (
    InvalidPromptInputError,
    PromptBuilder,
)
from enterprise_multi_agent_rag.retrieval.models import SearchResult


def _result(rank: int, content: str, embedding: list[float]) -> SearchResult:
    chunk = EmbeddedChunk(
        chunk_id=f"chunk-{rank}",
        document_id="generated-document",
        chunk_index=rank - 1,
        content=content,
        embedding=embedding,
        metadata={"generated": True},
    )
    return SearchResult(chunk=chunk, score=1.0 / rank, rank=rank)


def test_builds_valid_prompt_with_question_and_chunks() -> None:
    question = "How many vacation days do employees receive?"
    results = [
        _result(1, "Employees receive 15 vacation days per year.", [1.0, 0.0]),
        _result(2, "Vacation requests require manager approval.", [0.8, 0.2]),
    ]

    prompt = PromptBuilder().build(question, results)

    assert prompt.startswith("You are an enterprise knowledge assistant.")
    assert "Answer the question using only the provided context." in prompt
    assert f"Question:\n{question}" in prompt
    assert "[Context 1]\nEmployees receive 15 vacation days per year." in prompt
    assert "[Context 2]\nVacation requests require manager approval." in prompt
    assert prompt.endswith("Answer:")


def test_preserves_search_result_order() -> None:
    prompt = PromptBuilder().build(
        "What policies apply?",
        [
            _result(2, "Second-ranked content.", [0.2]),
            _result(1, "First-ranked content supplied second.", [0.9]),
        ],
    )

    assert prompt.index("Second-ranked content.") < prompt.index(
        "First-ranked content supplied second."
    )
    assert "[Context 1]\nSecond-ranked content." in prompt
    assert "[Context 2]\nFirst-ranked content supplied second." in prompt


def test_builds_valid_prompt_without_search_results() -> None:
    prompt = PromptBuilder().build("What is the policy?", [])

    assert "Context:\nNo relevant context was found." in prompt
    assert "Question:\nWhat is the policy?" in prompt
    assert prompt.endswith("Answer:")


@pytest.mark.parametrize("question", ["", " \n\t "])
def test_rejects_empty_or_whitespace_question(question: str) -> None:
    with pytest.raises(InvalidPromptInputError, match="must not be empty"):
        PromptBuilder().build(question, [])


def test_does_not_include_embeddings_or_object_representations() -> None:
    result = _result(1, "Only the chunk text belongs here.", [98765.4321, -12345.0])

    prompt = PromptBuilder().build("What belongs in the prompt?", [result])

    assert "98765.4321" not in prompt
    assert "-12345.0" not in prompt
    assert "EmbeddedChunk(" not in prompt
    assert "SearchResult(" not in prompt
    assert "Only the chunk text belongs here." in prompt
