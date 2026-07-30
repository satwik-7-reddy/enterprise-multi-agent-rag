"""Tests for end-to-end RAG component coordination."""

import pytest

from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk
from enterprise_multi_agent_rag.generation import RAGService
from enterprise_multi_agent_rag.retrieval.models import SearchResult


def _results() -> list[SearchResult]:
    chunk = EmbeddedChunk(
        chunk_id="generated-chunk",
        document_id="generated-document",
        chunk_index=0,
        content="Employees receive 15 vacation days.",
        embedding=[1.0, 0.0],
        metadata={"generated": True},
    )
    return [SearchResult(chunk=chunk, score=0.9, rank=1)]


class RecordingRetriever:
    """Return configured results while recording retrieval input."""

    def __init__(
        self,
        results: list[SearchResult],
        error: Exception | None = None,
    ) -> None:
        self.results = results
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, k: int = 5) -> list[SearchResult]:
        self.calls.append((question, k))
        if self.error is not None:
            raise self.error
        return self.results


class RecordingPromptBuilder:
    """Return a configured prompt while recording builder input."""

    def __init__(self, prompt: str, error: Exception | None = None) -> None:
        self.prompt = prompt
        self.error = error
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def build(self, question: str, search_results: list[SearchResult]) -> str:
        self.calls.append((question, search_results))
        if self.error is not None:
            raise self.error
        return self.prompt


class RecordingLLMService:
    """Return a configured answer while recording the generated prompt."""

    def __init__(self, answer: str, error: Exception | None = None) -> None:
        self.answer_text = answer
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.answer_text


def test_coordinates_components_and_returns_answer_unchanged() -> None:
    question = "How many vacation days are provided?"
    results = _results()
    prompt = "Final generated prompt."
    answer = "  Employees receive 15 vacation days.\n"
    retriever = RecordingRetriever(results)
    prompt_builder = RecordingPromptBuilder(prompt)
    llm_service = RecordingLLMService(answer)
    service = RAGService(retriever, prompt_builder, llm_service)  # type: ignore[arg-type]

    actual = service.answer(question, k=3)

    assert retriever.calls == [(question, 3)]
    assert prompt_builder.calls == [(question, results)]
    assert prompt_builder.calls[0][1] is results
    assert llm_service.prompts == [prompt]
    assert actual == answer


def test_retriever_errors_propagate_and_stop_flow() -> None:
    error = RuntimeError("retrieval failed")
    retriever = RecordingRetriever([], error)
    prompt_builder = RecordingPromptBuilder("unused")
    llm_service = RecordingLLMService("unused")
    service = RAGService(retriever, prompt_builder, llm_service)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="retrieval failed"):
        service.answer("question")

    assert prompt_builder.calls == []
    assert llm_service.prompts == []


def test_prompt_builder_errors_propagate_and_stop_generation() -> None:
    error = ValueError("prompt failed")
    prompt_builder = RecordingPromptBuilder("", error)
    llm_service = RecordingLLMService("unused")
    service = RAGService(  # type: ignore[arg-type]
        RecordingRetriever(_results()), prompt_builder, llm_service
    )

    with pytest.raises(ValueError, match="prompt failed"):
        service.answer("question")

    assert llm_service.prompts == []


def test_llm_service_errors_propagate() -> None:
    error = RuntimeError("generation failed")
    llm_service = RecordingLLMService("", error)
    service = RAGService(  # type: ignore[arg-type]
        RecordingRetriever(_results()),
        RecordingPromptBuilder("prompt"),
        llm_service,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        service.answer("question")
