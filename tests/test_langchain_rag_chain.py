"""Tests for the LangChain Runnable RAG composition."""

from typing import Any

import pytest

from enterprise_multi_agent_rag.chains import LangChainRAGChain


class FakeRetriever:
    def __init__(self, events: list[str], results: list[Any]) -> None:
        self.events = events
        self.results = results
        self.received: tuple[str, int] | None = None
        self.error: Exception | None = None

    def retrieve(self, question: str, k: int) -> list[Any]:
        self.events.append("retrieve")
        self.received = (question, k)
        if self.error is not None:
            raise self.error
        return self.results


class FakePromptBuilder:
    def __init__(self, events: list[str], prompt: str) -> None:
        self.events = events
        self.prompt = prompt
        self.received: tuple[str, list[Any]] | None = None
        self.error: Exception | None = None

    def build(self, question: str, results: list[Any]) -> str:
        self.events.append("prompt")
        self.received = (question, results)
        if self.error is not None:
            raise self.error
        return self.prompt


class FakeLLMService:
    def __init__(self, events: list[str], answer: str) -> None:
        self.events = events
        self.answer = answer
        self.received: str | None = None
        self.error: Exception | None = None

    def generate(self, prompt: str) -> str:
        self.events.append("generate")
        self.received = prompt
        if self.error is not None:
            raise self.error
        return self.answer


def _chain() -> tuple[
    LangChainRAGChain, FakeRetriever, FakePromptBuilder, FakeLLMService, list[str]
]:
    events: list[str] = []
    results = [object(), object()]
    retriever = FakeRetriever(events, results)
    prompt_builder = FakePromptBuilder(events, "generated prompt")
    llm_service = FakeLLMService(events, "final answer")
    chain = LangChainRAGChain(
        retriever=retriever,  # type: ignore[arg-type]
        prompt_builder=prompt_builder,  # type: ignore[arg-type]
        llm_service=llm_service,  # type: ignore[arg-type]
    )
    return chain, retriever, prompt_builder, llm_service, events


def test_public_invoke_passes_data_in_order_and_returns_answer() -> None:
    chain, retriever, prompt_builder, llm_service, events = _chain()

    answer = chain.invoke("How many vacation days?", k=3)

    assert retriever.received == ("How many vacation days?", 3)
    assert prompt_builder.received == ("How many vacation days?", retriever.results)
    assert llm_service.received == "generated prompt"
    assert events == ["retrieve", "prompt", "generate"]
    assert answer == "final answer"


def test_public_invoke_uses_default_k() -> None:
    chain, retriever, _, _, _ = _chain()

    chain.invoke("A question")

    assert retriever.received == ("A question", 5)


@pytest.mark.parametrize("dependency", ["retriever", "prompt_builder", "llm_service"])
def test_dependency_errors_propagate(dependency: str) -> None:
    chain, retriever, prompt_builder, llm_service, _ = _chain()
    dependencies = {
        "retriever": retriever,
        "prompt_builder": prompt_builder,
        "llm_service": llm_service,
    }
    dependencies[dependency].error = RuntimeError(f"{dependency} failed")

    with pytest.raises(RuntimeError, match=f"{dependency} failed"):
        chain.invoke("A question")
