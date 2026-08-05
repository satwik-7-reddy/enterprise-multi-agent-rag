"""Tests for the basic linear LangGraph RAG workflow."""

from typing import Any

import pytest

from enterprise_multi_agent_rag.graph import LangGraphRAGWorkflow


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
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.received: tuple[str, list[Any]] | None = None
        self.error: Exception | None = None

    def build(self, question: str, results: list[Any]) -> str:
        self.events.append("prompt")
        self.received = (question, results)
        if self.error is not None:
            raise self.error
        return "graph prompt"


class FakeLLMService:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.received: str | None = None
        self.error: Exception | None = None

    def generate(self, prompt: str) -> str:
        self.events.append("generate")
        self.received = prompt
        if self.error is not None:
            raise self.error
        return "graph answer"


def _workflow() -> tuple[
    LangGraphRAGWorkflow, FakeRetriever, FakePromptBuilder, FakeLLMService, list[str]
]:
    events: list[str] = []
    retriever = FakeRetriever(events, [object(), object()])
    prompt_builder = FakePromptBuilder(events)
    llm_service = FakeLLMService(events)
    workflow = LangGraphRAGWorkflow(
        retriever,  # type: ignore[arg-type]
        prompt_builder,  # type: ignore[arg-type]
        llm_service,  # type: ignore[arg-type]
    )
    return workflow, retriever, prompt_builder, llm_service, events


def test_run_passes_state_through_nodes_in_order() -> None:
    workflow, retriever, prompt_builder, llm_service, events = _workflow()

    answer = workflow.run("How many vacation days?", k=3)

    assert retriever.received == ("How many vacation days?", 3)
    assert prompt_builder.received == ("How many vacation days?", retriever.results)
    assert llm_service.received == "graph prompt"
    assert events == ["retrieve", "prompt", "generate"]
    assert answer == "graph answer"


def test_run_uses_default_k() -> None:
    workflow, retriever, _, _, _ = _workflow()

    workflow.run("A question")

    assert retriever.received == ("A question", 5)


def test_completed_graph_state_contains_all_workflow_values() -> None:
    workflow, retriever, _, _, _ = _workflow()

    state = workflow.graph.invoke({"question": "State question", "k": 2})

    assert state == {
        "question": "State question",
        "k": 2,
        "search_results": retriever.results,
        "prompt": "graph prompt",
        "answer": "graph answer",
    }


@pytest.mark.parametrize("dependency", ["retriever", "prompt_builder", "llm_service"])
def test_dependency_errors_propagate(dependency: str) -> None:
    workflow, retriever, prompt_builder, llm_service, _ = _workflow()
    dependencies = {
        "retriever": retriever,
        "prompt_builder": prompt_builder,
        "llm_service": llm_service,
    }
    dependencies[dependency].error = RuntimeError(f"{dependency} failed")

    with pytest.raises(RuntimeError, match=f"{dependency} failed"):
        workflow.run("A question")
