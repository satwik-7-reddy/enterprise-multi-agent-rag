"""Shared state for the linear LangGraph RAG workflow."""

from typing import NotRequired, TypedDict

from enterprise_multi_agent_rag.retrieval import SearchResult


class RAGState(TypedDict):
    """Values supplied to and produced by the RAG graph nodes."""

    question: str
    k: int
    search_results: NotRequired[list[SearchResult]]
    prompt: NotRequired[str]
    answer: NotRequired[str]
