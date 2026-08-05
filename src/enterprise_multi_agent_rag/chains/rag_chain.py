"""LangChain Runnable composition of the existing RAG operations."""

from typing import TypedDict

from langchain_core.runnables import RunnableLambda

from enterprise_multi_agent_rag.generation import LLMService, PromptBuilder
from enterprise_multi_agent_rag.retrieval import Retriever, SearchResult


class ChainInput(TypedDict):
    """Input accepted by the retrieval runnable."""

    question: str
    k: int


class PromptInput(TypedDict):
    """Question and results passed to the prompt runnable."""

    question: str
    search_results: list[SearchResult]


class LangChainRAGChain:
    """Express the existing linear RAG workflow as LangChain Runnables."""

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service

        retrieve_runnable = RunnableLambda(self._retrieve_step)
        prompt_runnable = RunnableLambda(self._prompt_step)
        generation_runnable = RunnableLambda(self._generation_step)
        self._chain = retrieve_runnable | prompt_runnable | generation_runnable

    def _retrieve_step(self, chain_input: ChainInput) -> PromptInput:
        results = self.retriever.retrieve(chain_input["question"], chain_input["k"])
        return {
            "question": chain_input["question"],
            "search_results": results,
        }

    def _prompt_step(self, prompt_input: PromptInput) -> str:
        return self.prompt_builder.build(
            prompt_input["question"], prompt_input["search_results"]
        )

    def _generation_step(self, prompt: str) -> str:
        return self.llm_service.generate(prompt)

    def invoke(self, question: str, k: int = 5) -> str:
        """Run retrieval, prompt building, and generation in sequence."""
        return self._chain.invoke({"question": question, "k": k})
