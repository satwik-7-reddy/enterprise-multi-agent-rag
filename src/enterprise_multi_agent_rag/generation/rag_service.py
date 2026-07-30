"""End-to-end coordination of retrieval, prompting, and generation."""

from enterprise_multi_agent_rag.generation.llm_service import LLMService
from enterprise_multi_agent_rag.generation.prompt_builder import PromptBuilder
from enterprise_multi_agent_rag.retrieval.retriever import Retriever


class RAGService:
    """Coordinate the existing RAG components without adding domain logic."""

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service

    def answer(self, question: str, k: int = 5) -> str:
        """Retrieve context, build a prompt, and return generated answer text."""
        search_results = self.retriever.retrieve(question, k)
        prompt = self.prompt_builder.build(question, search_results)
        return self.llm_service.generate(prompt)
