"""Linear LangGraph workflow over the existing RAG components."""

from langgraph.graph import END, START, StateGraph

from enterprise_multi_agent_rag.generation import LLMService, PromptBuilder
from enterprise_multi_agent_rag.graph.state import RAGState
from enterprise_multi_agent_rag.retrieval import Retriever


class LangGraphRAGWorkflow:
    """Coordinate retrieval, prompting, and generation as graph nodes."""

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service

        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("build_prompt", self._prompt_node)
        builder.add_node("generate_answer", self._generate_node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "build_prompt")
        builder.add_edge("build_prompt", "generate_answer")
        builder.add_edge("generate_answer", END)
        self.graph = builder.compile()

    def _retrieve_node(self, state: RAGState) -> dict[str, object]:
        results = self.retriever.retrieve(state["question"], state["k"])
        return {"search_results": results}

    def _prompt_node(self, state: RAGState) -> dict[str, object]:
        prompt = self.prompt_builder.build(
            state["question"], state["search_results"]
        )
        return {"prompt": prompt}

    def _generate_node(self, state: RAGState) -> dict[str, object]:
        answer = self.llm_service.generate(state["prompt"])
        return {"answer": answer}

    def run(self, question: str, k: int = 5) -> str:
        """Execute the compiled graph and return its generated answer."""
        completed_state = self.graph.invoke({"question": question, "k": k})
        return completed_state["answer"]
