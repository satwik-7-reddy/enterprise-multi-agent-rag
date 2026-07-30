"""Formatting retrieved context into a final prompt string."""

from enterprise_multi_agent_rag.retrieval.models import SearchResult


class InvalidPromptInputError(ValueError):
    """Raised when prompt input cannot produce a meaningful prompt."""


class PromptBuilder:
    """Combine a user question and ranked chunk text into one prompt."""

    def build(self, question: str, search_results: list[SearchResult]) -> str:
        """Build an answer-grounding prompt without calling an LLM."""
        if not question or not question.strip():
            raise InvalidPromptInputError(
                "Question must not be empty or whitespace-only."
            )

        if search_results:
            context = "\n\n".join(
                f"[Context {index}]\n{result.chunk.content}"
                for index, result in enumerate(search_results, start=1)
            )
        else:
            context = "No relevant context was found."

        return (
            "You are an enterprise knowledge assistant.\n\n"
            "Answer the question using only the provided context.\n"
            "If the answer is not present in the context, say that you do not "
            "have enough information.\n"
            "Do not invent details.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
