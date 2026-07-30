"""Natural-language question retrieval coordination."""

from enterprise_multi_agent_rag.embeddings.providers import BaseEmbeddingProvider
from enterprise_multi_agent_rag.retrieval.models import SearchResult
from enterprise_multi_agent_rag.retrieval.vector_store import FAISSVectorStore


class RetrieverError(Exception):
    """Base exception for retriever-specific errors."""


class InvalidQuestionError(RetrieverError):
    """Raised when a question contains no usable text."""


class Retriever:
    """Embed a question and search a configured vector store."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(self, question: str, k: int = 5) -> list[SearchResult]:
        """Return the most relevant chunks for a natural-language question."""
        if not question or not question.strip():
            raise InvalidQuestionError("Question must not be empty or whitespace-only.")

        query_embedding = self.embedding_provider.embed_text(question)
        return self.vector_store.search(query_embedding, k=k)
