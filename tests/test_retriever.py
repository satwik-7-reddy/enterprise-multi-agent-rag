"""Focused tests for natural-language chunk retrieval."""

import pytest

from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk
from enterprise_multi_agent_rag.embeddings.providers import BaseEmbeddingProvider
from enterprise_multi_agent_rag.retrieval import FAISSVectorStore, Retriever
from enterprise_multi_agent_rag.retrieval.retriever import InvalidQuestionError


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Return deterministic query vectors without external calls."""

    def __init__(
        self,
        vectors: dict[str, list[float]],
        error: Exception | None = None,
    ) -> None:
        self.vectors = vectors
        self.error = error
        self.questions: list[str] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def embed_text(self, text: str) -> list[float]:
        self.questions.append(text)
        if self.error is not None:
            raise self.error
        return self.vectors[text]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class RecordingVectorStore(FAISSVectorStore):
    """Real FAISS store that records the requested top-k value."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_k: int | None = None

    def search(self, query_embedding: list[float], k: int = 5):
        self.requested_k = k
        return super().search(query_embedding, k=k)


def _chunk(chunk_id: str, embedding: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id="generated-document",
        chunk_index=0,
        content=f"Generated content about {chunk_id}",
        embedding=embedding,
        metadata={"topic": chunk_id},
    )


@pytest.fixture
def vector_store() -> RecordingVectorStore:
    """Create a deterministic real FAISS index."""
    store = RecordingVectorStore()
    store.add_chunks(
        [
            _chunk("vacation", [1.0, 0.0, 0.0]),
            _chunk("insurance", [0.0, 1.0, 0.0]),
            _chunk("remote-work", [0.0, 0.0, 1.0]),
        ]
    )
    return store


def test_retrieves_ranked_chunks_for_valid_question(
    vector_store: RecordingVectorStore,
) -> None:
    question = "How many vacation days do employees get?"
    provider = FakeEmbeddingProvider({question: [0.9, 0.1, 0.0]})

    results = Retriever(provider, vector_store).retrieve(question, k=2)

    assert [result.chunk.chunk_id for result in results] == ["vacation", "insurance"]
    assert [result.rank for result in results] == [1, 2]


@pytest.mark.parametrize("question", ["", " \n\t "])
def test_rejects_empty_or_whitespace_question(question: str) -> None:
    provider = FakeEmbeddingProvider({})

    with pytest.raises(InvalidQuestionError, match="must not be empty"):
        Retriever(provider, FAISSVectorStore()).retrieve(question)

    assert provider.questions == []


def test_passes_exact_question_to_embedding_provider(
    vector_store: RecordingVectorStore,
) -> None:
    question = "  What is the remote work policy?  "
    provider = FakeEmbeddingProvider({question: [0.0, 0.0, 1.0]})

    Retriever(provider, vector_store).retrieve(question)

    assert provider.questions == [question]


def test_passes_requested_k_to_vector_store(
    vector_store: RecordingVectorStore,
) -> None:
    question = "vacation"
    provider = FakeEmbeddingProvider({question: [1.0, 0.0, 0.0]})

    results = Retriever(provider, vector_store).retrieve(question, k=1)

    assert vector_store.requested_k == 1
    assert len(results) == 1


def test_empty_vector_store_returns_empty_list() -> None:
    question = "Any policy?"
    provider = FakeEmbeddingProvider({question: [1.0, 0.0, 0.0]})

    results = Retriever(provider, FAISSVectorStore()).retrieve(question)

    assert results == []
    assert provider.questions == [question]


def test_embedding_provider_errors_propagate(
    vector_store: RecordingVectorStore,
) -> None:
    provider_error = RuntimeError("embedding unavailable")
    provider = FakeEmbeddingProvider({}, error=provider_error)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        Retriever(provider, vector_store).retrieve("question")
