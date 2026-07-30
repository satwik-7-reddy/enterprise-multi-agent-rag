"""Models returned by vector similarity search."""

from pydantic import BaseModel

from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk


class SearchResult(BaseModel):
    """One ranked chunk returned by a vector search."""

    chunk: EmbeddedChunk
    score: float
    rank: int
