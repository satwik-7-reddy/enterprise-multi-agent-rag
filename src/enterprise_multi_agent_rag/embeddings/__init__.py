"""Public provider-independent embedding API."""

from enterprise_multi_agent_rag.embeddings.factory import create_embedding_provider
from enterprise_multi_agent_rag.embeddings.models import EmbeddedChunk
from enterprise_multi_agent_rag.embeddings.service import EmbeddingService

__all__ = ["EmbeddedChunk", "EmbeddingService", "create_embedding_provider"]
