"""Configuration-driven embedding provider construction."""

from enterprise_multi_agent_rag.core.config import Settings
from enterprise_multi_agent_rag.embeddings.exceptions import (
    UnsupportedEmbeddingProviderError,
)
from enterprise_multi_agent_rag.embeddings.providers import (
    BaseEmbeddingProvider,
    BedrockEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


def create_embedding_provider(settings: Settings) -> BaseEmbeddingProvider:
    """Create the configured provider using case-insensitive matching."""
    provider = settings.embedding_provider.strip().lower()
    if provider == "bedrock":
        return BedrockEmbeddingProvider(
            region=settings.aws_region,
            model_id=settings.bedrock_embedding_model,
            dimensions=settings.bedrock_embedding_dimensions,
            normalize=settings.bedrock_embedding_normalize,
        )
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    raise UnsupportedEmbeddingProviderError(
        f"Unsupported embedding provider: '{settings.embedding_provider}'."
    )
