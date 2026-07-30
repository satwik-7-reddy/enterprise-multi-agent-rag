"""Exceptions raised by embedding configuration, providers, and services."""


class EmbeddingError(Exception):
    """Base exception for embedding generation errors."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider request or response fails."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding provider configuration is invalid."""


class EmptyEmbeddingInputError(EmbeddingError):
    """Raised when an input that should contain text is empty."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding vectors have unexpected dimensions."""


class UnsupportedEmbeddingProviderError(EmbeddingConfigurationError):
    """Raised when the configured embedding provider is not supported."""
