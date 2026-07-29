"""Exceptions raised while chunking documents."""


class DocumentChunkingError(Exception):
    """Base exception for document chunking errors."""


class InvalidChunkingConfigurationError(DocumentChunkingError):
    """Raised when chunk size or overlap settings are invalid."""


class EmptyDocumentContentError(DocumentChunkingError):
    """Raised when a document contains no usable text."""
