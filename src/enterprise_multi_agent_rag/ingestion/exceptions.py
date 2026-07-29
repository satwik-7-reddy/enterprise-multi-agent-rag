"""Exceptions raised while loading documents."""


class DocumentIngestionError(Exception):
    """Base exception for document ingestion errors."""


class DocumentNotFoundError(DocumentIngestionError):
    """Raised when a document path does not exist."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when a document has an unsupported extension."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when a document has no usable content."""


class DocumentParsingError(DocumentIngestionError):
    """Raised when a supported document cannot be parsed."""
