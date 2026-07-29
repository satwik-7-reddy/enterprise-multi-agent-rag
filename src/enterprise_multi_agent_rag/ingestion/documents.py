"""Public orchestration service for loading local documents."""

import hashlib
from pathlib import Path

from enterprise_multi_agent_rag.ingestion.exceptions import (
    DocumentNotFoundError,
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from enterprise_multi_agent_rag.ingestion.loaders import (
    DocumentParser,
    MarkdownParser,
    PdfParser,
    TextParser,
)
from enterprise_multi_agent_rag.ingestion.models import IngestedDocument


class DocumentLoader:
    """Validate and convert supported local files into ingested documents."""

    def __init__(self, parsers: dict[str, DocumentParser] | None = None) -> None:
        self._parsers = parsers or {
            ".pdf": PdfParser(),
            ".txt": TextParser(),
            ".md": MarkdownParser(),
        }

    def load(self, path: str | Path) -> IngestedDocument:
        """Load one PDF, text, or Markdown file."""
        source = Path(path).expanduser()
        if not source.exists():
            raise DocumentNotFoundError(f"Document not found: '{source}'.")
        if not source.is_file():
            raise DocumentParsingError(f"Document path is not a file: '{source}'.")

        extension = source.suffix.lower()
        parser = self._parsers.get(extension)
        if parser is None:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{extension or '<none>'}' for '{source}'."
            )

        try:
            raw_content = source.read_bytes()
        except OSError as exc:
            raise DocumentParsingError(f"Could not read '{source}': {exc}") from exc
        if not raw_content:
            raise EmptyDocumentError(f"Document is empty: '{source}'.")

        parsed = parser.parse(source)
        if not parsed.content.strip():
            raise EmptyDocumentError(
                f"Document contains no extractable non-whitespace content: '{source}'."
            )

        metadata = {
            "file_size_bytes": len(raw_content),
            "file_extension": extension,
            "character_count": len(parsed.content),
            "original_filename": source.name,
            **parsed.metadata,
        }
        return IngestedDocument(
            document_id=hashlib.sha256(raw_content).hexdigest(),
            filename=source.name,
            source_path=str(source.resolve()),
            file_type=extension.removeprefix("."),
            content=parsed.content,
            metadata=metadata,
        )


__all__ = ["DocumentLoader", "IngestedDocument"]
