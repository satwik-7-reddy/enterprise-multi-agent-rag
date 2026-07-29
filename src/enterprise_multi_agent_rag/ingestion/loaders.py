"""Format-specific document content loaders."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader

from enterprise_multi_agent_rag.ingestion.exceptions import DocumentParsingError


@dataclass(frozen=True)
class ParsedDocument:
    """Content and format-specific metadata returned by a parser."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(Protocol):
    """Interface implemented by format-specific parsers."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a local document."""


class Utf8TextParser:
    """Parse a UTF-8 encoded text-based document."""

    def parse(self, path: Path) -> ParsedDocument:
        """Read text without silently replacing invalid bytes."""
        try:
            return ParsedDocument(content=path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise DocumentParsingError(
                f"Could not decode '{path}' as UTF-8."
            ) from exc
        except OSError as exc:
            raise DocumentParsingError(f"Could not read '{path}': {exc}") from exc


class TextParser(Utf8TextParser):
    """Parser for plain-text files."""


class MarkdownParser(Utf8TextParser):
    """Parser for Markdown files."""


class PdfParser:
    """Extract PDF text page by page using pypdf."""

    def parse(self, path: Path) -> ParsedDocument:
        """Return page text joined by newlines and the number of pages."""
        try:
            reader = PdfReader(path)
            page_text = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            raise DocumentParsingError(f"Could not parse PDF '{path}': {exc}") from exc

        return ParsedDocument(
            content="\n".join(page_text),
            metadata={"page_count": len(reader.pages)},
        )
