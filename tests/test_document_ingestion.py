"""Unit tests for local document loading."""

from pathlib import Path

import pytest

from enterprise_multi_agent_rag.ingestion.documents import DocumentLoader
from enterprise_multi_agent_rag.ingestion.exceptions import (
    DocumentNotFoundError,
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)


def _write_test_pdf(path: Path, text: str = "Hello PDF") -> None:
    """Write a minimal one-page PDF without relying on an external fixture."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(pdf)


@pytest.fixture
def loader() -> DocumentLoader:
    """Return the public document loader."""
    return DocumentLoader()


def test_load_text_file(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Enterprise knowledge", encoding="utf-8")

    document = loader.load(path)

    assert document.content == "Enterprise knowledge"
    assert document.filename == "notes.txt"
    assert document.file_type == "txt"
    assert document.source_path == str(path.resolve())
    assert document.metadata == {
        "file_size_bytes": 20,
        "file_extension": ".txt",
        "character_count": 20,
        "original_filename": "notes.txt",
    }


def test_load_markdown_file(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "guide.MD"
    path.write_text("# Guide\n\nUse the loader.", encoding="utf-8")

    document = loader.load(path)

    assert document.content.startswith("# Guide")
    assert document.file_type == "md"
    assert document.metadata["file_extension"] == ".md"


def test_load_pdf_file(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    _write_test_pdf(path)

    document = loader.load(path)

    assert "Hello PDF" in document.content
    assert document.file_type == "pdf"
    assert document.metadata["page_count"] == 1
    assert document.metadata["character_count"] == len(document.content)


def test_document_id_is_deterministic(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "stable.txt"
    path.write_text("unchanged", encoding="utf-8")

    first = loader.load(path)
    second = loader.load(path)

    assert first.document_id == second.document_id
    assert len(first.document_id) == 64


def test_missing_file(loader: DocumentLoader, tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError, match="Document not found"):
        loader.load(tmp_path / "missing.txt")


def test_directory_is_not_a_document(loader: DocumentLoader, tmp_path: Path) -> None:
    with pytest.raises(DocumentParsingError, match="not a file"):
        loader.load(tmp_path)


def test_unsupported_extension(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("value", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="\\.csv"):
        loader.load(path)


@pytest.mark.parametrize("content", [b"", b" \n\t"])
def test_empty_or_whitespace_text(
    loader: DocumentLoader, tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(content)

    with pytest.raises(EmptyDocumentError):
        loader.load(path)


def test_invalid_utf8_text(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(DocumentParsingError, match="UTF-8"):
        loader.load(path)


def test_invalid_pdf(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")

    with pytest.raises(DocumentParsingError, match="Could not parse PDF"):
        loader.load(path)
