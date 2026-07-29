"""Document chunking orchestration."""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from enterprise_multi_agent_rag.chunking.exceptions import (
    DocumentChunkingError,
    EmptyDocumentContentError,
    InvalidChunkingConfigurationError,
)
from enterprise_multi_agent_rag.chunking.models import DocumentChunk
from enterprise_multi_agent_rag.ingestion.models import IngestedDocument


class DocumentChunker:
    """Split ingested document text into deterministic, overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise InvalidChunkingConfigurationError(
                "chunk_size must be greater than zero."
            )
        if chunk_overlap < 0:
            raise InvalidChunkingConfigurationError(
                "chunk_overlap must be zero or greater."
            )
        if chunk_overlap >= chunk_size:
            raise InvalidChunkingConfigurationError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=True,
            strip_whitespace=False,
        )

    def chunk(self, document: IngestedDocument) -> list[DocumentChunk]:
        """Split one ingested document while preserving source metadata."""
        if not document.content or not document.content.strip():
            raise EmptyDocumentContentError(
                f"Document '{document.document_id}' contains no chunkable content."
            )

        try:
            contents = self._splitter.split_text(document.content)
        except Exception as exc:
            raise DocumentChunkingError(
                f"Could not chunk document '{document.document_id}': {exc}"
            ) from exc

        if not contents:
            raise DocumentChunkingError(
                f"Chunking document '{document.document_id}' produced no chunks."
            )

        offsets = self._locate_chunks(document.content, contents)
        total_chunks = len(contents)
        chunks: list[DocumentChunk] = []
        for index, (content, (start, end)) in enumerate(zip(contents, offsets, strict=True)):
            stable_input = f"{document.document_id}\0{index}\0{content}".encode()
            metadata = {
                **document.metadata,
                "document_id": document.document_id,
                "chunk_index": index,
                "total_chunks": total_chunks,
                "filename": document.filename,
                "file_type": document.file_type,
                "source_path": document.source_path,
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=hashlib.sha256(stable_input).hexdigest(),
                    document_id=document.document_id,
                    chunk_index=index,
                    content=content,
                    start_character=start,
                    end_character=end,
                    metadata=metadata,
                )
            )
        return chunks

    def _locate_chunks(
        self, original: str, chunks: list[str]
    ) -> list[tuple[int | None, int | None]]:
        """Locate each exact chunk using a monotonic, overlap-aware search.

        Searching begins near the prior chunk's expected overlap. This prevents
        repeated text from resolving to an earlier occurrence. If an exact
        occurrence cannot be found, both offsets are left unknown.
        """
        offsets: list[tuple[int | None, int | None]] = []
        previous_start = 0
        previous_end = 0
        for index, content in enumerate(chunks):
            search_from = (
                0
                if index == 0
                else max(previous_start + 1, previous_end - self.chunk_overlap)
            )
            start = original.find(content, search_from)
            if start == -1:
                offsets.append((None, None))
                continue
            end = start + len(content)
            offsets.append((start, end))
            previous_start = start
            previous_end = end
        return offsets
