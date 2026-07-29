"""Models produced by the document chunking layer."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A standardized segment of an ingested document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    start_character: int | None
    end_character: int | None
    metadata: dict[str, Any] = Field(default_factory=dict)
