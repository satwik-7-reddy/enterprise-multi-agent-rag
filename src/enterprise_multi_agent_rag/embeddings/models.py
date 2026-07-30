"""Models produced by embedding generation."""

from typing import Any

from pydantic import BaseModel, Field


class EmbeddedChunk(BaseModel):
    """A document chunk paired with its numeric embedding vector."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
