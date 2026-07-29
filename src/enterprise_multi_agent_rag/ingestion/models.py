"""Internal models used by document ingestion."""

from typing import Any

from pydantic import BaseModel, Field


class IngestedDocument(BaseModel):
    """A consistently shaped document produced by all supported loaders."""

    document_id: str
    filename: str
    source_path: str
    file_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
