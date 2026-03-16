"""Pydantic models for the ingestion pipeline.

Every row in the SQLite ``state_db`` maps to a :class:`FileRecord`.
Parsing results are captured by :class:`ParseResult` before being
handed off to the vector store.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class ParsingStatus(str, enum.Enum):
    """Lifecycle states for a tracked file.

    Attributes:
        PENDING: File discovered but not yet parsed.
        IN_PROGRESS: Parsing has started.
        COMPLETED: Successfully parsed and embedded.
        FAILED: Parsing or embedding raised an error.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FileRecord(BaseModel):
    """Represents a single document tracked in the state database.

    Attributes:
        file_path: Absolute path to the source document.
        file_hash: MD5 hex-digest of the file contents.
        parsing_status: Current processing state.
        vector_id: ChromaDB document ID assigned after embedding
            (``None`` until embedding succeeds).
        error_message: Human-readable error captured on failure.
        created_at: Timestamp when the record was first inserted.
        updated_at: Timestamp of the most recent status change.
    """

    file_path: str
    file_hash: str
    parsing_status: ParsingStatus = ParsingStatus.PENDING
    vector_id: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ParseResult(BaseModel):
    """Output produced by a document parser.

    Attributes:
        file_path: Source document path.
        text: Extracted plain-text content.
        metadata: Arbitrary metadata dict (page count, format, etc.).
        success: Whether parsing completed without errors.
        error_message: Error string if ``success`` is False.
    """

    file_path: str
    text: str = ""
    metadata: dict = Field(default_factory=dict)
    success: bool = True
    error_message: str | None = None
