"""Pydantic data models used across the LocalVaultRAG pipeline.

Every data structure that crosses module boundaries is defined here to
enforce type-safety and provide self-documenting schemas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FileStatus(str, Enum):
    """Processing status of a tracked file."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TrackedFile(BaseModel):
    """Represents a file tracked in the state database.

    Attributes:
        path: Absolute path to the source file.
        md5_hash: MD5 hex-digest of the file contents.
        status: Current processing status.
        chunk_count: Number of text chunks produced from this file.
        error_message: Error details if processing failed.
        created_at: Timestamp when the record was first inserted.
        updated_at: Timestamp of the most recent update.
    """

    path: Path
    md5_hash: str
    status: FileStatus = FileStatus.PENDING
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """A chunk of text extracted from a document, ready for embedding.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        source_path: Path to the originating file.
        content: The text content of the chunk.
        metadata: Arbitrary metadata (page number, parser used, etc.).
        char_count: Number of characters in the content.
    """

    chunk_id: str
    source_path: Path
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return the character length of the content."""
        return len(self.content)


class ParseResult(BaseModel):
    """Result from parsing a single document.

    Attributes:
        source_path: Path to the parsed file.
        chunks: List of extracted text chunks.
        parser_used: Name of the parser that produced the result.
        page_count: Total pages in the source document (if applicable).
        success: Whether parsing completed without errors.
        error_message: Error details if parsing failed.
    """

    source_path: Path
    chunks: list[DocumentChunk] = Field(default_factory=list)
    parser_used: str = ""
    page_count: int = 0
    success: bool = True
    error_message: str | None = None


class IngestionStats(BaseModel):
    """Summary statistics for an ingestion run.

    Attributes:
        total_files: Total number of files discovered.
        skipped_unchanged: Files skipped because their MD5 hasn't changed.
        processed: Files successfully processed in this run.
        failed: Files that encountered errors during processing.
        total_chunks: Total chunks produced across all processed files.
    """

    total_files: int = 0
    skipped_unchanged: int = 0
    processed: int = 0
    failed: int = 0
    total_chunks: int = 0
