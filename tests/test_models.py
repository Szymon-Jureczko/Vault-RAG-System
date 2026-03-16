"""Tests for src.models — Pydantic data models."""

from __future__ import annotations

from pathlib import Path

from src.models import (
    DocumentChunk,
    FileStatus,
    IngestionStats,
    ParseResult,
    TrackedFile,
)


class TestTrackedFile:
    """Tests for the TrackedFile model."""

    def test_defaults(self) -> None:
        """Default status is PENDING and chunk_count is 0."""
        t = TrackedFile(path=Path("/a.pdf"), md5_hash="abc123")
        assert t.status == FileStatus.PENDING
        assert t.chunk_count == 0
        assert t.error_message is None

    def test_timestamps_populated(self) -> None:
        """created_at and updated_at are auto-populated."""
        t = TrackedFile(path=Path("/a.pdf"), md5_hash="abc123")
        assert t.created_at is not None
        assert t.updated_at is not None


class TestDocumentChunk:
    """Tests for the DocumentChunk model."""

    def test_char_count(self) -> None:
        """char_count reflects content length."""
        c = DocumentChunk(
            chunk_id="c1",
            source_path=Path("/a.txt"),
            content="hello world",
        )
        assert c.char_count == 11

    def test_metadata_default(self) -> None:
        """metadata defaults to an empty dict."""
        c = DocumentChunk(
            chunk_id="c1", source_path=Path("/a.txt"), content=""
        )
        assert c.metadata == {}


class TestParseResult:
    """Tests for the ParseResult model."""

    def test_default_success(self) -> None:
        """Default ParseResult is successful with no chunks."""
        r = ParseResult(source_path=Path("/a.pdf"))
        assert r.success is True
        assert r.chunks == []


class TestIngestionStats:
    """Tests for the IngestionStats model."""

    def test_all_zeros(self) -> None:
        """Freshly created stats are all zeros."""
        s = IngestionStats()
        assert s.total_files == 0
        assert s.skipped_unchanged == 0
        assert s.processed == 0
        assert s.failed == 0
        assert s.total_chunks == 0
