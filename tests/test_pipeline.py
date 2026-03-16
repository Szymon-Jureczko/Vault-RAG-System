"""Tests for ingestion.pipeline — batch committer and orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


from ingestion.pipeline import BatchCommitter, IngestionStats, discover_files


class TestBatchCommitter:
    """Tests for BatchCommitter OOM-safe batching."""

    def test_flush_empty_buffer(self) -> None:
        mock_coll = MagicMock()
        mock_embed = MagicMock(return_value=[[0.1, 0.2]])
        batcher = BatchCommitter(mock_coll, mock_embed, batch_size=10)
        assert batcher.flush() == 0

    def test_auto_flush_at_batch_size(self) -> None:
        mock_coll = MagicMock()
        mock_embed = MagicMock(return_value=[[0.1]] * 5)
        batcher = BatchCommitter(mock_coll, mock_embed, batch_size=5)

        for i in range(5):
            batcher.add(f"id_{i}", f"text_{i}", {"src": "test"})

        assert batcher.total_committed == 5
        mock_coll.add.assert_called_once()

    def test_finalize_flushes_remaining(self) -> None:
        mock_coll = MagicMock()
        mock_embed = MagicMock(return_value=[[0.1]] * 3)
        batcher = BatchCommitter(mock_coll, mock_embed, batch_size=100)

        for i in range(3):
            batcher.add(f"id_{i}", f"text_{i}", {"src": "test"})

        flushed = batcher.finalize()
        assert flushed == 3
        assert batcher.total_committed == 3


class TestDiscoverFiles:
    """Tests for file discovery."""

    def test_discovers_supported_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.pdf").write_bytes(b"fake")
        (tmp_path / "c.xyz").write_text("skip")

        files = discover_files(tmp_path)
        names = {f.name for f in files}

        assert "a.txt" in names
        assert "b.pdf" in names
        assert "c.xyz" not in names

    def test_discovers_nested_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.md").write_text("content")

        files = discover_files(tmp_path)
        assert any(f.name == "nested.md" for f in files)

    def test_empty_directory(self, tmp_path: Path) -> None:
        files = discover_files(tmp_path)
        assert files == []


class TestIngestionStats:
    """Tests for IngestionStats model."""

    def test_defaults(self) -> None:
        stats = IngestionStats()
        assert stats.total_discovered == 0
        assert stats.skipped_unchanged == 0
        assert stats.processed == 0
        assert stats.failed == 0
        assert stats.total_chunks == 0
