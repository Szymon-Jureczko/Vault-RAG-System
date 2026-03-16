"""Tests for ingestion.state_tracker — SQLite state management."""

import tempfile
from pathlib import Path

import pytest

from ingestion.models import ParsingStatus
from ingestion.state_tracker import StateTracker, compute_md5


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a path for a temporary SQLite database."""
    return tmp_path / "test_state.db"


@pytest.fixture()
def tracker(tmp_db: Path) -> StateTracker:
    """Return an initialised StateTracker backed by a temp database."""
    t = StateTracker(db_path=tmp_db)
    t.init_db()
    return t


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Create a small temporary file and return its path."""
    f = tmp_path / "sample.txt"
    f.write_text("hello world")
    return f


class TestComputeMd5:
    """Verify MD5 hashing utility."""

    def test_deterministic(self, sample_file: Path) -> None:
        h1 = compute_md5(sample_file)
        h2 = compute_md5(sample_file)
        assert h1 == h2

    def test_changes_on_modification(self, sample_file: Path) -> None:
        h1 = compute_md5(sample_file)
        sample_file.write_text("changed content")
        h2 = compute_md5(sample_file)
        assert h1 != h2

    def test_returns_hex_string(self, sample_file: Path) -> None:
        h = compute_md5(sample_file)
        assert len(h) == 32
        int(h, 16)  # should not raise


class TestStateTrackerInit:
    """Verify database initialisation."""

    def test_creates_db_file(self, tmp_db: Path) -> None:
        t = StateTracker(db_path=tmp_db)
        t.init_db()
        assert tmp_db.exists()

    def test_idempotent(self, tracker: StateTracker) -> None:
        tracker.init_db()
        tracker.init_db()  # should not raise


class TestStateTrackerCRUD:
    """Verify insert / read / update operations."""

    def test_upsert_and_get(self, tracker: StateTracker) -> None:
        tracker.upsert("/tmp/a.pdf", "aaa111")
        rec = tracker.get_record("/tmp/a.pdf")
        assert rec is not None
        assert rec.file_hash == "aaa111"
        assert rec.parsing_status is ParsingStatus.PENDING

    def test_get_missing_returns_none(self, tracker: StateTracker) -> None:
        assert tracker.get_record("/nonexistent") is None

    def test_mark_in_progress(self, tracker: StateTracker) -> None:
        tracker.mark_in_progress("/tmp/b.pdf", "bbb222")
        rec = tracker.get_record("/tmp/b.pdf")
        assert rec is not None
        assert rec.parsing_status is ParsingStatus.IN_PROGRESS

    def test_mark_completed(self, tracker: StateTracker) -> None:
        tracker.mark_in_progress("/tmp/c.pdf", "ccc333")
        tracker.mark_completed("/tmp/c.pdf", vector_id="vec-001")
        rec = tracker.get_record("/tmp/c.pdf")
        assert rec is not None
        assert rec.parsing_status is ParsingStatus.COMPLETED
        assert rec.vector_id == "vec-001"
        assert rec.error_message is None

    def test_mark_failed(self, tracker: StateTracker) -> None:
        tracker.mark_in_progress("/tmp/d.pdf", "ddd444")
        tracker.mark_failed("/tmp/d.pdf", error="timeout")
        rec = tracker.get_record("/tmp/d.pdf")
        assert rec is not None
        assert rec.parsing_status is ParsingStatus.FAILED
        assert rec.error_message == "timeout"


class TestNeedsProcessing:
    """Verify the resume-on-failure / incremental logic."""

    def test_new_file(self, tracker: StateTracker, sample_file: Path) -> None:
        assert tracker.needs_processing(sample_file) is True

    def test_completed_unchanged(
        self, tracker: StateTracker, sample_file: Path
    ) -> None:
        h = compute_md5(sample_file)
        tracker.mark_in_progress(str(sample_file), h)
        tracker.mark_completed(str(sample_file), vector_id="v1")
        assert tracker.needs_processing(sample_file) is False

    def test_completed_but_changed(
        self, tracker: StateTracker, sample_file: Path
    ) -> None:
        h = compute_md5(sample_file)
        tracker.mark_in_progress(str(sample_file), h)
        tracker.mark_completed(str(sample_file), vector_id="v1")
        # Modify the file so its hash changes
        sample_file.write_text("different content now")
        assert tracker.needs_processing(sample_file) is True

    def test_failed_needs_retry(self, tracker: StateTracker, sample_file: Path) -> None:
        h = compute_md5(sample_file)
        tracker.mark_in_progress(str(sample_file), h)
        tracker.mark_failed(str(sample_file), error="oops")
        assert tracker.needs_processing(sample_file) is True


class TestSummary:
    """Verify the summary / health-check method."""

    def test_empty_db(self, tracker: StateTracker) -> None:
        assert tracker.summary() == {}

    def test_counts(self, tracker: StateTracker) -> None:
        tracker.mark_in_progress("/a", "h1")
        tracker.mark_completed("/a", vector_id="v1")
        tracker.mark_in_progress("/b", "h2")
        tracker.mark_failed("/b", error="err")
        tracker.mark_in_progress("/c", "h3")
        s = tracker.summary()
        assert s.get("completed") == 1
        assert s.get("failed") == 1
        assert s.get("in_progress") == 1


class TestGetPendingFiles:
    """Verify filtered queries."""

    def test_excludes_completed(self, tracker: StateTracker) -> None:
        tracker.mark_in_progress("/a", "h1")
        tracker.mark_completed("/a", vector_id="v1")
        tracker.mark_in_progress("/b", "h2")
        pending = tracker.get_pending_files()
        paths = [r.file_path for r in pending]
        assert "/a" not in paths
        assert "/b" in paths
