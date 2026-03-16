"""Tests for src.state_db — SQLite incremental processing logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.models import FileStatus, TrackedFile
from src.state_db import StateDB, compute_md5


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Create Settings pointing at a temporary directory."""
    return Settings(
        state_db_path=tmp_path / "test_state.db",
        chroma_persist_path=tmp_path / "chroma",
    )


@pytest.fixture()
def db(tmp_settings: Settings) -> StateDB:
    """Create a fresh StateDB for each test."""
    state = StateDB(tmp_settings)
    yield state
    state.close()


class TestComputeMD5:
    """Tests for the compute_md5 helper."""

    def test_consistent_hash(self, tmp_path: Path) -> None:
        """Same content produces the same MD5."""
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        assert compute_md5(f) == compute_md5(f)

    def test_different_content(self, tmp_path: Path) -> None:
        """Different content produces different MD5s."""
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello")
        b.write_text("world")
        assert compute_md5(a) != compute_md5(b)


class TestStateDB:
    """Tests for StateDB CRUD operations."""

    def test_upsert_and_get(self, db: StateDB, tmp_path: Path) -> None:
        """Inserting a record and retrieving it returns matching data."""
        fp = tmp_path / "doc.pdf"
        fp.write_bytes(b"fake pdf")
        tracked = TrackedFile(path=fp, md5_hash=compute_md5(fp))

        db.upsert(tracked)
        result = db.get_file(fp)

        assert result is not None
        assert result.path == fp
        assert result.md5_hash == tracked.md5_hash
        assert result.status == FileStatus.PENDING

    def test_get_nonexistent(self, db: StateDB) -> None:
        """Querying a path that was never inserted returns None."""
        assert db.get_file(Path("/does/not/exist")) is None

    def test_is_unchanged_new_file(self, db: StateDB, tmp_path: Path) -> None:
        """A file not in the DB is not considered unchanged."""
        f = tmp_path / "new.txt"
        f.write_text("content")
        assert db.is_unchanged(f) is False

    def test_is_unchanged_same_content(
        self, db: StateDB, tmp_path: Path,
    ) -> None:
        """A file with matching MD5 is considered unchanged."""
        f = tmp_path / "same.txt"
        f.write_text("fixed content")
        tracked = TrackedFile(path=f, md5_hash=compute_md5(f))
        db.upsert(tracked)
        assert db.is_unchanged(f) is True

    def test_is_unchanged_modified(self, db: StateDB, tmp_path: Path) -> None:
        """After modifying a file its MD5 no longer matches."""
        f = tmp_path / "change.txt"
        f.write_text("original")
        tracked = TrackedFile(path=f, md5_hash=compute_md5(f))
        db.upsert(tracked)

        f.write_text("modified")
        assert db.is_unchanged(f) is False

    def test_mark_status(self, db: StateDB, tmp_path: Path) -> None:
        """mark_status updates the status field."""
        f = tmp_path / "s.txt"
        f.write_text("x")
        tracked = TrackedFile(path=f, md5_hash=compute_md5(f))
        db.upsert(tracked)

        db.mark_status(f, FileStatus.COMPLETED, chunk_count=10)
        result = db.get_file(f)
        assert result is not None
        assert result.status == FileStatus.COMPLETED
        assert result.chunk_count == 10

    def test_list_by_status(self, db: StateDB, tmp_path: Path) -> None:
        """list_by_status returns only files with the requested status."""
        for i, status in enumerate([FileStatus.PENDING, FileStatus.COMPLETED]):
            f = tmp_path / f"f{i}.txt"
            f.write_text(f"content {i}")
            t = TrackedFile(path=f, md5_hash=compute_md5(f), status=status)
            db.upsert(t)

        pending = db.list_by_status(FileStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].status == FileStatus.PENDING

    def test_reset_failed(self, db: StateDB, tmp_path: Path) -> None:
        """reset_failed moves all FAILED records back to PENDING."""
        f = tmp_path / "fail.txt"
        f.write_text("oops")
        t = TrackedFile(
            path=f, md5_hash=compute_md5(f), status=FileStatus.FAILED
        )
        db.upsert(t)

        count = db.reset_failed()
        assert count == 1
        result = db.get_file(f)
        assert result is not None
        assert result.status == FileStatus.PENDING
