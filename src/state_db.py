"""SQLite state database for incremental file processing.

Tracks every ingested file by its MD5 hash so unchanged files are never
re-processed. This is the single source of truth for "what has already
been ingested."
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings
from src.models import FileStatus, TrackedFile

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tracked_files (
    path         TEXT PRIMARY KEY,
    md5_hash     TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_tracked_files_status ON tracked_files(status);
"""


def compute_md5(file_path: Path, buf_size: int = 8192) -> str:
    """Compute the MD5 hex-digest of a file.

    Args:
        file_path: Path to the file to hash.
        buf_size: Read buffer size in bytes.

    Returns:
        Lowercase hex string of the MD5 digest.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as fh:
        while chunk := fh.read(buf_size):
            md5.update(chunk)
    return md5.hexdigest()


class StateDB:
    """Thin wrapper around SQLite for tracking file ingestion state.

    Args:
        settings: Application settings (provides ``state_db_path``).
    """

    def __init__(self, settings: Settings) -> None:
        self._db_path = settings.state_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the schema if it doesn't already exist."""
        with self._conn:
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_CREATE_INDEX_SQL)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Queries ─────────────────────────────────────────────────────────

    def get_file(self, path: Path) -> TrackedFile | None:
        """Retrieve a tracked file record by path.

        Args:
            path: Absolute path of the file.

        Returns:
            A TrackedFile instance or None if not tracked.
        """
        row = self._conn.execute(
            "SELECT * FROM tracked_files WHERE path = ?", (str(path),)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_model(row)

    def is_unchanged(self, path: Path) -> bool:
        """Check whether a file's MD5 matches its stored hash.

        Args:
            path: Path to the file on disk.

        Returns:
            True if the file exists in the DB **and** its current MD5
            matches the stored hash; False otherwise.
        """
        existing = self.get_file(path)
        if existing is None:
            return False
        current_hash = compute_md5(path)
        return existing.md5_hash == current_hash

    def list_by_status(self, status: FileStatus) -> list[TrackedFile]:
        """Return all tracked files with a given status.

        Args:
            status: The FileStatus value to filter on.

        Returns:
            List of matching TrackedFile instances.
        """
        rows = self._conn.execute(
            "SELECT * FROM tracked_files WHERE status = ?", (status.value,)
        ).fetchall()
        return [self._row_to_model(r) for r in rows]

    # ── Mutations ───────────────────────────────────────────────────────

    def upsert(self, tracked: TrackedFile) -> None:
        """Insert or update a tracked file record.

        Args:
            tracked: The TrackedFile to persist.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO tracked_files
                    (path, md5_hash, status, chunk_count, error_message,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    md5_hash      = excluded.md5_hash,
                    status        = excluded.status,
                    chunk_count   = excluded.chunk_count,
                    error_message = excluded.error_message,
                    updated_at    = excluded.updated_at
                """,
                (
                    str(tracked.path),
                    tracked.md5_hash,
                    tracked.status.value,
                    tracked.chunk_count,
                    tracked.error_message,
                    tracked.created_at.isoformat(),
                    now,
                ),
            )

    def mark_status(
        self,
        path: Path,
        status: FileStatus,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status (and optionally chunk count / error) of a file.

        Args:
            path: Path of the tracked file.
            status: New status value.
            chunk_count: If provided, update the chunk count.
            error_message: If provided, store error details.
        """
        now = datetime.now(timezone.utc).isoformat()
        sets = ["status = ?", "updated_at = ?"]
        params: list[str | int] = [status.value, now]

        if chunk_count is not None:
            sets.append("chunk_count = ?")
            params.append(chunk_count)
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)

        params.append(str(path))

        with self._conn:
            self._conn.execute(
                f"UPDATE tracked_files SET {', '.join(sets)} WHERE path = ?",
                params,
            )

    def reset_failed(self) -> int:
        """Set all FAILED files back to PENDING for retry.

        Returns:
            Number of rows updated.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            cur = self._conn.execute(
                "UPDATE tracked_files SET status = ?, updated_at = ? WHERE status = ?",
                (FileStatus.PENDING.value, now, FileStatus.FAILED.value),
            )
        return cur.rowcount

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> TrackedFile:
        """Convert a sqlite3.Row into a TrackedFile model."""
        return TrackedFile(
            path=Path(row["path"]),
            md5_hash=row["md5_hash"],
            status=FileStatus(row["status"]),
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
