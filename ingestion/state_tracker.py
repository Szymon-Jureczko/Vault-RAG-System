"""SQLite-backed state tracker for incremental document ingestion.

Provides resume-on-failure capability: every file is tracked by its MD5
hash so that unchanged documents are never re-processed, and partially
completed runs can be resumed from exactly where they left off.

Usage::

    from ingestion.state_tracker import StateTracker

    tracker = StateTracker()          # uses path from Settings
    tracker.init_db()                 # idempotent — safe to call repeatedly
    if tracker.needs_processing(path):
        tracker.mark_in_progress(path, md5_hash)
        ...                           # parse + embed
        tracker.mark_completed(path, vector_id="chroma-abc-123")
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ingestion.config import settings
from ingestion.models import FileRecord, ParsingStatus

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS file_state (
    file_path       TEXT PRIMARY KEY,
    file_hash       TEXT    NOT NULL,
    parsing_status  TEXT    NOT NULL DEFAULT 'pending',
    vector_id       TEXT,
    error_message   TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_state_status
    ON file_state (parsing_status);

CREATE INDEX IF NOT EXISTS idx_file_state_hash
    ON file_state (file_hash);
"""


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def compute_md5(file_path: str | Path) -> str:
    """Compute the MD5 hex-digest of a file, reading in 8 KiB chunks.

    Args:
        file_path: Path to the file on disk.

    Returns:
        Lowercase hex string of the MD5 hash.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class StateTracker:
    """SQLite state database for tracking document ingestion status.

    Args:
        db_path: Path to the SQLite database file.  Defaults to the
            value in :pydata:`ingestion.config.settings`.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else settings.state_db_path
        self._conn: sqlite3.Connection | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def init_db(self) -> None:
        """Create the database file, tables and indexes (idempotent).

        Parent directories are created automatically if they do not exist.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("State DB initialised at %s", self._db_path)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the active connection, initialising if necessary."""
        if self._conn is None:
            self.init_db()
        assert self._conn is not None
        return self._conn

    # ── queries ─────────────────────────────────────────────────────────────

    def get_record(self, file_path: str | Path) -> FileRecord | None:
        """Fetch the state record for a given file.

        Args:
            file_path: Absolute path to the document.

        Returns:
            A :class:`FileRecord` if the file is tracked, else ``None``.
        """
        row = self.conn.execute(
            "SELECT * FROM file_state WHERE file_path = ?",
            (str(file_path),),
        ).fetchone()
        if row is None:
            return None
        return FileRecord(**dict(row))

    def needs_processing(
        self,
        file_path: str | Path,
        *,
        key: str | Path | None = None,
    ) -> bool:
        """Determine whether *file_path* should be (re-)processed.

        A file needs processing when:
        * It has never been seen before, **or**
        * Its MD5 hash has changed since the last successful run, **or**
        * Its previous run failed or was interrupted.

        Args:
            file_path: Path to the source document on disk (used for
                computing the MD5 hash).
            key: State-DB key to look up.  Defaults to *file_path* when
                not provided.  Pass a source-relative path here so that
                Azure temp-dir resyncs reuse the same DB record.

        Returns:
            ``True`` if the file should be queued for parsing.
        """
        record = self.get_record(key or file_path)
        if record is None:
            return True
        current_hash = compute_md5(file_path)
        if record.file_hash != current_hash:
            return True
        if record.parsing_status in (
            ParsingStatus.PENDING,
            ParsingStatus.IN_PROGRESS,
            ParsingStatus.FAILED,
        ):
            return True
        return False

    def get_pending_files(self) -> list[FileRecord]:
        """Return all records that are not yet completed.

        Returns:
            List of :class:`FileRecord` instances whose status is
            ``pending``, ``in_progress``, or ``failed``.
        """
        rows = self.conn.execute(
            "SELECT * FROM file_state WHERE parsing_status != ?",
            (ParsingStatus.COMPLETED.value,),
        ).fetchall()
        return [FileRecord(**dict(r)) for r in rows]

    def get_all_records(self) -> list[FileRecord]:
        """Return every record in the state database.

        Returns:
            List of all :class:`FileRecord` instances.
        """
        rows = self.conn.execute("SELECT * FROM file_state").fetchall()
        return [FileRecord(**dict(r)) for r in rows]

    # ── mutations ───────────────────────────────────────────────────────────

    def upsert(self, file_path: str | Path, file_hash: str) -> None:
        """Insert a new record or update the hash of an existing one.

        Resets the ``parsing_status`` to ``pending`` whenever the hash
        differs from the stored value.

        Args:
            file_path: Absolute path to the document.
            file_hash: MD5 hex-digest of the current file contents.
        """
        now = _now_iso()
        self.conn.execute(
            """\
            INSERT INTO file_state
                (file_path, file_hash, parsing_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash      = excluded.file_hash,
                parsing_status = CASE
                    WHEN file_state.file_hash != excluded.file_hash
                    THEN 'pending'
                    ELSE file_state.parsing_status
                END,
                updated_at     = excluded.updated_at
            """,
            (str(file_path), file_hash, ParsingStatus.PENDING.value, now, now),
        )
        self.conn.commit()

    def mark_in_progress(self, file_path: str | Path, file_hash: str) -> None:
        """Record that parsing has started for *file_path*.

        If the file does not exist in the database it is inserted first.

        Args:
            file_path: Document being processed.
            file_hash: MD5 hex-digest of the file.
        """
        self.upsert(file_path, file_hash)
        self._update_status(file_path, ParsingStatus.IN_PROGRESS)

    def mark_completed(
        self,
        file_path: str | Path,
        vector_id: str | None = None,
    ) -> None:
        """Record that *file_path* was successfully embedded.

        Args:
            file_path: Completed document.
            vector_id: ChromaDB document ID assigned to the embedding.
        """
        now = _now_iso()
        self.conn.execute(
            """\
            UPDATE file_state
               SET parsing_status = ?, vector_id = ?, error_message = NULL,
                   updated_at = ?
             WHERE file_path = ?
            """,
            (ParsingStatus.COMPLETED.value, vector_id, now, str(file_path)),
        )
        self.conn.commit()

    def mark_failed(self, file_path: str | Path, error: str) -> None:
        """Record that processing of *file_path* failed.

        Args:
            file_path: Document that failed.
            error: Human-readable error description.
        """
        now = _now_iso()
        self.conn.execute(
            """\
            UPDATE file_state
               SET parsing_status = ?, error_message = ?, updated_at = ?
             WHERE file_path = ?
            """,
            (ParsingStatus.FAILED.value, error, now, str(file_path)),
        )
        self.conn.commit()

    # ── internals ───────────────────────────────────────────────────────────

    def _update_status(
        self,
        file_path: str | Path,
        status: ParsingStatus,
    ) -> None:
        """Set the status column for *file_path*.

        Args:
            file_path: Target document.
            status: New parsing status.
        """
        now = _now_iso()
        self.conn.execute(
            "UPDATE file_state SET parsing_status = ?, "
            "updated_at = ? WHERE file_path = ?",
            (status.value, now, str(file_path)),
        )
        self.conn.commit()

    # ── purge ──────────────────────────────────────────────────────────────

    def purge_deleted(
        self,
        known_paths: set[str],
        *,
        prefix: str = "",
    ) -> list[str]:
        """Remove state records for files no longer present on disk.

        When *prefix* is given, only records whose ``file_path`` starts
        with that prefix are considered.  This prevents an AZURE sync
        from purging LOCAL records (and vice-versa).

        Args:
            known_paths: Set of state-DB keys currently on disk.
            prefix: Only consider records starting with this string.

        Returns:
            List of file paths that were removed from the state DB.
        """
        if prefix:
            rows = self.conn.execute(
                "SELECT file_path FROM file_state WHERE file_path LIKE ?",
                (prefix + "%",),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT file_path FROM file_state").fetchall()
        stale = [
            row["file_path"] for row in rows if row["file_path"] not in known_paths
        ]
        if not stale:
            return []
        self.conn.executemany(
            "DELETE FROM file_state WHERE file_path = ?",
            [(p,) for p in stale],
        )
        self.conn.commit()
        logger.info("Purged %d stale records from state DB", len(stale))
        return stale

    # ── stats ───────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, int]:
        """Return a mapping of status → count for a quick health check.

        Returns:
            Dict like ``{"pending": 3, "completed": 42, ...}``.
        """
        rows = self.conn.execute(
            "SELECT parsing_status, COUNT(*) AS cnt "
            "FROM file_state GROUP BY parsing_status"
        ).fetchall()
        return {row["parsing_status"]: row["cnt"] for row in rows}
