"""Ingestion pipeline orchestrator.

Discovers files, checks them against the state DB for changes, parses
new/modified documents via concurrent workers, and commits chunks to
the vector store through the BatchCommitter.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.batch import BatchCommitter
from src.config import Settings
from src.embeddings import EmbeddingService
from src.models import FileStatus, IngestionStats, ParseResult
from src.parsers import parse_file
from src.state_db import StateDB, compute_md5
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# File extensions accepted by the pipeline (union of all parser extensions).
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
    ".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html",
}


def discover_files(directory: Path) -> list[Path]:
    """Recursively discover supported files in a directory.

    Args:
        directory: Root directory to scan.

    Returns:
        Sorted list of file paths with supported extensions.
    """
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(set(files))


def _parse_file_worker(file_path: str, chunk_size: int) -> ParseResult:
    """Worker function for ProcessPoolExecutor (must be top-level picklable).

    Args:
        file_path: String path to the file (Path can't be pickled across
            processes on all platforms).
        chunk_size: Target chunk size in characters.

    Returns:
        ParseResult from the parser.
    """
    return parse_file(Path(file_path), chunk_size=chunk_size)


class IngestionPipeline:
    """Orchestrates the full document ingestion workflow.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._state_db = StateDB(settings)
        self._embedding_service = EmbeddingService(settings)
        self._vector_store = VectorStore(settings, self._embedding_service)
        self._committer = BatchCommitter(
            self._vector_store,
            batch_size=settings.batch_size,
            gc_interval=settings.batch_gc_interval,
        )

    def run(
        self,
        source_dir: Path,
        chunk_size: int = 1000,
    ) -> IngestionStats:
        """Execute the ingestion pipeline on a directory of documents.

        Steps:
            1. Discover files with supported extensions.
            2. Filter out unchanged files (MD5 check via state DB).
            3. Parse changed/new files using concurrent workers.
            4. Commit chunks to the vector store through BatchCommitter.
            5. Update state DB with results.

        Args:
            source_dir: Directory containing documents to ingest.
            chunk_size: Approximate characters per text chunk.

        Returns:
            IngestionStats summarising the run.
        """
        stats = IngestionStats()

        all_files = discover_files(source_dir)
        stats.total_files = len(all_files)
        logger.info("Discovered %d supported files in %s", len(all_files), source_dir)

        # ── Filter unchanged ────────────────────────────────────────────
        files_to_process: list[Path] = []
        for fp in all_files:
            if self._state_db.is_unchanged(fp):
                stats.skipped_unchanged += 1
            else:
                files_to_process.append(fp)

        logger.info(
            "%d files to process (%d skipped as unchanged)",
            len(files_to_process),
            stats.skipped_unchanged,
        )

        if not files_to_process:
            return stats

        # ── Mark files as PROCESSING ────────────────────────────────────
        for fp in files_to_process:
            from src.models import TrackedFile

            md5 = compute_md5(fp)
            self._state_db.upsert(
                TrackedFile(path=fp, md5_hash=md5, status=FileStatus.PROCESSING)
            )

        # ── Parse concurrently ──────────────────────────────────────────
        max_workers = min(self._settings.max_workers, len(files_to_process))
        results: dict[str, ParseResult] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_parse_file_worker, str(fp), chunk_size): fp
                for fp in files_to_process
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    result = future.result()
                    results[str(fp)] = result
                except Exception as exc:
                    logger.error("Worker exception for %s: %s", fp, exc)
                    results[str(fp)] = ParseResult(
                        source_path=fp,
                        success=False,
                        error_message=str(exc),
                    )

        # ── Commit results ──────────────────────────────────────────────
        for fp in files_to_process:
            result = results[str(fp)]
            if result.success and result.chunks:
                self._committer.add_many(result.chunks)
                self._state_db.mark_status(
                    fp,
                    FileStatus.COMPLETED,
                    chunk_count=len(result.chunks),
                )
                stats.processed += 1
                stats.total_chunks += len(result.chunks)
            else:
                self._state_db.mark_status(
                    fp,
                    FileStatus.FAILED,
                    error_message=result.error_message or "Unknown error",
                )
                stats.failed += 1

        self._committer.finalize()

        logger.info(
            "Ingestion complete: %d processed, %d failed, %d chunks committed",
            stats.processed,
            stats.failed,
            stats.total_chunks,
        )
        return stats

    def close(self) -> None:
        """Release resources held by the pipeline."""
        self._state_db.close()
