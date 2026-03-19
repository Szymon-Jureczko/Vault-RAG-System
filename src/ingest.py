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
from src.models import FileStatus, IngestionStats, ParseResult, TrackedFile
from src.parsers import IMAGE_EXTENSIONS, _split_text, parse_file
from src.state_db import StateDB, compute_md5
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# File extensions accepted by the pipeline (union of all parser extensions).
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".json",
    ".xml",
    ".html",
}

# Parser names that produce OCR output and whose results should be cached.
_OCR_PARSER_NAMES: frozenset[str] = frozenset({"rapidocr", "docling"})


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
            2. Compute MD5 for each file; filter out unchanged files (single
               pass — eliminates the previous double-compute).
            3. Check OCR result cache for image files; skip executor for hits.
            4. Parse remaining files using concurrent workers.
            5. Store OCR results in cache for future retries.
            6. Commit chunks to the vector store through BatchCommitter.
            7. Update state DB with results.

        Args:
            source_dir: Directory containing documents to ingest.
            chunk_size: Approximate characters per text chunk.

        Returns:
            IngestionStats summarising the run.
        """
        stats = IngestionStats()

        all_files = discover_files(source_dir)
        stats.total_files = len(all_files)
        logger.info(
            "Discovered %d supported files in %s", len(all_files), source_dir
        )

        # ── Single-pass MD5 filter ───────────────────────────────────────
        # Compute MD5 once per file and reuse it for both the unchanged check
        # and the PROCESSING upsert.  Previously compute_md5() was called
        # twice: once inside is_unchanged() and again when marking PROCESSING.
        files_to_process: list[Path] = []
        md5_map: dict[str, str] = {}  # str(fp) -> md5 (files_to_process only)

        for fp in all_files:
            existing = self._state_db.get_file(fp)
            md5 = compute_md5(fp)
            if existing is not None and existing.md5_hash == md5:
                stats.skipped_unchanged += 1
            else:
                files_to_process.append(fp)
                md5_map[str(fp)] = md5

        logger.info(
            "%d files to process (%d skipped as unchanged)",
            len(files_to_process),
            stats.skipped_unchanged,
        )

        if not files_to_process:
            return stats

        # ── Mark files as PROCESSING ────────────────────────────────────
        for fp in files_to_process:
            self._state_db.upsert(
                TrackedFile(
                    path=fp,
                    md5_hash=md5_map[str(fp)],
                    status=FileStatus.PROCESSING,
                )
            )

        # ── OCR cache lookup (image files only) ─────────────────────────
        # If a previously OCR'd image is being retried (e.g. after
        # reset_failed()), retrieve the cached text instead of re-running OCR.
        results: dict[str, ParseResult] = {}
        files_needing_parse: list[Path] = []

        for fp in files_to_process:
            if fp.suffix.lower() in IMAGE_EXTENSIONS:
                cached_text = self._state_db.get_ocr_cache(md5_map[str(fp)])
                if cached_text is not None:
                    logger.info("OCR cache hit for %s", fp)
                    chunks = _split_text(cached_text, fp, chunk_size, "cache")
                    results[str(fp)] = ParseResult(
                        source_path=fp,
                        chunks=chunks,
                        parser_used="cache",
                        page_count=1,
                        success=True,
                    )
                    continue
            files_needing_parse.append(fp)

        # ── Parse concurrently ──────────────────────────────────────────
        max_workers = min(self._settings.max_workers, len(files_needing_parse))

        if files_needing_parse:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _parse_file_worker, str(fp), chunk_size
                    ): fp
                    for fp in files_needing_parse
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

        # ── Store OCR results in cache ───────────────────────────────────
        # Cache text for image files that were successfully OCR'd so that
        # future retries (after reset_failed()) skip the OCR engine entirely.
        for fp in files_needing_parse:
            result = results.get(str(fp))
            if (
                result
                and result.success
                and result.parser_used in _OCR_PARSER_NAMES
                and fp.suffix.lower() in IMAGE_EXTENSIONS
            ):
                combined_text = "\n".join(c.content for c in result.chunks)
                self._state_db.put_ocr_cache(
                    md5_map[str(fp)], combined_text, result.parser_used
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
