"""Ingestion pipeline orchestrator with two-phase parallel parsing.

Discovers files in ``data/``, checks each against the SQLite state tracker,
then runs two phases: fast formats (PDF, DOCX, XLSX, EML) with ``max_workers``
and OCR/image formats (PNG, JPEG, etc.) with the smaller ``ocr_workers`` pool.
Each phase commits to ChromaDB immediately so ``/stats`` shows incremental
progress rather than staying at 0% for the whole run.

Usage::

    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    stats = pipeline.run(Path("data/"))
    print(stats)
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

from ingestion.config import settings
from ingestion.parser import (
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    ParserResult,
    parse_file,
)
from ingestion.state_tracker import StateTracker, compute_md5

logger = logging.getLogger(__name__)


class IngestionStats(BaseModel):
    """Summary statistics for a pipeline run.

    Attributes:
        total_discovered: Total files found with supported extensions.
        skipped_unchanged: Files skipped (MD5 unchanged in state DB).
        processed: Files successfully parsed and committed.
        failed: Files that encountered errors.
        total_chunks: Total text chunks produced.
    """

    total_discovered: int = 0
    skipped_unchanged: int = 0
    processed: int = 0
    failed: int = 0
    total_chunks: int = 0


class BatchCommitter:
    """Commits chunks to ChromaDB in batches to prevent OOM on 16 GB RAM.

    Runs ``gc.collect()`` every ``gc_interval`` committed chunks and clears
    the internal buffer every ``batch_size`` chunks.

    Args:
        collection: ChromaDB collection to insert into.
        embedding_fn: Callable that embeds a list of texts.
        batch_size: Chunks per batch before flushing to ChromaDB.
        gc_interval: Run gc.collect() every N total committed chunks.
    """

    def __init__(
        self,
        collection,
        embedding_fn,
        batch_size: int = 100,
        gc_interval: int = 500,
    ) -> None:
        self._collection = collection
        self._embed = embedding_fn
        self._batch_size = batch_size
        self._gc_interval = gc_interval
        self._buffer: list[dict] = []
        self._total_committed: int = 0

    def delete_for_file(self, filename: str) -> None:
        """Delete all existing chunks for a file before re-ingesting.

        This prevents stale chunks from a previous ingestion from
        persisting alongside new ones when a file's content changes.

        Args:
            filename: Bare filename (e.g. ``"report.docx"``).
        """
        try:
            self._collection.delete(where={"filename": filename})
        except Exception as exc:
            logger.warning("Could not delete old chunks for %s: %s", filename, exc)

    def add(self, chunk_id: str, text: str, metadata: dict) -> None:
        """Buffer a single chunk for insertion.

        Args:
            chunk_id: Unique chunk identifier.
            text: Text content of the chunk.
            metadata: Source metadata dict.
        """
        self._buffer.append({"id": chunk_id, "text": text, "metadata": metadata})
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> int:
        """Commit all buffered chunks to ChromaDB.

        Returns:
            Number of chunks flushed.
        """
        if not self._buffer:
            return 0

        ids = [c["id"] for c in self._buffer]
        texts = [c["text"] for c in self._buffer]
        metas = [c["metadata"] for c in self._buffer]
        embeddings = self._embed(texts)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )

        count = len(self._buffer)
        self._total_committed += count
        self._buffer.clear()

        if self._total_committed % self._gc_interval < count:
            logger.info("GC checkpoint at %d chunks", self._total_committed)
            gc.collect()

        return count

    def finalize(self) -> int:
        """Flush remaining chunks and run a final GC pass.

        Returns:
            Number of chunks flushed.
        """
        flushed = self.flush()
        gc.collect()
        logger.info(
            "Batcher finalized — %d total chunks committed",
            self._total_committed,
        )
        return flushed

    @property
    def total_committed(self) -> int:
        """Total number of chunks committed so far."""
        return self._total_committed


def _worker(file_path: str, chunk_size: int) -> ParserResult:
    """Top-level picklable worker for ProcessPoolExecutor.

    Args:
        file_path: String path to the document.
        chunk_size: Target chunk size in characters.

    Returns:
        ParserResult from the parser.
    """
    return parse_file(Path(file_path), chunk_size=chunk_size)


def discover_files(directory: Path) -> list[Path]:
    """Recursively discover files with supported extensions.

    Args:
        directory: Root directory to scan.

    Returns:
        Sorted list of supported file paths.
    """
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(set(files))


class IngestionPipeline:
    """Orchestrates document discovery, parsing, and vector storage.

    Args:
        tracker: StateTracker instance (defaults to global settings).
        chroma_path: Path to ChromaDB persistent storage.
        max_workers: Max parallel workers for non-OCR parsers.
        ocr_workers: Max parallel workers for OCR/image parsers.
        batch_size: Chunks per ChromaDB commit batch.
        chunk_size: Target characters per text chunk.
    """

    def __init__(
        self,
        tracker: StateTracker | None = None,
        chroma_path: Path | None = None,
        max_workers: int | None = None,
        ocr_workers: int | None = None,
        batch_size: int = 100,
        chunk_size: int = 1000,
    ) -> None:
        self._tracker = tracker or StateTracker()
        self._tracker.init_db()
        self._chroma_path = chroma_path or settings.chroma_persist_path
        self._max_workers = max_workers or settings.max_workers
        self._ocr_workers = ocr_workers or settings.ocr_workers
        self._batch_size = batch_size
        self._chunk_size = chunk_size

    def _init_chroma(self):
        """Initialise ChromaDB collection and embedding function.

        Tries ONNX runtime for ~3x faster CPU inference, falls back
        to standard PyTorch SentenceTransformer.

        Returns:
            Tuple of (collection, embedding_fn).
        """
        import chromadb

        self._chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self._chroma_path))
        collection = client.get_or_create_collection(
            name="localvault_docs",
            metadata={"hnsw:space": "cosine"},
        )

        # Try ONNX backend first for faster CPU inference
        try:
            import numpy as np
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                f"sentence-transformers/{settings.embedding_model}"
            )
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                f"sentence-transformers/{settings.embedding_model}",
                export=True,
            )
            logger.info("Using ONNX runtime for embeddings")

            def embed_fn(texts: list[str]) -> list[list[float]]:
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )
                outputs = ort_model(**encoded)
                # Mean pooling over token embeddings
                mask = encoded["attention_mask"]
                embeddings = (outputs.last_hidden_state * mask[..., np.newaxis]).sum(
                    axis=1
                ) / mask.sum(axis=-1, keepdims=True)
                # Normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
                return embeddings.tolist()

        except Exception as exc:
            logger.info("ONNX unavailable (%s), using PyTorch", exc)
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(settings.embedding_model)

            def embed_fn(texts: list[str]) -> list[list[float]]:
                return model.encode(texts, show_progress_bar=False).tolist()

        return collection, embed_fn

    def _run_phase(
        self,
        files: list[Path],
        workers: int,
        batcher: "BatchCommitter",
        stats: IngestionStats,
    ) -> None:
        """Parse, embed, and commit one phase of files with per-file state updates.

        Each file is marked ``in_progress`` before its worker starts and
        transitions to ``completed`` or ``failed`` the moment the future
        resolves — giving /stats a live count that updates with every file
        rather than only at the end of the phase.

        Args:
            files:   Files to process in this phase.
            workers: Max parallel worker processes for this phase.
            batcher: Shared BatchCommitter; auto-flushes at batch_size.
            stats:   Mutable IngestionStats updated in place.
        """
        if not files:
            return

        for fp in files:
            md5 = compute_md5(fp)
            self._tracker.mark_in_progress(str(fp), md5)

        actual_workers = min(workers, len(files))
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_worker, str(fp), self._chunk_size): fp for fp in files
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Worker error for %s: %s", fp, exc)
                    self._tracker.mark_failed(str(fp), error=str(exc))
                    stats.failed += 1
                    continue

                if result.success and result.chunks:
                    batcher.delete_for_file(fp.name)
                    for chunk in result.chunks:
                        batcher.add(chunk.chunk_id, chunk.text, chunk.metadata)
                    self._tracker.mark_completed(
                        str(fp), vector_id=result.chunks[0].chunk_id
                    )
                    stats.processed += 1
                    stats.total_chunks += len(result.chunks)
                else:
                    self._tracker.mark_failed(
                        str(fp), error=result.error or "Unknown error"
                    )
                    stats.failed += 1

        batcher.flush()

    def run(self, source_dir: Path) -> IngestionStats:
        """Execute the two-phase ingestion pipeline.

        Phase 1 processes fast formats (DOCX, XLSX, EML, text-layer PDFs)
        with ``max_workers`` and commits before Phase 2 starts, so /stats
        shows incremental progress instead of staying at 0% for the whole run.
        Phase 2 processes OCR/image files with the smaller ``ocr_workers``
        pool to stay within memory limits.

        Args:
            source_dir: Directory containing documents to ingest.

        Returns:
            IngestionStats summarising the run.
        """
        stats = IngestionStats()

        # ── Discover ────────────────────────────────────────────────────
        all_files = discover_files(source_dir)
        stats.total_discovered = len(all_files)
        logger.info("Discovered %d files in %s", len(all_files), source_dir)

        # ── Filter unchanged ────────────────────────────────────────────
        to_process: list[Path] = []
        for fp in all_files:
            if self._tracker.needs_processing(fp):
                to_process.append(fp)
            else:
                stats.skipped_unchanged += 1

        logger.info(
            "%d to process, %d skipped",
            len(to_process),
            stats.skipped_unchanged,
        )
        if not to_process:
            return stats

        # ── Split into fast vs OCR file lists ───────────────────────────
        fast_files = [
            fp for fp in to_process if fp.suffix.lower() not in IMAGE_EXTENSIONS
        ]
        ocr_files = [fp for fp in to_process if fp.suffix.lower() in IMAGE_EXTENSIONS]

        # ── Shared batcher (initialised once, used across both phases) ──
        collection, embed_fn = self._init_chroma()
        batcher = BatchCommitter(collection, embed_fn, batch_size=self._batch_size)

        # ── Phase 1: fast formats ────────────────────────────────────────
        logger.info(
            "Phase 1 start: %d fast files with %d workers",
            len(fast_files),
            self._max_workers,
        )
        self._run_phase(fast_files, self._max_workers, batcher, stats)

        # ── Phase 2: OCR / image files ──────────────────────────────────
        if ocr_files:
            logger.info(
                "Phase 2 start: %d OCR files with %d workers",
                len(ocr_files),
                self._ocr_workers,
            )
            self._run_phase(ocr_files, self._ocr_workers, batcher, stats)

        batcher.finalize()

        logger.info(
            "Pipeline complete: %d processed, %d failed, %d chunks",
            stats.processed,
            stats.failed,
            stats.total_chunks,
        )
        return stats

    def close(self) -> None:
        """Release resources held by the pipeline."""
        self._tracker.close()
