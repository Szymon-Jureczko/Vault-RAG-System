"""Ingestion pipeline orchestrator with parallel parsing and batch committing.

Discovers files in ``data/``, checks each against the SQLite state tracker,
parses new/modified documents via ``ProcessPoolExecutor`` (8 workers max),
and commits chunks to ChromaDB through a memory-safe batcher.

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
from ingestion.parser import SUPPORTED_EXTENSIONS, ParserResult, parse_file
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

    def add(self, chunk_id: str, text: str, metadata: dict) -> None:
        """Buffer a single chunk for insertion.

        Args:
            chunk_id: Unique chunk identifier.
            text: Text content of the chunk.
            metadata: Source metadata dict.
        """
        self._buffer.append(
            {"id": chunk_id, "text": text, "metadata": metadata}
        )
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
        max_workers: Max parallel parsing workers.
        batch_size: Chunks per ChromaDB commit batch.
        chunk_size: Target characters per text chunk.
    """

    def __init__(
        self,
        tracker: StateTracker | None = None,
        chroma_path: Path | None = None,
        max_workers: int | None = None,
        batch_size: int = 100,
        chunk_size: int = 1000,
    ) -> None:
        self._tracker = tracker or StateTracker()
        self._tracker.init_db()
        self._chroma_path = chroma_path or settings.chroma_persist_path
        self._max_workers = max_workers or settings.max_workers
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
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            import numpy as np

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
                    texts, padding=True, truncation=True,
                    max_length=512, return_tensors="np",
                )
                outputs = ort_model(**encoded)
                # Mean pooling over token embeddings
                mask = encoded["attention_mask"]
                embeddings = (
                    outputs.last_hidden_state * mask[..., np.newaxis]
                ).sum(axis=1) / mask.sum(axis=-1, keepdims=True)
                # Normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
                return embeddings.tolist()

        except Exception as exc:
            logger.info("ONNX unavailable (%s), using PyTorch", exc)
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(settings.embedding_model)

            def embed_fn(texts: list[str]) -> list[list[float]]:
                return model.encode(
                    texts, show_progress_bar=False
                ).tolist()

        return collection, embed_fn

    def run(self, source_dir: Path) -> IngestionStats:
        """Execute the full ingestion pipeline.

        Steps:
            1. Discover files with supported extensions.
            2. Filter unchanged files via state tracker MD5 check.
            3. Parse changed/new files with ProcessPoolExecutor.
            4. Batch-commit chunks to ChromaDB.
            5. Update state tracker with results.

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

        # ── Mark in-progress ────────────────────────────────────────────
        for fp in to_process:
            md5 = compute_md5(fp)
            self._tracker.mark_in_progress(str(fp), md5)

        # ── Parse in parallel ───────────────────────────────────────────
        workers = min(self._max_workers, len(to_process))
        results: dict[str, ParserResult] = {}

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, str(fp), self._chunk_size): fp
                for fp in to_process
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    results[str(fp)] = future.result()
                except Exception as exc:
                    logger.error("Worker error for %s: %s", fp, exc)
                    results[str(fp)] = ParserResult(
                        file_path=str(fp), success=False, error=str(exc)
                    )

        # ── Commit to ChromaDB ──────────────────────────────────────────
        collection, embed_fn = self._init_chroma()
        batcher = BatchCommitter(
            collection, embed_fn, batch_size=self._batch_size
        )

        for fp in to_process:
            result = results[str(fp)]
            if result.success and result.chunks:
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
