"""Batch processing utilities for OOM-safe document ingestion.

The ``BatchCommitter`` accumulates processed chunks and periodically
flushes them to the vector store, running ``gc.collect()`` at configured
intervals to keep memory under control on 16 GB machines.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

from src.models import DocumentChunk

if TYPE_CHECKING:
    from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BatchCommitter:
    """Accumulates document chunks and flushes to the vector store in batches.

    Designed to prevent OOM on constrained hardware (16 GB RAM) by
    committing in fixed-size batches and explicitly clearing caches.

    Args:
        vector_store: The VectorStore instance to commit chunks into.
        batch_size: Number of chunks to accumulate before flushing.
        gc_interval: Run ``gc.collect()`` every N committed chunks.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        batch_size: int = 50,
        gc_interval: int = 500,
    ) -> None:
        self._store = vector_store
        self._batch_size = batch_size
        self._gc_interval = gc_interval
        self._buffer: list[DocumentChunk] = []
        self._total_committed: int = 0

    @property
    def total_committed(self) -> int:
        """Total number of chunks committed so far."""
        return self._total_committed

    def add(self, chunk: DocumentChunk) -> None:
        """Add a chunk to the buffer, flushing if the batch is full.

        Args:
            chunk: A document chunk to queue for insertion.
        """
        self._buffer.append(chunk)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def add_many(self, chunks: list[DocumentChunk]) -> None:
        """Add multiple chunks, flushing as needed.

        Args:
            chunks: List of document chunks to queue.
        """
        for chunk in chunks:
            self.add(chunk)

    def flush(self) -> int:
        """Commit all buffered chunks to the vector store.

        Returns:
            Number of chunks flushed in this call.
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)
        self._store.add_chunks(self._buffer)
        self._total_committed += count
        self._buffer.clear()

        logger.debug(
            "Flushed %d chunks (total committed: %d)",
            count,
            self._total_committed,
        )

        if self._total_committed % self._gc_interval < count:
            logger.info(
                "GC checkpoint at %d committed chunks", self._total_committed
            )
            gc.collect()

        return count

    def finalize(self) -> int:
        """Flush remaining chunks and run a final GC pass.

        Returns:
            Number of chunks flushed in the final call.
        """
        flushed = self.flush()
        gc.collect()
        logger.info(
            "BatchCommitter finalized — %d total chunks committed.",
            self._total_committed,
        )
        return flushed
