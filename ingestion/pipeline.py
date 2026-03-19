"""Thin adapter exposing ``IngestionPipeline`` under the ``ingestion`` package.

``api.main`` imports from this module; the implementation lives in
``src.ingest``.  This adapter constructs the pipeline via
``src.config.get_settings()`` so callers need no constructor arguments.
"""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings as _get_settings
from src.ingest import IngestionPipeline as _SrcPipeline
from src.models import IngestionStats  # re-export for api.main

__all__ = ["IngestionPipeline", "IngestionStats"]


class IngestionPipeline:
    """Zero-argument wrapper around ``src.ingest.IngestionPipeline``.

    Constructs the underlying pipeline with the application-wide
    ``src.config.settings`` singleton on first use.
    """

    def __init__(self) -> None:
        self._pipeline = _SrcPipeline(_get_settings())

    def run(self, source_dir: Path, chunk_size: int = 1000) -> IngestionStats:
        """Delegate to the underlying pipeline.

        Args:
            source_dir: Directory containing documents to ingest.
            chunk_size: Approximate characters per text chunk.

        Returns:
            IngestionStats summarising the completed run.
        """
        return self._pipeline.run(source_dir, chunk_size=chunk_size)
