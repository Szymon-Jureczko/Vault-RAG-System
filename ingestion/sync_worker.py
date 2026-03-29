"""Sync worker — runs in a **subprocess** spawned by the API server.

This module is deliberately kept **import-light**.  The ``spawn``
multiprocessing context resolves the target function by importing
the module that defines it.  Keeping the worker here (instead of in
``api.main``) avoids pulling FastAPI, httpx, PyTorch, ChromaDB, and
the rest of the API stack into the child process — saving ~300 MB
of RAM that is critical for OCR.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def sync_worker(
    source_dir: str,
    job_id: str,
    progress_file: str,
    ingestion_source: str = "LOCAL",
    azure_storage_connection_string: str = "",
    azure_container_name: str = "",
) -> None:
    """Ingestion worker that runs in a separate process.

    Writes progress snapshots to a JSON file so the API server
    (which has its own GIL) can serve poll requests without blocking.

    Env vars for the Azure source are set before any ingestion imports
    so the ``Settings()`` singleton in the subprocess picks them up.
    """
    os.environ["INGESTION_SOURCE"] = ingestion_source
    if azure_storage_connection_string:
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = azure_storage_connection_string
    if azure_container_name:
        os.environ["AZURE_CONTAINER_NAME"] = azure_container_name

    from ingestion.pipeline import IngestionPipeline, IngestionStats

    def _write(status: str, stats: IngestionStats | None) -> None:
        data: dict = {"status": status, "job_id": job_id}
        if stats is not None:
            data["stats"] = stats.model_dump()
        tmp = progress_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, progress_file)

    try:
        pipeline = IngestionPipeline()
        stats = pipeline.run(
            Path(source_dir),
            on_progress=lambda s: _write("running", s),
            ingestion_source=ingestion_source,
            azure_connection_string=azure_storage_connection_string,
            azure_container_name=azure_container_name,
        )
        pipeline.close()
        _write("completed", stats)
    except Exception as exc:
        logger.error("Sync worker %s failed: %s", job_id, exc)
        _write(f"failed: {exc}", None)
