"""FastAPI application — endpoints for ``/sync`` and ``/query``.

Provides a production API for triggering incremental ingestion and
querying the RAG system with hybrid retrieval and citations.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import threading
import uuid as _uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingestion.config import settings
from ingestion.pipeline import IngestionStats
from ingestion.state_tracker import StateTracker
from vector_db.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LocalVaultRAG",
    description="Privacy-first enterprise RAG — 100% local inference",
    version="0.1.0",
)

# ── Lazy singletons ─────────────────────────────────────────────────────────

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Lazily initialise the HybridRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ── Request / Response schemas ──────────────────────────────────────────────


class SyncRequest(BaseModel):
    """Request body for the /sync endpoint.

    Attributes:
        source_dir: Directory path containing documents to ingest.
    """

    source_dir: str = "data/"


class SyncResponse(BaseModel):
    """Response from the /sync endpoint.

    Attributes:
        status: Overall status of the sync operation.
        job_id: Unique identifier for the background sync job.
        stats: Detailed ingestion statistics (populated when complete).
    """

    status: str
    job_id: str = ""
    stats: IngestionStats | None = None


class StatsResponse(BaseModel):
    """Live ingestion progress from the state DB.

    Attributes:
        pending: Files not yet processed.
        in_progress: Files currently being parsed.
        completed: Successfully ingested files.
        failed: Files that errored during ingestion.
        total: Sum of all statuses.
        progress_pct: Percentage of non-pending files completed.
    """

    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    total: int = 0
    progress_pct: float = 0.0


# ── Background job store ──────────────────────────────────────────────────

_sync_jobs: dict[str, str] = {}  # job_id -> progress file path
_sync_lock = threading.Lock()


class QueryRequest(BaseModel):
    """Request body for the /query endpoint.

    Attributes:
        question: Natural language query.
        top_k: Number of results to return.
    """

    question: str
    top_k: int = 10


class CitationOut(BaseModel):
    """Citation in the API response.

    Attributes:
        filename: Source document name.
        page: Page number (if available).
        snippet: First 200 characters of the matched text.
    """

    filename: str
    page: int | None = None
    snippet: str = ""


class QueryResultOut(BaseModel):
    """A single query result in the API response.

    Attributes:
        text: Full text of the retrieved chunk.
        score: Relevance score.
        citation: Source citation.
    """

    text: str
    score: float
    citation: CitationOut


class QueryResponse(BaseModel):
    """Response from the /query endpoint.

    Attributes:
        question: The original question.
        results: Ranked retrieval results with citations.
        answer: LLM-generated answer (when available).
    """

    question: str
    results: list[QueryResultOut]
    answer: str = ""


# ── Sync worker (runs in a subprocess — its own GIL) ─────────────────────


def _sync_worker(source_dir: str, job_id: str, progress_file: str) -> None:
    """Ingestion worker that runs in a separate process.

    Writes progress snapshots to a JSON file so the API server
    (which has its own GIL) can serve poll requests without blocking.
    """
    from ingestion.pipeline import IngestionPipeline

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
        )
        pipeline.close()
        _write("completed", stats)
    except Exception as exc:
        logger.error("Sync worker %s failed: %s", job_id, exc)
        _write(f"failed: {exc}", None)


def _wait_and_refresh(
    proc: mp.Process,
    job_id: str,
    progress_file: str,
) -> None:
    """Wait for sync process to exit, then refresh retriever index."""
    proc.join()

    # Detect subprocess crash (OOM kill, segfault, etc.)
    if proc.exitcode and proc.exitcode != 0:
        logger.error(
            "Sync worker %s crashed with exit code %d",
            job_id,
            proc.exitcode,
        )
        data = {
            "status": f"failed: process killed (exit code {proc.exitcode})",
            "job_id": job_id,
        }
        tmp = progress_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, progress_file)
        return

    try:
        _get_retriever().refresh_index()
        logger.info("BM25 index refreshed after sync %s", job_id)
    except Exception as exc:
        logger.error("Index refresh failed after sync %s: %s", job_id, exc)


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.post("/sync", response_model=SyncResponse)
def sync_documents(request: SyncRequest) -> SyncResponse:
    """Trigger incremental document ingestion in the background.

    Returns immediately with a job_id. Poll ``GET /sync/{job_id}``
    to track progress.

    Args:
        request: SyncRequest with source directory path.

    Returns:
        SyncResponse with job_id and initial status.
    """
    source = Path(request.source_dir)
    if not source.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Source directory not found: {request.source_dir}",
        )

    job_id = _uuid.uuid4().hex[:12]
    progress_file = str(settings.chroma_persist_path.parent / f".sync_{job_id}.json")
    with _sync_lock:
        _sync_jobs[job_id] = progress_file

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_sync_worker,
        args=(str(source), job_id, progress_file),
    )
    proc.start()

    # Background thread waits for process exit then refreshes index
    threading.Thread(
        target=_wait_and_refresh,
        args=(proc, job_id, progress_file),
        daemon=True,
    ).start()

    return SyncResponse(status="running", job_id=job_id)


@app.get("/sync/{job_id}", response_model=SyncResponse)
async def sync_status(job_id: str) -> SyncResponse:
    """Check the status of a background sync job.

    Args:
        job_id: The job identifier returned by POST /sync.

    Returns:
        SyncResponse with current status and stats (when complete).
    """
    with _sync_lock:
        progress_file = _sync_jobs.get(job_id)
    if progress_file is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    try:
        raw = Path(progress_file).read_text()
        data = json.loads(raw)
        stats_data = data.get("stats")
        stats = IngestionStats(**stats_data) if stats_data else None
        return SyncResponse(
            status=data["status"],
            job_id=job_id,
            stats=stats,
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return SyncResponse(status="running", job_id=job_id)


@app.get("/stats", response_model=StatsResponse)
def ingestion_stats() -> StatsResponse:
    """Return live ingestion progress from the state DB.

    Reads counts directly from SQLite so results are real-time even
    while a background sync job is running.

    Returns:
        StatsResponse with per-status counts and overall progress %.
    """
    tracker = StateTracker(settings.state_db_path)
    counts = tracker.summary()
    total = sum(counts.values())
    completed = counts.get("completed", 0)
    progress_pct = round(completed / total * 100, 1) if total else 0.0
    return StatsResponse(
        pending=counts.get("pending", 0),
        in_progress=counts.get("in_progress", 0),
        completed=completed,
        failed=counts.get("failed", 0),
        total=total,
        progress_pct=progress_pct,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """Query the RAG system with hybrid retrieval and citations.

    Runs BM25 + semantic search, reranks with a cross-encoder,
    and optionally generates an LLM answer from the top context.

    Args:
        request: QueryRequest with question and top_k.

    Returns:
        QueryResponse with ranked results and citations.
    """
    retriever = _get_retriever()
    results = retriever.query(request.question, top_k=request.top_k)

    result_items = [
        QueryResultOut(
            text=r.text,
            score=r.score,
            citation=CitationOut(
                filename=r.citation.filename,
                page=r.citation.page,
                snippet=r.citation.snippet,
            ),
        )
        for r in results
    ]

    # ── Optional LLM answer generation ──────────────────────────────
    answer = ""
    if results:
        # Send only the top 5 chunks to the LLM to keep context small
        # enough for CPU inference.  All 10 are still shown as citations.
        answer = _generate_answer(request.question, results[:5])

    return QueryResponse(
        question=request.question,
        results=result_items,
        answer=answer,
    )


def _generate_answer(question: str, results: list[RetrievalResult]) -> str:
    """Generate an answer using Ollama from retrieved context.

    Args:
        question: The user's question.
        results: Retrieved context chunks.

    Returns:
        LLM-generated answer string.
    """
    context = "\n\n---\n\n".join(
        f"[Source: {r.citation.filename}, Page {r.citation.page}]\n{r.text}"
        for r in results
    )

    prompt = (
        "You are a research assistant. Answer the question based ONLY on "
        "the provided context. Rules:\n"
        "- If the context contains a list, enumeration, or bibliography, "
        "reproduce ALL items from ALL context chunks — never truncate.\n"
        "- Merge information from multiple chunks of the same document.\n"
        "- If the context does not contain enough information, say so.\n"
        "- Cite the source filename and page for each claim.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    try:
        import httpx

        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model_dev,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 1024,
                },
            },
            timeout=180.0,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as exc:
        logger.warning("LLM generation failed: %s", exc)
        return f"(LLM unavailable: {exc})"


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Dict with service status and document count.
    """
    try:
        retriever = _get_retriever()
        doc_count = retriever._collection.count()
        return {"status": "healthy", "documents": doc_count}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}
