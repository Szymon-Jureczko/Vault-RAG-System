"""FastAPI application — endpoints for ``/sync`` and ``/query``.

Provides a production API for triggering incremental ingestion and
querying the RAG system with hybrid retrieval and citations.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import threading
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingestion.config import settings
from ingestion.pipeline import IngestionPipeline, IngestionStats
from ingestion.state_tracker import StateTracker
from vector_db.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Singletons ───────────────────────────────────────────────────────────────

_retriever: HybridRetriever | None = None
_pipeline: IngestionPipeline | None = None


def _get_retriever() -> HybridRetriever:
    """Return the HybridRetriever singleton (initialised at startup)."""
    if _retriever is None:
        raise RuntimeError("Retriever not initialised — startup may have failed")
    return _retriever


def _get_pipeline() -> IngestionPipeline:
    """Lazily initialise the IngestionPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Eagerly initialise HybridRetriever at startup so BM25 is ready immediately."""
    global _retriever
    logger.info("Initialising HybridRetriever at startup…")
    _retriever = HybridRetriever()
    logger.info("HybridRetriever ready")
    yield


app = FastAPI(
    title="LocalVaultRAG",
    description="Privacy-first enterprise RAG — 100% local inference",
    version="0.1.0",
    lifespan=_lifespan,
)


# ── Request / Response schemas ──────────────────────────────────────────────


class SyncRequest(BaseModel):
    """Request body for the /sync endpoint.

    Attributes:
        source_dir: Directory path containing documents to ingest.
    """

    source_dir: str = "data/test_docs"


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

_sync_jobs: dict[str, SyncResponse] = {}
_sync_lock = threading.Lock()


class QueryRequest(BaseModel):
    """Request body for the /query endpoint.

    Attributes:
        question: Natural language query.
        top_k: Number of results to return.
    """

    question: str
    top_k: int = 5


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
    job = SyncResponse(status="running", job_id=job_id)
    with _sync_lock:
        _sync_jobs[job_id] = job

    def _run_sync() -> None:
        try:
            pipeline = _get_pipeline()
            stats = pipeline.run(source)
            _get_retriever().refresh_index()
            with _sync_lock:
                _sync_jobs[job_id] = SyncResponse(
                    status="completed",
                    job_id=job_id,
                    stats=stats,
                )
        except Exception as exc:
            logger.error("Sync job %s failed: %s", job_id, exc)
            with _sync_lock:
                _sync_jobs[job_id] = SyncResponse(
                    status=f"failed: {exc}",
                    job_id=job_id,
                )

    thread = threading.Thread(target=_run_sync, daemon=True)
    thread.start()

    return job


@app.get("/sync/{job_id}", response_model=SyncResponse)
def sync_status(job_id: str) -> SyncResponse:
    """Check the status of a background sync job.

    Args:
        job_id: The job identifier returned by POST /sync.

    Returns:
        SyncResponse with current status and stats (when complete).
    """
    with _sync_lock:
        job = _sync_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


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
        answer = _generate_answer(request.question, results)

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
    context_parts = []
    for r in results:
        source_label = r.citation.filename
        if r.citation.page is not None:
            source_label += f", Page {r.citation.page}"
        context_parts.append(f"[Source: {source_label}]\n{r.text}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a research assistant. Answer the question using ONLY the "
        "information in the provided context below. "
        "Do NOT fabricate or invent any text, quotes, or facts not present in "
        "the context. Do NOT put words in quotation marks unless you are "
        "copying them verbatim from the context. "
        "Cite the source filename for each claim you make. "
        "If the context does not contain enough information to answer, say so.\n\n"
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
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as exc:
        logger.warning("LLM generation failed: %s", exc)
        return f"(LLM unavailable: {exc})"


@app.get("/health")
def health_check() -> dict:
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
