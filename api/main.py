"""FastAPI application — endpoints for ``/sync`` and ``/query``.

Provides a production API for triggering incremental ingestion and
querying the RAG system with hybrid retrieval and citations.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingestion.config import settings
from ingestion.pipeline import IngestionPipeline, IngestionStats
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
_pipeline: IngestionPipeline | None = None


def _get_retriever() -> HybridRetriever:
    """Lazily initialise the HybridRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_pipeline() -> IngestionPipeline:
    """Lazily initialise the IngestionPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


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
        stats: Detailed ingestion statistics.
    """

    status: str
    stats: IngestionStats


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
    """Trigger incremental document ingestion.

    Scans the source directory, parses new/modified files, and commits
    chunks to the vector store. Unchanged files are skipped.

    Args:
        request: SyncRequest with source directory path.

    Returns:
        SyncResponse with status and ingestion statistics.
    """
    source = Path(request.source_dir)
    if not source.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Source directory not found: {request.source_dir}",
        )

    pipeline = _get_pipeline()
    stats = pipeline.run(source)

    return SyncResponse(status="completed", stats=stats)


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
    context = "\n\n---\n\n".join(
        f"[Source: {r.citation.filename}, Page {r.citation.page}]\n{r.text}"
        for r in results
    )

    prompt = (
        "You are a research assistant. Answer the question based ONLY on "
        "the provided context. If the context does not contain enough "
        "information, say so. Cite the source filename and page for each "
        "claim.\n\n"
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
