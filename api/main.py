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
import re
import threading
import uuid as _uuid
from pathlib import Path
from typing import Iterator, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ingestion.config import settings
from ingestion.pipeline import IngestionStats
from ingestion.state_tracker import StateTracker
from ingestion.sync_worker import sync_worker as _sync_target
from vector_db.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)

# Strips the "[Source: filename]\n" prefix that parser.py embeds in every
# chunk at ingestion time, so it is not duplicated in the LLM prompt.
_SOURCE_PREFIX_RE = re.compile(r"^\[Source:[^\]]*\]\n")

app = FastAPI(
    title="LocalVaultRAG",
    description="Privacy-first enterprise RAG — 100% local inference",
    version="0.1.0",
)

# ── Lazy singletons ─────────────────────────────────────────────────────────

_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def _get_retriever() -> HybridRetriever:
    """Lazily initialise the HybridRetriever singleton (thread-safe)."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = HybridRetriever()
    return _retriever


# ── Request / Response schemas ──────────────────────────────────────────────


class SyncRequest(BaseModel):
    """Request body for the /sync endpoint.

    Attributes:
        source_dir: Directory path containing documents to ingest (LOCAL mode).
        ingestion_source: Where to read documents from — ``'LOCAL'`` or ``'AZURE'``.
        azure_storage_connection_string: Azure storage account connection string.
            Required when ``ingestion_source`` is ``'AZURE'``.
        azure_container_name: Name of the Azure blob container to ingest from.
            Required when ``ingestion_source`` is ``'AZURE'``.
    """

    source_dir: str = "data/"
    ingestion_source: Literal["LOCAL", "AZURE"] = "LOCAL"
    azure_storage_connection_string: str = ""
    azure_container_name: str = ""


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
        ingestion_source: Optional source filter (``'local'`` or ``'azure'``).
            When set, only chunks from that ingestion source are returned.
            Defaults to ``None`` (no filtering).
    """

    question: str
    top_k: int = 10
    ingestion_source: str | None = None


class CitationOut(BaseModel):
    """Citation in the API response.

    Attributes:
        filename: Source document name.
        page: Page number (if available).
        snippet: Most relevant ~500 characters of the matched text.
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
# The worker lives in ingestion.sync_worker so that the subprocess
# (spawned via mp.get_context("spawn")) only imports lightweight
# ingestion modules — NOT api.main which pulls in PyTorch, ChromaDB,
# FastAPI, etc. (~300 MB saved).


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
    if request.ingestion_source == "LOCAL" and not source.is_dir():
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
        target=_sync_target,
        args=(
            str(source),
            job_id,
            progress_file,
            request.ingestion_source,
            request.azure_storage_connection_string,
            request.azure_container_name,
        ),
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
    results = retriever.query(
        request.question,
        top_k=request.top_k,
        source_filter=request.ingestion_source,
    )

    # Filter to chunks the cross-encoder considers relevant (score > 0).
    # Negative-score chunks are not useful as citations or LLM context.
    relevant = [r for r in results if r.score > 0] or results[:1]

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
        for r in relevant
    ]

    # ── Query Router: structured (SQL) vs unstructured (RAG) ────────
    answer = ""
    query_type = _classify_query(request.question)
    logger.info("Router classified %r as %s", request.question[:80], query_type)

    if query_type == "structured":
        answer, scored_tables = _handle_structured_query(request.question)
        if answer and scored_tables:
            result_items = [
                QueryResultOut(
                    text=c["text"],
                    score=c["score"],
                    citation=CitationOut(
                        filename=c["citation"]["filename"],
                        page=c["citation"]["page"],
                        snippet=c["citation"]["snippet"],
                    ),
                )
                for c in _build_table_citations(scored_tables)
            ]

    # Fallback to RAG if structured agent returned nothing, or if unstructured
    if not answer and relevant:
        top_score = relevant[0].score
        llm_chunks = [r for r in relevant[:5] if r.score >= top_score * 0.85]
        answer = _generate_answer(request.question, llm_chunks or relevant[:1])

    return QueryResponse(
        question=request.question,
        results=result_items,
        answer=answer,
    )


@app.post("/query/stream")
def query_documents_stream(request: QueryRequest) -> StreamingResponse:
    """Stream a RAG response: citations as JSON, then LLM tokens via SSE.

    Emits Server-Sent Events:
    - ``event: citations`` — JSON array of retrieval results (sent once).
    - ``event: token``     — a single LLM token in ``data:``.
    - ``event: done``      — signals end of stream.

    Args:
        request: QueryRequest with question and top_k.

    Returns:
        StreamingResponse with ``text/event-stream`` content type.
    """
    retriever = _get_retriever()
    results = retriever.query(
        request.question,
        top_k=request.top_k,
        source_filter=request.ingestion_source,
    )

    relevant = [r for r in results if r.score > 0] or results[:1]

    result_items = [
        {
            "text": r.text,
            "score": r.score,
            "citation": {
                "filename": r.citation.filename,
                "page": r.citation.page,
                "snippet": r.citation.snippet,
            },
        }
        for r in relevant
    ]

    top_score = relevant[0].score if relevant else 0
    llm_chunks = [r for r in relevant[:5] if r.score >= top_score * 0.85]

    # ── Query Router: structured (SQL) vs unstructured (RAG) ────────
    query_type = _classify_query(request.question)
    logger.info("Router classified %r as %s", request.question[:80], query_type)

    sql_stream: Iterator[str] | None = None
    if query_type == "structured":
        stream_result = _handle_structured_query_stream(request.question)
        if stream_result is not None:
            sql_stream, scored_tables = stream_result
            if scored_tables:
                result_items = _build_table_citations(scored_tables)

    def event_generator() -> Iterator[str]:
        # 1. Send citations immediately so the UI can show them
        yield f"event: citations\ndata: {json.dumps(result_items)}\n\n"

        # 2. Stream LLM tokens — SQL agent or RAG fallback
        if sql_stream is not None:
            for token in sql_stream:
                escaped = json.dumps(token)
                yield f"event: token\ndata: {escaped}\n\n"
        else:
            for token in _stream_answer(request.question, llm_chunks or relevant[:1]):
                escaped = json.dumps(token)
                yield f"event: token\ndata: {escaped}\n\n"

        # 3. Signal completion
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Query Router + Text-to-SQL Data Agent ─────────────────────────────────


def _classify_query(question: str) -> str:
    """Classify a question as 'structured' or 'unstructured' using Ollama.

    Uses a minimal prompt with aggressive context/predict limits for
    fast classification (~1 s on CPU).

    Args:
        question: The user's question.

    Returns:
        ``'structured'`` or ``'unstructured'``.
    """
    payload = {
        "model": settings.ollama_model_dev,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the user's question. Reply with exactly one word.\n"
                    "Reply STRUCTURED if the question asks for a number, "
                    "calculation, statistic, count, sum, average, comparison "
                    "of values, ranking, filtering, or any answer that requires "
                    "data from a table or spreadsheet.\n"
                    "Reply UNSTRUCTURED for everything else."
                ),
            },
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {"num_ctx": 256, "num_predict": 4, "temperature": 0.0},
    }
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=15.0,
        )
        resp.raise_for_status()
        answer = resp.json().get("message", {}).get("content", "").strip().upper()
        if answer.startswith("S"):
            return "structured"
    except Exception as exc:
        logger.warning(
            "Router classification failed: %s — defaulting to unstructured", exc
        )
    return "unstructured"


def _filter_relevant_tables(
    question: str, top_k: int | None = None
) -> list[tuple[str, float]] | None:
    """Pre-filter tables by BM25 relevance to the user's question.

    Args:
        question: The user's natural-language question.
        top_k: Number of tables to return.  Defaults to
            ``settings.structured_top_k``.

    Returns:
        List of ``(table_name, normalised_score)`` tuples sorted by
        relevance, or ``None`` if no tabular data exists.
    """
    import sqlite3

    from rank_bm25 import BM25Okapi

    if top_k is None:
        top_k = settings.structured_top_k

    db_path = settings.tabular_db_path
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT table_name, source_file, sheet_name, description, "
            "columns_json FROM _tabular_meta"
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()

    if not rows:
        return None

    if len(rows) <= top_k:
        return [(row["table_name"], 1.0) for row in rows]

    # Build one "document" per table from its metadata
    docs: list[list[str]] = []
    table_names: list[str] = []
    for row in rows:
        cols = json.loads(row["columns_json"])
        col_names = " ".join(cols.keys())
        text = " ".join(
            filter(
                None,
                [
                    row["description"],
                    row["source_file"],
                    row["sheet_name"],
                    col_names,
                ],
            )
        )
        docs.append(re.findall(r"\w+", text.lower()))
        table_names.append(row["table_name"])

    bm25 = BM25Okapi(docs)
    query_tokens = re.findall(r"\w+", question.lower())
    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]

    # Normalise scores to 0.0–1.0
    max_score = scores[top_indices[0]] if top_indices else 1.0
    if max_score <= 0:
        max_score = 1.0

    result = [(table_names[i], scores[i] / max_score) for i in top_indices]
    logger.info(
        "Table pre-filter: %d/%d tables selected (top scores: %s)",
        len(result),
        len(table_names),
        ", ".join(f"{scores[i]:.2f}" for i in top_indices[:3]),
    )
    return result


def _get_tabular_schema(
    scored_tables: list[tuple[str, float]] | None = None,
) -> str | None:
    """Read the tabular metadata table and build a schema prompt block.

    Args:
        scored_tables: Optional list of ``(table_name, score)`` tuples
            from :func:`_filter_relevant_tables`.  When provided only
            these tables are included.

    Returns:
        Schema description string, or ``None`` if no tabular data exists.
    """
    import json
    import sqlite3

    db_path = settings.tabular_db_path
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if scored_tables:
            names = [t[0] for t in scored_tables]
            placeholders = ",".join("?" for _ in names)
            rows = conn.execute(
                "SELECT table_name, source_file, sheet_name, description, "
                "columns_json, row_count FROM _tabular_meta "
                f"WHERE table_name IN ({placeholders}) "
                "ORDER BY source_file, table_name",
                names,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT table_name, source_file, sheet_name, description, "
                "columns_json, row_count FROM _tabular_meta "
                "ORDER BY source_file, table_name"
            ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return None

    if not rows:
        conn.close()
        return None

    parts = []
    for row in rows:
        cols = json.loads(row["columns_json"])
        col_str = ", ".join(f"{name} ({dtype})" for name, dtype in cols.items())
        sheet_info = f", sheet: {row['sheet_name']}" if row["sheet_name"] else ""
        desc = row["description"]
        desc_info = (
            f"\n  Description: {desc} — ALL rows belong to this dataset."
            if desc
            else ""
        )
        block = (
            f"Table: {row['table_name']} "
            f"(source: {row['source_file']}{sheet_info}, "
            f"{row['row_count']} rows)"
            f"{desc_info}\n"
            f"  Columns: {col_str}"
        )
        # Include sample rows so the LLM understands actual data values
        try:
            sample = conn.execute(
                f"SELECT * FROM \"{row['table_name']}\" LIMIT 3"
            ).fetchall()
            if sample:
                col_names = list(cols.keys())
                header = " | ".join(col_names)
                sample_lines = [
                    " | ".join(str(s[c]) for c in col_names) for s in sample
                ]
                block += f"\n  Sample rows:\n  {header}\n  " + "\n  ".join(sample_lines)
        except Exception:
            pass
        parts.append(block)
    conn.close()
    return "\n\n".join(parts)


def _generate_sql(question: str, schema: str) -> str:
    """Ask the LLM to generate a SQL SELECT for the given question.

    Args:
        question: The user's natural-language question.
        schema: Schema description of available tables.

    Returns:
        Raw SQL string from the LLM.
    """
    payload = {
        "model": settings.ollama_model_dev,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a SQL generator. Given the SQLite schema below, "
                    "write a single SELECT statement that answers the user's "
                    "question. Output ONLY the SQL query, nothing else. "
                    "Do not wrap it in markdown code fences.\n\n"
                    f"Schema:\n{schema}\n\n"
                    "Rules:\n"
                    "- Use only tables and columns shown above.\n"
                    "- SQLite syntax only.\n"
                    "- Always wrap column and table names in double quotes.\n"
                    "- IMPORTANT: Each table already contains ALL rows for "
                    "its dataset. Do NOT add a WHERE clause to filter by a "
                    "company name, dataset name, or source name. Only use "
                    "WHERE if the user asks to filter by a value that "
                    "actually appears in the sample rows (e.g. a specific "
                    "employee name, region, or date).\n"
                    "- SQLite has no MEDIAN(). To compute a median, use:\n"
                    "  SELECT val FROM tbl ORDER BY val "
                    "LIMIT 1 OFFSET (SELECT COUNT(*) FROM tbl) / 2;\n"
                    "- Never use INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.\n"
                    "- If the question cannot be answered, reply: "
                    "SELECT 'NOT_ANSWERABLE' AS error;"
                ),
            },
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {
            "num_ctx": settings.llm_num_ctx,
            "num_predict": 256,
            "temperature": 0.0,
        },
    }
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json=payload,
        timeout=settings.llm_timeout,
    )
    resp.raise_for_status()
    raw = resp.json().get("message", {}).get("content", "").strip()
    # Strip markdown code fences if the model includes them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def _execute_sql_safely(sql: str) -> list[dict]:
    """Execute a read-only SQL query against the tabular database.

    Three safety layers: ``PRAGMA query_only``, SELECT-only validation,
    and a 5-second progress-handler timeout.

    Args:
        sql: A SELECT statement.

    Returns:
        List of row dicts.

    Raises:
        ValueError: If the SQL is not a SELECT or is otherwise unsafe.
    """
    import sqlite3
    import time

    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned.upper().startswith("SELECT"):
        raise ValueError(f"Only SELECT statements are allowed, got: {cleaned[:50]}")
    if ";" in cleaned:
        raise ValueError("Multi-statement SQL is not allowed")

    conn = sqlite3.connect(str(settings.tabular_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")

    deadline = time.monotonic() + 5.0

    def _progress_check() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_progress_check, 1000)

    try:
        cursor = conn.execute(cleaned)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


_SQL_RESULT_SYSTEM_PROMPT = (
    "You are a data reporting assistant. The user asked a question "
    "and a SQL query was run to answer it. Summarize the results "
    "in a clear, natural-language response.\n\n"
    "RULES:\n"
    "1. Report the numbers directly from the query results.\n"
    "2. Be concise but complete.\n"
    "3. ALWAYS answer in the SAME LANGUAGE as the question.\n"
    "4. Do NOT show the SQL query.\n"
    "5. Do NOT add citations or source references."
)


def _format_sql_result(question: str, sql: str, rows: list[dict]) -> str:
    """Format SQL result rows into a natural-language answer via LLM.

    Args:
        question: Original user question.
        sql: The SQL that was executed.
        rows: Result rows as dicts.

    Returns:
        Natural language answer string.
    """
    result_text = json.dumps(rows[:50], default=str, ensure_ascii=False)
    if len(rows) > 50:
        result_text += f"\n... ({len(rows)} total rows, showing first 50)"

    payload = {
        "model": settings.ollama_model_dev,
        "messages": [
            {"role": "system", "content": _SQL_RESULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nQuery results:\n{result_text}",
            },
        ],
        "stream": False,
        "options": {
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict,
            "temperature": settings.llm_temperature,
            "num_thread": settings.llm_num_thread,
        },
    }
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json=payload,
        timeout=settings.llm_timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def _stream_sql_result(question: str, sql: str, rows: list[dict]) -> Iterator[str]:
    """Stream a natural-language answer from SQL results, token by token."""
    result_text = json.dumps(rows[:50], default=str, ensure_ascii=False)
    if len(rows) > 50:
        result_text += f"\n... ({len(rows)} total rows, showing first 50)"

    payload = {
        "model": settings.ollama_model_dev,
        "messages": [
            {"role": "system", "content": _SQL_RESULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nQuery results:\n{result_text}",
            },
        ],
        "stream": True,
        "options": {
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict,
            "temperature": settings.llm_temperature,
            "num_thread": settings.llm_num_thread,
        },
    }
    try:
        with httpx.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=httpx.Timeout(
                connect=10.0, read=settings.llm_timeout, write=10.0, pool=10.0
            ),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
    except Exception as exc:
        logger.warning("SQL result streaming failed: %s", exc)
        yield f"(LLM unavailable: {exc})"


def _sql_result_is_empty(rows: list[dict]) -> bool:
    """Check if SQL result is effectively empty (no rows, or all values NULL)."""
    if not rows:
        return True
    if len(rows) == 1:
        vals = list(rows[0].values())
        if all(v is None for v in vals):
            return True
    return False


def _build_table_citations(
    scored_tables: list[tuple[str, float]],
) -> list[dict]:
    """Build citation dicts for structured query source tables.

    Produces the same ``{text, score, citation: {filename, page, snippet}}``
    shape used by unstructured retrieval so the UI renders them identically.
    """
    import sqlite3

    if not scored_tables:
        return []

    db_path = settings.tabular_db_path
    if not db_path.exists():
        return []

    score_map = dict(scored_tables)
    names = [t[0] for t in scored_tables]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in names)
        rows = conn.execute(
            "SELECT table_name, source_file, sheet_name, description, "
            "columns_json FROM _tabular_meta "
            f"WHERE table_name IN ({placeholders})",
            names,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    # Deduplicate by source_file — keep the sheet with the highest score
    best_per_file: dict[str, dict] = {}
    for row in rows:
        fname = row["source_file"]
        tname = row["table_name"]
        score = score_map.get(tname, 0.0)
        cols = json.loads(row["columns_json"])
        col_list = ", ".join(f"{c} ({t})" for c, t in cols.items())
        sheet = row["sheet_name"]
        desc = row["description"] or ""

        snippet_parts = []
        if sheet:
            snippet_parts.append(f"Sheet: {sheet}")
        if desc:
            snippet_parts.append(desc)
        snippet_parts.append(f"Columns: {col_list}")
        snippet = "\n".join(snippet_parts)

        if fname not in best_per_file or score > best_per_file[fname]["score"]:
            best_per_file[fname] = {
                "text": desc,
                "score": round(score, 3),
                "citation": {
                    "filename": fname,
                    "page": None,
                    "snippet": snippet,
                },
            }

    result = sorted(best_per_file.values(), key=lambda x: x["score"], reverse=True)
    return result


def _handle_structured_query(
    question: str,
) -> tuple[str, list[tuple[str, float]] | None]:
    """Execute the full Text-to-SQL data agent pipeline (non-streaming).

    Returns:
        ``(answer, scored_tables)`` — answer is ``""`` on failure (signals
        RAG fallback).
    """
    relevant_tables = _filter_relevant_tables(question)
    schema = _get_tabular_schema(scored_tables=relevant_tables)
    if not schema:
        return "", relevant_tables

    try:
        sql = _generate_sql(question, schema)
        logger.info("Data agent SQL: %s", sql)
        rows = _execute_sql_safely(sql)

        if rows and "error" in rows[0] and rows[0]["error"] == "NOT_ANSWERABLE":
            return "", relevant_tables

        # Retry once if the result is empty — the model likely added a
        # spurious WHERE clause.
        if _sql_result_is_empty(rows):
            logger.info("Data agent got empty result, retrying without WHERE hint")
            retry_q = (
                f"{question}\n\n"
                f"(Hint: A previous query returned no results. "
                f"Do NOT filter by company or dataset name — "
                f"the table already contains all the data.)"
            )
            sql = _generate_sql(retry_q, schema)
            logger.info("Data agent retry SQL: %s", sql)
            rows = _execute_sql_safely(sql)

        if _sql_result_is_empty(rows):
            return "", relevant_tables

        return _format_sql_result(question, sql, rows), relevant_tables
    except Exception as exc:
        logger.warning("Structured query agent failed: %s", exc)
        return "", relevant_tables


def _handle_structured_query_stream(
    question: str,
) -> tuple[Iterator[str], list[tuple[str, float]] | None] | None:
    """Execute the Text-to-SQL data agent pipeline (streaming).

    Returns:
        ``(token_iterator, scored_tables)`` or ``None`` to signal fallback.
    """
    relevant_tables = _filter_relevant_tables(question)
    schema = _get_tabular_schema(scored_tables=relevant_tables)
    if not schema:
        return None

    try:
        sql = _generate_sql(question, schema)
        logger.info("Data agent SQL: %s", sql)
        rows = _execute_sql_safely(sql)

        if rows and "error" in rows[0] and rows[0]["error"] == "NOT_ANSWERABLE":
            return None

        # Retry once if the result is empty
        if _sql_result_is_empty(rows):
            logger.info("Data agent got empty result, retrying without WHERE hint")
            retry_q = (
                f"{question}\n\n"
                f"(Hint: A previous query returned no results. "
                f"Do NOT filter by company or dataset name — "
                f"the table already contains all the data.)"
            )
            sql = _generate_sql(retry_q, schema)
            logger.info("Data agent retry SQL: %s", sql)
            rows = _execute_sql_safely(sql)

        if _sql_result_is_empty(rows):
            return None

        return _stream_sql_result(question, sql, rows), relevant_tables
    except Exception as exc:
        logger.warning("Structured query agent failed: %s", exc)
        return None


# ── LLM prompt builder (unstructured RAG path) ───────────────────────────


def _build_llm_payload(
    question: str, results: list[RetrievalResult], *, stream: bool = False
) -> dict:
    """Build the Ollama /api/chat JSON payload for unstructured RAG.

    Args:
        question: The user's question.
        results: Retrieved context chunks.
        stream: Whether to request token-by-token streaming.

    Returns:
        dict ready for ``httpx.post(..., json=payload)``.
    """
    context = "\n\n---\n\n".join(
        "[Source: {fn}{pg}]\n{txt}".format(
            fn=r.citation.filename,
            pg=f", Page {r.citation.page}" if r.citation.page is not None else "",
            txt=_SOURCE_PREFIX_RE.sub("", r.text),
        )
        for r in results
    )

    system_prompt = (
        "You are a study assistant that answers ONLY from the provided "
        "context. You have NO knowledge of your own — treat the context as "
        "your only source of truth.\n\n"
        "STRICT RULES:\n"
        "1. Use ONLY facts that are explicitly stated in the context. Never "
        "add information from outside the context, even if you know it.\n"
        "2. If the context does not contain enough information to answer "
        'the question, say: "The provided documents do not contain enough '
        'information to answer this question."\n'
        "3. ALWAYS answer in the SAME LANGUAGE as the question.\n"
        "4. SYNTHESIZE information across ALL provided sources into one "
        "coherent answer — do not just parrot or summarise a single "
        "source. Merge overlapping ideas, resolve redundancies, and "
        "combine unique details from each source. If two sources say "
        "the same thing in different words, state the idea ONCE in your "
        "own words. Start with a clear definition, then explain key "
        "details, categories, applications, or examples. Aim for a "
        "comprehensive response that adds value beyond any single "
        "source alone.\n"
        "5. Do NOT add citations or source references — they are handled "
        "separately."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    return {
        "model": settings.ollama_model_dev,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": stream,
        "options": {
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict,
            "temperature": settings.llm_temperature,
            "num_thread": settings.llm_num_thread,
        },
    }


def _generate_answer(question: str, results: list[RetrievalResult]) -> str:
    """Generate an answer using Ollama from retrieved context.

    Args:
        question: The user's question.
        results: Retrieved context chunks.

    Returns:
        LLM-generated answer string.
    """
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=_build_llm_payload(question, results, stream=False),
            timeout=settings.llm_timeout,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    except Exception as exc:
        logger.warning("LLM generation failed: %s", exc)
        return f"(LLM unavailable: {exc})"


def _stream_answer(question: str, results: list[RetrievalResult]) -> Iterator[str]:
    """Stream an LLM answer token-by-token from Ollama.

    Yields:
        Individual text tokens as they arrive from the model.
    """
    try:
        with httpx.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json=_build_llm_payload(question, results, stream=True),
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.llm_timeout,
                write=10.0,
                pool=10.0,
            ),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
    except Exception as exc:
        logger.warning("LLM streaming failed: %s", exc)
        yield f"(LLM unavailable: {exc})"


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
