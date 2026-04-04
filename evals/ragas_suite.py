"""RAGAS evaluation suite — Faithfulness, Context Precision, Answer Relevancy.

Generates a golden dataset of questions from sample documents, evaluates the
RAG pipeline against it, and produces a benchmark report. Also measures
Files-Per-Minute ingestion throughput and Query Latency.

Usage::

    python -m evals.ragas_suite --data-dir data/test_docs/ \\
        --sample-size 50 --questions 20

    # Use Azure blob container as document source:
    python -m evals.ragas_suite --source azure \\
        --sample-size 50 --questions 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from ingestion.config import settings

logger = logging.getLogger(__name__)

# Suppress Azure SDK's verbose HTTP transport logs
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

API_BASE = "http://localhost:8000"


# ── Models ──────────────────────────────────────────────────────────────────


class GoldenQuestion(BaseModel):
    """A single question in the golden evaluation dataset.

    Attributes:
        question: The evaluation question.
        ground_truth: Expected answer (human-written).
        source_file: Document the question was derived from.
    """

    question: str
    ground_truth: str
    source_file: str


class EvalResult(BaseModel):
    """Evaluation scores for a single question.

    Attributes:
        question: The asked question.
        answer: RAG-generated answer.
        contexts: Retrieved context snippets.
        ground_truth: Expected answer.
        faithfulness: RAGAS faithfulness score (0-1).
        context_precision: RAGAS context precision score (0-1).
        answer_relevancy: RAGAS answer relevancy score (0-1).
    """

    question: str
    answer: str = ""
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str = ""
    faithfulness: float = 0.0
    context_precision: float = 0.0
    answer_relevancy: float = 0.0


class BenchmarkResult(BaseModel):
    """Ingestion and query latency benchmarks.

    Attributes:
        total_files: Number of files ingested.
        ingestion_seconds: Total ingestion time.
        files_per_minute: Throughput metric.
        avg_query_latency_ms: Average query response time.
        queries_tested: Number of queries benchmarked.
    """

    total_files: int = 0
    ingestion_seconds: float = 0.0
    files_per_minute: float = 0.0
    avg_query_latency_ms: float = 0.0
    queries_tested: int = 0


# ── Golden dataset generation ──────────────────────────────────────────────


def generate_golden_dataset(
    data_dir: Path,
    sample_size: int = 50,
    num_questions: int = 20,
) -> list[GoldenQuestion]:
    """Generate evaluation questions from sample documents via Ollama.

    Selects up to ``sample_size`` documents from ``data_dir``, extracts
    text snippets, and prompts the LLM to generate Q&A pairs.

    Args:
        data_dir: Directory containing source documents.
        sample_size: Maximum documents to sample.
        num_questions: Total questions to generate.

    Returns:
        List of GoldenQuestion instances.
    """
    from ingestion.parser import SUPPORTED_EXTENSIONS, parse_file

    files = sorted(
        f
        for f in data_dir.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
    )[:sample_size]

    if not files:
        logger.warning("No supported files found in %s", data_dir)
        return []

    # Extract text from each file
    file_texts: list[tuple[str, str]] = []
    for fp in files:
        result = parse_file(
            fp, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        if result.success and result.chunks:
            text = result.chunks[0].text[:1500]
            file_texts.append((fp.name, text))

    questions: list[GoldenQuestion] = []

    # Use OpenAI for question generation when API key is available
    openai_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    use_openai = bool(openai_key)
    if use_openai:
        logger.info("Using OpenAI for golden question generation")

    for filename, text in file_texts:
        if len(questions) >= num_questions:
            break

        prompt = (
            "Generate a factual question and answer based on this document "
            "excerpt. The question must be complete, well-formed, and in the "
            "same language as the excerpt. Return ONLY valid JSON: "
            '{"question": "...", "answer": "..."}\n\n'
            f"Document: {filename}\n\nExcerpt:\n{text}\n\nJSON:"
        )

        try:
            if use_openai:
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
            else:
                resp = httpx.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_model_dev,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                qa = json.loads(raw[start:end])
                questions.append(
                    GoldenQuestion(
                        question=qa["question"],
                        ground_truth=qa["answer"],
                        source_file=filename,
                    )
                )
        except Exception as exc:
            logger.warning(
                "Failed to generate question from %s: %s",
                filename,
                exc,
            )

    logger.info(
        "Generated %d golden questions from %d files",
        len(questions),
        len(file_texts),
    )
    return questions


# ── RAGAS evaluation ───────────────────────────────────────────────────────


def evaluate_ragas(questions: list[GoldenQuestion]) -> list[EvalResult]:
    """Evaluate the RAG pipeline using RAGAS metrics.

    For each golden question, queries the /query endpoint and computes
    Faithfulness, Context Precision, and Answer Relevancy.

    Args:
        questions: List of GoldenQuestion instances.

    Returns:
        List of EvalResult instances.
    """
    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from openai import AsyncOpenAI
        from ragas import evaluate
        from ragas.llms import llm_factory
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.run_config import RunConfig
    except ImportError:
        logger.error("RAGAS not installed. Run: pip install ragas datasets")
        return _fallback_evaluate(questions)

    # Prefer OpenAI for the judge LLM when an API key is available;
    # fall back to local Ollama otherwise.
    openai_key = settings.openai_api_key
    if openai_key:
        logger.info("Using OpenAI judge LLM (gpt-4o-mini)")
        client = AsyncOpenAI(api_key=openai_key)
        judge_llm = llm_factory("gpt-4o-mini", provider="openai", client=client)
    else:
        logger.info("Using Ollama judge LLM (%s)", settings.ollama_model_eval)
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
        )
        judge_llm = llm_factory(
            settings.ollama_model_eval,
            provider="openai",
            client=client,
        )
    # Use a multilingual model for answer relevancy so non-English questions
    # (Polish, etc.) are scored correctly instead of being penalised by the
    # English-only all-MiniLM-L6-v2 model.
    judge_embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
    ]

    # Collect RAG responses
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for q in questions:
        try:
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"question": q.question, "top_k": 8},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()

            rows["question"].append(q.question)
            rows["answer"].append(data.get("answer", ""))
            rows["contexts"].append(
                [
                    r["text"]
                    for r in data.get("results", [])
                    if r.get("text", "").strip()
                ]
            )
            rows["ground_truth"].append(q.ground_truth)
        except Exception as exc:
            logger.warning("Query failed for '%s': %s", q.question, exc)
            rows["question"].append(q.question)
            rows["answer"].append("")
            rows["contexts"].append([])
            rows["ground_truth"].append(q.ground_truth)

    dataset = Dataset.from_dict(rows)

    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=600,
            max_retries=1,
            max_wait=5,
            max_workers=1,
        ),
        raise_exceptions=False,
    )

    eval_results = []
    faith_scores = result["faithfulness"]
    cp_scores = result["context_precision"]
    ar_scores = result["answer_relevancy"]
    for i, q in enumerate(questions):
        eval_results.append(
            EvalResult(
                question=q.question,
                answer=rows["answer"][i],
                contexts=rows["contexts"][i],
                ground_truth=q.ground_truth,
                faithfulness=(
                    faith_scores[i]
                    if faith_scores[i] is not None and not math.isnan(faith_scores[i])
                    else 0.0
                ),
                context_precision=(
                    cp_scores[i]
                    if cp_scores[i] is not None and not math.isnan(cp_scores[i])
                    else 0.0
                ),
                answer_relevancy=(
                    ar_scores[i]
                    if ar_scores[i] is not None and not math.isnan(ar_scores[i])
                    else 0.0
                ),
            )
        )

    return eval_results


def _fallback_evaluate(questions: list[GoldenQuestion]) -> list[EvalResult]:
    """Simple fallback evaluation when RAGAS is not installed.

    Queries the API and reports raw results without RAGAS scoring.

    Args:
        questions: List of GoldenQuestion instances.

    Returns:
        List of EvalResult with zero scores.
    """
    results = []
    for q in questions:
        try:
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"question": q.question, "top_k": 8},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            results.append(
                EvalResult(
                    question=q.question,
                    answer=data.get("answer", ""),
                    contexts=[r["text"] for r in data.get("results", [])],
                    ground_truth=q.ground_truth,
                )
            )
        except Exception as exc:
            logger.warning("Fallback query failed: %s", exc)
            results.append(EvalResult(question=q.question, ground_truth=q.ground_truth))
    return results


# ── Benchmarking ───────────────────────────────────────────────────────────


def benchmark_ingestion(data_dir: str = "data/test_docs/") -> BenchmarkResult:
    """Measure ingestion throughput (Files-Per-Minute).

    Args:
        data_dir: Source directory for ingestion.

    Returns:
        BenchmarkResult with timing data.
    """
    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{API_BASE}/sync",
            json={"source_dir": data_dir},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("job_id")
        stats = data.get("stats")

        # POST /sync is async; poll until complete if we got a job_id.
        if job_id and stats is None:
            for _ in range(120):  # up to 10 minutes
                time.sleep(5)
                poll = httpx.get(f"{API_BASE}/sync/{job_id}", timeout=30)
                poll.raise_for_status()
                poll_data = poll.json()
                status = poll_data.get("status", "")
                if status == "completed" or status.startswith("failed"):
                    stats = poll_data.get("stats")
                    break
    except Exception as exc:
        logger.error("Ingestion benchmark failed: %s", exc)
        return BenchmarkResult()

    elapsed = time.perf_counter() - start
    if stats is None:
        logger.error("Ingestion benchmark: no stats returned")
        return BenchmarkResult()
    total = stats.get("total_discovered", 0)
    fpm = (total / elapsed) * 60 if elapsed > 0 else 0

    return BenchmarkResult(
        total_files=total,
        ingestion_seconds=round(elapsed, 2),
        files_per_minute=round(fpm, 2),
    )


def benchmark_queries(questions: list[GoldenQuestion]) -> BenchmarkResult:
    """Measure average query latency.

    Args:
        questions: Questions to benchmark.

    Returns:
        BenchmarkResult with latency data.
    """
    latencies: list[float] = []
    for q in questions:
        start = time.perf_counter()
        try:
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"question": q.question, "top_k": 8},
                timeout=120,
            )
            resp.raise_for_status()
            latencies.append((time.perf_counter() - start) * 1000)
        except Exception:
            pass

    avg = sum(latencies) / len(latencies) if latencies else 0
    return BenchmarkResult(
        avg_query_latency_ms=round(avg, 2),
        queries_tested=len(latencies),
    )


def _warmup_server(max_wait: int = 90) -> None:
    """Send a dummy query to wake the server before evaluation starts.

    The first real request often times out because the embedding model loads
    lazily on the first call.  This preflight query absorbs that cold-start
    cost so that no golden question is lost to a timeout.

    Args:
        max_wait: Seconds to keep retrying before giving up.
    """
    logger.info("Warming up server (max %ds)...", max_wait)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"question": "warmup", "top_k": 1},
                timeout=min(30.0, deadline - time.time()),
            )
            if resp.status_code == 200:
                logger.info("Server warmup complete")
                return
        except Exception:
            pass
        time.sleep(3)
    logger.warning("Server warmup timed out after %ds — proceeding anyway", max_wait)


# ── CLI entrypoint ─────────────────────────────────────────────────────────


def _resolve_azure_data_dir() -> Path:
    """Download blobs from Azure to the local staging path and return it.

    Validates that the required Azure settings are configured, creates the
    staging directory if absent, then delegates to ``download_azure_blobs``.

    Returns:
        Path to the local staging directory containing the downloaded blobs.

    Raises:
        SystemExit: If Azure credentials or container name are not configured.
    """
    from ingestion.pipeline import download_azure_blobs

    if not settings.azure_storage_connection_string:
        logger.error("AZURE_STORAGE_CONNECTION_STRING must be set for --source azure")
        raise SystemExit(1)
    if not settings.azure_container_name:
        logger.error("AZURE_CONTAINER_NAME must be set for --source azure")
        raise SystemExit(1)

    staging = settings.azure_staging_path
    staging.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading blobs from container '%s' to %s ...",
        settings.azure_container_name,
        staging,
    )
    count = download_azure_blobs(staging)
    logger.info("Downloaded %d blob(s) from Azure", count)
    return staging


def main() -> None:
    """Run the full evaluation and benchmarking suite."""
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Suite")
    parser.add_argument(
        "--source",
        choices=["local", "azure"],
        default="local",
        help="Document source: 'local' (default) or 'azure' (downloads from blob storage)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/test_docs/",
        help="Document source directory (local source only)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Max docs to sample",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=20,
        help="Golden questions to generate",
    )
    parser.add_argument(
        "--output",
        default="evals/results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--use-existing-dataset",
        action="store_true",
        help="Skip question generation and use the existing evals/golden_dataset.json as-is",
    )
    args = parser.parse_args()

    golden_path = Path("evals/golden_dataset.json")

    if args.use_existing_dataset:
        if not golden_path.exists():
            logger.error("--use-existing-dataset set but %s not found", golden_path)
            return
        golden = [GoldenQuestion(**q) for q in json.loads(golden_path.read_text())]
        logger.info("Loaded %d questions from existing %s", len(golden), golden_path)
        data_dir = Path(args.data_dir)  # still needed for ingestion benchmark
    else:
        if args.source == "azure":
            data_dir = _resolve_azure_data_dir()
        else:
            data_dir = Path(args.data_dir)

        # Step 1: Generate golden dataset
        logger.info("Generating golden dataset...")
        golden = generate_golden_dataset(
            data_dir,
            args.sample_size,
            args.questions,
        )

        if not golden:
            logger.error("No golden questions generated. Check data directory.")
            return

        # Save golden dataset
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps([q.model_dump() for q in golden], indent=2))
        logger.info("Golden dataset saved to %s", golden_path)

    # Step 2: RAGAS evaluation
    _warmup_server()
    logger.info("Running RAGAS evaluation...")
    eval_results = evaluate_ragas(golden)

    # Step 3: Benchmarks
    logger.info("Running ingestion benchmark...")
    ingest_bench = benchmark_ingestion(args.data_dir)

    logger.info("Running query latency benchmark...")
    query_bench = benchmark_queries(golden)

    # Step 4: Report
    report = {
        "golden_dataset_size": len(golden),
        "evaluation": [r.model_dump() for r in eval_results],
        "benchmarks": {
            "ingestion": ingest_bench.model_dump(),
            "query": query_bench.model_dump(),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Results saved to %s", output_path)

    # Summary
    if eval_results:
        n = len(eval_results)
        avg_faith = sum(r.faithfulness for r in eval_results) / n
        avg_prec = sum(r.context_precision for r in eval_results) / n
        avg_rel = sum(r.answer_relevancy for r in eval_results) / n
        print(f"\n{'='*50}")
        print("RAGAS Evaluation Summary")
        print(f"{'='*50}")
        print(f"  Faithfulness:       {avg_faith:.3f}")
        print(f"  Context Precision:  {avg_prec:.3f}")
        print(f"  Answer Relevancy:   {avg_rel:.3f}")

    print(f"\n  Ingestion: {ingest_bench.files_per_minute:.1f} files/min")
    print(f"  Query Latency: {query_bench.avg_query_latency_ms:.1f} ms avg")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
