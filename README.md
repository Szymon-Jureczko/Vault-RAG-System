# LocalVaultRAG

**Local private enterprise RAG system — 100% locally hosted**

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![LLM Llama 3.2](https://img.shields.io/badge/LLM-Llama%203.2-blueviolet)
![Vector DB ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)

---

## Overview
<img width="1782" height="937" alt="projimg3" src="https://github.com/user-attachments/assets/fa1036cc-d5ed-410e-b90d-44030c68ecf2" />
<img width="1292" height="732" alt="projimg4" src="https://github.com/user-attachments/assets/a84cef4a-6cda-418c-8fc8-0da2677880d7" />

LocalVaultRAG ingests thousands of messy enterprise documents — PDFs, Word files, spreadsheets, emails, and scanned images — indexes them with a hybrid search engine, and answers natural-language questions with cited sources. Everything runs on your machine. No OpenAI keys, no cloud APIs, no data ever leaves your network.

Built for commodity hardware: optimized to run on an Intel i5 with 16 GB RAM.

## Features

- **Multimodal ingestion** — PDF, DOCX, XLSX, EML, images (PNG/JPG/TIFF/BMP), and plain text (TXT, MD, CSV, JSON, XML, HTML)
- **Scanned document OCR** — RapidOCR (ONNX Runtime) with fast gating and text-density probing for PDFs
- **Hybrid retrieval** — BM25 keyword + semantic vector + filename lookup + proper-noun phrase matching, merged via Reciprocal Rank Fusion
- **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 refines top results for precision
- **Citations on every result** — filename, page number, and text snippet
- **Incremental processing** — SQLite state DB with MD5 change detection; never reprocesses unchanged files, resumes from failures
- **Memory-safe on 16 GB RAM** — two-phase pipeline, model swapping (unload embedder for OCR and vice versa), batch commits, `gc.collect()` checkpoints
- **RAGAS evaluation** — automated golden dataset generation, Faithfulness / Context Precision / Answer Relevancy scoring, ingestion throughput and query latency benchmarks
- **Production API + dashboard** — FastAPI REST endpoints and Streamlit researcher dashboard with chat interface

## Architecture

```
Documents on disk (data/)
         |
         v
 ┌─────────────────────────────────────────────────────┐
 │            Ingestion Pipeline                        │
 │  discover_files ──► StateTracker (SQLite + MD5)      │
 │        |                                             │
 │        ▼                                             │
 │  Phase 1: Fast formats    Phase 2: OCR formats       │
 │  (PDF, DOCX, XLSX, EML)   (scanned PDF, images)     │
 │  N workers in parallel     1 worker, mini-batches    │
 │        |                        |                    │
 │        └──────────┬─────────────┘                    │
 │                   ▼                                  │
 │          BatchCommitter ──► ChromaDB (HNSW cosine)   │
 │          all-MiniLM-L6-v2 embeddings (ONNX)          │
 └─────────────────────────────────────────────────────┘
                     |
                     v
 ┌─────────────────────────────────────────────────────┐
 │            Hybrid Retriever                          │
 │  Semantic search (ChromaDB cosine)                   │
 │  + BM25 keyword search (BM25Okapi)                   │
 │  + Filename / proper-noun phrase search              │
 │  ──► Reciprocal Rank Fusion ──► Cross-encoder rerank │
 │  ──► Citations (filename, page, snippet)             │
 └─────────────────────────────────────────────────────┘
                     |
                     v
 ┌─────────────────────────────────────────────────────┐
 │           Answer Generation                          │
 │  Top-5 chunks as context ──► Ollama (Llama 3.2)      │
 │  Research-assistant prompt ──► Cited answer           │
 └─────────────────────────────────────────────────────┘
                     |
                     v
 ┌─────────────────────────────────────────────────────┐
 │     FastAPI :8000         Streamlit :8888             │
 │   REST API endpoints    Researcher dashboard         │
 │   /sync /query /stats   Chat + citations + progress  │
 └─────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| LLM | Ollama + Llama 3.2 | 3b (dev) / 8b (eval), local CPU inference |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, ONNX-accelerated with PyTorch fallback |
| Vector DB | ChromaDB | HNSW index, cosine similarity, persistent disk mode |
| OCR | RapidOCR | ONNX Runtime, replaces Tesseract for 3-10x speed |
| PDF Parsing | PyMuPDF | Native text extraction + 300 DPI rendering for scanned pages |
| Keyword Search | BM25Okapi | Pickled to disk, auto-refreshed after ingestion |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Query-document pair scoring |
| API | FastAPI | Background sync, health checks, query endpoint |
| Dashboard | Streamlit | Chat interface, citation viewer, ingestion progress |
| State Tracking | SQLite | MD5 hashing, incremental processing, resume-on-failure |

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/localvaultrag.git
   cd localvaultrag
   ```

2. **Open in Dev Container:**

   Open the project in VS Code and select **"Reopen in Container"** when prompted (or run the `Dev Containers: Reopen in Container` command). This builds the Python 3.12 container and starts the Ollama sidecar automatically.

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   The defaults work out of the box — Ollama runs as a sidecar service on the Docker network.

4. **Generate test documents** (optional — creates ~1,000 synthetic documents):
   ```bash
   python scripts/generate_test_docs.py
   ```

5. **Start the API server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

6. **Start the Streamlit dashboard** (in a second terminal):
   ```bash
   streamlit run api/streamlit_app.py --server.port 8888
   ```

7. **Trigger ingestion** — click the Sync button in the dashboard, or:
   ```bash
   curl -X POST http://localhost:8000/sync -H "Content-Type: application/json" -d '{"source_dir": "data/"}'
   ```

## API Reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/sync` | `{"source_dir": "data/"}` | Trigger incremental document ingestion (background) |
| `GET` | `/sync/{job_id}` | — | Poll sync job progress |
| `GET` | `/stats` | — | Live ingestion counts from SQLite |
| `POST` | `/query` | `{"question": "...", "top_k": 10}` | Hybrid search + LLM-generated answer with citations |
| `GET` | `/health` | — | Health check with document count |

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the Q3 revenue figures?", "top_k": 5}'
```

Response includes ranked results with citations and an LLM-generated answer:

```json
{
  "question": "What were the Q3 revenue figures?",
  "results": [
    {
      "text": "...",
      "score": 0.87,
      "citation": {
        "filename": "quarterly_report_q3.pdf",
        "page": 4,
        "snippet": "Revenue for Q3 reached $2.4M, a 15% increase..."
      }
    }
  ],
  "answer": "According to quarterly_report_q3.pdf (page 4), revenue for Q3..."
}
```

## Project Structure

```
localvaultrag/
├── api/
│   ├── main.py                 # FastAPI REST server
│   └── streamlit_app.py        # Streamlit researcher dashboard
├── ingestion/
│   ├── config.py               # Pydantic Settings (reads .env)
│   ├── models.py               # FileRecord, ParseResult, ParsingStatus
│   ├── parser.py               # Multimodal parsers (PDF, DOCX, XLSX, EML, OCR, text)
│   ├── pipeline.py             # Two-phase ingestion orchestrator + BatchCommitter
│   └── state_tracker.py        # SQLite state DB with MD5 change detection
├── vector_db/
│   └── retriever.py            # Hybrid retriever (BM25 + semantic + reranking)
├── evals/
│   └── ragas_suite.py          # RAGAS evaluation + benchmarking CLI
├── scripts/
│   └── generate_test_docs.py   # Synthetic test document generator (Faker)
├── tests/
│   ├── test_parser.py          # Parser and chunking tests
│   ├── test_pipeline.py        # BatchCommitter and file discovery tests
│   ├── test_state_tracker.py   # SQLite state tracker tests
│   └── test_retriever.py       # Retriever unit + integration tests
├── data/                       # Runtime data (gitignored)
│   ├── chroma/                 # ChromaDB persistent index + BM25 pickle
│   ├── state.db                # SQLite state database
│   └── test_docs/              # Generated test documents
├── .devcontainer/
│   ├── devcontainer.json       # VS Code Dev Container config
│   ├── docker-compose.yml      # App (6 GB) + Ollama (8 GB) services
│   └── Dockerfile              # Python 3.12 + system deps
├── .env.example                # Environment variable template
├── pyproject.toml              # Build config, pytest/black/isort settings
└── requirements.txt            # Python dependencies
```

## How It Works

### Ingestion Pipeline

The pipeline runs in two phases to stay within 16 GB RAM:

1. **Phase 1 (fast formats):** PDF, DOCX, XLSX, and EML files are parsed in parallel using multiple workers. Each file is chunked (2,000 chars with 200-char overlap), embedded with all-MiniLM-L6-v2, and committed to ChromaDB in batches.

2. **Phase 2 (OCR formats):** Scanned PDFs and images are processed sequentially with a single worker. The embedding model is unloaded to free ~300 MB before OCR begins, and ORC models (~400 MB) are unloaded before reloading the embedder.

The SQLite state tracker records an MD5 hash for every file. On subsequent runs, unchanged files are skipped, and previously failed files are retried automatically.

### Hybrid Retrieval

Each query goes through a multi-signal pipeline:

1. **Semantic search** — cosine similarity against ChromaDB embeddings
2. **BM25 keyword search** — lexical matching via BM25Okapi
3. **Reciprocal Rank Fusion** — merges the two ranked lists (k=60)
4. **Filename / phrase search** — regex-detects filenames and proper nouns in the query for targeted metadata lookups
5. **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 scores each query-document pair
6. **Citation assembly** — attaches filename, page number, and text snippet to every result

### Answer Generation

The top 5 retrieved chunks are formatted as context and sent to the local Ollama instance (Llama 3.2). The system prompt instructs the LLM to answer only from provided context and cite sources.

## Configuration

All settings are managed via environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `OLLAMA_MODEL_DEV` | `llama3.2:3b` | LLM for development |
| `OLLAMA_MODEL_EVAL` | `llama3.2:8b` | LLM for evaluation runs |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHROMA_PERSIST_PATH` | `data/chroma` | ChromaDB storage directory |
| `STATE_DB_PATH` | `data/state.db` | SQLite state database path |
| `MAX_WORKERS` | `8` | Parallel workers for fast-format parsing |
| `APP_ENV` | `development` | Application environment |

## Testing

Run the full test suite:

```bash
pytest
```

66 tests cover parsing, chunking, pipeline orchestration, state tracking, and retrieval (unit + integration):

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_parser.py` | 12 | Text splitting, TextParser, parse_file dispatcher |
| `test_pipeline.py` | 7 | BatchCommitter batching, file discovery, IngestionStats |
| `test_state_tracker.py` | 16 | MD5 hashing, DB init, CRUD, resume logic, summaries |
| `test_retriever.py` | 31 | Pydantic models, RRF, BM25 tokenizer, regex patterns, integration |

> **Note:** Integration tests in `test_retriever.py` require a populated ChromaDB at `data/chroma/`.

## Evaluation

Run the RAGAS evaluation and benchmarking suite (requires the API server to be running):

```bash
python -m evals.ragas_suite --data-dir data/ --sample-size 50 --questions 20
```

This will:
1. **Generate a golden dataset** — sample documents, parse them, and use Ollama to create Q&A pairs
2. **Evaluate with RAGAS** — measure Faithfulness, Context Precision, and Answer Relevancy
3. **Benchmark ingestion** — measure files-per-minute throughput
4. **Benchmark queries** — measure average query latency

Results are saved to `evals/results.json`.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
