# Vault-RAG-System

**Private enterprise RAG system — local LLM inference, flexible document sources**

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![LLM Llama 3.2](https://img.shields.io/badge/LLM-Llama%203.2-blueviolet)
![Vector DB ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)

---

## Overview


Vault-RAG-System ingests enterprise documents at any scale — PDFs, Word files, spreadsheets, emails, and scanned images — indexes them with a hybrid search engine, and answers natural-language questions with cited sources. Documents can come from local disk or Azure Blob Storage. All LLM inference and vector search run on your machine via Ollama and ChromaDB — no OpenAI keys, no external AI APIs.

<img width="1855" height="949" alt="projimg2" src="https://github.com/user-attachments/assets/9fda9143-b161-415b-ab5e-ef8e1e424d6a" />

Built for commodity hardware: optimized to run on an Intel i5 with 16 GB RAM. There is no hard limit on document count — incremental processing, two-phase model swapping, and batch commits keep memory stable regardless of corpus size. Ingestion scales linearly: more documents means more time, not more memory.

On stronger hardware the pipeline scales up with a few parameter changes:

| Parameter | Default | Scale-up effect |
|-----------|---------|-----------------|
| `MAX_WORKERS` | `2` | Increase to match available CPU cores for faster parallel parsing |
| `OCR_WORKERS` | `1` | Increase when RAM allows simultaneous OCR + embedding |
| `LLM_NUM_THREAD` | `8` | Match to physical core count for faster LLM inference |
| `LLM_NUM_CTX` | `6144` | Expand context window for longer, more detailed answers |
| `LLM_NUM_PREDICT` | `768` | Allow longer LLM responses |
| `BATCH_COMMIT_SIZE` | `50` | Increase to reduce commit overhead with more RAM |
| `OLLAMA_MODEL_DEV` | `llama3.2:3b` | Swap to larger models (8b, 70b) for higher answer quality |

## Features

- **Query router** — LLM-based classifier routes each question as *structured* (analytical/tabular) or *unstructured* (knowledge), dispatching to the SQL agent or RAG pipeline accordingly
- **Text-to-SQL data agent** — generates SQL from natural language, executes safely against a read-only SQLite connection, formats results via LLM, and automatically retries with relaxed filters on empty results; returns exact values verbatim without rounding or approximation
- **BM25 table pre-filter** — BM25 scores all table metadata against the user query and feeds only the top-K most relevant tables into the LLM prompt, preventing context-window overflow on large corpora (e.g. 10 of 217 tables selected)
- **Scored table citations** — structured query responses include source table names, sheet, and BM25 relevance scores in the same Citations & Sources format used by unstructured queries
- **Tabular ingestion** — XLSX and CSV files are loaded into SQLite (`data/tabular.db`) with auto-detected header rows, sanitized column names, and numeric type coercion via pandas
- **Multimodal document ingestion** — PDF, DOCX, XLSX, EML, images (PNG/JPG/TIFF/BMP), and plain text (TXT, MD, CSV, JSON, XML, HTML)
- **Azure Blob Storage ingestion** — ingest documents directly from an Azure container; blobs are downloaded, parsed, and indexed with a single API call or dashboard button
- **Scanned document OCR** — RapidOCR (ONNX Runtime) with fast gating, text-density probing, reduced DPI/resolution limits, and `malloc_trim` to return freed heap pages to the OS
- **Hybrid retrieval** — BM25 keyword + semantic vector + filename lookup + proper-noun phrase matching, merged via Reciprocal Rank Fusion; optional `source_filter` to restrict results by ingestion source
- **Cross-encoder reranking** — multilingual mmarco-mMiniLMv2-L12-H384-v1 refines top results; chunks scoring below 0.3 are filtered out before LLM generation to reduce noise
- **Context quality filtering** — short or garbage chunks (< 50 chars after stripping source prefix) are dropped before passing context to the LLM, improving answer focus
- **Citations on every result** — filename, page number, and text snippet
- **Incremental processing** — SQLite state DB with MD5 change detection; source-prefixed keys keep LOCAL and Azure records independent across resyncs
- **Memory-safe on 16 GB RAM** — two-phase pipeline, model swapping, batch commits, `gc.collect()` checkpoints, and subprocess isolation of the sync worker to reduce import overhead from ~300 MB to ~2 MB
- **RAGAS evaluation** — curated golden dataset, Faithfulness / Context Precision / Answer Relevancy scoring with multilingual embeddings, ingestion throughput and query latency benchmarks; reusable via `--use-existing-dataset`
- **Production API + dashboard** — FastAPI REST endpoints with thread-safe singleton retriever, and Streamlit researcher dashboard with streaming chat interface

## Architecture

```
Documents on disk (data/)  ─or─  Azure Blob Storage
         |
         v
 ┌──────────────────────────────────────────────────────────────┐
 │            Ingestion Pipeline                                │
 │  discover_files ──► StateTracker (SQLite + MD5)              │
 │        |                                                     │
 │        ▼                                                     │
 │  Phase 1: Fast formats    Phase 2: OCR formats               │
 │  (PDF, DOCX, XLSX, EML)   (scanned PDF, images)             │
 │  N workers in parallel     1 worker, mini-batches            │
 │        |                        |                            │
 │        └──────────┬─────────────┘                            │
 │                   ▼                                          │
 │  BatchCommitter ──► ChromaDB       Tabular loader ──► SQLite │
 │  (HNSW cosine embeddings)          (XLSX/CSV → tabular.db)   │
 └──────────────────────────────────────────────────────────────┘
                     |
                     v
 ┌─────────────────────────────────────────────────────┐
 │              Query Router (LLM classifier)           │
 │  "structured" ──► Text-to-SQL   "unstructured" ──►  │
 │    Data Agent        │           Hybrid Retriever    │
 │    (SQLite query)    │     Semantic + BM25 + RRF     │
 │         |            │     + Cross-encoder rerank    │
 │         v            │              |                │
 │    LLM formatting    │              v                │
 │                      │     RAG answer generation     │
 └──────────────────────┴──────────────────────────────┘
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
| Query Router | Ollama LLM classifier | Routes structured (SQL) vs unstructured (RAG) queries |
| Text-to-SQL | Custom data agent | NL → SQL generation, safe execution, LLM-formatted results |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | 384-dim, multilingual, ONNX-accelerated with PyTorch fallback |
| Vector DB | ChromaDB | HNSW index, cosine similarity, persistent disk mode |
| Tabular DB | SQLite (`data/tabular.db`) | Auto-ingested XLSX/CSV with pandas, header detection, type coercion |
| OCR | RapidOCR | ONNX Runtime, replaces Tesseract for 3-10x speed |
| PDF Parsing | PyMuPDF | Native text extraction + 300 DPI rendering for scanned pages |
| Keyword Search | BM25Okapi | Pickled to disk, auto-refreshed after ingestion |
| Reranker | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Multilingual query-document pair scoring |
| API | FastAPI | Background sync, health checks, query + streaming endpoints |
| Dashboard | Streamlit | Chat interface, citation viewer, ingestion progress |
| State Tracking | SQLite | MD5 hashing, incremental processing, resume-on-failure |
| Cloud Storage (optional) | Azure Blob Storage | `azure-storage-blob>=12.0`, on-demand download |

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

### Option A — Docker Compose (standalone deployment)

1. **Clone and configure:**
   ```bash
   git clone https://github.com/<your-username>/localvaultrag.git
   cd localvaultrag
   cp .env.example .env
   ```

2. **Start all services:**
   ```bash
   docker compose up -d
   ```
   This builds the app image, starts the API server (`:8000`), Streamlit dashboard (`:8501`), and Ollama sidecar — then pulls the default LLM model automatically.

3. **Pull the LLM model** (first run only):
   ```bash
   docker compose exec ollama ollama pull llama3.2:3b
   ```

4. **Trigger ingestion** — place documents in the `data/` volume, then click Sync in the dashboard at `http://localhost:8501`, or:
   ```bash
   curl -X POST http://localhost:8000/sync -H "Content-Type: application/json" -d '{"source_dir": "data/"}'
   ```

### Option B — VS Code Dev Container (development)

1. Install [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

2. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/localvaultrag.git
   cd localvaultrag
   ```

3. **Open in Dev Container:**

   Open the project in VS Code and select **"Reopen in Container"** when prompted (or run the `Dev Containers: Reopen in Container` command). This builds the Python 3.12 container and starts the Ollama sidecar automatically.

4. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   The defaults work out of the box — Ollama runs as a sidecar service on the Docker network.

5. **Generate test documents** (optional — creates ~1,000 synthetic documents):
   ```bash
   python scripts/generate_test_docs.py
   ```

6. **Start the API server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

7. **Start the Streamlit dashboard** (in a second terminal):
   ```bash
   streamlit run api/streamlit_app.py --server.port 8888
   ```

8. **Trigger ingestion** — click the Sync button in the dashboard, or:
   ```bash
   curl -X POST http://localhost:8000/sync -H "Content-Type: application/json" -d '{"source_dir": "data/"}'
   ```

   For Azure Blob Storage ingestion:
   ```bash
   curl -X POST http://localhost:8000/sync \
     -H "Content-Type: application/json" \
     -d '{"ingestion_source": "AZURE", "azure_storage_connection_string": "DefaultEndpointsProtocol=...", "azure_container_name": "my-docs"}'
   ```

## API Reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/sync` | `{"source_dir": "data/", "ingestion_source": "LOCAL\|AZURE", "azure_storage_connection_string": "...", "azure_container_name": "..."}` | Trigger incremental document ingestion (background) |
| `GET` | `/sync/{job_id}` | — | Poll sync job progress |
| `GET` | `/stats` | — | Live ingestion counts from SQLite |
| `POST` | `/query` | `{"question": "...", "top_k": 10}` | Hybrid search + LLM-generated answer with citations |
| `POST` | `/query/stream` | `{"question": "...", "top_k": 10}` | Streaming variant — returns citations then LLM tokens via SSE |
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
│   ├── pipeline.py             # Two-phase ingestion orchestrator + BatchCommitter + tabular loader
│   ├── state_tracker.py        # SQLite state DB with MD5 change detection
│   └── sync_worker.py          # Import-light subprocess worker (prevents OOM)
├── vector_db/
│   └── retriever.py            # Hybrid retriever (BM25 + semantic + reranking)
├── evals/
│   ├── ragas_suite.py          # RAGAS evaluation + benchmarking CLI
│   ├── golden_dataset.json     # Curated 12-question evaluation dataset
│   └── results.json            # Latest evaluation results
├── scripts/
│   └── generate_test_docs.py   # Synthetic test document generator (Faker)
├── tests/
│   ├── test_parser.py          # Parser and chunking tests
│   ├── test_pipeline.py        # BatchCommitter and file discovery tests
│   ├── test_state_tracker.py   # SQLite state tracker tests
│   ├── test_retriever.py       # Retriever unit + integration tests
│   ├── test_generate_answer.py # LLM answer generation tests
│   └── test_azure_ingestion.py # Azure Blob Storage ingestion tests
├── data/                       # Runtime data (gitignored)
│   ├── chroma/                 # ChromaDB persistent index + BM25 pickle
│   ├── tabular.db              # SQLite tabular database (auto-ingested XLSX/CSV)
│   ├── state.db                # SQLite state database
│   └── test_docs/              # Generated test documents
├── .devcontainer/
│   ├── devcontainer.json       # VS Code Dev Container config
│   ├── docker-compose.yml      # App (6 GB) + Ollama (8 GB) services
│   └── Dockerfile              # Python 3.12 + system deps
├── .env.example                # Environment variable template
├── Dockerfile                  # Production container image
├── docker-compose.yml          # Standalone deployment (app + dashboard + Ollama)
├── .dockerignore               # Docker build exclusions
├── pyproject.toml              # Build config, pytest/black/isort settings
├── pyrightconfig.json          # Pyright/Pylance type-checking configuration
└── requirements.txt            # Python dependencies
```

## How It Works

### Ingestion Pipeline

The pipeline runs in two phases to stay within 16 GB RAM:

1. **Phase 1 (fast formats):** PDF, DOCX, XLSX, and EML files are parsed in parallel using multiple workers. Each file is chunked (2,000 chars with 400-char overlap), embedded with paraphrase-multilingual-MiniLM-L12-v2, and committed to ChromaDB in batches.

2. **Phase 2 (OCR formats):** Scanned PDFs and images are processed sequentially with a single worker. The embedding model is unloaded to free ~300 MB before OCR begins, and ORC models (~400 MB) are unloaded before reloading the embedder.

The SQLite state tracker records an MD5 hash for every file. On subsequent runs, unchanged files are skipped, and previously failed files are retried automatically. The sync worker runs in a dedicated subprocess with minimal imports to avoid pulling ~300 MB of API dependencies into the child process.

After vector ingestion, a **tabular loader** scans XLSX and CSV files and loads each sheet/file into SQLite (`data/tabular.db`). Header rows are auto-detected by scanning for non-"Unnamed" column labels, column names are sanitized to valid SQL identifiers, and numeric columns are coerced via `pd.to_numeric`.

### Query Router

Each incoming question is classified by the LLM as *structured* (analytical/tabular) or *unstructured* (knowledge/document). Structured questions are dispatched to the Text-to-SQL data agent; unstructured questions follow the hybrid retrieval + RAG path. If the SQL agent returns no results, the system falls back to RAG automatically.

### Text-to-SQL Data Agent

For structured queries the agent:

1. **Pre-filters tables** — BM25 scores all table metadata (names, column names, sample values) against the question and selects the top-K most relevant tables (default 10), keeping the schema prompt well within the LLM context window even with hundreds of tables
2. Reads the SQLite schema for the selected tables from `tabular.db`
3. Asks the LLM to generate a `SELECT` statement
4. Executes the SQL on a read-only connection
5. If the result is empty, retries once with relaxed WHERE filters
6. Formats the result rows via the LLM into a natural-language answer with exact values (no rounding or approximation)
7. Attaches scored table citations (source file, sheet, BM25 relevance) to the response

### Hybrid Retrieval

Each query goes through a multi-signal pipeline:

1. **Semantic search** — cosine similarity against ChromaDB embeddings
2. **BM25 keyword search** — lexical matching via BM25Okapi
3. **Reciprocal Rank Fusion** — merges the two ranked lists (k=60)
4. **Filename / phrase search** — regex-detects filenames and proper nouns in the query for targeted metadata lookups
5. **Cross-encoder reranking** — mmarco-mMiniLMv2-L12-H384-v1 (multilingual) scores each query-document pair
6. **Citation assembly** — attaches filename, page number, and text snippet to every result

### Answer Generation

Retrieved chunks are filtered by cross-encoder score (threshold 0.3) and short fragments (< 50 chars) are dropped. Up to 7 chunks are formatted as context and sent to the local Ollama instance (Llama 3.2). The system prompt instructs the LLM to answer only from provided context — extracting from headings, metadata, and footers when needed — and to cite sources. It only refuses when the context truly contains nothing relevant.

## Configuration

All settings are managed via environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `OLLAMA_MODEL_DEV` | `llama3.2:3b` | LLM for development |
| `OLLAMA_MODEL_EVAL` | `llama3.2:8b` | LLM for evaluation runs |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformer model (`.env.example` overrides to `all-MiniLM-L6-v2`) |
| `CHROMA_PERSIST_PATH` | `data/chroma` | ChromaDB storage directory |
| `STATE_DB_PATH` | `data/state.db` | SQLite state database path |
| `TABULAR_DB_PATH` | `data/tabular.db` | SQLite database for ingested XLSX/CSV tables |
| `STRUCTURED_TOP_K` | `10` | Max tables passed to the Text-to-SQL agent (BM25 pre-filter) |
| `MAX_WORKERS` | `2` | Parallel workers for fast-format parsing |
| `APP_ENV` | `development` | Application environment |
| `INGESTION_SOURCE` | `LOCAL` | Document source: `LOCAL` or `AZURE` |
| `AZURE_STORAGE_CONNECTION_STRING` | `""` | Azure storage account connection string |
| `AZURE_CONTAINER_NAME` | `""` | Azure blob container name |
| `AZURE_STAGING_PATH` | `data/azure_staging` | Local cache directory for Azure blobs |
| `LLM_NUM_CTX` | `6144` | Ollama context window (tokens) |
| `LLM_NUM_PREDICT` | `768` | Max tokens per LLM response |
| `LLM_TIMEOUT` | `120.0` | Ollama HTTP timeout (seconds) |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature (0.0 = greedy) |
| `LLM_NUM_THREAD` | `8` | CPU threads for Ollama inference |
| `OCR_WORKERS` | `1` | Parallel workers for OCR/image parsing |
| `BATCH_COMMIT_SIZE` | `50` | Vectors buffered before committing to ChromaDB |
| `CHUNK_OVERLAP` | `400` | Character overlap between consecutive text chunks |

## Testing

Run the full test suite:

```bash
pytest
```

97 tests cover parsing, chunking, pipeline orchestration, state tracking, retrieval, answer generation, and Azure ingestion (unit + integration):

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_retriever.py` | 31 | Pydantic models, RRF, BM25 tokenizer, regex patterns, source filtering, integration |
| `test_azure_ingestion.py` | 19 | Azure settings, blob download, pipeline Azure dispatch |
| `test_state_tracker.py` | 17 | MD5 hashing, DB init, CRUD, resume logic, summaries |
| `test_parser.py` | 13 | Text splitting, TextParser, parse_file dispatcher |
| `test_generate_answer.py` | 10 | Prompt construction, LLM settings passthrough, score filtering, query routing |
| `test_pipeline.py` | 7 | BatchCommitter batching, file discovery, IngestionStats |

> **Note:** Integration tests in `test_retriever.py` require a populated ChromaDB at `data/chroma/`.

## Evaluation

Run the RAGAS evaluation and benchmarking suite (requires the API server to be running):

```bash
# Generate a new golden dataset and evaluate
python -m evals.ragas_suite --data-dir data/ --sample-size 50 --questions 20

# Re-evaluate against the committed golden dataset (no generation step)
python -m evals.ragas_suite --use-existing-dataset
```

The suite:
1. **Generates or loads a golden dataset** — samples documents, extracts Q&A pairs via LLM (or loads `evals/golden_dataset.json` with `--use-existing-dataset`)
2. **Warms up the server** — sends a preflight query so the first real question isn't lost to cold-start model loading
3. **Evaluates with RAGAS** — measures Faithfulness, Context Precision, and Answer Relevancy using `gpt-4o-mini` as judge LLM and `paraphrase-multilingual-MiniLM-L12-v2` for answer relevancy embeddings (multilingual-aware scoring)
4. **Benchmarks ingestion** — measures files-per-minute throughput
5. **Benchmarks queries** — measures average query latency

Results are saved to `evals/results.json`.

### Latest results (12-question golden dataset, Ollama Llama 3.2 + OpenAI judge)

| Metric | Score |
|--------|-------|
| Faithfulness | **0.881** |
| Context Precision | **0.893** |
| Answer Relevancy | **0.848** |
| Ingestion throughput | 8,061 files/min |
| Avg query latency | 39,611 ms |

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
