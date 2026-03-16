# Task: Build Scalable Multimodal RAG Researcher 

Execute a systematic, 5-phase build of an enterprise-grade RAG system capable of handling 5,000-10,000 messy documents on commodity hardware.

## 🏗️ Target Architecture
- `data/`: Ingest root (supports PDF, DOCX, XLSX, EML, Scanned Images).
- `ingestion/`: Parallel Layout-Aware Parsers + SQLite Persistence.
- `vector_db/`: Persistent HNSW index (Chroma).
- `evals/`: RAGAS Faithfulness & Relevancy suite.
- `api/`: FastAPI + Streamlit Dashboard.

## Phase 1: Infrastructure & "State of Health"
- **Docker:** Setup Ollama (Llama 3.2) and ChromaDB.
- **State Tracker:** Create `ingestion/state_tracker.py`.
  - SQLite schema to track `file_hash`, `parsing_status`, and `vector_id`.
  - Logic: Enable "Resume-on-Failure" capability for large datasets.
- **Validation:** Initialize DB and spin up containers.

## Phase 2: High-Performance Multimodal Ingestion
- **Parser:** Implement `ingestion/parser.py`.
  - Integrate `Docling` for Table-to-Markdown reconstruction.
  - Integrate `Tesseract OCR` for embedded graphs and images.
- **Scaling:** Use `ProcessPoolExecutor` to parallelize parsing across 8 threads.
- **Batcher:** Commit vectors in batches of 100 to maintain 16GB RAM stability.

## Phase 3: Hybrid Retrieval & Citation Engine
- **Search:** Implement **Hybrid Retrieval** (BM25 Keyword + Semantic Vector).
- **Reranker:** Add a lightweight Cross-Encoder step to refine the Top-5 results.
- **Citations:** Every response MUST return source metadata (Filename, Page, Snippet).

## Phase 4: Production API & Streamlit UI
- **FastAPI:** Endpoints for `/sync` (trigger incremental ingest) and `/query`.
- **Streamlit:** Build a "Researcher Dashboard" with a chat interface, citation viewer, and ingestion progress bar.

##  Phase 5: RAGAS Validation & Benchmarking
- **Evaluation:** Setup `evals/ragas_suite.py`.
  - Generate a "Golden Dataset" of 20 questions from 50 sample files.
  - Calculate: **Faithfulness**, **Context Precision**, and **Answer Relevancy**.
- **Benchmark:** Create a script to measure "Files-Per-Minute" and "Query Latency."

**START PHASE 1 NOW:** Create the `docker-compose.yml` and the SQLite `state_tracker.py`. Focus on the MD5 hashing logic first.