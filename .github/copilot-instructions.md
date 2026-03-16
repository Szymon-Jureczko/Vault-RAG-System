# Portfolio Standard: High-Scale Enterprise RAG (Privacy-First)

## Priorities
- **Goal:** Demonstrate "Production Grade" AI Engineering.
- **Privacy:** 100% Local Inference. Zero data leakage. No OpenAI/Cloud APIs.
- **Hardware:** Optimized for Intel i5-1135G7 (8 threads) and 16GB RAM.

## Engineering Standards
- **Incremental Logic:** Must use SQLite `state_db` with MD5 hashing. Never re-process unchanged files.
- **Resource Management:** Implement `BatchCommitter` logic. Clear Python `gc.collect()` and vector-store cache every 500 files to prevent OOM on 16GB RAM.
- **Concurrency:** Use `concurrent.futures.ProcessPoolExecutor` for CPU-bound parsing (8 threads max).
- **Type Safety:** Strict use of Pydantic models and Google-style docstrings for all modules.

## Performance Tech Stack
- **Parser:** `Docling` (layout-aware) + `PyMuPDF` (speed) + `Tesseract` (OCR).
- **LLM:** Ollama (`llama3.2:3b` for dev, `8b` for evaluation).
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Best latency/RAM trade-off).
- **Vector DB:** ChromaDB (Persistent Disk Mode).