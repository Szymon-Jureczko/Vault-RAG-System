"""Application-wide configuration loaded from environment variables.

Uses ``pydantic-settings`` so every value can be overridden via ``.env``
or real environment variables without touching source code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the LocalVaultRAG system.

    Attributes:
        ollama_base_url: Base URL for the Ollama API server.
        ollama_model_dev: Smaller model used during development.
        ollama_model_eval: Larger model used for evaluation runs.
        embedding_model: HuggingFace sentence-transformer model name.
        chroma_persist_path: Directory where ChromaDB stores its index.
        state_db_path: Path to the SQLite state-tracking database.
        max_workers: Max workers for non-OCR ProcessPoolExecutor (CPU cores).
        ocr_workers: Max workers for OCR/image ProcessPoolExecutor (memory-constrained).
        batch_commit_size: Number of vectors to buffer before committing
            to ChromaDB.  Also the interval for ``gc.collect()`` calls.
        app_env: Current environment (development / production).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_model_dev: str = "llama3.2:3b"
    ollama_model_eval: str = "llama3.2:8b"

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_path: Path = Path("data/chroma")

    # ── State DB ────────────────────────────────────────────────────────────
    state_db_path: Path = Path("data/state.db")

    # ── Concurrency ─────────────────────────────────────────────────────────
    max_workers: int = 4  # non-OCR parsers (leave cores for API + embeddings)
    ocr_workers: int = 2  # OCR/image parsers (parse-then-embed to avoid OOM)

    # ── Batch / OOM-safety ──────────────────────────────────────────────────
    batch_commit_size: int = 500

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"


settings = Settings()
