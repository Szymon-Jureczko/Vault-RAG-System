"""Application configuration via Pydantic Settings.

Reads from environment variables / ``.env`` file.
All inference is local — no cloud API keys required.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the LocalVaultRAG system.

    Attributes:
        ollama_base_url: Base URL for the Ollama service.
        ollama_model_dev: Smaller model used during development.
        ollama_model_eval: Larger model used during evaluation / production.
        embedding_model: HuggingFace model ID for sentence embeddings.
        chroma_persist_path: Directory where ChromaDB stores its data.
        state_db_path: Path to the SQLite state database.
        max_workers: Max workers for ProcessPoolExecutor (CPU cores).
        app_env: Application environment (development | production).
        batch_size: Number of documents per batch before flushing.
        batch_gc_interval: Flush caches and run gc.collect() every N files.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_model_dev: str = "llama3.2:3b"
    ollama_model_eval: str = "llama3.2:8b"

    # ── Embeddings ──────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ────────────────────────────────────────────────────────
    chroma_persist_path: Path = Field(default=Path("data/chroma"))

    # ── State DB ────────────────────────────────────────────────────────
    state_db_path: Path = Field(default=Path("data/state.db"))

    # ── Concurrency ─────────────────────────────────────────────────────
    max_workers: int = 8

    # ── App ─────────────────────────────────────────────────────────────
    app_env: str = "development"

    # ── Batch Processing ────────────────────────────────────────────────
    batch_size: int = 50
    batch_gc_interval: int = 500

    @property
    def is_dev(self) -> bool:
        """Return True when running in development mode."""
        return self.app_env == "development"

    @property
    def active_model(self) -> str:
        """Return the LLM model name appropriate for the current environment."""
        return self.ollama_model_dev if self.is_dev else self.ollama_model_eval


def get_settings() -> Settings:
    """Create and return a Settings instance (reads .env on first call)."""
    return Settings()
