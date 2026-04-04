"""Application-wide configuration loaded from environment variables.

Uses ``pydantic-settings`` so every value can be overridden via ``.env``
or real environment variables without touching source code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
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
        chunk_overlap: Characters of overlap between consecutive chunks.
            400 ensures that sentences/list-items near a chunk boundary
            appear in both the preceding and following chunk, reducing the
            chance of a bibliography or section header being split.  Requires
            re-ingestion to take effect.
        app_env: Current environment (development / production).
        ingestion_source: Where to read documents from. ``'LOCAL'`` reads from
            the filesystem; ``'AZURE'`` downloads from Azure Blob Storage.
        azure_storage_connection_string: Azure storage account connection
            string.  Required when ``ingestion_source`` is ``'AZURE'``.
        azure_container_name: Name of the Azure blob container to ingest from.
            Required when ``ingestion_source`` is ``'AZURE'``.
        azure_staging_path: Persistent local directory where blobs are cached
            between Azure syncs.  Using a stable path (instead of a new temp
            dir each run) lets the state tracker skip unchanged files by MD5.
        llm_num_ctx: Ollama context window in tokens.  6144 provides
            headroom for 5 chunks of dense non-English content (5 × 2 020-char
            Polish chunks ÷ 3 chars/token ≈ 3 367 input tokens + prompt
            overhead and 512 generated tokens, well within 6 144).
        llm_num_predict: Maximum tokens Ollama will generate per response.
            512 allows complete enumeration of long bibliographies and lists.
            With 3 LLM chunks, total time is ~75 s (prompt eval ~49 s +
            decode ~26 s), well within the 120 s timeout.
        llm_timeout: HTTP timeout in seconds for the Ollama ``/api/generate``
            call.  120 s covers 3-chunk + 512-token generation on CPU.
        llm_temperature: Sampling temperature for Ollama generation.  0.0 means
            greedy decoding, which is faster and more factually consistent for
            RAG question-answering.
        llm_num_thread: Number of CPU threads for Ollama inference.  Defaults to
            4 to match the i5-1135G7 physical core count.  Hyperthreading adds
            overhead for LLM inference, so we use physical cores only.
        chunk_overlap: Character overlap between consecutive text chunks.
            400 chars keeps bibliography entries and section boundaries whole
            across chunk splits.  Re-ingestion is required after changing this.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_model_dev: str = "llama3.2:3b-instruct-q4_K_M"
    ollama_model_eval: str = "llama3.2:8b"

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_path: Path = Path("data/chroma")

    # ── State DB ────────────────────────────────────────────────────────────
    state_db_path: Path = Path("data/state.db")

    # ── Tabular DB (SQLite for Text-to-SQL agent) ─────────────────────────
    tabular_db_path: Path = Path("data/tabular.db")

    # ── Concurrency ─────────────────────────────────────────────────────────
    max_workers: int = (
        2  # non-OCR parsers; reduced to leave RAM for larger multilingual embedder
    )
    ocr_workers: int = 1  # OCR/image parsers — sequential is safer on 16 GB

    # ── Batch / OOM-safety ──────────────────────────────────────────────────
    batch_commit_size: int = 50

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"

    # ── Ingestion Source ─────────────────────────────────────────────────────
    ingestion_source: Literal["LOCAL", "AZURE"] = "LOCAL"

    # ── Azure Blob Storage (only required when ingestion_source = 'AZURE') ───
    azure_storage_connection_string: str = ""
    azure_container_name: str = ""
    azure_staging_path: Path = Path("data/azure_staging")

    # ── LLM generation parameters ────────────────────────────────────────────
    llm_num_ctx: int = 6144
    llm_num_predict: int = 768
    llm_timeout: float = 120.0
    llm_temperature: float = 0.0
    llm_num_thread: int = 4

    # ── Evaluation ───────────────────────────────────────────────────────────
    openai_api_key: str = ""
    structured_top_k: int = 10

    @field_validator("ingestion_source", mode="before")
    @classmethod
    def _normalise_ingestion_source(cls, v: object) -> str:
        """Uppercase so 'azure' and 'AZURE' are equivalent.

        Args:
            v: Raw value from environment variable.

        Returns:
            Uppercased string value.
        """
        if isinstance(v, str):
            return v.upper()
        return v  # type: ignore[return-value]


settings = Settings()
