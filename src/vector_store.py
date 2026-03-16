"""ChromaDB vector store — persistent disk mode.

Provides a thin wrapper around ChromaDB that accepts ``DocumentChunk``
objects and handles embedding + storage in a single call.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from src.config import Settings
from src.embeddings import EmbeddingService
from src.models import DocumentChunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "localvault_docs"


class VectorStore:
    """Manages a ChromaDB collection for document chunk storage and retrieval.

    Args:
        settings: Application settings (provides ``chroma_persist_path``).
        embedding_service: The EmbeddingService used to generate vectors.
    """

    def __init__(
        self, settings: Settings, embedding_service: EmbeddingService
    ) -> None:
        persist_path = Path(settings.chroma_persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._embedding_service = embedding_service
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d existing documents)",
            _COLLECTION_NAME,
            self._collection.count(),
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and insert document chunks into the collection.

        Args:
            chunks: List of DocumentChunk instances to store.
        """
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [
            {**c.metadata, "source_path": str(c.source_path)} for c in chunks
        ]

        embeddings = self._embedding_service.embed(texts)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.debug("Added %d chunks to ChromaDB", len(chunks))

    def query(
        self, query_text: str, n_results: int = 5
    ) -> dict:
        """Search the collection for chunks most similar to the query.

        Args:
            query_text: The natural-language query string.
            n_results: Maximum number of results to return.

        Returns:
            ChromaDB query results dict with keys ``ids``, ``documents``,
            ``metadatas``, ``distances``.
        """
        query_embedding = self._embedding_service.embed([query_text])
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
        return results

    def delete_by_source(self, source_path: Path) -> None:
        """Remove all chunks originating from a specific source file.

        Args:
            source_path: Path of the source file whose chunks to delete.
        """
        self._collection.delete(
            where={"source_path": str(source_path)},
        )
        logger.info("Deleted chunks for source: %s", source_path)

    @property
    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self._collection.count()
