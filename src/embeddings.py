"""Embedding service — wraps HuggingFace sentence-transformers.

Uses ``all-MiniLM-L6-v2`` by default (best latency/RAM trade-off for
the target hardware profile: i5-1135G7 + 16 GB RAM).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from src.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates vector embeddings from text using a local model.

    The model is lazily loaded on first call to ``embed`` to avoid
    wasting RAM if the service isn't used in a particular run.

    Args:
        settings: Application settings (provides ``embedding_model``).
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Lazily load the sentence-transformer model.

        Returns:
            Loaded SentenceTransformer instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality.

        Returns:
            Integer dimension of the embedding vectors.
        """
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
