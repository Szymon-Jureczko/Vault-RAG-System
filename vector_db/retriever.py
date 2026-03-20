"""Hybrid retrieval engine — BM25 + semantic + cross-encoder.

Implements the Phase 3 retrieval pipeline:
1. **BM25** keyword search over stored documents for lexical matching.
2. **Semantic** vector search via ChromaDB cosine similarity.
3. **Reciprocal Rank Fusion (RRF)** to merge both result sets.
4. **Cross-Encoder** reranker to refine the Top-5 results.
5. Every response includes **citations** (filename, page, snippet).

Usage::

    from vector_db.retriever import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.query("What are the key findings?", top_k=5)
    for r in results:
        print(r.text, r.citation)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import chromadb
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingestion.config import settings

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """Source citation for a retrieved chunk.

    Attributes:
        filename: Name of the source document.
        page: Page number (if available).
        snippet: First 200 characters of the matched text.
        source_path: Full path to the source file.
    """

    filename: str = ""
    page: int | None = None
    snippet: str = ""
    source_path: str = ""


class RetrievalResult(BaseModel):
    """A single retrieval result with text, score, and citation.

    Attributes:
        chunk_id: ChromaDB document ID.
        text: Full text of the retrieved chunk.
        score: Combined retrieval score (higher is more relevant).
        citation: Source citation metadata.
    """

    chunk_id: str
    text: str
    score: float = 0.0
    citation: Citation = Field(default_factory=Citation)


class HybridRetriever:
    """Hybrid BM25 + semantic retriever with cross-encoder reranking.

    Args:
        chroma_path: Path to ChromaDB persistent storage.
        embedding_model: HuggingFace model for semantic search.
        reranker_model: Cross-encoder model for reranking.
        collection_name: ChromaDB collection name.
    """

    _RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        chroma_path: Path | None = None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        collection_name: str = "localvault_docs",
    ) -> None:
        chroma_path = chroma_path or settings.chroma_persist_path
        self._embedder = self._init_embedder(
            embedding_model or settings.embedding_model
        )
        self._reranker = CrossEncoder(reranker_model or self._RERANKER_MODEL)

        client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metas: list[dict] = []
        self._bm25_index_path = Path(str(chroma_path)) / "bm25_index.pkl"

    @staticmethod
    def _init_embedder(model_name: str):
        """Initialise embedding backend (ONNX-first, PyTorch fallback).

        Mirrors the ingestion pipeline strategy so query and document
        embeddings are numerically identical.
        """
        hub_id = f"sentence-transformers/{model_name}"
        onnx_cache = settings.chroma_persist_path.parent / "onnx" / model_name
        try:
            import numpy as np
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            if onnx_cache.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(onnx_cache))
                ort_model = ORTModelForFeatureExtraction.from_pretrained(
                    str(onnx_cache)
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(hub_id)
                ort_model = ORTModelForFeatureExtraction.from_pretrained(
                    hub_id, export=True
                )
                onnx_cache.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(str(onnx_cache))
                ort_model.save_pretrained(str(onnx_cache))
            logger.info("Retriever using ONNX runtime for embeddings")

            class _OnnxEmbedder:
                """Thin wrapper matching SentenceTransformer.encode()."""

                def __init__(self, tok, mdl):
                    self._tok = tok
                    self._mdl = mdl

                def encode(self, texts, **_kw):
                    enc = self._tok(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="np",
                    )
                    out = self._mdl(**enc)
                    mask = enc["attention_mask"]
                    emb = (out.last_hidden_state * mask[..., np.newaxis]).sum(
                        axis=1
                    ) / mask.sum(axis=-1, keepdims=True)
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    return emb / norms

            return _OnnxEmbedder(tokenizer, ort_model)
        except Exception as exc:
            from sentence_transformers import SentenceTransformer

            logger.info("ONNX unavailable (%s), using PyTorch", exc)
            return SentenceTransformer(model_name)

    def _build_bm25_index(self) -> None:
        """Build/rebuild the BM25 index from all documents in ChromaDB."""
        all_docs = self._collection.get(include=["documents", "metadatas"])
        self._corpus_ids = all_docs["ids"]
        self._corpus_texts = all_docs["documents"]
        self._corpus_metas = all_docs["metadatas"]

        tokenized = [doc.lower().split() for doc in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "BM25 index built with %d documents",
            len(self._corpus_ids),
        )
        self._save_bm25_index()

    def _save_bm25_index(self) -> None:
        """Persist BM25 index and corpus to disk atomically."""
        payload = {
            "ids": self._corpus_ids,
            "texts": self._corpus_texts,
            "metas": self._corpus_metas,
            "bm25": self._bm25,
        }
        tmp = self._bm25_index_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._bm25_index_path)
        logger.info("BM25 index persisted to %s", self._bm25_index_path)

    def _load_bm25_index(self) -> bool:
        """Load BM25 index from disk if it exists and matches ChromaDB.

        Returns:
            True if loaded successfully, False if a rebuild is needed.
        """
        if not self._bm25_index_path.exists():
            return False
        try:
            with self._bm25_index_path.open("rb") as f:
                payload = pickle.load(f)
            chroma_count = self._collection.count()
            if len(payload["ids"]) != chroma_count:
                logger.info(
                    "BM25 index stale (%d docs on disk, %d in ChromaDB)"
                    " — rebuilding",
                    len(payload["ids"]),
                    chroma_count,
                )
                return False
            self._corpus_ids = payload["ids"]
            self._corpus_texts = payload["texts"]
            self._corpus_metas = payload["metas"]
            self._bm25 = payload["bm25"]
            logger.info("BM25 index loaded from disk (%d docs)", chroma_count)
            return True
        except Exception as exc:
            logger.warning("Could not load BM25 index: %s — rebuilding", exc)
            return False

    def _load_or_build_bm25_index(self) -> None:
        """Load BM25 index from disk, or build it fresh from ChromaDB."""
        if not self._load_bm25_index():
            if self._collection.count() > 0:
                self._build_bm25_index()
            else:
                logger.info("ChromaDB collection is empty — BM25 index deferred")

    def _semantic_search(
        self, query: str, n_results: int
    ) -> list[tuple[str, str, dict, float]]:
        """Run semantic vector search via ChromaDB.

        Args:
            query: Natural language query string.
            n_results: Number of results to fetch.

        Returns:
            List of (id, text, metadata, distance) tuples.
        """
        embedding = self._embedder.encode([query], show_progress_bar=False)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append(
                (
                    results["ids"][0][i],
                    results["documents"][0][i],
                    results["metadatas"][0][i],
                    results["distances"][0][i],
                )
            )
        return items

    def _bm25_search(
        self, query: str, n_results: int
    ) -> list[tuple[str, str, dict, float]]:
        """Run BM25 keyword search over the corpus.

        Args:
            query: Natural language query string.
            n_results: Number of results to fetch.

        Returns:
            List of (id, text, metadata, score) tuples.
        """
        if self._bm25 is None:
            self._build_bm25_index()

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :n_results
        ]

        items = []
        for idx in top_indices:
            if scores[idx] > 0:
                items.append(
                    (
                        self._corpus_ids[idx],
                        self._corpus_texts[idx],
                        self._corpus_metas[idx],
                        float(scores[idx]),
                    )
                )
        return items

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[tuple[str, str, dict, float]],
        k: int = 60,
    ) -> list[tuple[str, str, dict, float]]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            result_lists: Variable number of ranked result lists.
            k: RRF constant (default 60, per original paper).

        Returns:
            Fused and re-ranked list of results.
        """
        scores: dict[str, float] = {}
        docs: dict[str, tuple[str, dict]] = {}

        for results in result_lists:
            for rank, (doc_id, text, meta, _) in enumerate(results, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                docs[doc_id] = (text, meta)

        fused = [
            (doc_id, docs[doc_id][0], docs[doc_id][1], score)
            for doc_id, score in sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return fused

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, dict, float]],
        top_k: int,
    ) -> list[tuple[str, str, dict, float]]:
        """Rerank candidates using a cross-encoder model.

        Args:
            query: The original query.
            candidates: Fused candidate list.
            top_k: Number of results to return after reranking.

        Returns:
            Top-K reranked results.
        """
        if not candidates:
            return []

        pairs = [[query, text] for _, text, _, _ in candidates]
        ce_scores = self._reranker.predict(pairs)

        reranked = [
            (
                candidates[i][0],
                candidates[i][1],
                candidates[i][2],
                float(ce_scores[i]),
            )
            for i in range(len(candidates))
        ]
        reranked.sort(key=lambda x: x[3], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _build_citation(metadata: dict, text: str) -> Citation:
        """Build a Citation from chunk metadata.

        Args:
            metadata: Chunk metadata dict.
            text: Full chunk text.

        Returns:
            Citation with filename, page, and snippet.
        """
        return Citation(
            filename=metadata.get("filename", "unknown"),
            page=metadata.get("page"),
            snippet=text[:200].strip(),
            source_path=metadata.get("source", ""),
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        semantic_candidates: int = 50,
        bm25_candidates: int = 50,
    ) -> list[RetrievalResult]:
        """Execute a hybrid search with reranking and citations.

        Args:
            query_text: Natural language query.
            top_k: Number of final results to return.
            semantic_candidates: Candidates from semantic search.
            bm25_candidates: Candidates from BM25 search.

        Returns:
            Top-K RetrievalResult instances with citations.
        """
        if self._collection.count() == 0:
            logger.warning("Collection is empty — no results to return")
            return []

        # Step 1 & 2: Parallel retrieval
        semantic_results = self._semantic_search(
            query_text,
            semantic_candidates,
        )
        bm25_results = self._bm25_search(query_text, bm25_candidates)

        # Step 3: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(semantic_results, bm25_results)

        # Step 4: Cross-encoder reranking
        reranked = self._rerank(query_text, fused, top_k)

        # Step 5: Build results with citations
        results = []
        for doc_id, text, meta, score in reranked:
            results.append(
                RetrievalResult(
                    chunk_id=doc_id,
                    text=text,
                    score=score,
                    citation=self._build_citation(meta, text),
                )
            )

        logger.info(
            "Query returned %d results " "(from %d semantic + %d BM25 candidates)",
            len(results),
            len(semantic_results),
            len(bm25_results),
        )
        return results

    def refresh_index(self) -> None:
        """Rebuild the BM25 index from the current ChromaDB state."""
        self._build_bm25_index()
