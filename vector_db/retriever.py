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
import re
from pathlib import Path

import chromadb
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingestion.config import settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _extract_relevant_snippet(
    text: str, query: str, max_len: int = 500
) -> str:
    """Return the most query-relevant window of *text*, up to *max_len* chars.

    Uses a sliding-window approach scored by keyword overlap with the query.
    Falls back to the first *max_len* characters when no query is provided.
    The window boundaries are nudged to the nearest sentence-ending
    punctuation so snippets read more naturally.
    """
    # Strip the [Source: ...] prefix that ingestion prepends
    clean = re.sub(r"^\[Source:[^\]]*\]\s*", "", text)
    if not clean:
        return text[:max_len].strip()

    if len(clean) <= max_len:
        return clean.strip()

    if not query:
        return clean[:max_len].strip()

    query_words = {w.lower() for w in _WORD_RE.findall(query)}
    if not query_words:
        return clean[:max_len].strip()

    # Score each window position by keyword hit count
    words = _WORD_RE.findall(clean)
    word_starts: list[int] = []
    pos = 0
    for w in words:
        idx = clean.index(w, pos)
        word_starts.append(idx)
        pos = idx + len(w)

    best_start = 0
    best_score = -1
    step = max(1, len(words) // 60)  # coarse scan for speed

    for i in range(0, len(words), step):
        start_char = word_starts[i]
        end_char = min(start_char + max_len, len(clean))
        window_words = {
            w.lower() for w in _WORD_RE.findall(clean[start_char:end_char])
        }
        score = len(query_words & window_words)
        if score > best_score:
            best_score = score
            best_start = start_char

    # Snap start to beginning of sentence (look back for '. ' or newline)
    snap = clean.rfind(". ", max(0, best_start - 80), best_start)
    if snap != -1:
        best_start = snap + 2
    else:
        snap = clean.rfind("\n", max(0, best_start - 80), best_start)
        if snap != -1:
            best_start = snap + 1

    snippet = clean[best_start : best_start + max_len].strip()

    # Trim to last sentence-ending punctuation if possible
    for end_mark in (". ", ".\n", ".\t"):
        last = snippet.rfind(end_mark)
        if last > max_len // 2:
            snippet = snippet[: last + 1]
            break

    return snippet


class Citation(BaseModel):
    """Source citation for a retrieved chunk.

    Attributes:
        filename: Name of the source document.
        page: Page number (if available).
        snippet: Most relevant ~500 characters of the matched text.
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

    _RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    _FILENAME_RE = re.compile(
        r"\b([\w][\w\-]*\.(?:png|jpg|jpeg|pdf|eml|docx|txt|csv))\b",
        re.IGNORECASE,
    )
    _PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:[ \-][A-Z][a-z]+)+)\b")

    # Weight for the cross-encoder signal in the blended final score.
    # 60 % CE + 40 % RRF prevents the English-only CE model from
    # overriding strong BM25 keyword matches in non-English queries.
    _CE_WEIGHT: float = 0.6

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
        self._corpus_texts = all_docs["documents"] or []
        self._corpus_metas = [  # type: ignore[assignment]
            dict(m) for m in (all_docs["metadatas"] or [])
        ]

        tokenized = [re.findall(r"\w+", doc.lower()) for doc in self._corpus_texts]
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
        docs = results["documents"] or [[]]
        metas = results["metadatas"] or [[]]
        dists = results["distances"] or [[]]
        for i in range(len(results["ids"][0])):
            items.append(
                (
                    results["ids"][0][i],
                    docs[0][i],
                    metas[0][i],
                    dists[0][i],
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
            assert self._bm25 is not None

        tokens = re.findall(r"\w+", query.lower())
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

        candidates = [
            (cid, t, m, s) for cid, t, m, s in candidates if isinstance(t, str) and t
        ]
        if not candidates:
            return []

        pairs = [[query, text] for _, text, _, _ in candidates]
        ce_scores = self._reranker.predict(pairs)

        # Normalise CE scores to [0, 1] so they can be blended with the
        # incoming RRF scores without scale mismatch.  The English-only CE
        # model can otherwise suppress strong BM25/keyword matches for
        # non-English queries, causing the wrong source to rank first.
        ce_arr = np.array(ce_scores, dtype=float)
        ce_min, ce_max = ce_arr.min(), ce_arr.max()
        ce_norm = (ce_arr - ce_min) / (ce_max - ce_min + 1e-9)

        rrf_arr = np.array([s for _, _, _, s in candidates], dtype=float)
        rrf_min, rrf_max = rrf_arr.min(), rrf_arr.max()
        rrf_norm = (rrf_arr - rrf_min) / (rrf_max - rrf_min + 1e-9)

        blended = self._CE_WEIGHT * ce_norm + (1.0 - self._CE_WEIGHT) * rrf_norm

        reranked = [
            (
                candidates[i][0],
                candidates[i][1],
                candidates[i][2],
                float(blended[i]),
            )
            for i in range(len(candidates))
        ]
        reranked.sort(key=lambda x: x[3], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _build_citation(metadata: dict, text: str, query: str = "") -> Citation:
        """Build a Citation from chunk metadata.

        Selects a ~500-character snippet centred on the region of *text*
        that has the highest keyword overlap with *query*, so the
        citation surfaces the most relevant content rather than the
        first 200 characters.

        Args:
            metadata: Chunk metadata dict.
            text: Full chunk text.
            query: The user's query (used to pick a relevant window).

        Returns:
            Citation with filename, page, and snippet.
        """
        snippet = _extract_relevant_snippet(text, query, max_len=500)
        return Citation(
            filename=metadata.get("filename", "unknown"),
            page=metadata.get("page"),
            snippet=snippet,
            source_path=metadata.get("source", ""),
        )

    def _filename_search(self, query: str) -> list[tuple[str, str, dict, float]]:
        """Fetch all chunks for any filename explicitly mentioned in the query.

        Uses ChromaDB metadata filtering, bypassing BM25 and semantic search,
        so the exact file is always retrievable regardless of embedding similarity.

        Args:
            query: Natural language query string.

        Returns:
            List of (id, text, metadata, score) tuples with a high fixed score.
        """
        results = []
        for match in self._FILENAME_RE.finditer(query):
            fname = match.group(1)
            try:
                resp = self._collection.get(
                    where={"filename": fname},
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                logger.debug("Filename metadata lookup failed for %s: %s", fname, exc)
                continue
            for doc_id, text, meta in zip(
                resp["ids"], resp["documents"] or [], resp["metadatas"] or []
            ):
                results.append((doc_id, text, meta, 2.0))
        return results

    def _phrase_search(self, query: str) -> list[tuple[str, str, dict, float]]:
        """Fetch chunks containing proper-noun phrases found in the query.

        Detects sequences of 2+ consecutive title-case words (e.g. person names,
        company names) and does a ChromaDB verbatim substring search for each,
        guaranteeing that chunks containing the exact phrase reach the reranker.

        Args:
            query: Natural language query string.

        Returns:
            List of (id, text, metadata, score) tuples with a fixed score of 1.5.
        """
        results = []
        seen: set[str] = set()
        for match in self._PROPER_NOUN_RE.finditer(query):
            phrase = match.group(1)
            try:
                resp = self._collection.get(
                    where_document={"$contains": phrase},
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                logger.debug("Phrase lookup failed for %r: %s", phrase, exc)
                continue
            for doc_id, text, meta in zip(
                resp["ids"], resp["documents"] or [], resp["metadatas"] or []
            ):
                if doc_id not in seen:
                    seen.add(doc_id)
                    results.append((doc_id, text, meta, 1.5))
        return results

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        semantic_candidates: int = 20,
        bm25_candidates: int = 20,
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

        # Step 3b: Prepend any chunks for filenames explicitly named in the query
        filename_hits = self._filename_search(query_text)
        if filename_hits:
            seen_ids = {hit[0] for hit in filename_hits}
            fused = [c for c in fused if c[0] not in seen_ids]
            candidates = filename_hits + fused
        else:
            candidates = fused

        # Step 3c: Append chunks containing proper-noun phrases from the query
        phrase_hits = self._phrase_search(query_text)
        if phrase_hits:
            seen_ids = {c[0] for c in candidates}
            candidates += [c for c in phrase_hits if c[0] not in seen_ids]

        # Step 4: Cross-encoder reranking
        reranked = self._rerank(query_text, candidates, top_k)

        # Step 5: Build results with citations
        results = []
        for doc_id, text, meta, score in reranked:
            results.append(
                RetrievalResult(
                    chunk_id=doc_id,
                    text=text,
                    score=score,
                    citation=self._build_citation(meta, text, query_text),
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
