"""Tests for vector_db.retriever.HybridRetriever.

Three test classes:
- TestBM25Tokenizer   — pure unit tests for the regex tokenizer fix (no DB)
- TestFilenameRegex   — pure unit tests for the _FILENAME_RE class pattern
- TestHybridRetrieverIntegration — integration tests against the real data/chroma DB
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from vector_db.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Unit: BM25 tokenizer
# ---------------------------------------------------------------------------


class TestBM25Tokenizer:
    """Verify re.findall(r'\\w+', ...) strips brackets and punctuation that
    previously caused filename tokens to never match between corpus and query."""

    def test_corpus_token_strips_source_bracket(self):
        """[Source: img_0031.png] must tokenise to 'img_0031', not 'img_0031.png]'."""
        text = "[Source: img_0031.png]\nsome text"
        tokens = re.findall(r"\w+", text.lower())
        assert "img_0031" in tokens
        assert all("]" not in t for t in tokens)

    def test_query_token_strips_question_mark(self):
        """Trailing '?' in a filename query must not survive tokenisation."""
        query = "what can you tell me about img_0031.png?"
        tokens = re.findall(r"\w+", query.lower())
        assert "img_0031" in tokens
        assert all("?" not in t for t in tokens)

    def test_bm25_score_positive_for_filename_match(self):
        """The chunk whose [Source:] prefix matches the queried filename must
        rank above chunks that do not contain that filename.

        Uses 3 documents with distinct extensions so filename tokens are unique
        to one document (n=1, N=3 → IDF > 0 in BM25Okapi).
        """
        match_text = "[Source: img_0031.png]\nMeeting minutes quarterly review"
        other_a = "[Source: log_2024.txt]\nTechnical specification project"
        other_b = "[Source: report_data.csv]\nBudget analysis annual figures"
        tokenized = [
            re.findall(r"\w+", t.lower())
            for t in (match_text, other_a, other_b)
        ]
        bm25 = BM25Okapi(tokenized)
        query_tokens = re.findall(
            r"\w+", "what can you tell me about img_0031.png?".lower()
        )
        scores = bm25.get_scores(query_tokens)
        assert scores[0] > 0
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_old_split_would_fail(self):
        """Demonstrate the original bug: .split() leaves brackets/punctuation
        attached so tokens never match."""
        corpus_text = "[Source: img_0031.png]\nsome text"
        old_corpus_tokens = corpus_text.lower().split()
        old_query_tokens = "what can you tell me about img_0031.png?".lower().split()
        # The corpus has 'img_0031.png]', the query has 'img_0031.png?' — no overlap
        corpus_filename_tokens = {t for t in old_corpus_tokens if "img_0031" in t}
        query_filename_tokens = {t for t in old_query_tokens if "img_0031" in t}
        assert corpus_filename_tokens.isdisjoint(query_filename_tokens)


# ---------------------------------------------------------------------------
# Unit: _FILENAME_RE regex
# ---------------------------------------------------------------------------


class TestFilenameRegex:
    """Verify HybridRetriever._FILENAME_RE matches the expected filename patterns."""

    def test_matches_png(self):
        """PNG filenames are detected."""
        assert HybridRetriever._FILENAME_RE.search("tell me about img_0031.png")

    def test_matches_pdf(self):
        """PDF filenames are detected, including before a question mark."""
        assert HybridRetriever._FILENAME_RE.search("what is in scanned_pdf_0038.pdf?")

    def test_matches_eml(self):
        """Email filenames are detected."""
        assert HybridRetriever._FILENAME_RE.search("show me eml_0038.eml")

    def test_matches_docx(self):
        """DOCX filenames are detected."""
        assert HybridRetriever._FILENAME_RE.search("summarise docx_0010.docx")

    def test_no_match_plain_sentence(self):
        """Plain natural-language sentences without a filename are not matched."""
        assert not HybridRetriever._FILENAME_RE.search(
            "what is the quarterly budget proposal?"
        )

    def test_extracts_correct_filename(self):
        """The captured group is the bare filename without surrounding punctuation."""
        m = HybridRetriever._FILENAME_RE.search("about img_0031.png please")
        assert m is not None
        assert m.group(1) == "img_0031.png"

    def test_case_insensitive(self):
        """Matches regardless of extension casing."""
        assert HybridRetriever._FILENAME_RE.search("file.PDF")
        assert HybridRetriever._FILENAME_RE.search("file.Png")


# ---------------------------------------------------------------------------
# Unit: proper-noun / phrase regex
# ---------------------------------------------------------------------------


class TestProperNounRegex:
    """Verify HybridRetriever._PROPER_NOUN_RE captures multi-word title-case phrases."""

    def test_matches_full_name(self):
        """Two-word person names are captured."""
        m = HybridRetriever._PROPER_NOUN_RE.search("authored by Angela Graham")
        assert m is not None
        assert m.group(1) == "Angela Graham"

    def test_matches_hyphenated_company(self):
        """Hyphenated company names are captured."""
        m = HybridRetriever._PROPER_NOUN_RE.search("documents about Ward-Hardy")
        assert m is not None
        assert m.group(1) == "Ward-Hardy"

    def test_matches_three_word_name(self):
        """Three-word names are captured as a single phrase."""
        m = HybridRetriever._PROPER_NOUN_RE.search("by Angela Marie Graham")
        assert m is not None
        assert m.group(1) == "Angela Marie Graham"

    def test_no_match_single_word(self):
        """A single capitalised word does not match (too ambiguous)."""
        assert not HybridRetriever._PROPER_NOUN_RE.search("what is the Budget?")

    def test_no_match_lowercase_sentence(self):
        """Lowercase text is not matched."""
        assert not HybridRetriever._PROPER_NOUN_RE.search(
            "what do all documents concern?"
        )


# ---------------------------------------------------------------------------
# Integration: real data/chroma DB
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retriever():
    """Module-scoped HybridRetriever pointing at the real persistent DB.

    Module scope is used because loading CrossEncoder + the embedding model
    is expensive; this keeps the cost to one load per test session.
    """
    return HybridRetriever(chroma_path=Path("data/chroma"))


class TestHybridRetrieverIntegration:
    """End-to-end tests against the real data/chroma collection (1,405 chunks)."""

    def test_filename_search_direct(self, retriever):
        """_filename_search must return at least one chunk whose metadata
        filename matches exactly when queried by name."""
        hits = retriever._filename_search("what can you tell me about img_0031.png?")
        assert len(hits) > 0
        assert all(meta.get("filename") == "img_0031.png" for _, _, meta, _ in hits)

    def test_filename_query_returns_correct_file(self, retriever):
        """The main regression test: querying by filename must surface that file
        in the top results."""
        results = retriever.query("what can you tell me about img_0031.png?", top_k=5)
        filenames = [r.citation.filename for r in results]
        assert "img_0031.png" in filenames

    def test_semantic_query_still_works(self, retriever):
        """A regular semantic query (no filename) must still return top_k results."""
        results = retriever.query("quarterly budget proposal", top_k=5)
        assert len(results) == 5

    def test_nonexistent_filename_falls_back_gracefully(self, retriever):
        """A filename absent from the DB must not crash the pipeline; the
        retriever should fall back to semantic/BM25 results silently."""
        results = retriever.query("tell me about ghost_file_9999.png", top_k=5)
        assert isinstance(results, list)
        filenames = [r.citation.filename for r in results]
        assert "ghost_file_9999.png" not in filenames

    def test_phrase_search_direct(self, retriever):
        """_phrase_search must return at least one chunk whose text contains
        'Angela Graham' verbatim (stored in img_0037.png)."""
        hits = retriever._phrase_search(
            "What do documents authored by Angela Graham concern?"
        )
        assert len(hits) > 0
        filenames = {meta.get("filename") for _, _, meta, _ in hits}
        assert "img_0037.png" in filenames

    def test_author_query_returns_correct_file(self, retriever):
        """End-to-end: querying by author name must surface the correct document."""
        results = retriever.query(
            "What do documents authored by Angela Graham concern?", top_k=5
        )
        filenames = [r.citation.filename for r in results]
        assert "img_0037.png" in filenames
