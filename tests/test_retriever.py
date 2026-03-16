"""Tests for vector_db.retriever — citation model and RRF fusion."""

from __future__ import annotations


from vector_db.retriever import Citation, HybridRetriever, RetrievalResult


class TestCitation:
    """Tests for the Citation model."""

    def test_citation_defaults(self) -> None:
        c = Citation(filename="test.pdf")
        assert c.filename == "test.pdf"
        assert c.page is None
        assert c.snippet == ""

    def test_citation_with_page(self) -> None:
        c = Citation(
            filename="report.pdf",
            page=5,
            snippet="The key finding...",
        )
        assert c.page == 5
        assert "key finding" in c.snippet


class TestRetrievalResult:
    """Tests for the RetrievalResult model."""

    def test_result_defaults(self) -> None:
        r = RetrievalResult(chunk_id="abc", text="some text")
        assert r.score == 0.0
        assert r.citation.filename == ""

    def test_result_with_citation(self) -> None:
        r = RetrievalResult(
            chunk_id="abc",
            text="some text",
            score=0.95,
            citation=Citation(filename="doc.pdf", page=3),
        )
        assert r.score == 0.95
        assert r.citation.filename == "doc.pdf"


class TestReciprocalRankFusion:
    """Tests for the RRF merging algorithm."""

    def test_single_list(self) -> None:
        results = [
            ("id1", "text1", {"src": "a"}, 0.9),
            ("id2", "text2", {"src": "b"}, 0.8),
        ]
        fused = HybridRetriever._reciprocal_rank_fusion(results)
        assert len(fused) == 2
        assert fused[0][0] == "id1"  # higher rank first

    def test_two_lists_merges(self) -> None:
        list1 = [
            ("id1", "text1", {"src": "a"}, 0.9),
            ("id2", "text2", {"src": "b"}, 0.7),
        ]
        list2 = [
            ("id2", "text2", {"src": "b"}, 5.0),
            ("id3", "text3", {"src": "c"}, 3.0),
        ]
        fused = HybridRetriever._reciprocal_rank_fusion(list1, list2)
        ids = [f[0] for f in fused]
        # id2 appears in both lists so should rank higher
        assert ids.index("id2") < ids.index("id3")

    def test_empty_lists(self) -> None:
        fused = HybridRetriever._reciprocal_rank_fusion([], [])
        assert fused == []


class TestBuildCitation:
    """Tests for citation building from metadata."""

    def test_builds_from_metadata(self) -> None:
        meta = {
            "filename": "report.pdf",
            "page": 2,
            "source": "/data/report.pdf",
        }
        text = "This is the first chunk of text from the document."
        c = HybridRetriever._build_citation(meta, text)
        assert c.filename == "report.pdf"
        assert c.page == 2
        assert c.source_path == "/data/report.pdf"
        assert len(c.snippet) <= 200

    def test_missing_metadata_fields(self) -> None:
        c = HybridRetriever._build_citation({}, "hello")
        assert c.filename == "unknown"
        assert c.page is None
