"""Tests for ingestion.parser — multimodal document parsing."""

from __future__ import annotations

from pathlib import Path

from ingestion.parser import TextParser, _split_text, parse_file


class TestSplitText:
    """Tests for the _split_text helper."""

    def test_empty_text_returns_no_chunks(self, tmp_path: Path) -> None:
        chunks = _split_text("", tmp_path / "a.txt", 100, "test")
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self, tmp_path: Path) -> None:
        chunks = _split_text("   \n\n  ", tmp_path / "a.txt", 100, "test")
        assert chunks == []

    def test_single_chunk_for_short_text(self, tmp_path: Path) -> None:
        chunks = _split_text("hello world", tmp_path / "a.txt", 1000, "test")
        assert len(chunks) == 1
        assert "[Source: a.txt]" in chunks[0].text
        assert "hello world" in chunks[0].text

    def test_splits_into_multiple_chunks(
        self,
        tmp_path: Path,
    ) -> None:
        text = "a" * 250
        chunks = _split_text(text, tmp_path / "a.txt", 100, "test")
        assert len(chunks) > 1

    def test_metadata_includes_parser_name(self, tmp_path: Path) -> None:
        chunks = _split_text("hello", tmp_path / "a.txt", 100, "myparser")
        assert chunks[0].metadata["parser"] == "myparser"

    def test_metadata_includes_page_when_provided(
        self,
        tmp_path: Path,
    ) -> None:
        chunks = _split_text(
            "hello",
            tmp_path / "a.txt",
            100,
            "test",
            page=3,
        )
        assert chunks[0].metadata["page"] == 3

    def test_metadata_includes_filename(self, tmp_path: Path) -> None:
        chunks = _split_text("hello", tmp_path / "doc.pdf", 100, "test")
        assert chunks[0].metadata["filename"] == "doc.pdf"


class TestTextParser:
    """Tests for the TextParser."""

    def test_parse_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "sample.txt"
        f.write_text("This is a test document with some content.")
        parser = TextParser()
        result = parser.parse(f)
        assert result.success is True
        assert len(result.chunks) >= 1
        assert "test document" in result.chunks[0].text

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        parser = TextParser()
        result = parser.parse(tmp_path / "missing.txt")
        assert result.success is False
        assert result.error is not None


class TestParseFile:
    """Tests for the parse_file dispatcher."""

    def test_dispatches_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome markdown content.")
        result = parse_file(f)
        assert result.success is True
        assert result.parser_name == "text"

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("unknown format")
        result = parse_file(f)
        assert result.success is False
        assert "Unsupported" in (result.error or "")

    def test_csv_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\nval1,val2\n")
        result = parse_file(f)
        assert result.success is True
        assert result.parser_name == "text"

    def test_chunk_size_respected(self, tmp_path: Path) -> None:
        f = tmp_path / "long.txt"
        f.write_text("word " * 500)
        result = parse_file(f, chunk_size=100)
        assert result.success is True
        assert len(result.chunks) > 1
        # Each chunk is chunk_size content + [Source: filename] prefix
        prefix_len = len("[Source: long.txt]\n")
        assert all(
            len(c.text) <= 100 + prefix_len for c in result.chunks
        )
