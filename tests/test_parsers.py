"""Tests for src.parsers — document parsing and chunking."""

from __future__ import annotations

from pathlib import Path

from src.parsers import TextParser, get_parser, parse_file


class TestTextParser:
    """Tests for the plain-text parser."""

    def test_parse_text_file(self, tmp_path: Path) -> None:
        """TextParser should correctly parse a .txt file into chunks."""
        f = tmp_path / "sample.txt"
        f.write_text("A" * 2500)

        parser = TextParser()
        result = parser.parse(f, chunk_size=1000)

        assert result.success is True
        assert result.parser_used == "text"
        assert len(result.chunks) == 3  # 2500 / 1000 = 3 chunks

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty file produces zero chunks but succeeds."""
        f = tmp_path / "empty.txt"
        f.write_text("")

        parser = TextParser()
        result = parser.parse(f, chunk_size=1000)

        assert result.success is True
        assert len(result.chunks) == 0


class TestGetParser:
    """Tests for the parser dispatcher."""

    def test_pdf_extension(self) -> None:
        """PDF files return None from get_parser(); routing is done by parse_file().

        PDFs require content-aware routing (text-layer probe) that lives in
        parse_file(), not in get_parser(), so the registry intentionally
        excludes PDF extensions.
        """
        parser = get_parser(Path("doc.pdf"))
        assert parser is None

    def test_txt_extension(self) -> None:
        """Text files should resolve to TextParser."""
        parser = get_parser(Path("readme.txt"))
        assert parser is not None
        assert parser.name == "text"

    def test_image_extension(self) -> None:
        """Image files should resolve to RapidOCRParser."""
        parser = get_parser(Path("scan.png"))
        assert parser is not None
        assert parser.name == "rapidocr"

    def test_unsupported_extension(self) -> None:
        """Unsupported extensions return None."""
        assert get_parser(Path("file.xyz")) is None


class TestParseFile:
    """Tests for the top-level parse_file function."""

    def test_unsupported_returns_failure(self) -> None:
        """parse_file with unsupported extension returns a failed result."""
        result = parse_file(Path("file.xyz"))
        assert result.success is False
        assert "Unsupported" in (result.error_message or "")
