"""Document parsers — Docling (layout-aware), PyMuPDF (fast), Tesseract (OCR).

Each parser implements a common interface and returns a ``ParseResult``.
The module provides a ``parse_file`` dispatcher that selects the best
parser for a given file type.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from src.models import DocumentChunk, ParseResult

logger = logging.getLogger(__name__)

# ── Supported extensions ────────────────────────────────────────────────────
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html"}


def _make_chunk_id(source: Path, index: int) -> str:
    """Generate a deterministic-ish chunk ID."""
    return f"{source.stem}_{index}_{uuid.uuid4().hex[:8]}"


# ── Base class ──────────────────────────────────────────────────────────────


class BaseParser(ABC):
    """Abstract base class for document parsers.

    Subclasses must implement ``parse`` and ``supported_extensions``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable parser name."""

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Set of file extensions this parser handles (lowercase, with dot)."""

    @abstractmethod
    def parse(self, file_path: Path, chunk_size: int = 1000) -> ParseResult:
        """Parse a document and return chunked text.

        Args:
            file_path: Path to the file to parse.
            chunk_size: Approximate characters per chunk.

        Returns:
            A ParseResult with extracted chunks.
        """


# ── PyMuPDF parser (fast PDF) ──────────────────────────────────────────────


class PyMuPDFParser(BaseParser):
    """Fast PDF parser using PyMuPDF (fitz)."""

    @property
    def name(self) -> str:
        return "pymupdf"

    @property
    def supported_extensions(self) -> set[str]:
        return PDF_EXTENSIONS

    def parse(self, file_path: Path, chunk_size: int = 1000) -> ParseResult:
        """Extract text from PDF pages using PyMuPDF.

        Args:
            file_path: Path to the PDF file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParseResult containing document chunks.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            return ParseResult(
                source_path=file_path,
                success=False,
                error_message=f"PyMuPDF not installed: {exc}",
                parser_used=self.name,
            )

        try:
            doc = fitz.open(str(file_path))
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()

            chunks = _split_text(full_text, file_path, chunk_size, self.name)

            return ParseResult(
                source_path=file_path,
                chunks=chunks,
                parser_used=self.name,
                page_count=len(doc) if hasattr(doc, "__len__") else 0,
                success=True,
            )
        except Exception as exc:
            logger.error("PyMuPDF failed on %s: %s", file_path, exc)
            return ParseResult(
                source_path=file_path,
                success=False,
                error_message=str(exc),
                parser_used=self.name,
            )


# ── Tesseract OCR parser ───────────────────────────────────────────────────


class TesseractParser(BaseParser):
    """OCR parser using Tesseract via pytesseract."""

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def supported_extensions(self) -> set[str]:
        return IMAGE_EXTENSIONS

    def parse(self, file_path: Path, chunk_size: int = 1000) -> ParseResult:
        """Extract text from images using Tesseract OCR.

        Args:
            file_path: Path to the image file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParseResult containing document chunks.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            return ParseResult(
                source_path=file_path,
                success=False,
                error_message=f"pytesseract/Pillow not installed: {exc}",
                parser_used=self.name,
            )

        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            chunks = _split_text(text, file_path, chunk_size, self.name)

            return ParseResult(
                source_path=file_path,
                chunks=chunks,
                parser_used=self.name,
                page_count=1,
                success=True,
            )
        except Exception as exc:
            logger.error("Tesseract failed on %s: %s", file_path, exc)
            return ParseResult(
                source_path=file_path,
                success=False,
                error_message=str(exc),
                parser_used=self.name,
            )


# ── Plain-text parser ──────────────────────────────────────────────────────


class TextParser(BaseParser):
    """Simple parser for plain-text and structured-text files."""

    @property
    def name(self) -> str:
        return "text"

    @property
    def supported_extensions(self) -> set[str]:
        return TEXT_EXTENSIONS

    def parse(self, file_path: Path, chunk_size: int = 1000) -> ParseResult:
        """Read and chunk plain-text files.

        Args:
            file_path: Path to the text file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParseResult containing document chunks.
        """
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = _split_text(text, file_path, chunk_size, self.name)

            return ParseResult(
                source_path=file_path,
                chunks=chunks,
                parser_used=self.name,
                page_count=1,
                success=True,
            )
        except Exception as exc:
            logger.error("TextParser failed on %s: %s", file_path, exc)
            return ParseResult(
                source_path=file_path,
                success=False,
                error_message=str(exc),
                parser_used=self.name,
            )


# ── Helpers ─────────────────────────────────────────────────────────────────

_PARSERS: list[BaseParser] = [
    PyMuPDFParser(),
    TesseractParser(),
    TextParser(),
]


def _split_text(
    text: str, source: Path, chunk_size: int, parser_name: str
) -> list[DocumentChunk]:
    """Split a block of text into fixed-size chunks.

    Args:
        text: The full text to split.
        source: Source file path for metadata.
        chunk_size: Target chunk size in characters.
        parser_name: Name of the parser that produced the text.

    Returns:
        List of DocumentChunk instances.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[DocumentChunk] = []
    for i in range(0, len(text), chunk_size):
        segment = text[i : i + chunk_size]
        chunks.append(
            DocumentChunk(
                chunk_id=_make_chunk_id(source, len(chunks)),
                source_path=source,
                content=segment,
                metadata={
                    "parser": parser_name,
                    "chunk_index": len(chunks),
                    "char_offset": i,
                },
            )
        )
    return chunks


def get_parser(file_path: Path) -> BaseParser | None:
    """Select the appropriate parser for a file based on its extension.

    Args:
        file_path: Path to the file.

    Returns:
        A parser instance, or None if the extension is unsupported.
    """
    ext = file_path.suffix.lower()
    for parser in _PARSERS:
        if ext in parser.supported_extensions:
            return parser
    return None


def parse_file(file_path: Path, chunk_size: int = 1000) -> ParseResult:
    """Parse a file using the best available parser.

    Args:
        file_path: Path to the file to parse.
        chunk_size: Approximate characters per chunk.

    Returns:
        ParseResult from the selected parser, or a failed result if
        no parser supports the file extension.
    """
    parser = get_parser(file_path)
    if parser is None:
        return ParseResult(
            source_path=file_path,
            success=False,
            error_message=f"Unsupported file extension: {file_path.suffix}",
        )
    logger.info("Parsing %s with %s", file_path, parser.name)
    return parser.parse(file_path, chunk_size=chunk_size)
