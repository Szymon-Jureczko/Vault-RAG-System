"""Multimodal document parsers — Docling, PyMuPDF, Tesseract OCR.

Provides layout-aware parsing (Docling) for table-to-markdown reconstruction,
fast PDF extraction (PyMuPDF), and OCR for scanned images (Tesseract).
Each parser returns a list of text chunks with source metadata.

Usage::

    from ingestion.parser import parse_file

    result = parse_file(Path("report.pdf"), chunk_size=1000)
    for chunk in result.chunks:
        print(chunk.text, chunk.metadata)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Supported extension sets ────────────────────────────────────────────────
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html"}
OFFICE_EXTENSIONS = {".docx"}
XLSX_EXTENSIONS = {".xlsx"}
EML_EXTENSIONS = {".eml"}


class Chunk(BaseModel):
    """A single text chunk extracted from a document.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The extracted text content.
        metadata: Source metadata (filename, page, parser, etc.).
    """

    chunk_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class ParserResult(BaseModel):
    """Output from parsing a single document.

    Attributes:
        file_path: Path to the source document.
        chunks: Extracted text chunks.
        parser_name: Parser that produced the result.
        page_count: Number of pages in the source (if applicable).
        success: Whether parsing completed without errors.
        error: Error description if success is False.
    """

    file_path: str
    chunks: list[Chunk] = Field(default_factory=list)
    parser_name: str = ""
    page_count: int = 0
    success: bool = True
    error: str | None = None


def _chunk_id(source: Path, index: int) -> str:
    """Generate a unique chunk identifier."""
    return f"{source.stem}_{index}_{uuid.uuid4().hex[:8]}"


def _split_text(
    text: str,
    source: Path,
    chunk_size: int,
    parser_name: str,
    page: int | None = None,
) -> list[Chunk]:
    """Split text into fixed-size chunks with metadata.

    Args:
        text: Full extracted text.
        source: Source file path.
        chunk_size: Target characters per chunk.
        parser_name: Name of the parser that produced the text.
        page: Optional page number for page-level chunks.

    Returns:
        List of Chunk instances.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    for i in range(0, len(text), chunk_size):
        segment = text[i:i + chunk_size]
        meta: dict = {
            "source": str(source),
            "filename": source.name,
            "parser": parser_name,
            "chunk_index": len(chunks),
        }
        if page is not None:
            meta["page"] = page
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source, len(chunks)),
                text=segment,
                metadata=meta,
            )
        )
    return chunks


# ── Base parser ─────────────────────────────────────────────────────────────


class BaseParser(ABC):
    """Abstract base for document parsers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable parser name."""

    @property
    @abstractmethod
    def extensions(self) -> set[str]:
        """File extensions this parser handles (lowercase, with dot)."""

    @abstractmethod
    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Parse a document into text chunks.

        Args:
            path: Path to the source file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with extracted chunks.
        """


# ── Docling parser (layout-aware, table → markdown) ────────────────────────


class DoclingParser(BaseParser):
    """Layout-aware parser using Docling for tables."""

    @property
    def name(self) -> str:
        return "docling"

    @property
    def extensions(self) -> set[str]:
        return PDF_EXTENSIONS | OFFICE_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Parse PDF/DOCX/XLSX using Docling's layout engine.

        Args:
            path: Path to the document.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with markdown-formatted chunks.
        """
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            return ParserResult(
                file_path=str(path),
                success=False,
                error=f"Docling not installed: {exc}",
                parser_name=self.name,
            )

        try:
            converter = DocumentConverter()
            result = converter.convert(str(path))
            markdown_text = result.document.export_to_markdown()

            chunks = _split_text(markdown_text, path, chunk_size, self.name)
            return ParserResult(
                file_path=str(path),
                chunks=chunks,
                parser_name=self.name,
                page_count=result.document.num_pages(),
                success=True,
            )
        except Exception as exc:
            logger.error("Docling failed on %s: %s", path, exc)
            return ParserResult(
                file_path=str(path),
                success=False,
                error=str(exc),
                parser_name=self.name,
            )


# ── PyMuPDF parser (fast PDF) ──────────────────────────────────────────────


class PyMuPDFParser(BaseParser):
    """Fast PDF text extraction using PyMuPDF."""

    @property
    def name(self) -> str:
        return "pymupdf"

    @property
    def extensions(self) -> set[str]:
        return PDF_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Extract text from PDF pages using PyMuPDF (fitz).

        Args:
            path: Path to the PDF file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with page-level chunks.
        """
        try:
            import fitz
        except ImportError as exc:
            return ParserResult(
                file_path=str(path),
                success=False,
                error=f"PyMuPDF not installed: {exc}",
                parser_name=self.name,
            )

        try:
            doc = fitz.open(str(path))
            all_chunks: list[Chunk] = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                page_chunks = _split_text(
                    text, path, chunk_size, self.name, page=page_num
                )
                all_chunks.extend(page_chunks)
            page_count = len(doc)
            doc.close()

            return ParserResult(
                file_path=str(path),
                chunks=all_chunks,
                parser_name=self.name,
                page_count=page_count,
                success=True,
            )
        except Exception as exc:
            logger.error("PyMuPDF failed on %s: %s", path, exc)
            return ParserResult(
                file_path=str(path),
                success=False,
                error=str(exc),
                parser_name=self.name,
            )


# ── Tesseract OCR parser ───────────────────────────────────────────────────


class TesseractParser(BaseParser):
    """OCR parser for scanned images using Tesseract."""

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def extensions(self) -> set[str]:
        return IMAGE_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Extract text from images via Tesseract OCR.

        Args:
            path: Path to the image file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with OCR-extracted chunks.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            return ParserResult(
                file_path=str(path),
                success=False,
                error=f"pytesseract/Pillow not installed: {exc}",
                parser_name=self.name,
            )

        try:
            image = Image.open(path)
            # Resize large images to prevent OOM in worker processes
            max_dim = 2000
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim), Image.LANCZOS)
            text = pytesseract.image_to_string(image)
            image.close()
            chunks = _split_text(text, path, chunk_size, self.name, page=1)

            return ParserResult(
                file_path=str(path),
                chunks=chunks,
                parser_name=self.name,
                page_count=1,
                success=True,
            )
        except Exception as exc:
            logger.error("Tesseract failed on %s: %s", path, exc)
            return ParserResult(
                file_path=str(path),
                success=False,
                error=str(exc),
                parser_name=self.name,
            )


# ── Plain-text parser ──────────────────────────────────────────────────────


class TextParser(BaseParser):
    """Parser for plain-text and structured-text files."""

    @property
    def name(self) -> str:
        return "text"

    @property
    def extensions(self) -> set[str]:
        return TEXT_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Read and chunk plain-text files.

        Args:
            path: Path to the text file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with text chunks.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks = _split_text(text, path, chunk_size, self.name)

            return ParserResult(
                file_path=str(path),
                chunks=chunks,
                parser_name=self.name,
                page_count=1,
                success=True,
            )
        except Exception as exc:
            logger.error("TextParser failed on %s: %s", path, exc)
            return ParserResult(
                file_path=str(path),
                success=False,
                error=str(exc),
                parser_name=self.name,
            )


# ── XLSX parser (openpyxl, lightweight) ───────────────────────────────────


class OpenpyxlParser(BaseParser):
    """Lightweight XLSX parser using openpyxl instead of Docling."""

    @property
    def name(self) -> str:
        return "openpyxl"

    @property
    def extensions(self) -> set[str]:
        return XLSX_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            return ParserResult(
                file_path=str(path),
                success=False,
                error=f"openpyxl not installed: {exc}",
                parser_name=self.name,
            )

        try:
            wb = load_workbook(str(path), read_only=True, data_only=True)
            parts: list[str] = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows: list[str] = [f"## Sheet: {sheet_name}"]
                for row in ws.iter_rows(values_only=True):
                    cells = [str(c) if c is not None else "" for c in row]
                    rows.append(" | ".join(cells))
                parts.append("\n".join(rows))
            wb.close()

            full_text = "\n\n".join(parts)
            chunks = _split_text(full_text, path, chunk_size, self.name)
            return ParserResult(
                file_path=str(path),
                chunks=chunks,
                parser_name=self.name,
                page_count=len(wb.sheetnames) if parts else 0,
                success=True,
            )
        except Exception as exc:
            logger.error("OpenpyxlParser failed on %s: %s", path, exc)
            return ParserResult(
                file_path=str(path),
                success=False,
                error=str(exc),
                parser_name=self.name,
            )


# ── EML parser ─────────────────────────────────────────────────────────────


class EmlParser(BaseParser):
    """Parser for RFC-822 email files (.eml) using stdlib ``email``."""

    @property
    def name(self) -> str:
        return "eml"

    @property
    def extensions(self) -> set[str]:
        return EML_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Extract headers and body text from an EML file.

        Args:
            path: Path to the .eml file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with email content as text chunks.
        """
        import email as _email

        try:
            msg = _email.message_from_bytes(path.read_bytes())
            parts: list[str] = []

            # Include key headers as context
            for header in ("Subject", "From", "To", "CC", "Date"):
                val = msg.get(header)
                if val:
                    parts.append(f"{header}: {val}")

            # Extract all text/plain and text/html payloads
            for part in msg.walk():
                ct = part.get_content_type()
                if ct not in ("text/plain", "text/html"):
                    continue
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    parts.append(payload.decode(charset, errors="replace"))

            full_text = "\n\n".join(parts)
            chunks = _split_text(full_text, path, chunk_size, self.name)
            return ParserResult(
                file_path=str(path),
                chunks=chunks,
                parser_name=self.name,
                page_count=1,
                success=True,
            )
        except Exception as exc:
            logger.error("EmlParser failed on %s: %s", path, exc)
            return ParserResult(
                file_path=str(path),
                success=False,
                error=str(exc),
                parser_name=self.name,
            )


# ── Dispatcher ──────────────────────────────────────────────────────────────

_PARSERS: list[BaseParser] = [
    OpenpyxlParser(),
    DoclingParser(),
    PyMuPDFParser(),
    TesseractParser(),
    TextParser(),
    EmlParser(),
]

# Extension → parser lookup (first match wins; Docling for PDFs)
_EXT_MAP: dict[str, BaseParser] = {}
for _p in _PARSERS:
    for _ext in _p.extensions:
        if _ext not in _EXT_MAP:
            _EXT_MAP[_ext] = _p

SUPPORTED_EXTENSIONS = set(_EXT_MAP.keys())


def parse_file(path: Path, chunk_size: int = 1000) -> ParserResult:
    """Parse a file using the best available parser.

    Args:
        path: Path to the source document.
        chunk_size: Approximate characters per chunk.

    Returns:
        ParserResult from the matched parser.
    """
    ext = path.suffix.lower()
    parser = _EXT_MAP.get(ext)
    if parser is None:
        return ParserResult(
            file_path=str(path),
            success=False,
            error=f"Unsupported extension: {ext}",
        )
    logger.info("Parsing %s with %s", path.name, parser.name)
    return parser.parse(path, chunk_size=chunk_size)
