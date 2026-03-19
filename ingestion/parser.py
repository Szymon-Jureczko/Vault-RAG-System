"""Multimodal document parsers — PyMuPDF, Docling+RapidOCR, RapidOCR.

Provides layout-aware parsing (Docling) for table-to-markdown reconstruction,
fast PDF extraction (PyMuPDF), and OCR for scanned images (RapidOCR via
ONNX Runtime).  Each parser returns a list of text chunks with source
metadata.

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

# ── OCR gating constants ────────────────────────────────────────────────────
_OCR_MIN_FILE_BYTES: int = 10_000  # skip images < 10 KB
_OCR_VARIANCE_THRESHOLD: float = 100.0  # skip near-blank/solid-colour images
_OCR_TARGET_WIDTH: int = 1800  # rescale to ~300 DPI equivalent

# ── PDF text-density probe ──────────────────────────────────────────────────
_PDF_TEXT_THRESHOLD: int = 50  # avg chars/page below this → scanned


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
        segment = text[i : i + chunk_size]
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
        return OFFICE_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Parse DOCX using Docling's layout engine.

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


# ── Scanned PDF parser (Docling + RapidOCR) ────────────────────────────────


class ScannedPDFParser(BaseParser):
    """Parser for scanned PDFs using Docling with RapidOCR.

    Uses ``RapidOcrOptions`` so Docling's internal OCR uses RapidOCR
    (ONNX Runtime) instead of Tesseract.  The ``DocumentConverter`` is
    lazily initialised once per process and cached to avoid reloading
    ONNX models per file.
    """

    _converter = None  # class-level lazy singleton

    @classmethod
    def _get_converter(cls):
        if cls._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                RapidOcrOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                ocr_options=RapidOcrOptions(),
            )
            cls._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        return cls._converter

    @property
    def name(self) -> str:
        return "scanned_pdf"

    @property
    def extensions(self) -> set[str]:
        return PDF_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Extract text from a scanned PDF using Docling + RapidOCR.

        Args:
            path: Path to the PDF file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with OCR-extracted chunks.
        """
        try:
            conv = self._get_converter().convert(str(path))
            text = conv.document.export_to_markdown()
            page_count = (
                len(conv.document.pages) if hasattr(conv.document, "pages") else 0
            )
            chunks = _split_text(text, path, chunk_size, self.name)
            return ParserResult(
                file_path=str(path),
                chunks=chunks,
                parser_name=self.name,
                page_count=page_count,
                success=True,
            )
        except Exception as exc:
            logger.error("ScannedPDFParser failed on %s: %s", path, exc)
            return ParserResult(
                file_path=str(path),
                success=False,
                error=str(exc),
                parser_name=self.name,
            )


# ── RapidOCR parser (images, replaces Tesseract) ──────────────────────────


class RapidOCRParser(BaseParser):
    """OCR parser for scanned images using RapidOCR (ONNX Runtime).

    Replaces TesseractParser with in-process inference (no subprocess
    overhead) and two fast gates to skip non-text images.
    """

    _engine = None  # class-level lazy singleton per process

    @classmethod
    def _get_engine(cls):
        if cls._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            cls._engine = RapidOCR()
        return cls._engine

    @property
    def name(self) -> str:
        return "rapidocr"

    @property
    def extensions(self) -> set[str]:
        return IMAGE_EXTENSIONS

    def parse(self, path: Path, chunk_size: int = 1000) -> ParserResult:
        """Extract text from an image using RapidOCR.

        Args:
            path: Path to the image file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParserResult with OCR-extracted chunks.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            return ParserResult(
                file_path=str(path),
                success=False,
                error=f"numpy/Pillow not installed: {exc}",
                parser_name=self.name,
            )

        try:
            # Gate 1: skip tiny files (icons, bullets, decorative)
            if path.stat().st_size < _OCR_MIN_FILE_BYTES:
                logger.debug("Skipping OCR on undersized image %s", path)
                return ParserResult(
                    file_path=str(path),
                    chunks=[],
                    parser_name=self.name,
                    page_count=1,
                    success=True,
                )

            image = Image.open(path)

            # Gate 2: pixel-variance — blank/solid-colour images
            gray_arr = np.array(image.convert("L"))
            if float(gray_arr.var()) < _OCR_VARIANCE_THRESHOLD:
                logger.debug("OCR skipped: low-variance image %s", path)
                return ParserResult(
                    file_path=str(path),
                    chunks=[],
                    parser_name=self.name,
                    page_count=1,
                    success=True,
                )

            # Preprocessing: greyscale + rescale to optimal OCR width
            gray = image.convert("L")
            w, h = gray.size
            if w < 800 or w > 3000:
                scale = _OCR_TARGET_WIDTH / w
                gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            result, _ = self._get_engine()(np.array(gray))
            text = "\n".join(line[1] for line in result) if result else ""
            chunks = _split_text(text, path, chunk_size, self.name, page=1)

            return ParserResult(
                file_path=str(path),
                chunks=chunks,
                parser_name=self.name,
                page_count=1,
                success=True,
            )
        except Exception as exc:
            logger.error("RapidOCR failed on %s: %s", path, exc)
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


# ── PDF text-density probe ──────────────────────────────────────────────────


def is_scanned_pdf(file_path: Path) -> bool:
    """Probe PDF text density to decide whether OCR is needed.

    Opens the file, extracts raw text from all pages, and returns True
    if the average character count per page is below
    ``_PDF_TEXT_THRESHOLD``.  The probe costs ~5-20 ms versus 1-30 s
    for a full OCR pass.

    Args:
        file_path: Path to the PDF.

    Returns:
        True if the PDF is likely scanned / requires OCR.
    """
    try:
        import fitz

        doc = fitz.open(str(file_path))
        total_chars = sum(len(page.get_text().strip()) for page in doc)
        avg = total_chars / max(len(doc), 1)
        doc.close()
        return avg < _PDF_TEXT_THRESHOLD
    except Exception:
        return True  # assume scanned — OCR attempt is safer than losing content


# ── Dispatcher ──────────────────────────────────────────────────────────────

# Non-PDF parsers — PDFs are routed dynamically by parse_file().
_PARSERS: list[BaseParser] = [
    OpenpyxlParser(),
    DoclingParser(),
    RapidOCRParser(),
    TextParser(),
    EmlParser(),
]

_EXT_MAP: dict[str, BaseParser] = {}
for _p in _PARSERS:
    for _ext in _p.extensions:
        if _ext not in _EXT_MAP:
            _EXT_MAP[_ext] = _p

SUPPORTED_EXTENSIONS = set(_EXT_MAP.keys()) | PDF_EXTENSIONS


def parse_file(path: Path, chunk_size: int = 1000) -> ParserResult:
    """Parse a file using the best available parser.

    PDFs are routed via a cheap text-density probe:
    - Text-layer PDFs  -> ``PyMuPDFParser`` (fast, no OCR)
    - Scanned PDFs     -> ``ScannedPDFParser`` (Docling + RapidOCR)

    All other files are routed by extension via ``_EXT_MAP``.

    Args:
        path: Path to the source document.
        chunk_size: Approximate characters per chunk.

    Returns:
        ParserResult from the matched parser.
    """
    ext = path.suffix.lower()

    if ext in PDF_EXTENSIONS:
        if is_scanned_pdf(path):
            parser: BaseParser = ScannedPDFParser()
        else:
            parser = PyMuPDFParser()
    else:
        parser = _EXT_MAP.get(ext)  # type: ignore[assignment]

    if parser is None:
        return ParserResult(
            file_path=str(path),
            success=False,
            error=f"Unsupported extension: {ext}",
        )
    logger.info("Parsing %s with %s", path.name, parser.name)
    return parser.parse(path, chunk_size=chunk_size)
