"""Document parsers: PyMuPDF (fast PDF), Docling (scanned), RapidOCR (images).

Each parser implements a common interface and returns a ``ParseResult``.
The module provides a ``parse_file`` dispatcher that selects the best
parser for a given file type.

OCR is only invoked when text cannot be extracted directly:
- Image files always require OCR.
- PDFs are first probed for embedded text; only scanned PDFs fall through to
  ``DoclingParser`` with RapidOCR.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from src.models import DocumentChunk, ParseResult

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)

# ── Supported extensions ────────────────────────────────────────────────────
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html"}

# ── OCR gating constants (tune via env / settings if needed) ────────────────
_OCR_MIN_FILE_BYTES: int = 10_000       # skip images < 10 KB (icons, logos)
_OCR_VARIANCE_THRESHOLD: float = 100.0  # skip near-blank/solid-colour images
_OCR_TARGET_WIDTH: int = 1800           # rescale images to ~300 DPI equivalent


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


# ── PyMuPDF parser (fast PDF text extraction) ──────────────────────────────


class PyMuPDFParser(BaseParser):
    """Fast PDF parser using PyMuPDF (fitz).

    Only used for PDFs that have an embedded text layer.  Scanned PDFs are
    routed to ``DoclingParser`` by ``parse_file()``.
    """

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


# ── Docling parser (layout-aware, scanned PDFs) ─────────────────────────────


class DoclingParser(BaseParser):
    """Layout-aware parser for scanned / mixed-content PDFs using Docling.

    Handles PDFs where PyMuPDF returns insufficient text (scanned documents,
    mixed text+image pages).  Uses ``RapidOcrOptions`` explicitly so that
    Docling's internal OCR always runs RapidOCR — never Tesseract.

    The ``DocumentConverter`` is lazily initialised once per worker process
    and cached as a class-level attribute to avoid reloading models per file.
    """

    _converter: "DocumentConverter | None" = None  # class-level lazy singleton

    @classmethod
    def _get_converter(cls) -> "DocumentConverter":
        if cls._converter is None:
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                RapidOcrOptions,
            )
            from docling.document_converter import DocumentConverter

            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                ocr_options=RapidOcrOptions(),  # use RapidOCR, not Tesseract
            )
            cls._converter = DocumentConverter(
                format_options={"pdf": {"pipeline_options": pipeline_options}}
            )
        return cls._converter

    @property
    def name(self) -> str:
        return "docling"

    @property
    def supported_extensions(self) -> set[str]:
        return PDF_EXTENSIONS

    def parse(self, file_path: Path, chunk_size: int = 1000) -> ParseResult:
        """Extract text from a scanned / mixed PDF using Docling + RapidOCR.

        Args:
            file_path: Path to the PDF file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParseResult containing document chunks.
        """
        try:
            conv = self._get_converter().convert(str(file_path))
            text = conv.document.export_to_text()
            page_count = (
                len(conv.document.pages)
                if hasattr(conv.document, "pages")
                else 0
            )
            chunks = _split_text(text, file_path, chunk_size, self.name)
            return ParseResult(
                source_path=file_path,
                chunks=chunks,
                parser_used=self.name,
                page_count=page_count,
                success=True,
            )
        except Exception as exc:
            logger.error("Docling failed on %s: %s", file_path, exc)
            return ParseResult(
                source_path=file_path,
                success=False,
                error_message=str(exc),
                parser_used=self.name,
            )


# ── RapidOCR parser (standalone images) ────────────────────────────────────


class RapidOCRParser(BaseParser):
    """OCR parser for standalone image files using RapidOCR (ONNX Runtime).

    Replaces ``TesseractParser``.  Key improvements:

    * **In-process inference** — no subprocess fork/exec overhead per file.
    * **3-10x faster on CPU** than pytesseract (same PaddleOCR model weights,
      delivered via ONNX Runtime instead of spawning a tesseract binary).
    * **Image gating** — skips OCR on tiny files and near-blank images before
      any expensive decode or inference.
    * **Preprocessing** — converts to greyscale and rescales to the optimal
      OCR input resolution before inference.

    The ``RapidOCR`` engine is lazily initialised once per worker process and
    cached as a class-level attribute to avoid reloading ONNX models per file.
    """

    _engine: "RapidOCR | None" = None  # class-level lazy singleton per process

    @classmethod
    def _get_engine(cls) -> "RapidOCR":
        if cls._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            cls._engine = RapidOCR()
        return cls._engine

    @property
    def name(self) -> str:
        return "rapidocr"

    @property
    def supported_extensions(self) -> set[str]:
        return IMAGE_EXTENSIONS

    def parse(self, file_path: Path, chunk_size: int = 1000) -> ParseResult:
        """Extract text from an image using RapidOCR.

        Applies two fast gates before invoking the OCR engine:
        1. File-size gate — skips files below ``_OCR_MIN_FILE_BYTES``.
        2. Pixel-variance gate — skips blank / solid-colour images.

        Args:
            file_path: Path to the image file.
            chunk_size: Approximate characters per chunk.

        Returns:
            ParseResult containing document chunks.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            return ParseResult(
                source_path=file_path,
                success=False,
                error_message=f"numpy/Pillow not installed: {exc}",
                parser_used=self.name,
            )

        try:
            # Gate 1: skip tiny files (icons, bullets, decorative graphics)
            if file_path.stat().st_size < _OCR_MIN_FILE_BYTES:
                logger.debug("Skipping OCR on undersized image %s", file_path)
                return ParseResult(
                    source_path=file_path,
                    chunks=[],
                    parser_used=self.name,
                    page_count=1,
                    success=True,
                )

            image = Image.open(file_path)

            # Gate 2: pixel-variance proxy — blank/solid-colour images have
            # near-zero variance and contain no meaningful text.
            gray_arr = np.array(image.convert("L"))
            if float(gray_arr.var()) < _OCR_VARIANCE_THRESHOLD:
                logger.debug("OCR skipped: low-variance image %s", file_path)
                return ParseResult(
                    source_path=file_path,
                    chunks=[],
                    parser_used=self.name,
                    page_count=1,
                    success=True,
                )

            # Preprocessing: greyscale + rescale to optimal OCR resolution.
            # RapidOCR's detection network works best on images ~1800 px wide
            # (~300 DPI for A4).  Very narrow images are upsampled; very wide
            # images are downsampled to avoid unnecessary computation.
            gray = image.convert("L")
            w, h = gray.size
            if w < 800 or w > 3000:
                scale = _OCR_TARGET_WIDTH / w
                new_w, new_h = int(w * scale), int(h * scale)
                gray = gray.resize((new_w, new_h), Image.LANCZOS)

            result, _ = self._get_engine()(np.array(gray))
            text = "\n".join(line[1] for line in result) if result else ""
            chunks = _split_text(text, file_path, chunk_size, self.name)

            return ParseResult(
                source_path=file_path,
                chunks=chunks,
                parser_used=self.name,
                page_count=1,
                success=True,
            )
        except Exception as exc:
            logger.error("RapidOCR failed on %s: %s", file_path, exc)
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

# Registry used by get_parser() for non-PDF extensions.
# PDFs bypass this list and are routed by parse_file() using _is_scanned_pdf().
_PARSERS: list[BaseParser] = [
    RapidOCRParser(),
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
        segment = text[i:i + chunk_size]
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


# ── PDF text-density probe ──────────────────────────────────────────────────

_PDF_TEXT_THRESHOLD: int = 50  # avg chars/page below this → treat as scanned


def _is_scanned_pdf(file_path: Path) -> bool:
    """Probe PDF text density to decide whether OCR is needed.

    Opens the file, extracts raw text from all pages, and returns True if the
    average character count per page is below ``_PDF_TEXT_THRESHOLD``.  The
    probe costs ~5-20 ms (text extraction only, no rendering) versus 1-30 s
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
        return False  # conservative: let PyMuPDF attempt text extraction


def get_parser(file_path: Path) -> BaseParser | None:
    """Select the appropriate parser for a non-PDF file based on extension.

    PDFs are handled directly by ``parse_file()`` using content-aware routing;
    they do not pass through this function.

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

    PDFs are routed via a cheap text-density probe:
    - Text-layer PDFs  → ``PyMuPDFParser`` (fast, no OCR)
    - Scanned PDFs     → ``DoclingParser`` (layout-aware, RapidOCR)

    All other files are routed by extension via ``get_parser()``.

    Args:
        file_path: Path to the file to parse.
        chunk_size: Approximate characters per chunk.

    Returns:
        ParseResult from the selected parser, or a failed result if
        no parser supports the file extension.
    """
    if file_path.suffix.lower() in PDF_EXTENSIONS:
        parser: BaseParser = (
            DoclingParser() if _is_scanned_pdf(file_path) else PyMuPDFParser()
        )
    else:
        parser = get_parser(file_path)

    if parser is None:
        return ParseResult(
            source_path=file_path,
            success=False,
            error_message=f"Unsupported file extension: {file_path.suffix}",
        )
    logger.info("Parsing %s with %s", file_path, parser.name)
    return parser.parse(file_path, chunk_size=chunk_size)
