"""Provenance-first PDF text extraction (Phase 1, task T2).

Primary PDF backbone (PLAN Sections 4-6): native text blocks with
coordinates, document metadata, image signals, and per-page quality
signals for the extraction-quality router.

Dependency-free: standard library + PyMuPDF only
(``import pymupdf``, never the deprecated ``import fitz``).

API discoveries (verified empirically against the installed PyMuPDF):
- Page labels are document-level ``/PageLabels`` entries exposed via
  ``Document.get_page_labels()`` (``[]`` when the PDF defines none) and
  the per-page ``Page.get_label()`` (``""`` when undefined or on error).
  ``source_page_number`` is therefore best-effort: ``None`` unless the
  document actually defines page labels. It is NEVER synthesized from
  the physical index.
- ``pymupdf.open(stream=data, filetype="pdf")`` raises
  ``pymupdf.EmptyFileError`` for empty input and
  ``pymupdf.FileDataError`` for non-PDF bytes; both are mapped to
  ``ValueError`` here so the caller can mark the document FAILED.
- ``page.get_text("dict")`` returns ``{"width", "height", "blocks"}``;
  text blocks have ``type == 0`` with ``lines -> spans -> text``.
- ``page.get_images(full=True)`` returns the image list for the signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

#: Replacement character — a strong signal of broken text extraction.
_REPLACEMENT_CHAR = "\ufffd"

#: ASCII control codes that are legitimate inside block text.
_ALLOWED_CONTROLS = frozenset((0x09, 0x0A, 0x0D))  # \t \n \r


@dataclass
class TextBlock:
    """One native-text block: stripped text plus its PDF coordinates."""

    text: str
    bbox: tuple[float, float, float, float]
    block_index: int


@dataclass
class PageExtraction:
    """Per-page extraction result; ``error`` is set instead of raising."""

    pdf_page_number: int
    source_page_number: str | None
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)
    image_count: int = 0
    char_count: int = 0
    word_count: int = 0
    suspicious_ratio: float = 0.0
    has_images: bool = False
    error: str | None = None


@dataclass
class PdfMetadata:
    """Document-level metadata (empty strings normalized to ``None``)."""

    title: str | None
    author: str | None
    page_count: int


def _is_suspicious_char(ch: str) -> bool:
    """True for U+FFFD or ASCII control chars other than \\t, \\n, \\r."""
    if ch == _REPLACEMENT_CHAR:
        return True
    code = ord(ch)
    if code in _ALLOWED_CONTROLS:
        return False
    return code < 0x20 or code == 0x7F


def _block_text(block: dict) -> str:
    """Join a ``get_text("dict")`` text block's spans; strip the result."""
    lines = []
    for line in block.get("lines") or []:
        spans = line.get("spans") or []
        lines.append("".join(span.get("text") or "" for span in spans))
    return "\n".join(lines).strip()


def _page_labels(doc: pymupdf.Document) -> list[str | None]:
    """Best-effort source page numbers from PDF page labels only.

    Returns ``None`` per page when the document defines no ``/PageLabels``
    (the common case) or when label lookup fails. Never synthesizes.
    """
    try:
        defined = doc.get_page_labels()
    except Exception:
        return [None] * len(doc)
    if not defined:
        return [None] * len(doc)
    labels: list[str | None] = []
    for index in range(len(doc)):
        try:
            label = doc[index].get_label()
        except Exception:
            label = ""
        labels.append(label or None)
    return labels


def _extract_page(doc: pymupdf.Document, index: int, label: str | None) -> PageExtraction:
    """Extract one page; any failure is captured in ``error``."""
    try:
        page = doc[index]
        width = float(page.rect.width)
        height = float(page.rect.height)

        blocks: list[TextBlock] = []
        raw = page.get_text("dict")
        for block in raw.get("blocks") or []:
            if block.get("type") != 0:
                continue
            text = _block_text(block)
            if not text:
                continue
            bbox = tuple(float(v) for v in block["bbox"])
            blocks.append(
                TextBlock(
                    text=text,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    block_index=len(blocks),
                )
            )

        char_count = sum(len(block.text) for block in blocks)
        word_count = sum(len(block.text.split()) for block in blocks)
        suspicious = sum(
            1 for block in blocks for ch in block.text if _is_suspicious_char(ch)
        )
        suspicious_ratio = (suspicious / char_count) if char_count else 0.0

        images = page.get_images(full=True) or []
        image_count = len(images)

        return PageExtraction(
            pdf_page_number=index,
            source_page_number=label,
            width=width,
            height=height,
            blocks=blocks,
            image_count=image_count,
            char_count=char_count,
            word_count=word_count,
            suspicious_ratio=suspicious_ratio,
            has_images=bool(images),
            error=None,
        )
    except Exception as exc:
        return PageExtraction(
            pdf_page_number=index,
            source_page_number=None,
            width=0.0,
            height=0.0,
            blocks=[],
            image_count=0,
            char_count=0,
            word_count=0,
            suspicious_ratio=0.0,
            has_images=False,
            error=f"page {index}: {type(exc).__name__}: {exc}",
        )


def extract_pdf(data: bytes) -> tuple[PdfMetadata, list[PageExtraction]]:
    """Extract native text blocks, signals, and metadata from PDF bytes.

    Raises:
        ValueError: If the input is empty, not bytes, or not a readable
            PDF. The caller maps this to ``IngestionStatus.FAILED``.
    """
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes) or len(data) == 0:
        raise ValueError("extract_pdf requires non-empty PDF bytes")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(
            f"unable to open PDF stream ({type(exc).__name__}): {exc}"
        ) from exc

    try:
        if len(doc) == 0:
            raise ValueError("PDF contains no pages")
        raw_meta = doc.metadata or {}
        metadata = PdfMetadata(
            title=(raw_meta.get("title") or "").strip() or None,
            author=(raw_meta.get("author") or "").strip() or None,
            page_count=len(doc),
        )
        labels = _page_labels(doc)
        pages = [
            _extract_page(doc, index, labels[index]) for index in range(len(doc))
        ]
        return metadata, pages
    finally:
        doc.close()
