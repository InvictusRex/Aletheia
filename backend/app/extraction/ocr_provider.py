"""OCR provider abstraction (task T-O1).

Freezes the contract that concrete OCR engines (e.g. the sibling
``paddle_ocr.py`` module) implement and that the extraction pipeline
calls. Deliberately stdlib-only: no paddle / boto / AWS imports here,
so the pipeline can import this module without pulling in heavy engine
dependencies.

Mapping to the canonical evidence layer (read-only reference, NOT
imported here to keep this module dependency-free):

- each :class:`OcrBlock` maps to one ``EvidenceUnit`` with
  ``EvidenceType.OCR_TEXT``;
- ``OcrBlock.bbox`` uses the same ``(x0, y0, x1, y1)`` PDF-points
  convention as ``EvidenceUnit.bbox``;
- ``OcrBlock.confidence`` maps to ``EvidenceUnit.extraction_quality``.

Provider-abstraction pattern follows docs/PLAN.md sections 5 (canonical
evidence layer: PyMuPDF / pdfplumber / OCR all converge) and 13 (small
``Protocol`` interface with swappable concrete providers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "OcrBlock",
    "OcrResult",
    "OcrError",
    "OcrUnavailableError",
    "OcrInitError",
    "OcrModelMissingError",
    "OcrRuntimeError",
    "OcrMalformedError",
    "OcrProvider",
    "NullProvider",
]


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class OcrError(Exception):
    """Base class for all OCR failures.

    Carries a plain message string (``str(exception)``). Callers (the
    pipeline) catch this base class around ``extract_page``.
    """


class OcrUnavailableError(OcrError):
    """Engine/weights unavailable — caller skips OCR (silently-ish)."""


class OcrInitError(OcrError):
    """Engine failed to initialise (bad config, missing dependency, ...)."""


class OcrModelMissingError(OcrError):
    """Model weights / data files not found locally."""


class OcrRuntimeError(OcrError):
    """Engine failed while processing a page (crash, timeout, OOM, ...)."""


class OcrMalformedError(OcrError):
    """Engine output was unparseable — caller must NOT fabricate blocks."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class OcrBlock:
    """One recognized text block.

    ``text`` is stripped (leading/trailing whitespace removed) but
    otherwise verbatim — never cleaned, corrected, or hallucinated.

    ``bbox`` is ``(x0, y0, x1, y1)`` in PDF points, ordered so that
    ``x0 <= x1`` and ``y0 <= y1``; ``None`` when the engine reports no
    geometry.

    ``confidence`` is in ``0..1`` when the engine reports it, else
    ``None``.
    """

    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"OcrBlock.text must be str, got {type(self.text).__name__}")
        self.text = self.text.strip()
        if self.bbox is not None:
            try:
                x0, y0, x1, y1 = self.bbox
            except (TypeError, ValueError):
                raise ValueError(f"OcrBlock.bbox must be (x0, y0, x1, y1), got {self.bbox!r}")
            try:
                x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
            except (TypeError, ValueError):
                raise ValueError(f"OcrBlock.bbox coordinates must be numeric, got {self.bbox!r}")
            self.bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise ValueError(
                    f"OcrBlock.confidence must be a number in 0..1, got {self.confidence!r}"
                )
            value = float(self.confidence)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"OcrBlock.confidence must be in 0..1, got {self.confidence!r}"
                )
            self.confidence = value


@dataclass
class OcrResult:
    """Result of OCR on a single page.

    ``blocks`` contains non-empty-text blocks only; an empty list means
    "nothing recognized", NOT an error (providers raise :class:`OcrError`
    subclasses for actual failures instead of returning partial garbage).

    ``provider`` is the engine name (e.g. ``"paddleocr"``).
    """

    blocks: list[OcrBlock]
    provider: str

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, list):
            raise TypeError(
                f"OcrResult.blocks must be a list, got {type(self.blocks).__name__}"
            )
        for block in self.blocks:
            if not isinstance(block, OcrBlock):
                raise TypeError(
                    f"OcrResult.blocks must contain OcrBlock, got {type(block).__name__}"
                )
        # Defensive: keep only non-empty-text blocks per the contract.
        self.blocks = [block for block in self.blocks if block.text]
        if not isinstance(self.provider, str):
            raise TypeError(
                f"OcrResult.provider must be str, got {type(self.provider).__name__}"
            )


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


@runtime_checkable
class OcrProvider(Protocol):
    """Structural interface for page-level OCR engines.

    Implementations raise only :class:`OcrError` subclasses for OCR
    failures (unavailable / init / missing model / runtime / malformed).

    ``ValueError`` may additionally be raised for invalid input (e.g.
    ``pdf_bytes`` not non-empty ``bytes``/``bytearray``, or
    ``pdf_page_number`` negative) — this signals a caller bug, not an
    OCR failure, and must NOT be caught as an OCR error.
    """

    def extract_page(self, pdf_bytes: bytes, pdf_page_number: int) -> OcrResult:
        """Recognize text on one PDF page.

        :param pdf_bytes: raw bytes of the whole PDF document.
        :param pdf_page_number: 0-based index of the page to process.
        :returns: :class:`OcrResult` (possibly with zero blocks).
        :raises OcrError: on any OCR failure.
        :raises ValueError: on invalid input (non-bytes/empty bytes,
            negative page number).
        """
        ...  # pragma: no cover


class NullProvider:
    """Trivial :class:`OcrProvider` with no engine behind it.

    Gives the pipeline a safe default and tests a deterministic
    unavailable path: :meth:`extract_page` always raises
    :class:`OcrUnavailableError`.
    """

    def extract_page(self, pdf_bytes: bytes, pdf_page_number: int) -> OcrResult:
        """Always raise ``OcrUnavailableError("no OCR provider configured")``."""
        raise OcrUnavailableError("no OCR provider configured")
