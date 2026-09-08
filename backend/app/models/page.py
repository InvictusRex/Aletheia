"""Page model.

Distinguishes the physical PDF page index (``pdf_page_number``, 0-based)
from the printed/source page number (``source_page_number``), which may
be discontinuous in curated excerpts. The two must never be conflated:
``source_page_number`` is best-effort metadata and may be ``None``.
"""

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class QualityVerdict(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"


class Page(BaseModel):
    document_id: UUID
    pdf_page_number: int = Field(ge=0)

    source_page_number: str | None = None

    # Dimensions are 0 only when page extraction itself failed (error set).
    width: float = Field(ge=0)
    height: float = Field(ge=0)

    char_count: int = Field(ge=0, default=0)
    word_count: int = Field(ge=0, default=0)
    suspicious_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    has_images: bool = False

    extraction_quality: float = Field(ge=0.0, le=1.0, default=0.0)
    quality_verdict: QualityVerdict = QualityVerdict.BAD
    error: str | None = None
