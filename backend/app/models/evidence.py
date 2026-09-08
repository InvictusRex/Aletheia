"""Canonical EvidenceUnit (PLAN Section 5).

Extractor-agnostic: PyMuPDF today, pdfplumber/Textract later — all must
produce this representation. Provenance fields are mandatory; layout
information (bbox) is preserved, never flattened away.
"""

from enum import Enum
from uuid import UUID, uuid5

from pydantic import BaseModel, Field

EVIDENCE_ID_NAMESPACE = uuid5(UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"), "aletheia.evidence")


class EvidenceType(str, Enum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    TABLE_ROW = "TABLE_ROW"
    TABLE_CELL = "TABLE_CELL"
    IMAGE = "IMAGE"
    OCR_TEXT = "OCR_TEXT"
    HEADING = "HEADING"
    FOOTNOTE = "FOOTNOTE"
    CAPTION = "CAPTION"


class ExtractionMethod(str, Enum):
    PYMUPDF = "PYMUPDF"
    PDFPLUMBER = "PDFPLUMBER"
    TEXTRACT = "TEXTRACT"


class EvidenceUnit(BaseModel):
    id: UUID
    document_id: UUID
    pdf_page_number: int = Field(ge=0)

    source_page_number: str | None = None

    type: EvidenceType
    text: str

    bbox: tuple[float, float, float, float] | None = None

    extraction_method: ExtractionMethod
    extraction_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    meta: dict = Field(default_factory=dict)


def make_evidence_id(
    document_id: UUID,
    pdf_page_number: int,
    block_index: int,
    evidence_type: EvidenceType,
) -> UUID:
    """Deterministic evidence ID: stable across re-ingests of a document."""
    return uuid5(
        EVIDENCE_ID_NAMESPACE,
        f"{document_id}:{pdf_page_number}:{block_index}:{evidence_type.value}",
    )
