"""Canonical Fact representation (fact layer).

Separation of concerns:
- The LLM proposes :class:`FactDraft` objects scoped to supplied evidence.
- Deterministic code validates drafts into :class:`Fact` objects
  (evidence-ID enforcement, numeric/date parsing, confidence clamping).
- Unsupported or fabricated content is rejected or flagged, never stored.

``EvidenceChunk`` / ``ChunkUnit`` are the cross-layer contract between
fact chunking (``app.facts``) and LLM providers (``app.llm``): every
unit carries its canonical evidence ID so facts link exact provenance.
"""

from datetime import date
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ValueKind(str, Enum):
    NUMERIC = "NUMERIC"
    PERCENTAGE = "PERCENTAGE"
    TEXT = "TEXT"


class TimeKind(str, Enum):
    FISCAL_YEAR = "FISCAL_YEAR"
    QUARTER = "QUARTER"
    DATE = "DATE"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class EstimateStatus(str, Enum):
    ACTUAL = "ACTUAL"
    ESTIMATE = "ESTIMATE"
    FORECAST = "FORECAST"
    PRELIMINARY = "PRELIMINARY"
    REVISED = "REVISED"
    UNKNOWN = "UNKNOWN"


class FactStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"


class ChunkUnit(BaseModel):
    """One evidence unit as supplied to the extractor.

    Cell-level units preserve row/column/header references so facts can
    link the most specific available evidence without a reasoning layer.
    """

    evidence_id: UUID
    evidence_type: str
    text: str

    row: int | None = None
    col: int | None = None
    column_header: str | None = None
    row_header: str | None = None
    table_index: int | None = None


class EvidenceChunk(BaseModel):
    """One page-scoped extraction input with explicit overflow accounting."""

    document_id: UUID
    pdf_page_number: int = Field(ge=0)
    source_page_number: str | None = None

    units: list[ChunkUnit] = Field(default_factory=list)
    dropped_evidence_ids: list[UUID] = Field(default_factory=list)


class FactDraft(BaseModel):
    """Raw LLM proposal. All strings verbatim; nothing here is trusted."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    value_text: str = Field(min_length=1)

    value_kind: ValueKind | None = None
    unit: str | None = None

    time_text: str | None = None
    scope_text: str | None = None
    estimate_status: EstimateStatus | None = None
    geography: str | None = None
    context: dict = Field(default_factory=dict)

    evidence_ids: list[UUID] = Field(min_length=1)

    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous: bool = False
    notes: str = ""


class Fact(BaseModel):
    """Validated canonical fact. Every fact is grounded in evidence."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID

    subject: str = Field(min_length=1)
    canonical_subject: str | None = None

    predicate: str = Field(min_length=1)
    canonical_predicate: str | None = None

    value_kind: ValueKind
    value_text: str = Field(min_length=1)
    value_number: float | None = None
    unit: str | None = None

    normalized_number: float | None = None
    normalized_unit: str | None = None

    time_text: str | None = None
    time_kind: TimeKind = TimeKind.UNKNOWN
    time_start: date | None = None
    time_end: date | None = None

    scope_text: str | None = None
    estimate_status: EstimateStatus = EstimateStatus.UNKNOWN
    geography: str | None = None
    context: dict = Field(default_factory=dict)

    evidence_ids: list[UUID] = Field(min_length=1)

    extraction_confidence: float = Field(ge=0.0, le=1.0)
    ambiguity_flags: list[str] = Field(default_factory=list)
    status: FactStatus = FactStatus.CONFIRMED
