"""Canonical domain models (Phase 1).

Pydantic contracts shared by extraction, persistence, and API layers.
Downstream components consume these — never extractor internals.
"""

from app.models.document import Document, IngestionStatus
from app.models.evidence import (
    EvidenceType,
    EvidenceUnit,
    ExtractionMethod,
    make_evidence_id,
)
from app.models.fact import (
    ChunkUnit,
    EstimateStatus,
    EvidenceChunk,
    Fact,
    FactDraft,
    FactStatus,
    TimeKind,
    ValueKind,
)
from app.models.page import Page, QualityVerdict

__all__ = [
    "ChunkUnit",
    "Document",
    "EstimateStatus",
    "EvidenceChunk",
    "EvidenceType",
    "EvidenceUnit",
    "ExtractionMethod",
    "Fact",
    "FactDraft",
    "FactStatus",
    "IngestionStatus",
    "Page",
    "QualityVerdict",
    "TimeKind",
    "ValueKind",
    "make_evidence_id",
]
