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
from app.models.relationship import (
    CandidatePair,
    ContextComparison,
    DimensionVerdict,
    NumericVerdict,
    Relationship,
    RelationshipJudgment,
    RelationshipStatus,
    RelationshipType,
)

__all__ = [
    "CandidatePair",
    "ChunkUnit",
    "ContextComparison",
    "DimensionVerdict",
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
    "NumericVerdict",
    "Page",
    "QualityVerdict",
    "Relationship",
    "RelationshipJudgment",
    "RelationshipStatus",
    "RelationshipType",
    "TimeKind",
    "ValueKind",
    "make_evidence_id",
]
