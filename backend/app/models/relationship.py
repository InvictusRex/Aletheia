"""Relationship contracts (matching layer).

Five relationship types exactly — no LIKELY_CONTRADICTION; likelihood is
carried by confidence. Two statuses: RESOLVED vs NEEDS_REVIEW (explicit
uncertainty, never a synonym for RELATED). UNRELATED pairs are counted
but never persisted.
"""

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    CONTEXTUAL_DIFFERENCE = "CONTEXTUAL_DIFFERENCE"
    RELATED = "RELATED"
    UNRELATED = "UNRELATED"


class RelationshipStatus(str, Enum):
    RESOLVED = "RESOLVED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class NumericVerdict(str, Enum):
    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"
    DIFFERENT = "DIFFERENT"
    INCOMPARABLE = "INCOMPARABLE"


class DimensionVerdict(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class ContextComparison(BaseModel):
    """Per-dimension tri-state outcome. Unknown is never incompatibility."""

    dimensions: dict[str, DimensionVerdict] = Field(default_factory=dict)

    @property
    def incompatible(self) -> list[str]:
        return sorted(
            name
            for name, verdict in self.dimensions.items()
            if verdict == DimensionVerdict.INCOMPATIBLE
        )

    @property
    def unknown(self) -> list[str]:
        return sorted(
            name
            for name, verdict in self.dimensions.items()
            if verdict == DimensionVerdict.UNKNOWN
        )


class CandidatePair(BaseModel):
    """One ordered candidate pair. Scores rank only — never classify."""

    fact_a_id: UUID
    fact_b_id: UUID
    score: float = Field(ge=0.0, le=1.0)
    source: str = "vector"


class Relationship(BaseModel):
    """Persisted relationship. fact_a_id/fact_b_id stored id-ordered."""

    id: UUID = Field(default_factory=uuid4)
    fact_a_id: UUID
    fact_b_id: UUID

    relationship_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    reasoning_metadata: dict = Field(default_factory=dict)
    status: RelationshipStatus = RelationshipStatus.RESOLVED


class RelationshipJudgment(BaseModel):
    """Validated LLM output for genuinely ambiguous pairs only."""

    relationship_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
