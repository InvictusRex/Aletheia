"""Relationship endpoints (explicit generation, explicit inspection).

POST /documents/{id}/relationships discovers candidates for that
document's facts against all other documents and persists classified
relationships. GET routes inspect persisted relationships with both
facts and their evidence.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.repositories import (
    get_document_bundle,
    get_fact,
    get_relationship,
    list_relationships_for_document,
)
from app.db.session import get_db
from app.matching.service import MatchingReport, run_matching_for_document
from app.models import EvidenceUnit, Fact, Relationship

router = APIRouter(tags=["relationships"])


class RelationshipDetail(BaseModel):
    relationship: Relationship
    fact_a: Fact
    fact_b: Fact
    evidence_a: list[EvidenceUnit] = Field(default_factory=list)
    evidence_b: list[EvidenceUnit] = Field(default_factory=list)


@router.post(
    "/documents/{document_id}/relationships", response_model=MatchingReport
)
def trigger_matching(
    document_id: UUID, db: Session = Depends(get_db)
) -> MatchingReport:
    if get_document_bundle(db, document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    try:
        report = run_matching_for_document(db, document_id)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"relationship matching failed: {exc}"
        ) from exc
    return report


@router.get(
    "/documents/{document_id}/relationships",
    response_model=list[Relationship],
)
def list_document_relationships(
    document_id: UUID,
    db: Session = Depends(get_db),
    relationship_type: str | None = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
) -> list[Relationship]:
    if get_document_bundle(db, document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    return list_relationships_for_document(
        db, document_id, relationship_type, min_confidence
    )


@router.get("/relationships/{relationship_id}", response_model=RelationshipDetail)
def get_relationship_detail(
    relationship_id: UUID, db: Session = Depends(get_db)
) -> RelationshipDetail:
    rel = get_relationship(db, relationship_id)
    if rel is None:
        raise HTTPException(status_code=404, detail="relationship not found")

    fact_a = get_fact(db, rel.fact_a_id)
    fact_b = get_fact(db, rel.fact_b_id)
    if fact_a is None or fact_b is None:
        raise HTTPException(status_code=404, detail="related fact not found")

    def evidence_for(fact: Fact) -> list[EvidenceUnit]:
        bundle = get_document_bundle(db, fact.document_id)
        if bundle is None:
            return []
        by_id = {e.id: e for e in bundle[2]}
        return [by_id[eid] for eid in fact.evidence_ids if eid in by_id]

    return RelationshipDetail(
        relationship=rel,
        fact_a=fact_a,
        fact_b=fact_b,
        evidence_a=evidence_for(fact_a),
        evidence_b=evidence_for(fact_b),
    )
