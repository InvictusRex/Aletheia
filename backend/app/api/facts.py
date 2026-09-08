"""Fact endpoints (minimal).

POST /documents/{id}/facts runs evidence-scoped LLM extraction
explicitly (never at upload time). GET lists persisted facts.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.repositories import get_document_bundle, list_facts_for_document
from app.db.session import get_db
from app.facts.normalization import NormalizationReport, normalize_document_facts
from app.facts.service import FactExtractionReport, extract_facts_for_document
from app.models import Fact

router = APIRouter(prefix="/documents", tags=["facts"])


@router.post("/{document_id}/facts", response_model=FactExtractionReport)
def trigger_fact_extraction(
    document_id: UUID, db: Session = Depends(get_db)
) -> FactExtractionReport:
    if get_document_bundle(db, document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    try:
        report = extract_facts_for_document(db, document_id)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"fact extraction failed: {exc}"
        ) from exc
    if report.chunks_processed == 0 and any(
        "unavailable" in e or "skipped" in e for e in report.errors
    ):
        raise HTTPException(
            status_code=503,
            detail="LLM extraction unavailable (no API key configured)",
        )
    return report


@router.get("/{document_id}/facts", response_model=list[Fact])
def list_document_facts(
    document_id: UUID, db: Session = Depends(get_db)
) -> list[Fact]:
    if get_document_bundle(db, document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    return list_facts_for_document(db, document_id)


@router.post("/{document_id}/normalize", response_model=NormalizationReport)
def trigger_fact_normalization(
    document_id: UUID, db: Session = Depends(get_db)
) -> NormalizationReport:
    if get_document_bundle(db, document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    try:
        report = normalize_document_facts(db, document_id)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"fact normalization failed: {exc}"
        ) from exc
    return report
