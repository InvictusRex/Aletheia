"""Fact endpoints (minimal).

POST /documents/{id}/facts runs evidence-scoped LLM extraction
explicitly (never at upload time). GET lists persisted facts.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.repositories import get_document_bundle, list_facts_for_document
from app.db.session import get_db
from app.facts.normalization import NormalizationReport, normalize_document_facts
from app.facts.service import FactExtractionReport, extract_facts_for_document
from app.models import Fact

router = APIRouter(prefix="/documents", tags=["facts"])


@router.post("/{document_id}/facts", response_model=FactExtractionReport)
def trigger_fact_extraction(
    document_id: UUID,
    start_page: int | None = Query(
        default=None, ge=0, description="0-based first pdf page (inclusive)"
    ),
    end_page: int | None = Query(
        default=None, ge=0, description="0-based last pdf page (inclusive)"
    ),
    db: Session = Depends(get_db),
) -> FactExtractionReport:
    """Run evidence-scoped LLM extraction, optionally over a page window.

    Without page parameters the whole document is processed; with them
    only chunks on ``start_page..end_page`` run. Progress commits per
    chunk, so reruns skip already-completed chunks without duplicating
    facts.
    """
    if get_document_bundle(db, document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    try:
        report = extract_facts_for_document(
            db, document_id, start_page=start_page, end_page=end_page
        )
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
