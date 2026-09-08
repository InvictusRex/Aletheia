"""Document ingestion endpoints (Phase 1, minimal).

POST /documents ingests a PDF synchronously (prototype scale) and
persists the canonical evidence bundle. GET /documents/{id} inspects it.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.repositories import get_document_bundle, save_ingestion
from app.db.session import get_db
from app.extraction.pipeline import run_ingestion
from app.models import Document, EvidenceUnit, Page

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class IngestResponse(BaseModel):
    document: Document
    page_count: int
    evidence_count: int
    bad_pages: int


class DocumentBundle(BaseModel):
    document: Document
    pages: list[Page]
    evidence: list[EvidenceUnit]


@router.post("", response_model=IngestResponse, status_code=201)
async def ingest_document(
    file: UploadFile, db: Session = Depends(get_db)
) -> IngestResponse:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 50 MB limit")
    if len(data) < 4 or data[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="uploaded file is not a PDF")

    document, pages, evidence = run_ingestion(file.filename or "upload.pdf", data)
    try:
        save_ingestion(db, document, pages, evidence)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"failed to persist ingestion: {exc}"
        ) from exc

    bad_pages = sum(1 for p in pages if p.error is not None)
    return IngestResponse(
        document=document,
        page_count=len(pages),
        evidence_count=len(evidence),
        bad_pages=bad_pages,
    )


@router.get("/{document_id}", response_model=DocumentBundle)
def get_document(document_id: UUID, db: Session = Depends(get_db)) -> DocumentBundle:
    bundle = get_document_bundle(db, document_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="document not found")
    document, pages, evidence = bundle
    return DocumentBundle(document=document, pages=pages, evidence=evidence)
