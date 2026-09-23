"""Document ingestion endpoints (Phase 1, minimal).

POST /documents ingests a PDF synchronously (prototype scale) and
persists the canonical evidence bundle. GET /documents/{id} inspects it.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.repositories import (
    get_document_bundle,
    get_document_file,
    list_documents,
    save_document_file,
    save_ingestion,
)
from app.extraction.render import RenderError, render_page_png
from app.db.session import get_db
from app.core.config import settings
from app.extraction.pipeline import run_ingestion
from app.ml.client import MLServiceClient
from app.ml.wake import wake_and_wait
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
def ingest_document(
    file: UploadFile, db: Session = Depends(get_db)
) -> IngestResponse:
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 50 MB limit")
    if len(data) < 4 or data[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="uploaded file is not a PDF")

    # A new document is the event that drives the ML pipeline, so the
    # service is asked for here and ingestion waits for it. If it never
    # arrives, native extraction still succeeds and only the OCR fallback
    # is missing.
    if settings.ocr_enabled:
        wake_and_wait(MLServiceClient(base_url=settings.ml_service_url))

    document, pages, evidence = run_ingestion(file.filename or "upload.pdf", data)
    try:
        save_ingestion(db, document, pages, evidence)
        # Keep the PDF: a bounding box is only meaningful drawn on the
        # page it came from, and the page cannot be re-rendered later
        # without the original bytes.
        save_document_file(db, document.id, data)
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


@router.get("", response_model=list[Document])
def list_all_documents(db: Session = Depends(get_db)) -> list[Document]:
    """List all ingested documents (read-only workspace overview)."""
    return list_documents(db)


@router.get("/{document_id}", response_model=DocumentBundle)
def get_document(document_id: UUID, db: Session = Depends(get_db)) -> DocumentBundle:
    bundle = get_document_bundle(db, document_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="document not found")
    document, pages, evidence = bundle
    return DocumentBundle(document=document, pages=pages, evidence=evidence)


@router.get("/{document_id}/pages/{pdf_page_number}/image")
def get_page_image(
    document_id: UUID,
    pdf_page_number: int,
    evidence_id: UUID | None = Query(
        default=None, description="Highlight this evidence unit's bounding box"
    ),
    dpi: int = Query(default=110, ge=40, le=300),
    db: Session = Depends(get_db),
) -> Response:
    """Render one page as PNG, optionally with an evidence box drawn on it.

    This is the last step of the provenance chain: fact -> evidence ->
    page -> the exact rectangle the text was read from.
    """
    bundle = get_document_bundle(db, document_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="document not found")
    content = get_document_file(db, document_id)
    if content is None:
        raise HTTPException(
            status_code=404,
            detail="original PDF not stored for this document; re-upload to enable page rendering",
        )
    bbox = None
    if evidence_id is not None:
        match = next((e for e in bundle[2] if e.id == evidence_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        if match.pdf_page_number != pdf_page_number:
            raise HTTPException(
                status_code=400,
                detail="evidence is not on the requested page",
            )
        bbox = match.bbox
    try:
        png = render_page_png(content, pdf_page_number, bbox=bbox, dpi=dpi)
    except RenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")
