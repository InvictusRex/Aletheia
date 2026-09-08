"""Persistence helpers for Phase 1 ingestion bundle."""

from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DocumentRow, EvidenceUnitRow, PageRow
from app.models import Document, EvidenceUnit, Page


def _status_str(document: Document) -> str:
    status = document.ingestion_status
    return status.value if hasattr(status, "value") else str(status)


def _verdict_str(page: Page) -> str:
    verdict = page.quality_verdict
    return verdict.value if hasattr(verdict, "value") else str(verdict)


def _evidence_type_str(evidence: EvidenceUnit) -> str:
    value = evidence.type
    return value.value if hasattr(value, "value") else str(value)


def _method_str(evidence: EvidenceUnit) -> str:
    value = evidence.extraction_method
    return value.value if hasattr(value, "value") else str(value)


def save_ingestion(
    session: Session,
    document: Document,
    pages: list[Page],
    evidence: list[EvidenceUnit],
) -> None:
    """Persist a full ingestion bundle.

    Adds all rows with a single flush; the caller commits.
    """
    doc_row = DocumentRow(
        id=document.id,
        filename=document.filename,
        title=document.title,
        source=document.source,
        page_count=document.page_count,
        file_sha256=document.file_sha256,
        ingestion_status=_status_str(document),
        error=document.error,
        created_at=document.created_at,
    )
    page_rows = [
        PageRow(
            id=uuid4(),
            document_id=p.document_id,
            pdf_page_number=p.pdf_page_number,
            source_page_number=p.source_page_number,
            width=p.width,
            height=p.height,
            char_count=p.char_count,
            word_count=p.word_count,
            suspicious_ratio=p.suspicious_ratio,
            extraction_quality=p.extraction_quality,
            has_images=p.has_images,
            quality_verdict=_verdict_str(p),
            error=p.error,
        )
        for p in pages
    ]
    evidence_rows = [
        EvidenceUnitRow(
            id=e.id,
            document_id=e.document_id,
            pdf_page_number=e.pdf_page_number,
            source_page_number=e.source_page_number,
            type=_evidence_type_str(e),
            text=e.text,
            bbox=list(e.bbox) if e.bbox is not None else None,
            extraction_method=_method_str(e),
            extraction_quality=e.extraction_quality,
            meta=dict(e.meta) if e.meta is not None else {},
        )
        for e in evidence
    ]
    session.add(doc_row)
    session.add_all(page_rows)
    session.add_all(evidence_rows)
    session.flush()


def get_document_bundle(
    session: Session, document_id: UUID
) -> tuple[Document, list[Page], list[EvidenceUnit]] | None:
    """Load a document plus its pages and evidence units.

    Pages are ordered by ``pdf_page_number``; evidence is ordered by
    ``pdf_page_number``, then evidence id string.
    """
    doc_row = session.get(DocumentRow, document_id)
    if doc_row is None:
        return None

    page_rows = list(
        session.scalars(
            select(PageRow)
            .where(PageRow.document_id == document_id)
            .order_by(PageRow.pdf_page_number)
        ).all()
    )
    ev_rows = list(
        session.scalars(
            select(EvidenceUnitRow)
            .where(EvidenceUnitRow.document_id == document_id)
            .order_by(EvidenceUnitRow.pdf_page_number, EvidenceUnitRow.id)
        ).all()
    )
    # Ensure evidence id-string ordering within a page (UUID order matches
    # lexicographic string order, but sort explicitly per the contract).
    ev_rows.sort(key=lambda r: (r.pdf_page_number, str(r.id)))

    document = Document(
        id=doc_row.id,
        filename=doc_row.filename,
        title=doc_row.title,
        source=doc_row.source,
        page_count=doc_row.page_count,
        file_sha256=doc_row.file_sha256,
        ingestion_status=doc_row.ingestion_status,
        error=doc_row.error,
        created_at=doc_row.created_at,
    )
    pages = [
        Page(
            document_id=r.document_id,
            pdf_page_number=r.pdf_page_number,
            source_page_number=r.source_page_number,
            width=r.width,
            height=r.height,
            char_count=r.char_count,
            word_count=r.word_count,
            suspicious_ratio=r.suspicious_ratio,
            has_images=r.has_images,
            extraction_quality=r.extraction_quality,
            quality_verdict=r.quality_verdict,
            error=r.error,
        )
        for r in page_rows
    ]
    evidence = [
        EvidenceUnit(
            id=r.id,
            document_id=r.document_id,
            pdf_page_number=r.pdf_page_number,
            source_page_number=r.source_page_number,
            type=r.type,
            text=r.text,
            bbox=tuple(r.bbox) if r.bbox is not None else None,  # type: ignore[arg-type]
            extraction_method=r.extraction_method,
            extraction_quality=r.extraction_quality,
            meta=dict(r.meta) if r.meta is not None else {},
        )
        for r in ev_rows
    ]
    return document, pages, evidence
