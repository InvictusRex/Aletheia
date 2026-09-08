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


from app.db.models import FactEvidenceRow, FactRow  # noqa: E402 -- appended; existing imports above untouched
from app.models.fact import Fact  # noqa: E402 -- appended; existing imports above untouched


def _value_kind_str(fact: Fact) -> str:
    value = fact.value_kind
    return value.value if hasattr(value, "value") else str(value)


def _time_kind_str(fact: Fact) -> str:
    value = fact.time_kind
    return value.value if hasattr(value, "value") else str(value)


def _estimate_status_str(fact: Fact) -> str:
    value = fact.estimate_status
    return value.value if hasattr(value, "value") else str(value)


def _fact_status_str(fact: Fact) -> str:
    value = fact.status
    return value.value if hasattr(value, "value") else str(value)


def _fact_from_row(row: FactRow, evidence_ids: list[UUID]) -> Fact:
    """Rebuild a :class:`Fact` from a row plus its ordered evidence ids."""
    return Fact(
        id=row.id,
        document_id=row.document_id,
        subject=row.subject,
        canonical_subject=row.canonical_subject,
        predicate=row.predicate,
        canonical_predicate=row.canonical_predicate,
        value_kind=row.value_kind,  # type: ignore[arg-type]
        value_text=row.value_text,
        value_number=row.value_number,
        unit=row.unit,
        normalized_number=row.normalized_number,
        normalized_unit=row.normalized_unit,
        time_text=row.time_text,
        time_kind=row.time_kind,  # type: ignore[arg-type]
        time_start=row.time_start,
        time_end=row.time_end,
        scope_text=row.scope_text,
        estimate_status=row.estimate_status,  # type: ignore[arg-type]
        geography=row.geography,
        context=dict(row.context) if row.context is not None else {},
        evidence_ids=evidence_ids,
        extraction_confidence=row.extraction_confidence,
        ambiguity_flags=list(row.ambiguity_flags)
        if row.ambiguity_flags is not None
        else [],
        status=row.status,  # type: ignore[arg-type]
    )


def save_facts(session: Session, facts: list[Fact]) -> None:
    """Persist facts plus fact-evidence link rows.

    Adds all rows with a single flush; the caller commits.

    NOTE: the ``fact_evidence`` link table is unordered, so input order of
    ``evidence_ids`` cannot be preserved relationally. Retrieval
    (:func:`list_facts_for_document`, :func:`get_fact`) orders
    ``evidence_ids`` by evidence id string for determinism.
    """
    fact_rows = [
        FactRow(
            id=f.id,
            document_id=f.document_id,
            subject=f.subject,
            canonical_subject=f.canonical_subject,
            predicate=f.predicate,
            canonical_predicate=f.canonical_predicate,
            value_kind=_value_kind_str(f),
            value_text=f.value_text,
            value_number=f.value_number,
            unit=f.unit,
            normalized_number=f.normalized_number,
            normalized_unit=f.normalized_unit,
            time_text=f.time_text,
            time_kind=_time_kind_str(f),
            time_start=f.time_start,
            time_end=f.time_end,
            scope_text=f.scope_text,
            estimate_status=_estimate_status_str(f),
            geography=f.geography,
            context=dict(f.context) if f.context is not None else {},
            extraction_confidence=f.extraction_confidence,
            ambiguity_flags=list(f.ambiguity_flags)
            if f.ambiguity_flags is not None
            else [],
            status=_fact_status_str(f),
        )
        for f in facts
    ]
    link_rows = [
        FactEvidenceRow(fact_id=f.id, evidence_id=eid)
        for f in facts
        for eid in f.evidence_ids
    ]
    session.add_all(fact_rows)
    # Flush facts BEFORE link rows: the link rows reference FactRow only
    # by raw foreign key (no ORM relationship), so a single flush may emit
    # link INSERTs first, which PostgreSQL rejects. Two flushes keep the
    # ordering explicit; the caller still commits.
    session.flush()
    session.add_all(link_rows)
    session.flush()


def list_facts_for_document(session: Session, document_id: UUID) -> list[Fact]:
    """List facts for a document, ordered by fact id string for determinism.

    Each fact's ``evidence_ids`` are rebuilt from the link table ordered by
    evidence id string.
    """
    fact_rows = list(
        session.scalars(
            select(FactRow)
            .where(FactRow.document_id == document_id)
            .order_by(FactRow.id)
        ).all()
    )
    # Ensure id-string ordering for determinism (UUID order matches
    # lexicographic string order, but sort explicitly per the contract).
    fact_rows.sort(key=lambda r: str(r.id))
    if not fact_rows:
        return []
    fact_ids = [r.id for r in fact_rows]
    link_rows = list(
        session.scalars(
            select(FactEvidenceRow)
            .where(FactEvidenceRow.fact_id.in_(fact_ids))
            .order_by(FactEvidenceRow.evidence_id)
        ).all()
    )
    # Ensure evidence id-string ordering within each fact.
    link_rows.sort(key=lambda r: (str(r.fact_id), str(r.evidence_id)))
    by_fact: dict[UUID, list[UUID]] = {fid: [] for fid in fact_ids}
    for link in link_rows:
        by_fact.setdefault(link.fact_id, []).append(link.evidence_id)
    return [_fact_from_row(r, by_fact.get(r.id, [])) for r in fact_rows]


def get_fact(session: Session, fact_id: UUID) -> Fact | None:
    """Load a single fact by id, or ``None`` if missing.

    ``evidence_ids`` are rebuilt from the link table ordered by evidence id
    string.
    """
    row = session.get(FactRow, fact_id)
    if row is None:
        return None
    link_rows = list(
        session.scalars(
            select(FactEvidenceRow)
            .where(FactEvidenceRow.fact_id == fact_id)
            .order_by(FactEvidenceRow.evidence_id)
        ).all()
    )
    link_rows.sort(key=lambda r: str(r.evidence_id))
    evidence_ids = [r.evidence_id for r in link_rows]
    return _fact_from_row(row, evidence_ids)


def update_fact_normalization(session: Session, fact: Fact) -> bool:
    """Persist a normalized fact's reserved columns by id.

    Updates ONLY normalized_number, normalized_unit, canonical_subject,
    canonical_predicate, and ambiguity_flags. Returns False when the
    fact row does not exist; the caller commits.
    """
    row = session.get(FactRow, fact.id)
    if row is None:
        return False
    row.normalized_number = fact.normalized_number
    row.normalized_unit = fact.normalized_unit
    row.canonical_subject = fact.canonical_subject
    row.canonical_predicate = fact.canonical_predicate
    row.ambiguity_flags = list(fact.ambiguity_flags)
    session.flush()
    return True
