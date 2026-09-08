"""Ingestion pipeline (T4): PDF bytes → canonical evidence bundle.

Pure function — no database access. Maps PyMuPDF extraction output into
the canonical Pydantic models (Document / Page / EvidenceUnit) with
quality assessment. Partial failures are captured explicitly:
a document with some failed pages is PARTIAL, with none usable FAILED.
"""

from hashlib import sha256
from uuid import uuid4

from app.extraction.pymupdf import extract_pdf
from app.extraction.quality import assess_quality
from app.models import (
    Document,
    EvidenceType,
    EvidenceUnit,
    ExtractionMethod,
    IngestionStatus,
    Page,
    QualityVerdict,
    make_evidence_id,
)


def run_ingestion(
    filename: str, data: bytes, source: str | None = None
) -> tuple[Document, list[Page], list[EvidenceUnit]]:
    document_id = uuid4()
    file_sha256 = sha256(data).hexdigest()

    try:
        metadata, extracted = extract_pdf(data)
    except ValueError as exc:
        document = Document(
            id=document_id,
            filename=filename,
            source=source,
            page_count=0,
            file_sha256=file_sha256,
            ingestion_status=IngestionStatus.FAILED,
            error=str(exc),
        )
        return document, [], []

    pages: list[Page] = []
    evidence: list[EvidenceUnit] = []
    failed = 0

    for pe in extracted:
        if pe.error is not None and not pe.blocks:
            failed += 1
            pages.append(
                Page(
                    document_id=document_id,
                    pdf_page_number=pe.pdf_page_number,
                    source_page_number=pe.source_page_number,
                    width=0.0,
                    height=0.0,
                    quality_verdict=QualityVerdict.BAD,
                    error=pe.error,
                )
            )
            continue

        score, verdict = assess_quality(
            pe.char_count, pe.word_count, pe.suspicious_ratio, pe.has_images
        )
        pages.append(
            Page(
                document_id=document_id,
                pdf_page_number=pe.pdf_page_number,
                source_page_number=pe.source_page_number,
                width=pe.width,
                height=pe.height,
                char_count=pe.char_count,
                word_count=pe.word_count,
                suspicious_ratio=pe.suspicious_ratio,
                has_images=pe.has_images,
                extraction_quality=score,
                quality_verdict=verdict,
                error=pe.error,
            )
        )
        for block in pe.blocks:
            evidence.append(
                EvidenceUnit(
                    id=make_evidence_id(
                        document_id, pe.pdf_page_number, block.block_index,
                        EvidenceType.TEXT,
                    ),
                    document_id=document_id,
                    pdf_page_number=pe.pdf_page_number,
                    source_page_number=pe.source_page_number,
                    type=EvidenceType.TEXT,
                    text=block.text,
                    bbox=block.bbox,
                    extraction_method=ExtractionMethod.PYMUPDF,
                    extraction_quality=score,
                    meta={"block_index": block.block_index},
                )
            )
        if pe.has_images and not pe.blocks:
            # Explicit marker so image-only pages are never silent gaps.
            evidence.append(
                EvidenceUnit(
                    id=make_evidence_id(
                        document_id, pe.pdf_page_number, 0, EvidenceType.IMAGE
                    ),
                    document_id=document_id,
                    pdf_page_number=pe.pdf_page_number,
                    source_page_number=pe.source_page_number,
                    type=EvidenceType.IMAGE,
                    text="",
                    bbox=None,
                    extraction_method=ExtractionMethod.PYMUPDF,
                    extraction_quality=0.0,
                    meta={"image_count": pe.image_count, "note": "NEEDS_OCR"},
                )
            )

    if failed == 0:
        status = IngestionStatus.COMPLETED
        error = None
    elif evidence:
        status = IngestionStatus.PARTIAL
        error = f"{failed} of {len(extracted)} pages failed extraction"
    else:
        status = IngestionStatus.FAILED
        error = f"all {len(extracted)} pages failed extraction"

    document = Document(
        id=document_id,
        filename=filename,
        title=metadata.title,
        source=source,
        page_count=metadata.page_count,
        file_sha256=file_sha256,
        ingestion_status=status,
        error=error,
    )
    return document, pages, evidence
