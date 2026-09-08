"""Ingestion pipeline (T4): PDF bytes → canonical evidence bundle.

Pure function — no database access. Maps PyMuPDF extraction output into
the canonical Pydantic models (Document / Page / EvidenceUnit) with
quality assessment, then adds ADDITIVE structured table evidence
(pdfplumber via app.extraction.tables). TEXT evidence is never replaced
or suppressed: a table page yields both representations. Partial
failures are captured explicitly: a document with some failed pages is
PARTIAL, with none usable FAILED.
"""

import logging
from hashlib import sha256
from uuid import uuid4

from app.core.config import settings
from app.extraction.pymupdf import extract_pdf
from app.extraction.quality import assess_quality
from app.extraction.tables import StructuredTable, extract_tables
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

logger = logging.getLogger(__name__)


def _grid_text(headers: list[str], table: StructuredTable) -> str:
    """Deterministic tab/newline rendering of the cleaned grid."""
    grid = [["" for _ in range(table.n_cols)] for _ in range(table.n_rows)]
    if table.n_rows and table.n_cols:
        for i, header in enumerate(headers[: table.n_cols]):
            grid[0][i] = header
    for cell in table.cells:
        if 0 <= cell.row < table.n_rows and 0 <= cell.col < table.n_cols:
            grid[cell.row][cell.col] = cell.text
    return "\n".join("\t".join(row) for row in grid)


def _table_evidence(
    document_id,
    table: StructuredTable,
    page_score: float,
    start_index: int,
) -> tuple[list[EvidenceUnit], int]:
    """Build TABLE + TABLE_CELL units for one structured table.

    Returns the units and the next free per-page evidence index.
    Tables that failed with no salvaged cells yield nothing (the failure
    is logged by the caller); partial cells are still emitted honestly
    with the parse error recorded in TABLE meta.
    """
    if not table.cells and not table.n_rows:
        return [], start_index

    index = start_index
    units: list[EvidenceUnit] = []

    row_first_col = {
        cell.row: cell.text for cell in table.cells if cell.col == 0
    }
    table_meta: dict = {
        "table_index": table.table_index,
        "n_rows": table.n_rows,
        "n_cols": table.n_cols,
        "headers": list(table.headers),
    }
    if table.error is not None:
        table_meta["parse_error"] = table.error

    units.append(
        EvidenceUnit(
            id=make_evidence_id(
                document_id, table.pdf_page_number, index, EvidenceType.TABLE
            ),
            document_id=document_id,
            pdf_page_number=table.pdf_page_number,
            source_page_number=table.source_page_number,
            type=EvidenceType.TABLE,
            text=_grid_text(table.headers, table),
            bbox=table.bbox,
            extraction_method=ExtractionMethod.PDFPLUMBER,
            extraction_quality=page_score,
            meta=table_meta,
        )
    )
    index += 1

    for cell in table.cells:
        column_header = (
            table.headers[cell.col]
            if 0 <= cell.col < len(table.headers)
            else ""
        )
        units.append(
            EvidenceUnit(
                id=make_evidence_id(
                    document_id,
                    table.pdf_page_number,
                    index,
                    EvidenceType.TABLE_CELL,
                ),
                document_id=document_id,
                pdf_page_number=table.pdf_page_number,
                source_page_number=table.source_page_number,
                type=EvidenceType.TABLE_CELL,
                text=cell.text,
                bbox=cell.bbox,
                extraction_method=ExtractionMethod.PDFPLUMBER,
                extraction_quality=page_score,
                meta={
                    "table_index": table.table_index,
                    "row": cell.row,
                    "col": cell.col,
                    "n_rows": table.n_rows,
                    "n_cols": table.n_cols,
                    "column_header": column_header,
                    "row_header": row_first_col.get(cell.row, ""),
                },
            )
        )
        index += 1
    return units, index


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

    tables_by_page: dict[int, list[StructuredTable]] = {}
    if settings.tables_enabled:
        try:
            for table in extract_tables(data):
                tables_by_page.setdefault(table.pdf_page_number, []).append(table)
        except ValueError as exc:
            logger.warning("table extraction skipped for %s: %s", filename, exc)

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

        # ADDITIVE table evidence: TEXT above is preserved untouched.
        table_index_counter = 0
        for table in tables_by_page.get(pe.pdf_page_number, []):
            if table.error is not None:
                logger.warning(
                    "table %d on page %d of %s: %s",
                    table.table_index,
                    pe.pdf_page_number,
                    filename,
                    table.error,
                )
            table_units, table_index_counter = _table_evidence(
                document_id, table, score, table_index_counter
            )
            evidence.extend(table_units)

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
