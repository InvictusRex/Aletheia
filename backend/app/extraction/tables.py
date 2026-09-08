"""Structured table extraction with pdfplumber (Phase 1, task T-A).

Pipeline position (PLAN Section 40)::

    PDF
      ↓
    PyMuPDF (candidate table detection)
      ↓
    pdfplumber (structured extraction)
      ↓
    StructuredTable
      ↓
    Evidence Units (built by the pipeline, not here)

This module is document-agnostic: it contains no dataset names, metric
names, years, or filenames. Cell text is preserved verbatim (stripped of
surrounding whitespace only); no semantic claims are made about headers
or values.

Dependency notes: ``import pymupdf`` is used (never the deprecated
``import fitz``). pdfplumber is used only for structured tables, per
PLAN Section 4.2.

Decisions taken where the contract left a choice (all documented here):

1. ``pages`` validation: ``None`` means all pages. Any entry that is
   not an ``int`` (``bool`` counts as invalid), is negative, or is
   ``>=`` the page count raises ``ValueError``. An empty list yields
   ``[]``. Duplicate indices are de-duplicated preserving first-seen
   order, so a page is never extracted twice.
2. Grid cleaning: ragged rows are padded with ``""`` to the max width,
   then fully-empty rows and fully-empty columns are dropped, then
   ``row``/``col`` are re-indexed onto the cleaned grid. ``n_rows`` /
   ``n_cols`` describe the cleaned grid (header row included in
   ``n_rows``). ``headers`` is the cleaned first row *including* ``""``
   placeholders, so ``headers[c]`` always aligns with column ``c``.
3. Candidate gating fail-open: if PyMuPDF ``page.find_tables()`` itself
   raises for a page, the page is treated as a candidate and pdfplumber
   is still attempted (gating must not silently suppress tables on a
   page PyMuPDF cannot analyse).
4. Retry strategy: default ``find_tables()`` first; when a candidate
   page yields nothing, exactly one retry with
   ``{"text_x_tolerance": 3, "text_y_tolerance": 3}`` (line strategy
   unchanged). Broader fallbacks (e.g. ``strategy="text"`` for
   borderless tables) are deliberately out of scope for this task.
   Which attempts ran is recorded in the ``error`` string whenever a
   candidate page ultimately yields nothing.
5. A candidate page where pdfplumber finds nothing after both attempts
   emits ONE ``StructuredTable`` with ``error`` set (the attempts are
   recorded there) and empty ``cells``. Pages with no PyMuPDF
   candidates are skipped entirely and contribute nothing, so an empty
   return list unambiguously means "no tables found".
6. Coordinates: pdfplumber cell tuples ``(x0, top, x1, bottom)`` are in
   the same PDF-point space as PyMuPDF bboxes, so they are stored
   directly after ordering (``x0 <= x1``, ``y0 <= y1``) and
   float-casting. ``None`` is stored when geometry is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import pdfplumber
import pymupdf

#: Retry settings used when a candidate page yields no tables with the
#: default pdfplumber settings. Keeps the default line-based strategy;
#: only the text-extraction tolerances are made explicit.
_RETRY_TABLE_SETTINGS: dict = {"text_x_tolerance": 3, "text_y_tolerance": 3}

#: Text-extraction kwargs mirroring what ``Page.extract_tables`` passes
#: to ``Table.extract`` for a given settings dict (``text_*`` keys with
#: the prefix stripped).
_DEFAULT_TEXT_KWARGS: dict = {"x_tolerance": 3, "y_tolerance": 3}
_RETRY_TEXT_KWARGS: dict = {"x_tolerance": 3, "y_tolerance": 3}


@dataclass
class TableCell:
    """One non-empty table cell: raw text plus its grid position."""

    text: str
    bbox: tuple[float, float, float, float] | None
    row: int
    col: int


@dataclass
class StructuredTable:
    """One extracted table. ``error`` set instead of raising."""

    pdf_page_number: int
    source_page_number: str | None
    table_index: int
    bbox: tuple[float, float, float, float] | None
    headers: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    cells: list[TableCell] = field(default_factory=list)
    error: str | None = None


def _page_labels(doc: pymupdf.Document) -> list[str | None]:
    """Best-effort source page numbers from PDF page labels only.

    Mirrors ``pymupdf.py``: ``None`` per page when the document defines
    no ``/PageLabels`` or when lookup fails. Never synthesizes.
    """
    try:
        defined = doc.get_page_labels()
    except Exception:
        return [None] * len(doc)
    if not defined:
        return [None] * len(doc)
    labels: list[str | None] = []
    for index in range(len(doc)):
        try:
            label = doc[index].get_label()
        except Exception:
            label = ""
        labels.append(label or None)
    return labels


def _ordered_bbox(bbox) -> tuple[float, float, float, float] | None:
    """Order a ``(x0, y0, x1, y1)``-style tuple; ``None`` when unusable."""
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except Exception:
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _candidate_count(pymu_page: pymupdf.Page) -> int:
    """Number of PyMuPDF candidate tables on a page (fail-open).

    Raises are swallowed and reported as "has candidates" so that a
    gating failure never silently suppresses a page (see module
    docstring, decision 3).
    """
    try:
        finder = pymu_page.find_tables()
    except Exception:
        return 1
    try:
        tables = getattr(finder, "tables", None)
        if tables is None:
            tables = list(finder)
        return len(tables)
    except Exception:
        return 1


def _safe_table_bbox(table_obj) -> tuple[float, float, float, float] | None:
    """Best-effort table bbox; ``None`` when unavailable."""
    try:
        return _ordered_bbox(table_obj.bbox)
    except Exception:
        return None


def _build_table(
    table_obj,
    pdf_page_number: int,
    source_page_number: str | None,
    table_index: int,
    strategy_desc: str,
) -> StructuredTable:
    """Build one ``StructuredTable`` from a pdfplumber ``Table`` object.

    Never raises: any mid-parse failure is captured in ``error`` with
    whatever cells were salvaged (possibly empty).
    """
    try:
        text_kwargs = (
            _RETRY_TEXT_KWARGS
            if strategy_desc == "retry"
            else _DEFAULT_TEXT_KWARGS
        )
        raw_grid = table_obj.extract(**text_kwargs)
    except Exception as exc:
        return StructuredTable(
            pdf_page_number=pdf_page_number,
            source_page_number=source_page_number,
            table_index=table_index,
            bbox=_safe_table_bbox(table_obj),
            headers=[],
            n_rows=0,
            n_cols=0,
            cells=[],
            error=(
                f"page {pdf_page_number} table {table_index}: "
                f"text extraction failed ({strategy_desc} settings) "
                f"({type(exc).__name__}: {exc})"
            ),
        )

    bbox = _safe_table_bbox(table_obj)

    try:
        rows = list(table_obj.rows)
    except Exception:
        rows = []
    # Geometry grid aligned 1:1 with ``raw_grid`` (``Table.extract``
    # iterates ``row.cells``); ``None`` entries mark missing cells.
    try:
        geom_grid: list[list] = [list(row.cells) for row in rows]
    except Exception:
        geom_grid = []

    # Normalize text grid: ``None`` -> ``""``, strip, pad ragged rows.
    text_grid: list[list[str]] = []
    for raw_row in raw_grid or []:
        try:
            cells = list(raw_row)
        except TypeError:
            continue
        text_grid.append(
            [c.strip() if isinstance(c, str) else "" for c in cells]
        )
    width = max((len(r) for r in text_grid), default=0)
    for r in text_grid:
        r.extend([""] * (width - len(r)))
    for g in geom_grid:
        g.extend([None] * (width - len(g)))
    while len(geom_grid) < len(text_grid):
        geom_grid.append([None] * width)

    # Drop fully-empty rows, then fully-empty columns.
    keep_rows = [
        i for i, r in enumerate(text_grid) if any(c != "" for c in r)
    ]
    keep_cols = [
        j
        for j in range(width)
        if any(text_grid[i][j] != "" for i in keep_rows)
    ]
    cleaned = [[text_grid[i][j] for j in keep_cols] for i in keep_rows]
    cleaned_geom = [[geom_grid[i][j] for j in keep_cols] for i in keep_rows]

    n_rows = len(cleaned)
    n_cols = len(keep_cols)
    headers = list(cleaned[0]) if cleaned else []

    cells: list[TableCell] = []
    try:
        for r, (row, grow) in enumerate(zip(cleaned, cleaned_geom)):
            for c, (text, geom) in enumerate(zip(row, grow)):
                if text == "":
                    continue
                cells.append(
                    TableCell(
                        text=text,
                        bbox=_ordered_bbox(geom) if geom is not None else None,
                        row=r,
                        col=c,
                    )
                )
    except Exception as exc:
        return StructuredTable(
            pdf_page_number=pdf_page_number,
            source_page_number=source_page_number,
            table_index=table_index,
            bbox=bbox,
            headers=headers,
            n_rows=n_rows,
            n_cols=n_cols,
            cells=cells,
            error=(
                f"page {pdf_page_number} table {table_index}: "
                f"partial parse ({strategy_desc} settings; salvaged "
                f"{len(cells)} cells) "
                f"({type(exc).__name__}: {exc})"
            ),
        )

    return StructuredTable(
        pdf_page_number=pdf_page_number,
        source_page_number=source_page_number,
        table_index=table_index,
        bbox=bbox,
        headers=headers,
        n_rows=n_rows,
        n_cols=n_cols,
        cells=cells,
        error=None,
    )


def _validate_pages(pages: list[int] | None, page_count: int) -> list[int]:
    """Validate the ``pages`` argument; return deduped target indices.

    Raises:
        ValueError: For non-integer entries (``bool`` included),
            negative indices, or indices beyond the page count.
    """
    if pages is None:
        return list(range(page_count))
    try:
        requested = list(pages)
    except TypeError as exc:
        raise ValueError(
            f"pages must be a list of 0-based page indices: {exc}"
        ) from exc
    targets: list[int] = []
    for entry in requested:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(
                f"invalid page index {entry!r}: expected int in "
                f"[0, {page_count})"
            )
        if entry < 0 or entry >= page_count:
            raise ValueError(
                f"page index {entry} out of range for {page_count}-page document"
            )
        if entry not in targets:
            targets.append(entry)
    return targets


def extract_tables(
    data: bytes, pages: list[int] | None = None
) -> list[StructuredTable]:
    """Extract structured tables from PDF bytes with pdfplumber.

    Gating (PLAN Section 40): PyMuPDF ``page.find_tables()`` decides
    which pages deserve pdfplumber extraction; pages without candidates
    are skipped without running pdfplumber table search on them.

    Strategy per candidate page: default ``find_tables()`` first (the
    same search ``extract_tables()`` performs, but keeping the
    ``Table`` objects so cell bboxes are preserved); if that yields no
    tables, one retry with ``_RETRY_TABLE_SETTINGS``. A candidate page
    that still yields nothing emits one ``StructuredTable`` with
    ``error`` set recording both attempts.

    Args:
        data: Raw PDF bytes.
        pages: Optional 0-based page indices to restrict extraction to.

    Returns:
        List of ``StructuredTable`` (empty when no tables found).

    Raises:
        ValueError: If the input is empty, not bytes, or not a readable
            PDF, or if any requested page index is out of range.
    """
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes) or len(data) == 0:
        raise ValueError("extract_tables requires non-empty PDF bytes")

    # --- Gate with PyMupdf (opened once, always closed). ---
    try:
        pymu_doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(
            f"unable to open PDF stream ({type(exc).__name__}): {exc}"
        ) from exc

    try:
        if len(pymu_doc) == 0:
            raise ValueError("PDF contains no pages")
        targets = _validate_pages(pages, len(pymu_doc))
        labels = _page_labels(pymu_doc)
        candidates: list[int] = [
            i for i in targets if _candidate_count(pymu_doc[i]) > 0
        ]
    finally:
        pymu_doc.close()

    if not candidates:
        return []

    # --- Extract with pdfplumber (bytes opened ONCE, always closed). ---
    try:
        plumber_doc = pdfplumber.open(BytesIO(data))
    except Exception as exc:
        raise ValueError(
            f"unable to open PDF stream with pdfplumber "
            f"({type(exc).__name__}): {exc}"
        ) from exc

    try:
        results: list[StructuredTable] = []
        for index in candidates:
            label = labels[index]
            try:
                plumber_page = plumber_doc.pages[index]
            except Exception as exc:
                results.append(
                    StructuredTable(
                        pdf_page_number=index,
                        source_page_number=label,
                        table_index=0,
                        bbox=None,
                        headers=[],
                        n_rows=0,
                        n_cols=0,
                        cells=[],
                        error=f"page {index}: cannot access pdfplumber page "
                        f"({type(exc).__name__}: {exc})",
                    )
                )
                continue
            try:
                found = plumber_page.find_tables()
                strategy = "default"
                if not found:
                    found = plumber_page.find_tables(
                        table_settings=dict(_RETRY_TABLE_SETTINGS)
                    )
                    strategy = "retry"
                    attempts = (
                        "default find_tables() + retry "
                        f"find_tables({_RETRY_TABLE_SETTINGS})"
                    )
                else:
                    attempts = "default find_tables()"
            except Exception as exc:
                results.append(
                    StructuredTable(
                        pdf_page_number=index,
                        source_page_number=label,
                        table_index=0,
                        bbox=None,
                        headers=[],
                        n_rows=0,
                        n_cols=0,
                        cells=[],
                        error=f"page {index}: pdfplumber table search failed "
                        f"({type(exc).__name__}: {exc})",
                    )
                )
                continue

            if not found:
                results.append(
                    StructuredTable(
                        pdf_page_number=index,
                        source_page_number=label,
                        table_index=0,
                        bbox=None,
                        headers=[],
                        n_rows=0,
                        n_cols=0,
                        cells=[],
                        error=(
                            f"page {index}: PyMuPDF reported candidate "
                            f"table(s) but pdfplumber found none "
                            f"(tried {attempts})"
                        ),
                    )
                )
                continue

            next_index = 0
            for table_obj in found:
                try:
                    results.append(
                        _build_table(
                            table_obj,
                            pdf_page_number=index,
                            source_page_number=label,
                            table_index=next_index,
                            strategy_desc=strategy,
                        )
                    )
                except Exception as exc:  # never let one table kill the page
                    results.append(
                        StructuredTable(
                            pdf_page_number=index,
                            source_page_number=label,
                            table_index=next_index,
                            bbox=_safe_table_bbox(table_obj),
                            headers=[],
                            n_rows=0,
                            n_cols=0,
                            cells=[],
                            error=(
                                f"page {index} table {next_index}: "
                                f"unexpected build failure ({strategy} "
                                f"settings) "
                                f"({type(exc).__name__}: {exc})"
                            ),
                        )
                    )
                next_index += 1
        return results
    finally:
        try:
            plumber_doc.close()
        except Exception:
            pass
