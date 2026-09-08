"""T-D unit tests: table-aware PDF extraction (additive pdfplumber step).

All PDFs are generated in-memory with PyMuPDF ruled grids
(page.draw_line + page.insert_text, then tobytes()); no binary fixtures.
"""

import pymupdf
import pytest

import app.extraction.pipeline as pipeline_module
from app.extraction.pipeline import run_ingestion
from app.extraction.tables import extract_tables
from app.models.evidence import EvidenceType, ExtractionMethod

HEADER_A = "Metric"
HEADER_B = "FY24"
HEADER_C = "FY25"


def _draw_grid(page, headers, rows, origin=(72, 72), col_width=120, row_h=28):
    """Draw a ruled grid with text centred-ish inside each cell."""
    x0, y0 = origin
    ncols = len(headers)
    nrows = len(rows) + 1
    x1 = x0 + col_width * ncols
    y1 = y0 + row_h * nrows
    for r in range(nrows + 1):
        y = y0 + r * row_h
        page.draw_line((x0, y), (x1, y))
    for c in range(ncols + 1):
        x = x0 + c * col_width
        page.draw_line((x, y0), (x, y1))
    grid = [headers] + rows
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            page.insert_text(
                (x0 + c * col_width + 4, y0 + r * row_h + 18),
                value,
                fontsize=10,
            )


def _make_table_pdf(headers, rows, origin=(72, 72)):
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        _draw_grid(page, headers, rows, origin=origin)
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _make_multi_table_pdf():
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        _draw_grid(
            page, [HEADER_A, HEADER_B], [["Revenue", "100"]], origin=(72, 72)
        )
        _draw_grid(
            page, [HEADER_A, HEADER_C], [["Cost", "40"]], origin=(72, 260)
        )
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _make_text_pdf(text="Hello plain world. " * 30):
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _good_table(tables):
    """Return the single successfully parsed table (all fixtures are clean)."""
    ok = [t for t in tables if t.error is None]
    assert ok, f"expected a parsed table, got errors: {[t.error for t in tables]}"
    return ok[0]


def test_extract_tables_ruled_grid_headers_dims_bboxes():
    data = _make_table_pdf(
        [HEADER_A, HEADER_B], [["Revenue", "100"], ["Cost", "40"]]
    )
    tables = extract_tables(data)
    assert len(tables) == 1
    table = tables[0]
    assert table.error is None
    assert table.headers == [HEADER_A, HEADER_B]
    assert table.n_rows == 3  # header row included
    assert table.n_cols == 2
    assert len(table.cells) == 6  # no empty cells in this fixture
    for cell in table.cells:
        assert 0 <= cell.row < table.n_rows
        assert 0 <= cell.col < table.n_cols
        assert cell.text != ""
        assert cell.bbox is not None, "ruled-grid cells must carry geometry"
        assert len(cell.bbox) == 4
        x0, y0, x1, y1 = cell.bbox
        assert all(isinstance(v, float) for v in cell.bbox)
        assert x0 <= x1
        assert y0 <= y1
    if table.bbox is not None:
        x0, y0, x1, y1 = table.bbox
        assert x0 <= x1
        assert y0 <= y1


def test_value_to_header_association_from_cells_only():
    data = _make_table_pdf(
        [HEADER_A, HEADER_B], [["Revenue", "100"], ["Cost", "40"]]
    )
    table = _good_table(extract_tables(data))
    # Reconstruct the grid position purely from public cells/meta.
    by_pos = {(c.row, c.col): c.text for c in table.cells}
    value_cells = [c for c in table.cells if c.text == "100"]
    assert len(value_cells) == 1
    value = value_cells[0]
    assert value.row > 0, "data value must sit below the header row"
    header_text = by_pos[(0, value.col)]
    assert header_text == HEADER_B
    assert table.headers[value.col] == HEADER_B


def test_table_free_text_page_returns_empty():
    data = _make_text_pdf()
    assert extract_tables(data) == []


@pytest.mark.parametrize("bad", [b"", b"not a pdf at all", b"\x00\x01\x02\x03"])
def test_garbage_bytes_raise_valueerror(bad):
    with pytest.raises(ValueError):
        extract_tables(bad)


def test_multiple_tables_distinct_table_index():
    data = _make_multi_table_pdf()
    tables = extract_tables(data)
    ok = [t for t in tables if t.error is None]
    assert len(ok) == 2
    indices = sorted(t.table_index for t in ok)
    assert indices == [0, 1]
    assert len({t.table_index for t in ok}) == 2
    assert {t.pdf_page_number for t in ok} == {0}
    headers = sorted(tuple(t.headers) for t in ok)
    assert headers == [
        (HEADER_A, HEADER_B),
        (HEADER_A, HEADER_C),
    ]


def test_run_ingestion_table_page_is_additive():
    data = _make_table_pdf(
        [HEADER_A, HEADER_B], [["Revenue", "100"], ["Cost", "40"]]
    )
    document, pages, evidence = run_ingestion("table.pdf", data)
    by_type = {}
    for unit in evidence:
        by_type.setdefault(unit.type, []).append(unit)
    assert by_type.get(EvidenceType.TEXT), "TEXT evidence must be preserved"
    assert by_type.get(EvidenceType.TABLE), "expected TABLE evidence"
    assert by_type.get(EvidenceType.TABLE_CELL), "expected TABLE_CELL evidence"

    assert len(pages) == 1
    page_score = pages[0].extraction_quality

    table_units = by_type[EvidenceType.TABLE]
    assert len(table_units) == 1
    table_meta = table_units[0].meta
    assert table_meta["table_index"] == 0
    assert table_meta["n_rows"] == 3
    assert table_meta["n_cols"] == 2
    assert table_meta["headers"] == [HEADER_A, HEADER_B]

    for unit in by_type[EvidenceType.TABLE_CELL]:
        assert unit.extraction_method == ExtractionMethod.PDFPLUMBER
        assert unit.document_id == document.id
        assert unit.extraction_quality == page_score
        for key in (
            "table_index",
            "row",
            "col",
            "n_rows",
            "n_cols",
            "column_header",
            "row_header",
        ):
            assert key in unit.meta, f"TABLE_CELL meta missing {key}"
        # Context headers must be non-empty strings for this dense fixture.
        assert isinstance(unit.meta["column_header"], str)
        assert unit.meta["column_header"] != ""
        assert isinstance(unit.meta["row_header"], str)
        assert unit.meta["row_header"] != ""
        # Column context must agree with the TABLE headers.
        assert unit.meta["column_header"] == table_meta["headers"][unit.meta["col"]]


def test_table_unit_text_contains_headers_and_values():
    data = _make_table_pdf(
        [HEADER_A, HEADER_B], [["Revenue", "100"], ["Cost", "40"]]
    )
    _, _, evidence = run_ingestion("table.pdf", data)
    table_units = [e for e in evidence if e.type == EvidenceType.TABLE]
    assert len(table_units) == 1
    text = table_units[0].text
    for token in (HEADER_A, HEADER_B, "Revenue", "100"):
        assert token in text


def test_table_evidence_ids_differ_from_text_ids():
    data = _make_table_pdf(
        [HEADER_A, HEADER_B], [["Revenue", "100"], ["Cost", "40"]]
    )
    _, _, evidence = run_ingestion("table.pdf", data)
    text_ids = {e.id for e in evidence if e.type == EvidenceType.TEXT}
    table_ids = {
        e.id for e in evidence if e.type in (EvidenceType.TABLE, EvidenceType.TABLE_CELL)
    }
    assert text_ids, "precondition: TEXT evidence expected"
    assert table_ids, "precondition: TABLE evidence expected"
    assert text_ids.isdisjoint(table_ids), "evidence IDs must not collide"
    assert len(table_ids) == len(
        [e for e in evidence if e.type in (EvidenceType.TABLE, EvidenceType.TABLE_CELL)]
    ), "TABLE evidence IDs must be unique"


def test_tables_disabled_yields_text_only(monkeypatch):
    # Pipeline reads `settings.tables_enabled` via the object imported as
    # `app.extraction.pipeline.settings` (from app.core.config import settings),
    # so patch that exact reference.
    monkeypatch.setattr(pipeline_module.settings, "tables_enabled", False)
    data = _make_table_pdf(
        [HEADER_A, HEADER_B], [["Revenue", "100"], ["Cost", "40"]]
    )
    _, _, evidence = run_ingestion("table.pdf", data)
    assert evidence, "TEXT evidence must remain with tables disabled"
    assert all(
        e.type not in (EvidenceType.TABLE, EvidenceType.TABLE_CELL) for e in evidence
    )
    assert any(e.type == EvidenceType.TEXT for e in evidence)


def test_pipeline_cell_meta_binds_value_to_period_header():
    """Pipeline-level value→header check via TABLE_CELL meta (period column)."""
    data = _make_table_pdf(
        [HEADER_A, HEADER_B], [["Revenue", "100"], ["Cost", "40"]]
    )
    _, _, evidence = run_ingestion("table.pdf", data)
    cells = [e for e in evidence if e.type == EvidenceType.TABLE_CELL]
    matches = [c for c in cells if c.text == "100"]
    assert len(matches) == 1
    unit = matches[0]
    assert unit.meta["column_header"] == HEADER_B
    assert unit.meta["row_header"] == "Revenue"
    assert unit.meta["row"] > 0, "data value must sit below the header row"
