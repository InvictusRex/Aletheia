"""Evidence chunking for fact extraction (task F-C1).

Pure, deterministic function: groups canonical :class:`EvidenceUnit`
objects into page-scoped :class:`EvidenceChunk` inputs for the LLM
extractor. One chunk per page unless the page exceeds ``max_chars``.

Rules (frozen contract with the service layer):
- TEXT and friends become one :class:`ChunkUnit` each, in page order.
- TABLE units are prefixed with a ``Table {index} (columns: ...)``
  header line built from the TABLE meta ``headers``.
- TABLE_CELL units keep their row/col/header/table_index refs.
- Zero-text IMAGE units are SKIPPED (documented choice below): an
  IMAGE unit carries ``text == ""`` (image-only page marker, see
  ``app.extraction.pipeline``); there is no extractable content to
  cite, so emitting an empty unit would only spend an LLM call.
- Tables are ATOMIC: a TABLE unit plus all of its CELL units move
  together. Oversize tables split ONLY at row boundaries.

INVARIANT — never silently discard: every citable unit that cannot
fit the current chunk goes through the atomic-overflow path (alone
in its own chunk, budget exceeded rather than content lost). Any
future skip logic MUST append the skipped IDs to
``dropped_evidence_ids`` instead of dropping them quietly; in the
current logic that field is always empty. The zero-text IMAGE skip
above is an intentional, documented exclusion (no extractable
content), not a drop.
"""

from uuid import UUID

from app.models import ChunkUnit, EvidenceChunk, EvidenceType, EvidenceUnit, Page

__all__ = ["build_chunks"]


def _table_header_line(table_index: object, headers: object) -> str:
    """Render ``Table {i} (columns: h1 | h2 | ...):`` deterministically."""
    names = [str(h) for h in headers] if isinstance(headers, list) else []
    label = str(table_index) if table_index is not None else "?"
    return f"Table {label} (columns: {' | '.join(names)}):"


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _skip_unit(unit: EvidenceUnit) -> bool:
    """True only for zero-text IMAGE units (no extractable content).

    Documented choice: IMAGE units are page markers with ``text == ""``;
    including them would add a citable ID with nothing behind it, while
    skipping them loses no facts. Every other type is always kept.
    """
    return unit.type == EvidenceType.IMAGE and not unit.text.strip()


def _table_chunk_unit(unit: EvidenceUnit, body: str | None = None) -> ChunkUnit:
    meta = unit.meta or {}
    headers = meta.get("headers")
    names = [str(h) for h in headers] if isinstance(headers, list) else []
    header_line = _table_header_line(meta.get("table_index"), names)
    grid = body if body is not None else unit.text
    text = header_line if not grid else f"{header_line}\n{grid}"
    return ChunkUnit(
        evidence_id=unit.id,
        evidence_type=EvidenceType.TABLE.value,
        text=text,
        table_index=_as_int(meta.get("table_index")),
    )


def _cell_chunk_unit(unit: EvidenceUnit) -> ChunkUnit:
    meta = unit.meta or {}
    return ChunkUnit(
        evidence_id=unit.id,
        evidence_type=EvidenceType.TABLE_CELL.value,
        text=unit.text,
        row=_as_int(meta.get("row")),
        col=_as_int(meta.get("col")),
        column_header=_as_str(meta.get("column_header")),
        row_header=_as_str(meta.get("row_header")),
        table_index=_as_int(meta.get("table_index")),
    )


def _single_chunk_unit(unit: EvidenceUnit) -> ChunkUnit:
    """One ChunkUnit for TEXT-like units (TEXT/OCR_TEXT/IMAGE/other)."""
    return ChunkUnit(
        evidence_id=unit.id,
        evidence_type=unit.type.value,
        text=unit.text,
    )


def _cost(units: list[ChunkUnit]) -> int:
    return sum(len(u.text) for u in units)


def _split_table_group(
    table_unit: EvidenceUnit,
    cell_units: list[EvidenceUnit],
    max_chars: int,
) -> list[list[ChunkUnit]]:
    """Split an oversize table ONLY at row boundaries.

    Each split gets its own TABLE ChunkUnit (same evidence ID, repeated
    column-header line plus only that split's rows) followed by the CELL
    ChunkUnits for those rows. Row ranges are disjoint and cover every
    cell, so zero cell evidence IDs are lost or duplicated. A single
    row that alone exceeds ``max_chars`` goes alone (overflow).
    """
    meta = table_unit.meta or {}
    headers = meta.get("headers")
    names = [str(h) for h in headers] if isinstance(headers, list) else []
    header_line = _table_header_line(meta.get("table_index"), names)

    if not cell_units:
        # No rows to split on: the TABLE unit stands alone (overflow).
        return [[_table_chunk_unit(table_unit)]]

    def row_key(unit: EvidenceUnit) -> tuple[bool, int]:
        row = (unit.meta or {}).get("row")
        return (True, 0) if not isinstance(row, int) or isinstance(row, bool) else (False, row)

    def col_key(unit: EvidenceUnit) -> tuple[bool, int]:
        col = (unit.meta or {}).get("col")
        return (True, 0) if not isinstance(col, int) or isinstance(col, bool) else (False, col)

    ordered = sorted(cell_units, key=lambda u: (row_key(u), col_key(u)))
    rows: list[list[EvidenceUnit]] = []
    for unit in ordered:
        key = row_key(unit)
        if not rows or row_key(rows[-1][-1]) != key:
            rows.append([unit])
        else:
            rows[-1].append(unit)

    splits: list[list[ChunkUnit]] = []
    current_rows: list[list[EvidenceUnit]] = []

    def split_cost(row_groups: list[list[EvidenceUnit]]) -> int:
        body = "\n".join(
            " | ".join(cell.text for cell in group) for group in row_groups
        )
        table_text = header_line if not body else f"{header_line}\n{body}"
        cells_cost = sum(len(cell.text) for group in row_groups for cell in group)
        return len(table_text) + cells_cost

    def flush_current() -> None:
        if not current_rows:
            return
        body = "\n".join(
            " | ".join(cell.text for cell in group) for group in current_rows
        )
        chunk_units = [_table_chunk_unit(table_unit, body=body)]
        for group in current_rows:
            chunk_units.extend(_cell_chunk_unit(cell) for cell in group)
        splits.append(chunk_units)
        del current_rows[:]

    for group in rows:
        if current_rows and split_cost([*current_rows, group]) > max_chars:
            # Row boundary split: seal the current split; the row starts
            # the next one (alone, as overflow, if it fits nowhere).
            flush_current()
        current_rows.append(group)
    flush_current()
    return splits


def build_chunks(
    document_id: UUID,
    pages: list[Page],
    evidence: list[EvidenceUnit],
    max_chars: int = 2500,
) -> list[EvidenceChunk]:
    """Group evidence into page-scoped extraction chunks (pure function).

    Deterministic: pages iterate in ascending ``pdf_page_number`` order
    and units keep their input order within a page.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be > 0, got {max_chars!r}")

    page_by_number: dict[int, Page] = {}
    for page in pages:
        page_by_number.setdefault(page.pdf_page_number, page)

    by_page: dict[int, list[EvidenceUnit]] = {}
    for unit in evidence:
        by_page.setdefault(unit.pdf_page_number, []).append(unit)

    chunks: list[EvidenceChunk] = []
    for pdf_page_number in sorted(set(by_page) | set(page_by_number)):
        page = page_by_number.get(pdf_page_number)
        page_units = by_page.get(pdf_page_number, [])
        citable = [u for u in page_units if not _skip_unit(u)]
        if not citable:
            # Empty page (no citable units): no chunk, no LLM call.
            continue
        source_page_number = (
            page.source_page_number
            if page is not None
            else (citable[0].source_page_number if citable else None)
        )

        # Partition page units into atomic placeable items, preserving
        # page order: singletons plus TABLE groups (TABLE + its CELLs,
        # claimed by matching meta table_index wherever they sit).
        cells_by_table: dict[int | None, list[EvidenceUnit]] = {}
        for unit in citable:
            if unit.type == EvidenceType.TABLE_CELL:
                key = (unit.meta or {}).get("table_index")
                key = key if isinstance(key, int) and not isinstance(key, bool) else None
                cells_by_table.setdefault(key, []).append(unit)
        claimed: set[int] = set()  # id() of consumed cell units
        items: list[tuple[list[ChunkUnit], bool, EvidenceUnit | None, list[EvidenceUnit]]] = []
        for unit in citable:
            if unit.type == EvidenceType.TABLE:
                key = (unit.meta or {}).get("table_index")
                key = key if isinstance(key, int) and not isinstance(key, bool) else None
                group_cells = [
                    c for c in cells_by_table.get(key, []) if id(c) not in claimed
                ]
                for c in group_cells:
                    claimed.add(id(c))
                group = [_table_chunk_unit(unit)] + [
                    _cell_chunk_unit(c) for c in group_cells
                ]
                items.append((group, True, unit, group_cells))
            elif unit.type == EvidenceType.TABLE_CELL:
                if id(unit) in claimed:
                    continue  # carried inside its TABLE group
                # Orphan cell (no matching TABLE on this page): keep as a
                # singleton so its ID is never lost.
                items.append(([_cell_chunk_unit(unit)], False, None, []))
            else:
                items.append(([_single_chunk_unit(unit)], False, None, []))

        current: list[ChunkUnit] = []
        current_cost = 0

        def flush() -> None:
            nonlocal current, current_cost
            if current:
                chunks.append(
                    EvidenceChunk(
                        document_id=document_id,
                        pdf_page_number=pdf_page_number,
                        source_page_number=source_page_number,
                        units=current,
                        dropped_evidence_ids=[],
                    )
                )
                current = []
                current_cost = 0

        def place(units: list[ChunkUnit]) -> None:
            """Place one atomic item; overflow stands alone, never dropped."""
            nonlocal current, current_cost
            item_cost = _cost(units)
            if current and current_cost + item_cost > max_chars:
                flush()
            current.extend(units)
            current_cost += item_cost
            if item_cost > max_chars:
                # Atomic-overflow path: a single over-long block (or row)
                # goes alone in its own chunk; atomicity beats the budget.
                flush()

        for group, is_table, table_unit, group_cells in items:
            if is_table and table_unit is not None and _cost(group) > max_chars:
                flush()
                for split in _split_table_group(table_unit, group_cells, max_chars):
                    place(split)
            else:
                place(group)
        flush()

    return chunks
