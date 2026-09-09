"""Extraction prompt builder for fact extraction (task F-C1).

Pure, deterministic function: renders one :class:`EvidenceChunk` as a
self-contained extractor prompt. The service layer sends the returned
string to the LLM provider verbatim.

The instructions implement the extraction checklist: only facts
supported by the supplied evidence, never invented; verbatim values;
subject/predicate/value/unit/time/scope/qualifiers; evidence IDs
cited from the supplied list ONLY; confidence in 0..1; ambiguity
flagged. The prompt carries no knowledge of any particular source
collection, indicator, period, or file — everything the extractor may
use is inside the evidence blocks below.
"""

from app.models import EvidenceChunk
from app.models.fact import ChunkUnit

__all__ = ["build_extraction_prompt"]


def _render_unit(unit: ChunkUnit) -> str:
    """Render one block as ``[<evidence_id>] (<TYPE>) <text>``.

    TABLE_CELL blocks carry their row/col/header refs so the extractor
    can cite the most specific unit available.
    """
    tag = f"[{unit.evidence_id}] ({unit.evidence_type})"
    if unit.evidence_type == "TABLE_CELL":
        refs = (
            f"table={unit.table_index if unit.table_index is not None else '?'} "
            f"row={unit.row if unit.row is not None else '?'} "
            f"col={unit.col if unit.col is not None else '?'} "
            f"column_header={unit.column_header if unit.column_header else '?'} "
            f"row_header={unit.row_header if unit.row_header else '?'}"
        )
        return f"{tag} {refs} :: {unit.text}"
    return f"{tag} {unit.text}"


def build_extraction_prompt(chunk: EvidenceChunk) -> str:
    """Build the deterministic extractor prompt for one chunk."""
    lines: list[str] = []
    lines.append("Extract structured facts from the evidence blocks below.")
    lines.append("")
    lines.append("Rules:")
    lines.append(
        "1. Propose a fact ONLY when it is directly supported by the "
        "supplied evidence. If no supported fact exists, return empty drafts."
    )
    lines.append(
        "2. Never invent, infer beyond what is written, or use outside "
        "knowledge. Every claim must trace to cited evidence. Do not "
        "infer missing dates, units, entities, or values."
    )
    lines.append(
        "3. Preserve values verbatim as written in the evidence, including "
        "their units and qualifiers; do not convert, round, or rephrase values. "
        "Do NOT perform unit conversion."
    )
    lines.append(
        "4. Each draft states a subject, a predicate, and a value, plus the "
        "unit, time reference, scope, and any qualifiers exactly as given."
    )
    lines.append(
        "5. Cite evidence by listing the evidence IDs shown in brackets "
        "below. Cite ONLY IDs from this list; citing any other identifier "
        "is an error. Cite the most specific unit (cell over table)."
    )
    lines.append(
        "6. Score confidence between 0 and 1 for how strongly the cited "
        "evidence supports the draft."
    )
    lines.append(
        "7. Flag ambiguity: set ambiguous to true and explain in notes "
        "whenever the evidence is unclear, conflicting, or admits "
        "multiple readings."
    )
    lines.append(
        "8. Prefer fewer highly reliable facts over many speculative facts. "
        "If a chunk is ambiguous, omit the fact."
    )
    lines.append("")
    lines.append(
        f"Context (not citable): document {chunk.document_id}, "
        f"pdf page {chunk.pdf_page_number}, "
        f"source page {chunk.source_page_number if chunk.source_page_number is not None else '?'}."
    )
    lines.append(
        "The document and page identifiers above are context only and "
        "must never appear in evidence_ids."
    )
    lines.append("")
    lines.append("Evidence:")
    for unit in chunk.units:
        lines.append(_render_unit(unit))
    lines.append("")
    lines.append("Return exactly one JSON object with this shape:")
    lines.append('{"drafts": [...]}')
    lines.append("Each entry of drafts has this shape:")
    lines.append("{")
    lines.append('  "subject": "who or what the claim is about",')
    lines.append('  "predicate": "the relation or attribute claimed",')
    lines.append('  "value_text": "the value exactly as written in the evidence",')
    lines.append('  "value_kind": "NUMERIC or PERCENTAGE or TEXT or null",')
    lines.append('  "unit": "unit as written, or null",')
    lines.append('  "time_text": "time reference as written, or null",')
    lines.append('  "scope_text": "scope as written, or null",')
    lines.append(
        '  "estimate_status": '
        '"ACTUAL or ESTIMATE or FORECAST or PRELIMINARY or REVISED or UNKNOWN or null",'
    )
    lines.append('  "geography": "place as written, or null",')
    lines.append('  "context": {},')
    lines.append('  "evidence_ids": ["one or more IDs from the list above"],')
    lines.append('  "confidence": 0.0,')
    lines.append('  "ambiguous": false,')
    lines.append('  "notes": ""')
    lines.append("}")
    lines.append(
        "Return only the JSON object, with no surrounding commentary, "
        "no explanations, no reasoning traces, and no markdown fences."
    )
    return "\n".join(lines) + "\n"
