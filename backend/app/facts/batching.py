"""Token-aware batching of semantic chunks into bounded LLM requests.

Sits between page-scoped chunking (``app.facts.chunking``) and the LLM
provider: accumulates whole chunks until another would exceed
``max_input_tokens``. Never splits a chunk (chunking owns safe splits),
never drops content (an oversize chunk travels alone as overflow),
never reorders units, never rewrites evidence IDs.

Token sizes are estimated from the exact rendered prompt text, so the
budget accounts for instructions plus evidence.
"""

from __future__ import annotations

from app.facts.prompts import build_extraction_prompt
from app.llm.rate_limit import estimate_tokens
from app.models import EvidenceChunk

__all__ = ["batch_chunks", "merge_chunks"]


def merge_chunks(chunks: list[EvidenceChunk]) -> EvidenceChunk:
    """Merge batches into one chunk for prompting/validation.

    Units concatenate in order; the merged chunk carries the first
    chunk's page numbers (used only for log/error attribution) and the
    union of dropped IDs. Validators check membership against the full
    unit set, so provenance is preserved exactly.
    """
    first = chunks[0]
    units = [unit for chunk in chunks for unit in chunk.units]
    dropped = [eid for chunk in chunks for eid in chunk.dropped_evidence_ids]
    return EvidenceChunk(
        document_id=first.document_id,
        pdf_page_number=first.pdf_page_number,
        source_page_number=first.source_page_number,
        units=units,
        dropped_evidence_ids=dropped,
    )


def batch_chunks(
    chunks: list[EvidenceChunk], max_input_tokens: int
) -> list[list[EvidenceChunk]]:
    """Group chunks into request-sized batches by estimated prompt tokens.

    Deterministic and order-preserving. A chunk whose own rendered
    prompt exceeds the budget goes alone (atomic overflow); empty input
    yields no batches. Raises ``ValueError`` for non-positive budgets.
    """
    if max_input_tokens <= 0:
        raise ValueError(f"max_input_tokens must be > 0, got {max_input_tokens!r}")
    batches: list[list[EvidenceChunk]] = []
    current: list[EvidenceChunk] = []
    current_tokens = 0
    for chunk in chunks:
        cost = estimate_tokens(build_extraction_prompt(chunk))
        if current and current_tokens + cost > max_input_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(chunk)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches
