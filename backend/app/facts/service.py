"""Fact extraction orchestration: chunks → LLM → validated facts.

The service wires chunking, prompting, the LLM provider, and
validation. It never fabricates: provider failures mark chunks failed
with recorded errors, and validator rejections are counted, never stored.

Resumability: each completed chunk persists its facts AND a progress
row (``extraction_chunks``) in the same per-chunk commit, so rerunning
extraction skips already-completed chunks instead of repeating Groq
calls or duplicating facts. A failed chunk never invalidates completed
ones. One chunk per request, sequentially — no batching, no concurrency.
"""

import hashlib
import logging

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.repositories import (
    get_chunk_statuses,
    get_document_bundle,
    get_fact,
    list_facts_for_document,
    save_facts,
    upsert_chunk_status,
)
from app.facts.chunking import build_chunks, select_representative_chunks
from app.facts.prompts import build_extraction_prompt
from app.facts.validator import build_fact
from app.llm.groq import GroqFactsProvider
from app.llm.provider import LLMError, LLMProvider
from app.llm.rate_limit import get_shared_limiter
from app.models import EvidenceChunk, Fact

logger = logging.getLogger(__name__)

#: Progress states stored in ``extraction_chunks.status``.
CHUNK_COMPLETED = "COMPLETED"
CHUNK_FAILED = "FAILED"


class FactExtractionReport(BaseModel):
    document_id: str
    facts: list[Fact] = Field(default_factory=list)
    chunks_processed: int = 0
    chunks_failed: int = 0
    chunks_skipped: int = 0
    chunks_capped: int = 0
    drafts_rejected: int = 0
    facts_skipped_duplicate: int = 0
    errors: list[str] = Field(default_factory=list)


def get_llm_provider() -> LLMProvider | None:
    """Resolve the configured LLM provider, or None when unavailable.

    Extraction stays optional: missing keys, unknown providers, or a
    missing SDK disable the layer without affecting ingestion.
    """
    if settings.llm_provider != "groq":
        logger.warning(
            "unknown LLM provider %r: fact extraction disabled",
            settings.llm_provider,
        )
        return None
    if not settings.groq_api_key:
        logger.info("no GROQ_API_KEY: fact extraction unavailable")
        return None
    return GroqFactsProvider(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        timeout_s=settings.fact_llm_timeout_s,
        max_retries=settings.groq_max_retries,
        limiter=get_shared_limiter(),
        backoff_base_s=settings.groq_backoff_base_s,
        backoff_max_s=settings.groq_backoff_max_s,
    )


def chunk_evidence_hash(chunk: EvidenceChunk) -> str:
    """Stable identity hash for a chunk's evidence set (pure function).

    Guards resume-skips against stale progress rows when the chunking
    bound changed between runs: same (page, index) but different content
    is reprocessed rather than skipped.
    """
    digest = hashlib.sha256()
    for eid in sorted(str(u.evidence_id) for u in chunk.units):
        digest.update(eid.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def _fact_identity(fact: Fact) -> tuple:
    """Semantic identity of a fact for duplicate suppression.

    Excludes the random row id (and confidence): temperature-0
    re-extraction of the same chunk yields identical drafts, so a
    rerun over already-persisted facts dedupes exactly instead of
    inserting duplicates. This also protects facts persisted before
    chunk-progress rows existed.
    """
    return (
        fact.subject,
        fact.predicate,
        fact.value_kind,
        fact.value_text,
        fact.unit,
        fact.time_text,
        fact.scope_text,
        fact.estimate_status,
        fact.geography,
        tuple(sorted(str(eid) for eid in fact.evidence_ids)),
    )


def _validate_page_range(
    start_page: int | None, end_page: int | None
) -> None:
    """Reject nonsense page windows before any Groq call (pure validation)."""
    if start_page is not None and start_page < 0:
        raise ValueError(f"start_page must be >= 0, got {start_page!r}")
    if end_page is not None and end_page < 0:
        raise ValueError(f"end_page must be >= 0, got {end_page!r}")
    if (
        start_page is not None
        and end_page is not None
        and start_page > end_page
    ):
        raise ValueError(
            f"start_page ({start_page}) must be <= end_page ({end_page})"
        )


def extract_facts_for_document(
    session: Session,
    document_id: str,
    provider: LLMProvider | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
) -> FactExtractionReport:
    """Run fact extraction over a persisted document's evidence.

    Each completed chunk commits its facts plus its progress row, so an
    interrupted run resumes by skipping ``COMPLETED`` chunks. Optional
    ``start_page``/``end_page`` (0-based ``pdf_page_number``, inclusive)
    bound the run to a page window; ``None`` means unbounded and
    preserves the whole-document default. Chunk ordinals are assigned
    per page over the FULL document chunking so scoped and full runs
    share stable chunk identities.
    """
    _validate_page_range(start_page, end_page)
    bundle = get_document_bundle(session, document_id)
    if bundle is None:
        raise KeyError(f"document not found: {document_id}")
    document, pages, evidence = bundle

    active = provider if provider is not None else get_llm_provider()
    report = FactExtractionReport(document_id=str(document.id))
    if active is None:
        report.errors.append("LLM provider unavailable: fact extraction skipped")
        return report

    chunks = build_chunks(
        document.id, pages, evidence, max_chars=settings.fact_chunk_max_chars
    )
    # Ordinals per page over the FULL chunking (before window filtering)
    # so scoped and whole-document runs share chunk identities.
    seen_per_page: dict[int, int] = {}
    indexed: list[tuple[EvidenceChunk, int]] = []
    for chunk in chunks:
        idx = seen_per_page.get(chunk.pdf_page_number, 0)
        seen_per_page[chunk.pdf_page_number] = idx + 1
        indexed.append((chunk, idx))
    if start_page is not None:
        indexed = [(c, i) for c, i in indexed if c.pdf_page_number >= start_page]
    if end_page is not None:
        indexed = [(c, i) for c, i in indexed if c.pdf_page_number <= end_page]

    # Operability bound: exhaustive LLM extraction is not viable on the
    # free TPM tier, so each run sends at most a representative subset
    # (distributed across the run's chunks, tables/numeric preferred).
    # Selection runs over the full eligible set BEFORE resume-skipping,
    # so reruns deterministically select the same chunks and skip the
    # completed ones instead of drifting to new chunks.
    max_chunks = settings.fact_max_chunks_per_document
    if max_chunks < 1:
        raise ValueError(
            "fact_max_chunks_per_document must be >= 1, "
            f"got {max_chunks!r}"
        )
    selected = select_representative_chunks(
        [chunk for chunk, _ in indexed], max_chunks
    )
    selected_ids = {id(chunk) for chunk in selected}
    report.chunks_capped = len(indexed) - len(selected)
    indexed = [(c, i) for c, i in indexed if id(c) in selected_ids]

    # One chunk per request, sequentially: no cross-chunk batching, so
    # table/row boundaries and evidence identity can never be disturbed
    # by request packing. Request size stays small because chunking
    # already bounds chunks; pacing and retries live in the provider.
    statuses = get_chunk_statuses(session, document.id)
    existing_by_identity: dict[tuple, Fact] = {}
    for existing in list_facts_for_document(session, document.id):
        existing_by_identity.setdefault(_fact_identity(existing), existing)
    for chunk, chunk_index in indexed:
        key = (chunk.pdf_page_number, chunk_index)
        digest = chunk_evidence_hash(chunk)
        prior = statuses.get(key)
        if (
            prior is not None
            and prior.status == CHUNK_COMPLETED
            and (prior.evidence_hash or "") == digest
        ):
            # Resume-skip: no Groq call, no new rows; rehydrate the
            # report from the persisted fact ids.
            report.chunks_skipped += 1
            for raw_fid in prior.fact_ids or []:
                try:
                    from uuid import UUID as _UUID

                    fact = get_fact(session, _UUID(str(raw_fid)))
                except (ValueError, AttributeError):
                    continue
                if fact is not None:
                    report.facts.append(fact)
            continue
        try:
            result = active.extract_facts(build_extraction_prompt(chunk))
        except LLMError as exc:
            session.rollback()
            upsert_chunk_status(
                session,
                document.id,
                chunk.pdf_page_number,
                chunk_index,
                CHUNK_FAILED,
                f"{type(exc).__name__}: {exc}",
                [],
                digest,
            )
            session.commit()
            report.chunks_failed += 1
            report.errors.append(
                f"page {chunk.pdf_page_number}: {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "fact extraction failed for page %d of %s: %s",
                chunk.pdf_page_number, document.filename, exc,
            )
            continue
        report.chunks_processed += 1
        fresh: list[Fact] = []
        matched: list[Fact] = []
        for draft in result.drafts:
            fact, reason = build_fact(draft, document.id, chunk)
            if fact is None:
                report.drafts_rejected += 1
                report.errors.append(
                    f"page {chunk.pdf_page_number}: rejected draft: {reason}"
                )
                continue
            identity = _fact_identity(fact)
            if identity in existing_by_identity:
                # Already persisted (legacy row or earlier chunk/run):
                # keep the link, never insert a duplicate.
                report.facts_skipped_duplicate += 1
                matched.append(existing_by_identity[identity])
                continue
            existing_by_identity[identity] = fact
            fresh.append(fact)
        if fresh:
            save_facts(session, fresh)
        upsert_chunk_status(
            session,
            document.id,
            chunk.pdf_page_number,
            chunk_index,
            CHUNK_COMPLETED,
            None,
            [f.id for f in fresh] + [f.id for f in matched],
            digest,
        )
        session.commit()
        report.facts.extend(fresh)
        report.facts.extend(matched)

    return report
