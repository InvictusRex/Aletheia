"""Fact extraction orchestration: chunks → LLM → validated facts.

The service wires chunking, prompting, the LLM provider, and
validation. It never fabricates: provider failures skip chunks with
recorded errors, and validator rejections are counted, never stored.
"""

import logging

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.repositories import get_document_bundle, save_facts
from app.facts.batching import batch_chunks, merge_chunks
from app.facts.chunking import build_chunks
from app.facts.prompts import build_extraction_prompt
from app.facts.validator import build_fact
from app.llm.groq import GroqFactsProvider
from app.llm.provider import LLMError, LLMProvider
from app.llm.rate_limit import get_shared_limiter
from app.models import Fact

logger = logging.getLogger(__name__)


class FactExtractionReport(BaseModel):
    document_id: str
    facts: list[Fact] = Field(default_factory=list)
    chunks_processed: int = 0
    chunks_failed: int = 0
    drafts_rejected: int = 0
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
        expected_output_tokens=settings.groq_expected_output_tokens,
        backoff_base_s=settings.groq_backoff_base_s,
        backoff_max_s=settings.groq_backoff_max_s,
    )


def extract_facts_for_document(
    session: Session, document_id: str, provider: LLMProvider | None = None
) -> FactExtractionReport:
    """Run fact extraction over a persisted document's evidence."""
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
    batches = batch_chunks(
        chunks, max_input_tokens=settings.groq_max_input_tokens_per_request
    )
    logger.info(
        "fact extraction batching for %s: %d chunks in %d request batches",
        document.filename, len(chunks), len(batches),
    )
    all_facts: list[Fact] = []
    for batch_chunks_list in batches:
        batch = merge_chunks(batch_chunks_list)
        try:
            result = active.extract_facts(build_extraction_prompt(batch))
        except LLMError as exc:
            report.chunks_failed += 1
            report.errors.append(
                f"page {batch.pdf_page_number}: {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "fact extraction failed for page %d of %s: %s",
                batch.pdf_page_number, document.filename, exc,
            )
            continue
        report.chunks_processed += 1
        for draft in result.drafts:
            fact, reason = build_fact(draft, document.id, batch)
            if fact is None:
                report.drafts_rejected += 1
                report.errors.append(
                    f"page {batch.pdf_page_number}: rejected draft: {reason}"
                )
                continue
            all_facts.append(fact)

    if all_facts:
        save_facts(session, all_facts)
    report.facts = all_facts
    return report
