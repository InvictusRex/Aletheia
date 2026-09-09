"""Candidate discovery + relationship reasoning orchestration.

Staged pipeline: ensure embeddings → cross-document top-k discovery
(vector, lexical fallback) → deterministic classify → conditional LLM
judgment → persist. Fail closed everywhere; one bad candidate never
kills the run; similarity ranks only and never classifies.
"""

import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.repositories import (
    get_document_bundle,
    get_embeddings,
    list_all_facts,
    relationship_exists,
    save_embedding,
    save_relationship,
)
from app.llm.judgment import GroqJudgmentTransport, judge_pair
from app.llm.provider import LLMError
from app.llm.rate_limit import get_shared_limiter
from app.matching.compare import classify, compare_context, compare_numeric
from app.matching.embeddings import (
    FactEmbedder,
    build_embedding_text,
    lexical_score,
    rank_candidates,
)
from app.models import (
    EvidenceUnit,
    Fact,
    Relationship,
    RelationshipStatus,
    RelationshipType,
)

logger = logging.getLogger(__name__)


class MatchingReport(BaseModel):
    document_id: str
    relationships: list[Relationship] = Field(default_factory=list)
    candidates_evaluated: int = 0
    unrelated_count: int = 0
    skipped_existing: int = 0
    errors: list[str] = Field(default_factory=list)


def get_fact_embedder() -> FactEmbedder | None:
    """Resolve the embedding engine, or None when unavailable.

    Missing engine degrades discovery to the lexical path; matching
    never fails merely because embeddings are unavailable.
    """
    embedder = FactEmbedder(model_name=settings.matching_embedding_model)
    available, reason = FactEmbedder.available()
    if not available:
        logger.info("embeddings unavailable, using lexical discovery: %s", reason)
        return None
    return embedder


def get_judgment_transport() -> GroqJudgmentTransport | None:
    """Resolve the LLM judgment transport, or None without credentials."""
    if settings.llm_provider != "groq":
        logger.warning(
            "unknown LLM provider %r: relationship judgment disabled",
            settings.llm_provider,
        )
        return None
    if not settings.groq_api_key:
        logger.info("no GROQ_API_KEY: relationship judgment unavailable")
        return None
    return GroqJudgmentTransport(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        timeout_s=settings.fact_llm_timeout_s,
        max_retries=settings.groq_max_retries,
        limiter=get_shared_limiter(),
        backoff_base_s=settings.groq_backoff_base_s,
        backoff_max_s=settings.groq_backoff_max_s,
    )


def _ordered(a: UUID, b: UUID) -> tuple[UUID, UUID]:
    return (a, b) if str(a) < str(b) else (b, a)


def ensure_embeddings(
    session: Session, facts: list[Fact], embedder: FactEmbedder | None
) -> dict[UUID, list[float]]:
    """Load stored vectors, embedding and persisting any missing ones."""
    ids = [f.id for f in facts]
    vectors = get_embeddings(session, ids)
    missing = [f for f in facts if f.id not in vectors]
    if not missing or embedder is None:
        return vectors
    try:
        fresh = embedder.embed([build_embedding_text(f) for f in missing])
    except Exception as exc:
        logger.warning("embedding failed, falling back to lexical: %s", exc)
        return vectors
    for fact, vector in zip(missing, fresh):
        save_embedding(session, fact.id, vector)
        vectors[fact.id] = vector
    return vectors


def discover_candidates(
    doc_facts: list[Fact],
    pool: list[Fact],
    vectors: dict[UUID, list[float]],
    top_k: int,
    floor: float,
) -> list[tuple[UUID, UUID, float]]:
    """Cross-document top-k candidate pairs with deterministic order.

    Vector ranking when both sides have vectors, else lexical scoring.
    Unknown fields never eliminate candidates; the floor and top-k bound
    the comparison space (never all-pairs).
    """
    by_id = {f.id: f for f in pool}
    doc_ids = {f.id for f in doc_facts}
    vector_mode = all(f.id in vectors for f in pool)
    pairs: list[tuple[UUID, UUID, float]] = []
    for query in doc_facts:
        scored: list[tuple[UUID, float]] = []
        for cand in pool:
            if cand.id in doc_ids or cand.document_id == query.document_id:
                continue
            if vector_mode:
                scored.append((cand.id, None))  # placeholder, ranked below
            else:
                scored.append((cand.id, lexical_score(query, cand)))
        if vector_mode:
            pool_vecs = [
                (fid, vectors[fid]) for fid, _ in scored if fid in vectors
            ]
            ranked = rank_candidates(
                query.id, vectors[query.id], pool_vecs, top_k, floor
            )
        else:
            scored.sort(key=lambda item: (-item[1], str(item[0])))
            ranked = [(fid, s) for fid, s in scored if s >= floor][:top_k]
        for fid, score in ranked:
            a, b = _ordered(query.id, fid)
            pairs.append((a, b, score))
    # Deduplicate (same pair reachable from either side only in
    # multi-document runs) keeping the highest score, deterministic order.
    best: dict[tuple[UUID, UUID], float] = {}
    for a, b, score in pairs:
        if (a, b) not in best or score > best[(a, b)]:
            best[(a, b)] = score
    return sorted(
        [(a, b, s) for (a, b), s in best.items()],
        key=lambda item: (str(item[0]), str(item[1])),
    )


def _evidence_texts(
    bundle_evidence: list[EvidenceUnit], fact: Fact, limit: int = 4
) -> list[str]:
    by_id = {e.id: e for e in bundle_evidence}
    return [
        by_id[eid].text[:500] for eid in fact.evidence_ids if eid in by_id
    ][:limit]


def run_matching_for_document(
    session: Session,
    document_id: str,
    embedder: FactEmbedder | None = None,
    transport: GroqJudgmentTransport | None = None,
) -> MatchingReport:
    """Discover candidates for one document's facts and classify them."""
    pool = list_all_facts(session)
    doc_facts = [f for f in pool if str(f.document_id) == str(document_id)]
    report = MatchingReport(document_id=str(document_id))
    if not doc_facts:
        return report

    active_embedder = embedder if embedder is not None else get_fact_embedder()
    vectors = ensure_embeddings(session, pool, active_embedder)
    candidates = discover_candidates(
        doc_facts,
        pool,
        vectors,
        settings.matching_top_k,
        settings.matching_similarity_floor,
    )
    by_id = {f.id: f for f in pool}
    bundles: dict[str, list[EvidenceUnit]] = {}

    def evidence_for(fact: Fact) -> list[str]:
        key = str(fact.document_id)
        if key not in bundles:
            bundle = get_document_bundle(session, fact.document_id)
            bundles[key] = bundle[2] if bundle else []
        return _evidence_texts(bundles[key], fact)

    for a_id, b_id, _ in candidates:
        try:
            if relationship_exists(session, a_id, b_id):
                report.skipped_existing += 1
                continue
            fact_a, fact_b = by_id[a_id], by_id[b_id]
            rtype, confidence, explanation, metadata, needs_llm = classify(
                fact_a, fact_b, settings.matching_numeric_tolerance
            )
            report.candidates_evaluated += 1
            if rtype == RelationshipType.UNRELATED:
                report.unrelated_count += 1
                continue
            status = RelationshipStatus.RESOLVED
            if needs_llm:
                active_transport = (
                    transport if transport is not None
                    else get_judgment_transport()
                )
                if active_transport is None:
                    status = RelationshipStatus.NEEDS_REVIEW
                    confidence = min(confidence, 0.5)
                    metadata["needs_review_reason"] = "llm_unavailable"
                else:
                    try:
                        judgment = judge_pair(
                            active_transport,
                            fact_a,
                            fact_b,
                            compare_numeric(
                                fact_a, fact_b, settings.matching_numeric_tolerance
                            ),
                            compare_context(fact_a, fact_b),
                            evidence_for(fact_a),
                            evidence_for(fact_b),
                        )
                    except LLMError as exc:
                        status = RelationshipStatus.NEEDS_REVIEW
                        confidence = min(confidence, 0.5)
                        metadata["needs_review_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        rtype = judgment.relationship_type
                        confidence = judgment.confidence
                        explanation = judgment.explanation
                        metadata["llm_model"] = settings.groq_model
            rel = Relationship(
                fact_a_id=a_id,
                fact_b_id=b_id,
                relationship_type=rtype,
                confidence=confidence,
                explanation=explanation,
                reasoning_metadata=metadata,
                status=status,
            )
            save_relationship(session, rel)
            report.relationships.append(rel)
        except Exception as exc:
            report.errors.append(f"pair {a_id}/{b_id}: {type(exc).__name__}: {exc}")
            logger.warning("matching failed for pair %s/%s: %s", a_id, b_id, exc)
            continue
    return report
