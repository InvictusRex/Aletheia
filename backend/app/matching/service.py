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
    replace_relationship,
    save_embedding,
    save_relationship,
)
from app.llm.judgment import GroqJudgmentTransport, judge_pair
from app.llm.ollama import OllamaJudgmentTransport
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
    recomputed: int = 0
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


def get_judgment_transport() -> GroqJudgmentTransport | OllamaJudgmentTransport | None:
    """Resolve the LLM judgment transport, or None without credentials.

    ``LLM_PROVIDER=groq`` needs ``GROQ_API_KEY``; ``LLM_PROVIDER=ollama``
    talks to ``OLLAMA_BASE_URL`` (no key). Anything else is a clear
    configuration error. A missing Groq key degrades gracefully to
    ``None`` (pairs stay ``NEEDS_REVIEW``) without affecting matching.
    """
    if settings.llm_provider == "groq":
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
    if settings.llm_provider == "ollama":
        return OllamaJudgmentTransport(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_s=settings.ollama_timeout_s,
            max_retries=settings.ollama_max_retries,
            think=settings.ollama_think,
            backoff_base_s=settings.ollama_backoff_base_s,
            backoff_max_s=settings.ollama_backoff_max_s,
            keep_alive=settings.ollama_keep_alive,
            context_tokens=settings.llm_context_tokens,
        )
    raise ValueError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r}: "
        "expected 'groq' or 'ollama'"
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
    exhaustive_max_pairs: int | None = None,
    include_same_document: bool = False,
) -> list[tuple[UUID, UUID, float]]:
    """Cross-document candidate pairs with deterministic order.

    Ranking exists to bound the comparison space, but it also hides
    valid pairs: a correct partner that falls outside a fact's top-k is
    never compared at all. When the whole cross-document space is
    smaller than ``exhaustive_max_pairs`` this returns every pair and
    skips ranking, so recall is complete and top-k degrades to a pure
    scale guard. Similarity still never classifies anything.

    ``include_same_document`` also pairs facts drawn from one document.
    Off by default, because a single document restates its own figures
    constantly (a table and the prose describing it), but a document
    that contradicts ITSELF is invisible without it.
    """
    doc_ids = {f.id for f in doc_facts}

    def eligible(query: Fact, cand: Fact) -> bool:
        if cand.id == query.id:
            return False
        if include_same_document:
            # Same-document pairs are allowed, but each unordered pair
            # must still be offered once.
            return str(cand.id) > str(query.id) or cand.id not in doc_ids
        return cand.id not in doc_ids and cand.document_id != query.document_id
    candidate_space = sum(
        1 for query in doc_facts for cand in pool if eligible(query, cand)
    )
    if exhaustive_max_pairs is not None and candidate_space <= exhaustive_max_pairs:
        every: set[tuple[UUID, UUID]] = set()
        for query in doc_facts:
            for cand in pool:
                if not eligible(query, cand):
                    continue
                every.add(_ordered(query.id, cand.id))
        # Score 1.0 for every pair: nothing was ranked, so no pair is
        # preferred over another and the score carries no meaning here.
        return [
            (a, b, 1.0)
            for a, b in sorted(every, key=lambda p: (str(p[0]), str(p[1])))
        ]

    pairs: list[tuple[UUID, UUID, float]] = []
    for query in doc_facts:
        eligible_ids = [cand.id for cand in pool if eligible(query, cand)]
        # Vector ranking is decided PER QUERY, not for the whole run: a
        # single fact without an embedding used to drop every comparison
        # to the lexical path, discarding vectors that were available
        # for everything else.
        query_vec = vectors.get(query.id)
        pool_vecs = [
            (fid, vectors[fid]) for fid in eligible_ids if fid in vectors
        ]
        if query_vec is not None and pool_vecs:
            ranked = rank_candidates(query.id, query_vec, pool_vecs, top_k, floor)
            ranked_ids = {fid for fid, _ in ranked}
            # Anything without a vector still gets a lexical chance
            # rather than being silently unreachable.
            by_id = {f.id: f for f in pool}
            leftover = [
                (fid, lexical_score(query, by_id[fid]))
                for fid in eligible_ids
                if fid not in vectors and fid not in ranked_ids
            ]
            leftover.sort(key=lambda item: (-item[1], str(item[0])))
            ranked = ranked + [(f, s) for f, s in leftover if s >= floor][
                : max(0, top_k - len(ranked))
            ]
        else:
            by_id = {f.id: f for f in pool}
            scored = [(fid, lexical_score(query, by_id[fid])) for fid in eligible_ids]
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
    transport: GroqJudgmentTransport | OllamaJudgmentTransport | None = None,
    compare_with: list[UUID] | None = None,
    recompute: bool = False,
    include_same_document: bool = False,
) -> MatchingReport:
    """Discover candidates for one document's facts and classify them.

    ``compare_with`` bounds the candidate side to those document ids
    (the query document is always included). ``None`` compares against
    every other document, preserving the unbounded default.

    ``recompute`` re-classifies pairs that are already stored and
    overwrites their verdict in place. It is off by default: an ordinary
    run never silently rewrites a stored judgment, but a stored verdict
    computed from since-corrected facts would otherwise be permanent.
    """
    pool = list_all_facts(session)
    if compare_with is not None:
        allowed = {str(d) for d in compare_with} | {str(document_id)}
        pool = [f for f in pool if str(f.document_id) in allowed]
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
        settings.matching_exhaustive_max_pairs,
        include_same_document,
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
            already_stored = relationship_exists(session, a_id, b_id)
            if already_stored and not recompute:
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
                        withheld = metadata.get("withheld_verdict")
                        if withheld and judgment.relationship_type.value == withheld:
                            # The deterministic layer withheld this verdict
                            # because a precondition was not met (e.g. neither
                            # side states a period). The judge is handed the
                            # pair as given and is not asked to establish that
                            # precondition, so it may refine the typing but
                            # must not reinstate the very verdict that was
                            # withheld for lack of evidence.
                            metadata["llm_overruled"] = withheld
                            logger.info(
                                "judgment proposed the withheld verdict %s for %s/%s; keeping %s",
                                withheld, a_id, b_id, rtype.value,
                            )
                        else:
                            rtype = judgment.relationship_type
                        if metadata.get("match_tier") == "weak":
                            # On the weak tier the open question is whether
                            # these are even the same claim, and the judge
                            # is not asked that -- it is handed the pair as
                            # given. So it may pick the type and argue it,
                            # but it cannot upgrade an inexact match into a
                            # confident verdict. Other tiers already
                            # established the match deterministically, so
                            # the judge's own confidence stands there.
                            confidence = min(judgment.confidence, confidence)
                        else:
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
            if already_stored:
                replace_relationship(session, rel)
                report.recomputed += 1
            else:
                save_relationship(session, rel)
            report.relationships.append(rel)
        except Exception as exc:
            report.errors.append(f"pair {a_id}/{b_id}: {type(exc).__name__}: {exc}")
            logger.warning("matching failed for pair %s/%s: %s", a_id, b_id, exc)
            continue
    return report
