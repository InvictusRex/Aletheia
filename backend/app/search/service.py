"""Knowledge-layer retrieval: user query → ranked persisted facts.

Retrieval-only. This module never calls an LLM, never mutates stored
rows, never creates embeddings as a side effect, and never generates
natural-language answers. It returns persisted objects with provenance.

Score semantics (all in [0, 1]):
- lexical-only hit: token-Jaccard between query tokens and the fact's
  subject/predicate/value/unit tokens;
- vector-only hit: max(0, 1 - cosine_distance) between the query vector
  and the fact's stored embedding;
- hybrid hit (both): arithmetic mean of the two;
- ``min_score`` filters on the final score; hits require score > 0;
- deterministic order: score descending, then fact id string ascending.
"""

import re
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.repositories import (
    get_document_bundle,
    list_relationships_for_document,
    nearest_fact_ids_by_vector,
    search_fact_rows_by_tokens,
)
from app.matching.embeddings import FactEmbedder, build_embedding_text
from app.models import Document, EvidenceUnit, Fact, Relationship

_TOKEN_RE = re.compile(r"[a-z0-9]+")  # same convention as matching lexical scoring

_VECTOR_POOL_MULTIPLIER = 5
_LEXICAL_POOL_MULTIPLIER = 5


def tokenize_query(query: str) -> list[str]:
    """Lowercase alphanumeric tokens, order-preserving, deduplicated."""
    seen: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        if token not in seen:
            seen.append(token)
    return seen


def _fact_tokens(fact: Fact) -> set[str]:
    parts = [fact.subject, fact.predicate, fact.value_text, fact.unit or ""]
    return set(_TOKEN_RE.findall(" ".join(parts).lower()))


def _jaccard(query_tokens: set[str], fact_tokens: set[str]) -> float:
    if not query_tokens or not fact_tokens:
        return 0.0
    return len(query_tokens & fact_tokens) / len(query_tokens | fact_tokens)


def _combine_scores(
    lexical: float | None, vector: float | None
) -> tuple[float, str] | None:
    """Merge per-path scores into (final score, match kind)."""
    present = [(s, k) for s, k in ((lexical, "lexical"), (vector, "vector")) if s is not None and s > 0]
    if not present:
        return None
    if len(present) == 2:
        return ((present[0][0] + present[1][0]) / 2.0, "hybrid")
    return present[0]


def search_facts(
    session: Session,
    query: str,
    document_id: UUID | None = None,
    limit: int = 20,
    min_score: float = 0.0,
    embedder: FactEmbedder | None = None,
) -> list[tuple[Fact, float, str]]:
    """Retrieve ranked (fact, score, match_kind) hits. Read-only.

    Independent lexical + vector discovery over persisted facts, merged
    and deduplicated, bounded by limit. Never touches the LLM layer and
    never writes. ``embedder=None`` forces the lexical path (tests).
    """
    tokens = tokenize_query(query)
    pool_limit = max(limit * _LEXICAL_POOL_MULTIPLIER, 50)
    lexical_facts = (
        search_fact_rows_by_tokens(session, tokens, pool_limit, document_id)
        if tokens
        else []
    )
    by_id: dict[UUID, Fact] = {f.id: f for f in lexical_facts}
    lexical_scores = {
        f.id: _jaccard(set(tokens), _fact_tokens(f)) for f in lexical_facts
    }

    vector_scores: dict[UUID, float] = {}
    active = embedder
    if active is None:
        active = FactEmbedder()
        available, _ = FactEmbedder.available()
        if not available:
            active = None
    if active is not None and tokens:
        try:
            query_vec = active.embed([" ".join(tokens)])[0]
            for fid, distance in nearest_fact_ids_by_vector(
                session, query_vec, max(limit * _VECTOR_POOL_MULTIPLIER, 50),
                document_id,
            ):
                vector_scores[fid] = max(0.0, 1.0 - distance)
        except Exception:
            vector_scores = {}
    if vector_scores:
        extra_ids = [fid for fid in vector_scores if fid not in by_id]
        if extra_ids:
            from app.db.repositories import get_fact

            for fid in extra_ids:
                fact = get_fact(session, fid)
                if fact is not None and (
                    document_id is None or fact.document_id == document_id
                ):
                    by_id[fid] = fact

    hits: list[tuple[Fact, float, str]] = []
    for fid, fact in by_id.items():
        if document_id is not None and fact.document_id != document_id:
            continue
        merged = _combine_scores(lexical_scores.get(fid), vector_scores.get(fid))
        if merged is None:
            continue
        score, kind = merged
        if score > 0 and score >= min_score:
            hits.append((fact, score, kind))
    hits.sort(key=lambda item: (-item[1], str(item[0].id)))
    return hits[:limit]


def hydrate_hit(
    session: Session, fact: Fact
) -> tuple[Document | None, list[EvidenceUnit], list[Relationship]]:
    """Load a hit's document, evidence, and persisted relationships."""
    bundle = get_document_bundle(session, fact.document_id)
    if bundle is None:
        return None, [], []
    document, _, evidence = bundle
    by_id = {e.id: e for e in evidence}
    fact_evidence = [by_id[eid] for eid in fact.evidence_ids if eid in by_id]
    all_rels = list_relationships_for_document(session, fact.document_id)
    rels = [r for r in all_rels if fact.id in (r.fact_a_id, r.fact_b_id)]
    rels.sort(key=lambda r: str(r.id))
    return document, fact_evidence, rels
