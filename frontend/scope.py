from __future__ import annotations


def prune_selection(selected_ids: list[str], known_ids: list[str]) -> list[str]:
    known = set(known_ids)
    ordered: list[str] = []
    for doc_id in selected_ids:
        if doc_id in known and doc_id not in ordered:
            ordered.append(doc_id)
    return ordered


def filter_facts_by_scope(facts: list[dict], selected_ids: list[str]) -> list[dict]:
    scope = set(selected_ids)
    return [f for f in facts if str(f.get("document_id")) in scope]


def relationship_in_scope(detail: dict, selected_ids: list[str]) -> bool:
    scope = set(selected_ids)
    fact_a = (detail or {}).get("fact_a", {}) or {}
    fact_b = (detail or {}).get("fact_b", {}) or {}
    doc_a = str(fact_a.get("document_id", ""))
    doc_b = str(fact_b.get("document_id", ""))
    return bool(doc_a and doc_b) and doc_a in scope and doc_b in scope


def filter_relationships_by_scope(
    details: list[dict], selected_ids: list[str]
) -> list[dict]:
    return [d for d in details if relationship_in_scope(d, selected_ids)]


def selection_label(count: int) -> str:
    if count == 1:
        return "1 document selected"
    return f"{count} documents selected"


def rel_scope_key(selected_ids: list[str]) -> str:
    return ",".join(sorted(set(str(i) for i in selected_ids)))


def rels_in_scope(rels: list[dict], scope_fact_ids: set[str]) -> list[dict]:
    return [
        rel for rel in rels
        if str(rel.get("fact_a_id", "")) in scope_fact_ids
        and str(rel.get("fact_b_id", "")) in scope_fact_ids
    ]


def apply_relationship_view(
    rels: list[dict], rel_type: str | None, min_conf: float
) -> list[dict]:
    kept = []
    for rel in rels:
        if rel_type is not None and str(rel.get("relationship_type")) != rel_type:
            continue
        try:
            conf = float(rel.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            continue
        if conf < min_conf:
            continue
        kept.append(rel)
    kept.sort(
        key=lambda rel: (
            -float(rel.get("confidence", 0) or 0),
            str(rel.get("id")),
        )
    )
    return kept


def _is_normalized_fact(fact: dict) -> bool:
    return (
        fact.get("normalized_number") is not None
        or bool(fact.get("normalized_unit"))
        or bool(fact.get("canonical_subject"))
        or bool(fact.get("canonical_predicate"))
    )


def normalization_summary(facts: list[dict]) -> dict:
    total = len(facts)
    normalized = sum(1 for f in facts if _is_normalized_fact(f))
    return {"total": total, "normalized": normalized}


def normalize_blocked(has_documents: bool, fact_total: int | None) -> bool:
    return not has_documents or fact_total == 0


def unnormalized_documents(stats: list[dict]) -> list[str]:
    """Names of in-scope documents holding facts of which none are normalized.

    Checked per document, not over the scope total: one normalized
    document must not mask an unnormalized one, because relationship
    typing compares normalized values and would silently skip every pair
    drawn from the unnormalized side. Unknown counts are not treated as
    unnormalized.
    """
    pending: list[str] = []
    for entry in stats:
        facts = entry.get("facts")
        normalized = entry.get("normalized")
        if facts is None or normalized is None:
            continue
        if facts > 0 and normalized == 0:
            pending.append(str(entry.get("name") or "—"))
    return pending


def relationships_blocked(
    has_documents: bool, fact_total: int | None, unnormalized: list[str]
) -> bool:
    if normalize_blocked(has_documents, fact_total):
        return True
    return bool(unnormalized)


def relationship_block_reason(
    has_documents: bool, fact_total: int | None, unnormalized: list[str]
) -> str | None:
    if not has_documents:
        return None
    if fact_total == 0:
        return "No extracted facts in scope yet."
    if unnormalized:
        return (
            f"Run NORMALIZE first — {', '.join(unnormalized)} "
            f"{'has' if len(unnormalized) == 1 else 'have'} facts but none "
            "normalized. Relationships compare normalized values, so those "
            "facts can never be typed."
        )
    return None
