import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "frontend_scope",
    Path(__file__).resolve().parents[2] / "frontend" / "scope.py",
)
scope = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scope)


def test_prune_selection_drops_unknown_and_dupes():
    assert scope.prune_selection(["b", "a", "b", "zzz"], ["a", "b"]) == ["b", "a"]
    assert scope.prune_selection([], ["a"]) == []


def test_filter_facts_by_scope():
    facts = [
        {"id": "1", "document_id": "a"},
        {"id": "2", "document_id": "b"},
        {"id": "3", "document_id": "c"},
    ]
    kept = scope.filter_facts_by_scope(facts, ["a", "c"])
    assert [f["id"] for f in kept] == ["1", "3"]


def _detail(a_doc, b_doc):
    return {
        "fact_a": {"id": "fa", "document_id": a_doc},
        "fact_b": {"id": "fb", "document_id": b_doc},
    }


def test_relationship_scope_requires_both_sides():
    both = _detail("a", "b")
    assert scope.relationship_in_scope(both, ["a", "b"]) is True
    assert scope.relationship_in_scope(both, ["a"]) is False
    assert scope.relationship_in_scope(both, ["a", "c"]) is False
    same = _detail("a", "a")
    assert scope.relationship_in_scope(same, ["a"]) is True
    assert scope.relationship_in_scope(same, ["b"]) is False
    assert scope.relationship_in_scope({}, ["a"]) is False
    assert scope.relationship_in_scope(
        {"fact_a": {"id": "x"}, "fact_b": {"id": "y"}}, ["x", "y"]
    ) is False


def test_filter_relationships_by_scope():
    details = [_detail("a", "b"), _detail("a", "c"), _detail("b", "c")]
    kept = scope.filter_relationships_by_scope(details, ["a", "b"])
    assert kept == [_detail("a", "b")]


def test_selection_label():
    assert scope.selection_label(0) == "0 documents selected"
    assert scope.selection_label(1) == "1 document selected"
    assert scope.selection_label(3) == "3 documents selected"


def test_rel_scope_key_stable_and_order_free():
    assert scope.rel_scope_key(["b", "a", "b"]) == scope.rel_scope_key(["a", "b"])
    assert scope.rel_scope_key([]) == ""


def _view_rel(rel_id, rtype, conf, fact_a="fa", fact_b="fb"):
    return {"id": rel_id, "relationship_type": rtype, "confidence": conf,
            "fact_a_id": fact_a, "fact_b_id": fact_b}


def test_rels_in_scope_requires_both_fact_ids():
    rels = [
        _view_rel("1", "CORROBORATES", 0.9, "fa", "fb"),
        _view_rel("2", "CONTRADICTS", 0.7, "fa", "fx"),
        _view_rel("3", "CORROBORATES", 0.4, "fx", "fy"),
    ]
    assert [r["id"] for r in scope.rels_in_scope(rels, {"fa", "fb"})] == ["1"]
    assert scope.rels_in_scope(rels, set()) == []
    assert len(scope.rels_in_scope(rels, {"fa", "fb", "fx", "fy"})) == 3


def test_apply_relationship_view_filters_and_sorts():
    rels = [
        _view_rel("1", "CORROBORATES", 0.9),
        _view_rel("2", "CONTRADICTS", 0.7),
        _view_rel("3", "CORROBORATES", 0.4),
    ]
    assert [r["id"] for r in scope.apply_relationship_view(rels, None, 0.0)] == ["1", "2", "3"]
    assert [r["id"] for r in scope.apply_relationship_view(rels, "CONTRADICTS", 0.0)] == ["2"]
    assert [r["id"] for r in scope.apply_relationship_view(rels, None, 0.8)] == ["1"]
    assert scope.apply_relationship_view(rels, "RELATED", 0.0) == []
    assert scope.apply_relationship_view(rels, None, 2.0) == []


def test_normalization_summary_counts_normalized_facts():
    facts = [
        {"id": "1", "normalized_number": 1.5},
        {"id": "2", "canonical_subject": "Revenue"},
        {"id": "3", "subject": "X"},
        {"id": "4", "normalized_unit": "INR crore"},
        {"id": "5", "canonical_predicate": "total"},
    ]
    assert scope.normalization_summary(facts) == {"total": 5, "normalized": 4}
    assert scope.normalization_summary([]) == {"total": 0, "normalized": 0}



def test_normalize_blocked_without_documents_or_facts():
    assert scope.normalize_blocked(False, 12) is True
    assert scope.normalize_blocked(True, 0) is True
    assert scope.normalize_blocked(True, 12) is False
    assert scope.normalize_blocked(True, None) is False


def test_unnormalized_documents_flags_only_fact_bearing_documents():
    stats = [
        {"name": "a.pdf", "facts": 1178, "normalized": 1178},
        {"name": "b.pdf", "facts": 1155, "normalized": 0},
        {"name": "c.pdf", "facts": 0, "normalized": 0},
    ]
    assert scope.unnormalized_documents(stats) == ["b.pdf"]


def test_unnormalized_documents_ignores_unknown_counts():
    stats = [
        {"name": "a.pdf", "facts": None, "normalized": None},
        {"name": "b.pdf", "facts": 10, "normalized": None},
    ]
    assert scope.unnormalized_documents(stats) == []


def test_one_normalized_document_does_not_mask_an_unnormalized_one():
    stats = [
        {"name": "normalized.pdf", "facts": 1178, "normalized": 1178},
        {"name": "raw.pdf", "facts": 1155, "normalized": 0},
    ]
    pending = scope.unnormalized_documents(stats)
    assert scope.relationships_blocked(True, 2333, pending) is True
    assert "raw.pdf" in (scope.relationship_block_reason(True, 2333, pending) or "")


def test_relationships_unblocked_once_every_document_is_normalized():
    stats = [{"name": "a.pdf", "facts": 10, "normalized": 10}]
    pending = scope.unnormalized_documents(stats)
    assert pending == []
    assert scope.relationships_blocked(True, 10, pending) is False
    assert scope.relationship_block_reason(True, 10, pending) is None


def test_relationships_blocked_inherits_normalize_preconditions():
    assert scope.relationships_blocked(False, 12, []) is True
    assert scope.relationships_blocked(True, 0, []) is True


def test_relationship_block_reason_names_the_missing_step():
    assert scope.relationship_block_reason(True, 0, []) == (
        "No extracted facts in scope yet."
    )
    assert scope.relationship_block_reason(False, 0, []) is None
    reason = scope.relationship_block_reason(True, 12, ["x.pdf", "y.pdf"])
    assert "have facts but none" in reason
    single = scope.relationship_block_reason(True, 12, ["x.pdf"])
    assert "has facts but none" in single


def test_comparability_label_names_what_is_missing():
    assert scope.comparability_label(
        {"comparability": "COMPARABLE", "missing_for_comparison": []}
    ) == ("COMPARABLE", "badge-grn")
    label, _cls = scope.comparability_label(
        {"comparability": "PARTIAL", "missing_for_comparison": ["period"]}
    )
    assert "period" in label
    label, _cls = scope.comparability_label(
        {"comparability": "NOT_COMPARABLE", "missing_for_comparison": ["unit", "period"]}
    )
    assert "unit" in label and "period" in label


def test_comparability_label_is_blank_when_absent():
    assert scope.comparability_label({}) == ("", "")


def test_comparability_summary_counts_each_verdict():
    facts = [
        {"comparability": "COMPARABLE"},
        {"comparability": "COMPARABLE"},
        {"comparability": "PARTIAL"},
        {},
    ]
    assert scope.comparability_summary(facts) == {
        "COMPARABLE": 2, "PARTIAL": 1, "NOT_COMPARABLE": 0
    }


def test_withheld_note_explains_a_refused_verdict():
    note = scope.withheld_note({"reasoning_metadata": {
        "withheld_verdict": "CONTRADICTS",
        "withheld_reason": "scope stated on only one side",
    }})
    assert note == "CONTRADICTS withheld — scope stated on only one side"


def test_withheld_note_is_none_for_an_ordinary_relationship():
    assert scope.withheld_note({"reasoning_metadata": {}}) is None
    assert scope.withheld_note({}) is None
