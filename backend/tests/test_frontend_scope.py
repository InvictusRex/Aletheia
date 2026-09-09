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
