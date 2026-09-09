import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

sys.modules.setdefault("streamlit", ModuleType("streamlit"))
fake_api = ModuleType("frontend_test_api doubles live transport")
sys.modules["api"] = fake_api

_SPEC = importlib.util.spec_from_file_location("aletheia_frontend_app_test", FRONTEND_DIR / "app.py")
app = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(app)


class FakeSession(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class FakeRerun(Exception):
    pass


class FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeColumn:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, text, *args, **kwargs):
        self.owner.markdown_calls.append(text)

    def button(self, label, *args, **kwargs):
        self.owner.button_calls.append((label, kwargs.get("key")))
        return self.owner.pressed.get(kwargs.get("key"), False)

    def checkbox(self, label, *args, **kwargs):
        self.owner.checkbox_calls.append((label, kwargs.get("key")))
        return self.owner.checked.get(kwargs.get("key"), False)

    def caption(self, text, *args, **kwargs):
        self.owner.captions.append(text)


class FakeUI:
    def __init__(self):
        self.session_state = FakeSession()
        self.markdown_calls = []
        self.button_calls = []
        self.checkbox_calls = []
        self.caption_calls = []
        self.infos = []
        self.errors = []
        self.jsons = []
        self.pressed = {}
        self.checked = {}

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [FakeColumn(self) for _ in range(count)]

    def markdown(self, text, *args, **kwargs):
        self.markdown_calls.append(text)

    def button(self, label, *args, **kwargs):
        self.button_calls.append((label, kwargs.get("key")))
        return self.pressed.get(kwargs.get("key"), False)

    def checkbox(self, label, *args, **kwargs):
        self.checkbox_calls.append((label, kwargs.get("key")))
        return self.checked.get(kwargs.get("key"), False)

    def caption(self, text, *args, **kwargs):
        self.caption_calls.append(text)

    def info(self, text, *args, **kwargs):
        self.infos.append(text)

    def error(self, text, *args, **kwargs):
        self.errors.append(text)

    def json(self, payload, *args, **kwargs):
        self.jsons.append(payload)

    def slider(self, label, minimum, maximum, value, step):
        return value

    def text_input(self, label, *args, **kwargs):
        return kwargs.get("value", "")

    def spinner(self, text):
        return FakeContext()

    def expander(self, label, expanded=False):
        return FakeContext()

    def rerun(self):
        raise FakeRerun()


class FakeAPI:
    def __init__(self):
        self.calls = Counter()
        self.facts_by_doc = {}
        self.rels_by_doc = {}
        self.details = {}
        self.bundles = {}
        self.searches = {}
        self.errors = {}

    def list_facts(self, doc_id):
        self.calls["list_facts"] += 1
        if doc_id in self.errors.get("list_facts", {}):
            return None, self.errors["list_facts"][doc_id]
        return list(self.facts_by_doc.get(doc_id, [])), None

    def list_relationships(self, doc_id, *args, **kwargs):
        self.calls["list_relationships"] += 1
        if doc_id in self.errors.get("list_relationships", {}):
            return None, self.errors["list_relationships"][doc_id]
        return list(self.rels_by_doc.get(doc_id, [])), None

    def get_relationship_detail(self, rel_id):
        self.calls["get_relationship_detail"] += 1
        return self.details[rel_id], None

    def get_bundle(self, doc_id):
        self.calls["get_bundle"] += 1
        if doc_id in self.errors.get("get_bundle", {}):
            return None, self.errors["get_bundle"][doc_id]
        return self.bundles[doc_id], None

    def search(self, query, doc_id, limit=30):
        self.calls["search"] += 1
        return {"hits": list(self.searches.get((query, doc_id), []))}, None


def _bind(ui, api):
    app.st = ui
    app.api_client = api
    return ui.session_state


def _fact(fact_id, doc_id, normalized=False):
    fact = {"id": fact_id, "document_id": doc_id, "subject": "Revenue",
            "predicate": "value", "value_text": "100"}
    if normalized:
        fact["normalized_number"] = 100.0
        fact["normalized_unit"] = "INR crore"
    return fact


def test_doc_stats_use_per_document_counts_and_retry_errors():
    ui, api = FakeUI(), FakeAPI()
    state = _bind(ui, api)
    state.doc_stats = {}
    state.fact_cache = {}
    api.facts_by_doc = {"d1": [_fact("f1", "d1"), _fact("f2", "d1", True)]}
    api.rels_by_doc = {"d1": [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}]}

    stats = app._doc_stats("d1")
    assert stats == {"facts": 2, "normalized": 1, "relationships": 3}
    app._doc_stats("d1")
    assert api.calls["list_facts"] == 1
    assert api.calls["list_relationships"] == 1

    api.errors = {"list_facts": {"d2": "boom"}, "list_relationships": {}}
    api.facts_by_doc["d2"] = [_fact("f9", "d2", True)]
    api.rels_by_doc["d2"] = [{"id": "r9"}]
    failed = app._doc_stats("d2")
    assert failed == {"facts": None, "normalized": None, "relationships": 1}
    assert "d2" not in state.doc_stats
    api.errors = {}
    recovered = app._doc_stats("d2")
    assert recovered == {"facts": 1, "normalized": 1, "relationships": 1}
    assert api.calls["list_facts"] == 3


def test_fact_bundle_and_search_caches_avoid_repeat_transport():
    ui, api = FakeUI(), FakeAPI()
    state = _bind(ui, api)
    state.fact_cache = {}
    state.bundle_cache = {}
    state.search_cache = {}
    api.facts_by_doc = {"d1": [_fact("f1", "d1")]}
    api.bundles = {"d1": {"document": {"id": "d1"}, "evidence": [{"id": "e1"}]}}
    api.searches = {("revenue", "d1"): [{"fact": {"id": "f1"}}]}

    facts, err = app._load_doc_facts("d1")
    bundle, bundle_err = app._cached_bundle("d1")
    hits, search_err = app._cached_search("  revenue ", [{"id": "d1"}], limit=30)
    assert err is None and bundle_err is None and search_err is None
    assert [f["id"] for f in facts] == ["f1"]
    assert bundle["evidence"] == [{"id": "e1"}]
    assert hits == [{"fact": {"id": "f1"}}]

    app._load_doc_facts("d1")
    app._cached_bundle("d1")
    app._cached_search("REVENUE", [{"id": "d1"}], limit=30)
    assert api.calls["list_facts"] == 1
    assert api.calls["get_bundle"] == 1
    assert api.calls["search"] == 1


def test_relationship_filters_use_cache_and_details_stay_on_demand():
    ui, api = FakeUI(), FakeAPI()
    state = _bind(ui, api)
    state.update({
        "rel_type_filter": "All",
        "rel_min_conf": 0.0,
        "rel_cache": {"key": None, "rels": []},
        "rel_details": {},
        "fact_cache": {},
        "selected_rel_id": None,
    })
    docs = [{"id": "d1"}, {"id": "d2"}]
    api.facts_by_doc = {
        "d1": [_fact("f1", "d1"), _fact("f2", "d1")],
        "d2": [_fact("f3", "d2")],
    }
    api.rels_by_doc = {
        "d1": [
            {"id": "r1", "relationship_type": "CORROBORATES", "confidence": 0.9,
             "fact_a_id": "f1", "fact_b_id": "f3", "explanation": "same"},
            {"id": "r2", "relationship_type": "CONTRADICTS", "confidence": 0.7,
             "fact_a_id": "f1", "fact_b_id": "f2", "explanation": "different"},
        ],
        "d2": [
            {"id": "r1", "relationship_type": "CORROBORATES", "confidence": 0.9,
             "fact_a_id": "f1", "fact_b_id": "f3", "explanation": "same"},
            {"id": "outside", "relationship_type": "CORROBORATES", "confidence": 0.9,
             "fact_a_id": "f1", "fact_b_id": "missing", "explanation": "outside"},
        ],
    }
    api.details = {
        "r1": {
            "relationship": {"reasoning_metadata": {"numeric_verdict": "INCOMPARABLE"}},
            "fact_a": {"subject": "A", "value_text": "1"},
            "fact_b": {"subject": "B", "value_text": "1"},
            "evidence_a": [{"id": "e1", "text": "proof", "type": "text",
                            "extraction_method": "pdf", "pdf_page_number": 0}],
            "evidence_b": [],
        }
    }

    app._render_relationships_section(docs, {"d1": "one", "d2": "two"})
    assert api.calls["list_relationships"] == 2
    assert api.calls["list_facts"] == 2
    assert api.calls["get_relationship_detail"] == 0

    state.rel_type_filter = "CONTRADICTS"
    app._render_relationships_section(docs, {"d1": "one", "d2": "two"})
    assert api.calls["list_relationships"] == 2
    assert api.calls["list_facts"] == 2
    assert api.calls["get_relationship_detail"] == 0
    assert ui.infos == []

    state.rel_type_filter = "All"
    state.rel_min_conf = 0.95
    app._render_relationships_section(docs, {"d1": "one", "d2": "two"})
    assert api.calls["list_relationships"] == 2
    assert api.calls["list_facts"] == 2
    assert api.calls["get_relationship_detail"] == 0
    assert ui.infos and "No persisted" in ui.infos[-1]

    state.rel_min_conf = 0.0
    state.selected_rel_id = "r1"
    app._render_relationships_section(docs, {"d1": "one", "d2": "two"})
    app._render_relationships_section(docs, {"d1": "one", "d2": "two"})
    assert api.calls["get_relationship_detail"] == 1
    assert any("MATCH SIGNALS" in text for text in ui.markdown_calls)

    app._render_relationships_section([{"id": "d1"}], {"d1": "one"})
    assert api.calls["list_relationships"] == 3
    assert api.calls["list_facts"] == 2


def test_documents_selected_and_processing_have_expected_rows_only():
    ui, api = FakeUI(), FakeAPI()
    state = _bind(ui, api)
    docs = [
        {"id": "d1", "filename": "annual.pdf", "page_count": 10,
         "created_at": "2024-01-01T00:00:00Z", "ingestion_status": "COMPLETED"},
        {"id": "d2", "filename": "prospectus.pdf", "page_count": 20,
         "created_at": "2024-01-02T00:00:00Z", "ingestion_status": "COMPLETED"},
        {"id": "d3", "filename": "quarterly.pdf", "page_count": 5,
         "created_at": "2024-01-03T00:00:00Z", "ingestion_status": "COMPLETED"},
    ]
    state.update({
        "selected_doc_ids": ["d1", "d2"],
        "selected_fact_id": None,
        "selected_rel_id": None,
        "doc_stats": {},
        "fact_cache": {},
        "pipeline_reports": {
            "d1": {"facts": {"report": {}}, "normalize": {"report": {}},
                   "relationships": {"report": {}}},
            "d3": {"facts": {"report": None, "error": "extraction failed"}},
        },
        "show_rel_scope": False,
        "rel_scope_ids": [],
    })
    ui.checked = {"sel_d1": True, "sel_d2": True, "sel_d3": False}
    api.facts_by_doc = {
        "d1": [_fact("f1", "d1", True) for _ in range(3)],
        "d2": [],
    }
    api.rels_by_doc = {"d1": [{"id": "r1"}], "d2": []}
    api.errors = {"list_facts": {"d3": "unavailable"},
                  "list_relationships": {"d3": "unavailable"}}

    app._render_documents_section(docs)
    assert api.calls["list_facts"] == 3
    assert api.calls["list_relationships"] == 3
    table = "\n".join(ui.markdown_calls)
    assert "3" in table and "1" in table and "—" in table

    app._render_overview(docs)
    overview = "\n".join(ui.markdown_calls)
    assert overview.count("doc-row-sel") == 3
    assert "FACTS EXTRACTED" in overview
    assert "FACTS PENDING" in overview
    assert "EXTRACTION FAILED" in overview
    assert "1178" not in overview

    selected = [docs[0], docs[1]]
    ui.markdown_calls = []
    app._render_processing(selected, docs)
    processing = "\n".join(ui.markdown_calls)
    assert [label for label, key in ui.button_calls if key in
            {"pipe_extract", "pipe_normalize", "pipe_relate_open"}] == [
        "EXTRACT FACTS", "NORMALIZE", "FORM RELATIONS"]
    assert "annual.pdf" not in processing
    assert "PENDING" not in processing
    assert "FAILED" not in processing


def test_search_key_ignores_scope_order_and_query_spacing():
    first = app._search_key("  Revenue Growth ", ["b", "a"], 30)
    second = app._search_key("revenue   growth", ["a", "b"], 30)
    third = app._search_key("revenue growth", ["a"], 30)
    assert first == second
    assert first != third
