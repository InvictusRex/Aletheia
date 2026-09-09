from uuid import uuid4
from app.facts.prompts import build_extraction_prompt
from app.facts.validator import build_fact
from app.models import ChunkUnit, EvidenceChunk, FactDraft, FactStatus, ValueKind


def _chunk(document_id, *evidence_ids):
    return EvidenceChunk(
        document_id=document_id,
        pdf_page_number=0,
        units=[
            ChunkUnit(evidence_id=eid, evidence_type="TEXT", text="synthetic unit")
            for eid in evidence_ids
        ],
    )


def _draft(evidence_id, **overrides):
    kwargs = {
        "subject": "Synthetic subject",
        "predicate": "reported total",
        "value_text": "740",
        "value_kind": ValueKind.NUMERIC,
        "time_text": None,
        "evidence_ids": [evidence_id],
        "confidence": 0.8,
    }
    kwargs.update(overrides)
    return FactDraft(**kwargs)


def test_prompt_excludes_boilerplate_and_requires_verbatim_currency():
    doc_id = uuid4()
    eid = uuid4()
    prompt = build_extraction_prompt(_chunk(doc_id, eid))
    lowered = prompt.lower()
    assert "urls" in lowered
    assert "contact details" in lowered
    assert "filing identifiers" in lowered
    assert "scrip codes" in lowered
    assert "boilerplate addresses" in lowered
    assert "administrative metadata" in lowered
    assert "substantive" in lowered
    assert "verbatim" in lowered
    assert "₹" in prompt
    assert "$" in prompt
    assert "€" in prompt
    assert "£" in prompt


def test_mangled_currency_flagged_ambiguous_preserving_text():
    doc_id = uuid4()
    eid = uuid4()
    raw = "I 8,836.14"
    draft = _draft(eid, value_text=raw, value_kind=ValueKind.NUMERIC)
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.value_text == raw
    assert fact.value_text == "I 8,836.14"
    assert "currency_symbol_suspect" in fact.ambiguity_flags
    assert fact.status == FactStatus.AMBIGUOUS


def test_clean_numeric_confirmed():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(eid, value_text="740", value_kind=ValueKind.NUMERIC)
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.status == FactStatus.CONFIRMED
    assert fact.ambiguity_flags == []
    assert fact.value_text == "740"
    assert fact.value_number == 740.0
    assert fact.value_kind == ValueKind.NUMERIC


def test_clean_percentage_confirmed():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(eid, value_text="1.6%", value_kind=ValueKind.PERCENTAGE)
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.status == FactStatus.CONFIRMED
    assert fact.ambiguity_flags == []
    assert fact.value_text == "1.6%"
    assert fact.value_number == 1.6
    assert fact.value_kind == ValueKind.PERCENTAGE


def test_text_grade_unflagged():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(eid, value_text="A grade", value_kind=ValueKind.TEXT)
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.status == FactStatus.CONFIRMED
    assert fact.ambiguity_flags == []
    assert fact.value_text == "A grade"
    assert fact.value_kind == ValueKind.TEXT


def test_unknown_evidence_rejected():
    doc_id = uuid4()
    known = uuid4()
    draft = _draft(uuid4())
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, known))
    assert fact is None
    assert reason is not None
    assert "unknown" in reason.lower()
