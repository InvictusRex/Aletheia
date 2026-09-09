"""Tests: token-aware batching of semantic chunks (no network, no LLM)."""

import uuid

import pytest

from app.facts.batching import batch_chunks, merge_chunks
from app.models import ChunkUnit, EvidenceChunk


def _unit(eid=None, text="evidence text here", etype="TEXT"):
    return ChunkUnit(
        evidence_id=eid or uuid.uuid4(), evidence_type=etype, text=text
    )


def _chunk(doc=None, page=0, texts=("alpha beta gamma",)):
    return EvidenceChunk(
        document_id=doc or uuid.uuid4(),
        pdf_page_number=page,
        units=[_unit(text=t) for t in texts],
    )


def test_small_chunks_merge_into_one_request():
    doc = uuid.uuid4()
    chunks = [_chunk(doc, 0, ("short text one",)), _chunk(doc, 1, ("short text two",))]
    batches = batch_chunks(chunks, max_input_tokens=2000)
    assert len(batches) == 1
    assert [u.text for u in merge_chunks(batches[0]).units] == [
        "short text one",
        "short text two",
    ]


def test_batch_splits_when_budget_would_be_exceeded():
    doc = uuid.uuid4()
    big = "word " * 3000  # ~15000 chars, well over a 2000-token budget
    chunks = [_chunk(doc, 0, (big,)), _chunk(doc, 1, ("tiny",))]
    batches = batch_chunks(chunks, max_input_tokens=2000)
    assert len(batches) == 2
    assert len(batches[0]) == 1 and len(batches[1]) == 1


def test_evidence_ids_survive_batching_unchanged():
    doc = uuid.uuid4()
    id_a, id_b = uuid.uuid4(), uuid.uuid4()
    chunks = [
        EvidenceChunk(document_id=doc, pdf_page_number=0,
                      units=[_unit(eid=id_a, text="aaa")]),
        EvidenceChunk(document_id=doc, pdf_page_number=1,
                      units=[_unit(eid=id_b, text="bbb")]),
    ]
    merged = merge_chunks(batch_chunks(chunks, max_input_tokens=2000)[0])
    assert [u.evidence_id for u in merged.units] == [id_a, id_b]
    assert merged.document_id == doc
    assert merged.pdf_page_number == 0


def test_table_group_never_splits_across_batches():
    doc = uuid.uuid4()
    table_id = uuid.uuid4()
    table_text = "Table 0 (columns: h1 | h2):\n" + ("row text here\n" * 5)
    table_chunk = EvidenceChunk(
        document_id=doc, pdf_page_number=3,
        units=[_unit(eid=table_id, text=table_text, etype="TABLE")]
        + [_unit(text=f"cell {i}", etype="TABLE_CELL") for i in range(6)],
    )
    other = _chunk(doc, 4, ("unrelated",))
    batches = batch_chunks([table_chunk, other], max_input_tokens=2000)
    flat = [u for b in batches for u in merge_chunks(b).units]
    table_units = [u for u in flat if u.evidence_id == table_id]
    assert len(table_units) == 1
    cell_ids = {u.evidence_id for u in flat if u.evidence_type == "TABLE_CELL"}
    assert len(cell_ids) == 6


def test_oversize_chunk_travels_alone_explicitly():
    doc = uuid.uuid4()
    huge = "word " * 20000
    chunks = [_chunk(doc, 0, ("small",)), _chunk(doc, 1, (huge,)), _chunk(doc, 2, ("small",))]
    batches = batch_chunks(chunks, max_input_tokens=2000)
    assert len(batches) == 3
    assert len(batches[1]) == 1  # the oversize chunk stands alone


def test_empty_input_yields_no_batches():
    assert batch_chunks([], max_input_tokens=2000) == []


def test_nonpositive_budget_rejected():
    with pytest.raises(ValueError):
        batch_chunks([_chunk()], max_input_tokens=0)
    with pytest.raises(ValueError):
        batch_chunks([_chunk()], max_input_tokens=-5)
