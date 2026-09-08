"""Focused tests: Groq strict JSON Schema transform.

Groq strict mode requires ``additionalProperties: false`` on every object,
all properties listed in ``required``, and no unresolved ``$ref`` — while
Pydantic v2 emits partial ``required`` lists, ``$defs`` references, and
wire-level hints (``default``, ``title``, ``format``). These tests prove
``to_strict_schema`` performs exactly that mechanical transform without
changing validation semantics (Pydantic re-validates every item anyway).
No network, no API key.
"""

import json

import pytest

from app.llm.groq import build_groq_schema, to_strict_schema


def _count_keys(node, key):
    if isinstance(node, dict):
        return sum(
            (1 if k == key else 0) + _count_keys(v, key)
            for k, v in node.items()
        )
    if isinstance(node, list):
        return sum(_count_keys(v, key) for v in node)
    return 0


def _collect_objects(node):
    found = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_collect_objects(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_collect_objects(value))
    return found


def test_strict_pins_additional_properties_false_everywhere():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "object", "properties": {}}},
        },
    }
    cleaned = to_strict_schema(schema)
    for obj in _collect_objects(cleaned):
        assert obj["additionalProperties"] is False


def test_strict_requires_all_properties():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string", "default": None},
        },
        "required": ["a"],
    }
    cleaned = to_strict_schema(schema)
    assert cleaned["required"] == ["a", "b"]


def test_strict_inlines_local_refs_and_drops_defs():
    schema = {
        "type": "object",
        "properties": {"kind": {"$ref": "#/$defs/Kind"}},
        "$defs": {"Kind": {"type": "string", "enum": ["X", "Y"]}},
    }
    cleaned = to_strict_schema(schema)
    assert "$defs" not in cleaned
    assert _count_keys(cleaned, "$ref") == 0
    assert cleaned["properties"]["kind"] == {
        "type": "string",
        "enum": ["X", "Y"],
    }
    assert cleaned["required"] == ["kind"]
    assert cleaned["additionalProperties"] is False


def test_strict_rejects_non_local_refs():
    with pytest.raises(ValueError, match="unsupported non-local"):
        to_strict_schema({
            "type": "object",
            "properties": {"x": {"$ref": "https://example.com/s.json"}},
        })


def test_strict_rejects_non_object_root():
    with pytest.raises(ValueError, match="must be a JSON object schema"):
        to_strict_schema({"type": "array", "items": {"type": "string"}})


def test_strict_drops_wire_hints_but_keeps_semantics():
    schema = {
        "type": "object",
        "title": "Thing",
        "description": "A thing.",
        "properties": {
            "name": {"type": "string", "title": "Name", "minLength": 1, "default": None},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "format": "float"},
        },
    }
    cleaned = to_strict_schema(schema)
    assert _count_keys(cleaned, "title") == 0
    assert _count_keys(cleaned, "description") == 0
    assert _count_keys(cleaned, "default") == 0
    assert _count_keys(cleaned, "format") == 0
    assert cleaned["properties"]["name"]["type"] == "string"
    assert cleaned["properties"]["score"]["type"] == "number"


def test_strict_preserves_anyof_and_does_not_mutate_input():
    schema = {
        "type": "object",
        "properties": {
            "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }
    snapshot = json.dumps(schema, sort_keys=True)
    cleaned = to_strict_schema(schema)
    assert cleaned["properties"]["unit"]["anyOf"] == [
        {"type": "string"}, {"type": "null"}
    ]
    assert json.dumps(schema, sort_keys=True) == snapshot


def test_built_groq_schema_is_strict_clean():
    schema = build_groq_schema()
    assert schema["required"] == ["drafts"]
    assert schema["properties"]["drafts"]["type"] == "array"
    assert "$defs" not in schema
    assert _count_keys(schema, "$ref") == 0
    for obj in _collect_objects(schema):
        assert obj["additionalProperties"] is False
        if "properties" in obj:
            assert obj["required"] == sorted(obj["properties"])


def test_freeform_object_uses_minimal_closed_shape():
    schema = build_groq_schema()
    item_props = schema["properties"]["drafts"]["items"]["properties"]
    assert item_props["context"] == {
        "type": "object",
        "additionalProperties": False,
    }
