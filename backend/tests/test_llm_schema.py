"""Focused tests: Gemini response-schema sanitization.

The Gemini Developer API rejects ``additionalProperties`` (Enterprise-only),
while Pydantic v2 emits it on object schemas. These tests prove the
recursive sanitizer removes those keys at every depth without changing
any other schema semantics. No network, no API key.
"""

import json

from app.llm.gemini import build_response_schema, sanitize_response_schema


def _count_keys(node, key):
    if isinstance(node, dict):
        return sum(
            (1 if k == key else 0) + _count_keys(v, key)
            for k, v in node.items()
        )
    if isinstance(node, list):
        return sum(_count_keys(v, key) for v in node)
    return 0


def test_sanitizer_removes_top_level_key():
    assert sanitize_response_schema(
        {"type": "object", "additionalProperties": False, "properties": {}}
    ) == {"type": "object", "properties": {}}


def test_sanitizer_recurses_into_properties_and_items():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "drafts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
                },
            }
        },
    }
    cleaned = sanitize_response_schema(schema)
    assert _count_keys(cleaned, "additionalProperties") == 0
    assert cleaned["properties"]["drafts"]["items"]["properties"]["tags"]["type"] == "array"


def test_sanitizer_recurses_into_defs():
    schema = {
        "$defs": {"Kind": {"type": "string", "additionalProperties": True}},
        "type": "object",
    }
    cleaned = sanitize_response_schema(schema)
    assert _count_keys(cleaned, "additionalProperties") == 0
    assert cleaned["$defs"]["Kind"] == {"type": "string"}


def test_sanitizer_preserves_other_semantics_and_input():
    schema = {
        "type": "object",
        "required": ["drafts"],
        "properties": {"drafts": {"type": "array", "minItems": 1}},
    }
    snapshot = json.dumps(schema, sort_keys=True)
    cleaned = sanitize_response_schema(schema)
    assert json.dumps(cleaned, sort_keys=True) == snapshot
    assert json.dumps(schema, sort_keys=True) == snapshot


def test_sanitizer_passes_through_scalars():
    assert sanitize_response_schema("object") == "object"
    assert sanitize_response_schema(42) == 42
    assert sanitize_response_schema(None) is None


def test_built_response_schema_has_no_additional_properties():
    schema = build_response_schema()
    assert _count_keys(schema, "additionalProperties") == 0
    assert schema["required"] == ["drafts"]
    assert schema["properties"]["drafts"]["type"] == "array"
