"""Fact chunking + extraction prompts (task F-C1)."""

from app.facts.chunking import build_chunks
from app.facts.prompts import build_extraction_prompt

__all__ = ["build_chunks", "build_extraction_prompt"]
