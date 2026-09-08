"""LLM provider abstraction (transport-only, document-agnostic).

Sync interface. Prompt construction lives in ``app.facts.prompts``
(not here); providers receive a ready-made prompt string and return
validated :class:`FactExtractionBatch` objects.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.models.fact import FactDraft


class LLMError(Exception):
    """Base class for all LLM provider failures."""


class LLMUnavailableError(LLMError):
    """Provider unusable (no API key / SDK missing) — caller skips extraction."""


class LLMTimeoutError(LLMError):
    """Provider request timed out."""


class LLMTransportError(LLMError):
    """Network / service-level transport failure."""


class LLMMalformedError(LLMError):
    """Unparseable or validator-rejected-at-transport output — caller must not fabricate."""


class FactExtractionBatch(BaseModel):
    """Validated batch of fact drafts from a single model call."""

    drafts: list[FactDraft]
    model: str


@runtime_checkable
class LLMProvider(Protocol):
    """Sync transport-only contract for LLM fact extraction.

    Document-agnostic: implementations must not parse documents or
    build prompts — prompt construction lives in ``app.facts.prompts``.

    Implementations must raise only :class:`LLMError` subclasses for
    provider failures.
    """

    model_name: str

    def extract_facts(self, prompt: str) -> FactExtractionBatch:
        """Send a ready-made prompt and return the validated batch."""
        ...
