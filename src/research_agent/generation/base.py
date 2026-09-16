"""Generation interface.

Kept behind a small `Protocol` so the LLM provider/model can be swapped
without touching pipeline code -- see CLAUDE.md: provider choice is an open
design decision, not a settled one.
"""

from __future__ import annotations

from typing import Protocol

from research_agent.models import RetrievedChunk


class GenerationConfigError(Exception):
    """Missing/invalid configuration (e.g. an unset API key).

    Distinct from a runtime API failure -- callers can treat this as "cannot
    run generation right now" rather than a pipeline bug.
    """


class Generator(Protocol):
    """Produces a grounded answer from a question and supporting chunks."""

    model_name: str

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Return the generated answer text."""
        ...
