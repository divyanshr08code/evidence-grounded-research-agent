"""Prompt construction for grounded generation.

Kept separate from the provider client so the grounding instructions and
context format can be inspected/tested without calling an API.
"""

from __future__ import annotations

from research_agent.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the "
    "numbered source excerpts provided below. Every factual claim in your "
    "answer must be attributable to at least one excerpt; cite it inline "
    "using its bracketed number, e.g. [2]. If the excerpts do not contain "
    "enough information to answer, say so explicitly instead of guessing."
)


def format_source_label(retrieved: RetrievedChunk) -> str:
    location = f"p.{retrieved.chunk.page}" if retrieved.chunk.page is not None else "n/a"
    return f"{retrieved.chunk.filename} ({location})"


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] Source: {format_source_label(retrieved)}\n{retrieved.chunk.text}"
        for i, retrieved in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context(chunks)
    return f"Question: {question}\n\nSource excerpts:\n\n{context}"
