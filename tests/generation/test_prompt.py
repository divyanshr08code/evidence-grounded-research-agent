from __future__ import annotations

from research_agent.generation.prompt import SYSTEM_PROMPT, build_context, build_user_prompt
from research_agent.models import Chunk, RetrievedChunk


def _retrieved(chunk_id: str, text: str, page: int | None, rank: int, score: float) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        filename="doc1.pdf" if page is not None else "doc1.txt",
        text=text,
        chunk_index=0,
        page=page,
        start_char=0,
        end_char=len(text),
    )
    return RetrievedChunk(chunk=chunk, score=score, rank=rank)


def test_system_prompt_instructs_grounding_and_citation() -> None:
    assert "ONLY" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()
    assert "insufficient" in SYSTEM_PROMPT.lower() or "enough information" in SYSTEM_PROMPT.lower()


def test_build_context_numbers_and_labels_sources() -> None:
    chunks = [
        _retrieved("a", "Glaciers store water.", page=3, rank=1, score=0.9),
        _retrieved("b", "Corals can bleach.", page=None, rank=2, score=0.5),
    ]

    context = build_context(chunks)

    assert "[1] Source: doc1.pdf (p.3)" in context
    assert "Glaciers store water." in context
    assert "[2] Source: doc1.txt (n/a)" in context
    assert "Corals can bleach." in context


def test_build_context_empty_list_returns_empty_string() -> None:
    assert build_context([]) == ""


def test_build_user_prompt_includes_question_and_context() -> None:
    chunks = [_retrieved("a", "Some fact.", page=1, rank=1, score=0.8)]
    prompt = build_user_prompt("What is X?", chunks)

    assert "What is X?" in prompt
    assert "Some fact." in prompt
