"""Deterministic, configurable text chunking.

Chunking is intentionally simple for the Phase 1 baseline: a sliding
character window with overlap, snapped to the nearest preceding whitespace
when possible so words aren't split mid-token. This is not claimed to be an
optimal chunking strategy -- it is a documented, reproducible starting
point that later experiments can swap out via `ChunkingConfig`.
"""

from __future__ import annotations

from research_agent.config import ChunkingConfig
from research_agent.models import Chunk, Document

# How far back (in characters) to look for a whitespace boundary before
# accepting a hard, mid-word cut. Keeps the snap-to-boundary search bounded.
_BOUNDARY_LOOKBACK = 40


def _snap_to_boundary(text: str, end: int) -> int:
    """Move `end` left to the nearest whitespace, within a bounded lookback.

    Falls back to the original `end` (a hard cut) if no whitespace is found
    within the lookback window, or if `end` already lands on one.
    """
    if end >= len(text) or text[end].isspace():
        return end

    lookback_limit = max(0, end - _BOUNDARY_LOOKBACK)
    i = end
    while i > lookback_limit:
        if text[i - 1].isspace():
            return i
        i -= 1
    return end


def chunk_text(text: str, config: ChunkingConfig) -> list[tuple[str, int, int]]:
    """Split `text` into (chunk_text, start_char, end_char) spans.

    Deterministic sliding window: each span is up to `config.chunk_size`
    characters, advancing by `chunk_size - chunk_overlap` each step, with
    the end boundary snapped to nearby whitespace where practical. Spans
    that are empty after stripping (e.g. pure whitespace runs) are dropped.

    `start_char`/`end_char` are adjusted to bound the *stored* (stripped)
    text exactly, so `text[start_char:end_char] == chunk_text` always holds
    -- the window itself (where a chunk starts, how far it advances) is
    unaffected; only the reported boundary of the stripped result is
    corrected.
    """
    if not text:
        return []

    spans: list[tuple[str, int, int]] = []
    step = config.chunk_size - config.chunk_overlap
    text_len = len(text)
    start = 0

    while start < text_len:
        raw_end = min(start + config.chunk_size, text_len)
        end = _snap_to_boundary(text, raw_end) if raw_end < text_len else raw_end
        if end <= start:  # guard against a degenerate snap on pathological input
            end = raw_end

        raw_span = text[start:end]
        span_text = raw_span.strip()
        if span_text:
            leading_ws = len(raw_span) - len(raw_span.lstrip())
            actual_start = start + leading_ws
            actual_end = actual_start + len(span_text)
            spans.append((span_text, actual_start, actual_end))

        if end >= text_len:
            break
        start += step

    return spans


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Chunk every page of `document`, preserving page and offset provenance.

    `chunk_index` is a single counter across the whole document (not reset
    per page), so it -- combined with `doc_id` -- is a stable, unique chunk
    identifier.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, page_text in enumerate(document.pages, start=1):
        page = page_number if document.doc_type == "pdf" else None
        for span_text, start, end in chunk_text(page_text, config):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::c{chunk_index}",
                    doc_id=document.doc_id,
                    filename=document.filename,
                    text=span_text,
                    chunk_index=chunk_index,
                    page=page,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_index += 1

    return chunks
