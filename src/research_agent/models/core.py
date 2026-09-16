"""Core data structures shared across the pipeline.

These are the traceability backbone: every downstream artifact (a citation,
a verification result, an evaluation metric) will eventually need to point
back to a `Chunk`, and every `Chunk` must be traceable to the `Document` and
page it came from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """A single ingested source file, before chunking."""

    doc_id: str
    filename: str
    path: str
    doc_type: str  # "txt" | "pdf"
    pages: tuple[str, ...]  # extracted text per page; one element for .txt


@dataclass(frozen=True)
class Chunk:
    """A contiguous span of text produced by chunking a `Document`."""

    chunk_id: str
    doc_id: str
    filename: str
    text: str
    chunk_index: int  # position of this chunk within the document
    page: int | None  # 1-indexed page number; None when the format has none
    start_char: int  # offset of `text` within the source page/document text
    end_char: int


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its rank and similarity score."""

    chunk: Chunk
    score: float
    rank: int  # 1-indexed rank within the result set


@dataclass
class PipelineResult:
    """The full traceable output of a single question run."""

    question: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str | None = None
    generation_error: str | None = None
    generation_model: str | None = None
