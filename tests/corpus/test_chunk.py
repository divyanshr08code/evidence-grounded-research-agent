from __future__ import annotations

import pytest

from research_agent.config import ChunkingConfig
from research_agent.corpus.chunk import chunk_document, chunk_text
from research_agent.models import Document


def test_chunk_text_empty_string_returns_no_spans() -> None:
    assert chunk_text("", ChunkingConfig()) == []


def test_chunk_text_short_text_returns_single_span() -> None:
    config = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    spans = chunk_text("short text", config)

    assert len(spans) == 1
    text, start, end = spans[0]
    assert text == "short text"
    assert (start, end) == (0, 10)


def test_chunk_text_is_deterministic() -> None:
    text = "word " * 500
    config = ChunkingConfig(chunk_size=80, chunk_overlap=20)

    assert chunk_text(text, config) == chunk_text(text, config)


def test_chunk_text_respects_overlap() -> None:
    text = "abcdefgh " * 30  # 240 chars, no long words to trigger boundary snapping
    config = ChunkingConfig(chunk_size=50, chunk_overlap=10)

    spans = chunk_text(text, config)

    assert len(spans) > 1
    for (_, _, end_a), (_, start_b, _) in zip(spans, spans[1:]):
        # consecutive spans should overlap by roughly `chunk_overlap` characters
        assert 0 < end_a - start_b <= config.chunk_overlap + 1


def test_chunk_text_snaps_to_whitespace_instead_of_splitting_a_word() -> None:
    # A hard cut at chunk_size=12 would land inside "cccccccccc" (chars 10-19).
    # Boundary-snapping should push the cut back to the preceding space instead.
    text = "aaaa bbbb cccccccccc dddd"
    config = ChunkingConfig(chunk_size=12, chunk_overlap=2)

    spans = chunk_text(text, config)
    span_texts = [t for t, _, _ in spans]

    assert "cccccccccc" in span_texts  # the word survives intact as its own chunk
    assert not any(s not in ("cccccccccc",) and "ccccccc" in s for s in span_texts)


def test_chunk_text_covers_entire_input() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 10
    config = ChunkingConfig(chunk_size=60, chunk_overlap=15)

    spans = chunk_text(text, config)

    assert spans[0][1] == 0
    # Not len(text): the source text ends in whitespace ("dog. " repeated),
    # and end_char must bound the *stripped* stored text (see
    # test_chunk_text_offsets_exactly_match_stored_text), not the raw window.
    assert spans[-1][2] == len(text.rstrip())


def test_chunk_document_assigns_unique_deterministic_chunk_ids() -> None:
    document = Document(
        doc_id="doc1",
        filename="doc1.txt",
        path="/tmp/doc1.txt",
        doc_type="txt",
        pages=("word " * 200,),
    )
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)

    chunks_a = chunk_document(document, config)
    chunks_b = chunk_document(document, config)

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert len(set(c.chunk_id for c in chunks_a)) == len(chunks_a)


def test_chunk_document_txt_has_no_page_number() -> None:
    document = Document(
        doc_id="doc1", filename="doc1.txt", path="p", doc_type="txt", pages=("hello world",)
    )
    chunks = chunk_document(document, ChunkingConfig())

    assert all(c.page is None for c in chunks)


def test_chunk_document_pdf_preserves_page_numbers() -> None:
    document = Document(
        doc_id="doc1",
        filename="doc1.pdf",
        path="p",
        doc_type="pdf",
        pages=("page one text", "page two text"),
    )
    chunks = chunk_document(document, ChunkingConfig(chunk_size=100, chunk_overlap=10))

    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2}
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.filename == "doc1.pdf" for c in chunks)


def test_chunk_document_chunk_index_is_global_across_pages() -> None:
    document = Document(
        doc_id="doc1", filename="doc1.pdf", path="p", doc_type="pdf", pages=("aaa", "bbb")
    )
    chunks = chunk_document(document, ChunkingConfig(chunk_size=100, chunk_overlap=10))

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_text_offsets_exactly_match_stored_text() -> None:
    """Regression test for the offset/text provenance bug: (start, end) must
    exactly bound the *stored* (stripped) text, not the pre-strip window.

    chunk_size=12/overlap=2 over this text forces a hard boundary snap
    mid-word (see test_chunk_text_snaps_to_whitespace_instead_of_splitting_a_word),
    which is exactly the case where the old implementation returned an
    end_char one character past the actual (stripped) end of the text.
    """
    text = "aaaa bbbb cccccccccc dddd"
    config = ChunkingConfig(chunk_size=12, chunk_overlap=2)

    spans = chunk_text(text, config)

    assert len(spans) > 1  # sanity: boundary snapping actually occurs
    for span_text, start, end in spans:
        assert text[start:end] == span_text


def test_chunk_text_offsets_match_stored_text_with_overlap_and_many_chunks() -> None:
    """Same invariant, over a longer text producing several overlapping
    chunks, most of which have a trailing-whitespace-stripped boundary."""
    text = "abcdefgh " * 30
    config = ChunkingConfig(chunk_size=50, chunk_overlap=10)

    spans = chunk_text(text, config)

    assert len(spans) > 1
    for span_text, start, end in spans:
        assert text[start:end] == span_text


def test_chunk_document_offsets_exactly_match_source_page_text() -> None:
    """Same invariant at the Chunk/Document level: chunk.text must match
    the corresponding page text sliced at (start_char, end_char)."""
    page_text = "abcdefgh " * 30
    document = Document(
        doc_id="doc1", filename="doc1.txt", path="p", doc_type="txt", pages=(page_text,)
    )
    config = ChunkingConfig(chunk_size=50, chunk_overlap=10)

    chunks = chunk_document(document, config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert page_text[chunk.start_char : chunk.end_char] == chunk.text


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [(0, 0), (-5, 0), (10, 10), (10, 15)],
)
def test_chunking_config_rejects_invalid_values(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
