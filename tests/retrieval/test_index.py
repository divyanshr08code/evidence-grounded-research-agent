from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from research_agent.models import Chunk
from research_agent.retrieval.index import EmbeddingModelMismatchError, VectorIndex


def _deterministic_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """n random-ish vectors without touching numpy.random (see note in
    test_search_respects_top_k -- numpy's random submodule can be blocked
    by DLL/application-control policies in locked-down environments)."""
    rng = random.Random(seed)
    return np.array([[rng.random() for _ in range(dim)] for _ in range(n)], dtype=np.float32)


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        filename="doc1.txt",
        text=text,
        chunk_index=0,
        page=None,
        start_char=0,
        end_char=len(text),
    )


def test_search_on_empty_index_returns_empty_list() -> None:
    index = VectorIndex()
    result = index.search(np.array([1.0, 0.0]), top_k=5)
    assert result == []


def test_build_rejects_mismatched_chunk_and_embedding_counts() -> None:
    index = VectorIndex()
    chunks = [_chunk("a"), _chunk("b")]
    embeddings = np.zeros((1, 4), dtype=np.float32)  # only 1 row for 2 chunks

    with pytest.raises(ValueError):
        index.build(chunks, embeddings)


def test_search_ranks_by_cosine_similarity() -> None:
    index = VectorIndex()
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = np.array(
        [
            [1.0, 0.0],  # a: identical direction to the query
            [0.0, 1.0],  # b: orthogonal to the query
            [-1.0, 0.0],  # c: opposite direction to the query
        ],
        dtype=np.float32,
    )
    index.build(chunks, embeddings)

    results = index.search(np.array([1.0, 0.0]), top_k=3)

    assert [r.chunk.chunk_id for r in results] == ["a", "b", "c"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)
    assert results[2].score == pytest.approx(-1.0)


def test_search_respects_top_k() -> None:
    index = VectorIndex()
    chunks = [_chunk(str(i)) for i in range(5)]
    embeddings = _deterministic_vectors(5, 4)
    index.build(chunks, embeddings)

    results = index.search(embeddings[0], top_k=2)

    assert len(results) == 2


def test_search_top_k_larger_than_corpus_returns_all() -> None:
    index = VectorIndex()
    chunks = [_chunk("a"), _chunk("b")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.build(chunks, embeddings)

    results = index.search(np.array([1.0, 0.0]), top_k=100)

    assert len(results) == 2


def test_document_ids_returns_distinct_doc_ids() -> None:
    index = VectorIndex()
    chunks = [
        Chunk("a::0", "a", "a.txt", "t", 0, None, 0, 1),
        Chunk("a::1", "a", "a.txt", "t", 1, None, 0, 1),
        Chunk("b::0", "b", "b.txt", "t", 0, None, 0, 1),
    ]
    embeddings = _deterministic_vectors(3, 4)
    index.build(chunks, embeddings)

    assert index.document_ids() == {"a", "b"}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = VectorIndex()
    chunks = [_chunk("a", "alpha"), _chunk("b", "beta")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.build(chunks, embeddings)

    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)

    assert len(loaded) == 2
    assert loaded.document_ids() == index.document_ids()

    original_results = index.search(np.array([1.0, 0.0]), top_k=2)
    loaded_results = loaded.search(np.array([1.0, 0.0]), top_k=2)
    assert [r.chunk.chunk_id for r in original_results] == [
        r.chunk.chunk_id for r in loaded_results
    ]


def test_save_and_load_round_trip_empty_index(tmp_path: Path) -> None:
    index = VectorIndex()
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)

    assert len(loaded) == 0
    assert loaded.search(np.array([1.0, 0.0]), top_k=5) == []


def test_build_records_embedding_model() -> None:
    index = VectorIndex()
    chunks = [_chunk("a")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    index.build(chunks, embeddings, embedding_model="model-a")

    assert index.embedding_model == "model-a"


def test_save_and_load_preserves_embedding_model(tmp_path: Path) -> None:
    index = VectorIndex()
    chunks = [_chunk("a")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    index.build(chunks, embeddings, embedding_model="model-a")

    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)

    assert loaded.embedding_model == "model-a"


def test_load_raises_on_embedding_model_mismatch(tmp_path: Path) -> None:
    index = VectorIndex()
    chunks = [_chunk("a")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    index.build(chunks, embeddings, embedding_model="model-a")
    index.save(tmp_path)

    with pytest.raises(EmbeddingModelMismatchError):
        VectorIndex.load(tmp_path, embedding_model="model-b")


def test_load_does_not_raise_when_requested_model_matches(tmp_path: Path) -> None:
    index = VectorIndex()
    chunks = [_chunk("a")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    index.build(chunks, embeddings, embedding_model="model-a")
    index.save(tmp_path)

    loaded = VectorIndex.load(tmp_path, embedding_model="model-a")

    assert len(loaded) == 1


def test_load_does_not_raise_when_no_model_requested(tmp_path: Path) -> None:
    # Backward-compatible: callers that don't pass embedding_model (or an
    # index saved without one recorded) skip the check rather than erroring.
    index = VectorIndex()
    chunks = [_chunk("a")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    index.build(chunks, embeddings, embedding_model="model-a")
    index.save(tmp_path)

    loaded = VectorIndex.load(tmp_path)

    assert len(loaded) == 1
