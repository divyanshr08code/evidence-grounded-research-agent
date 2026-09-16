"""Integration test for the real local embedder.

Loading `all-MiniLM-L6-v2` requires the model weights to be present or
downloadable (first run needs network access to Hugging Face Hub). This
test is skipped rather than failed when that's not available, so the rest
of the suite stays runnable offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_agent.retrieval.embedder import SentenceTransformerEmbedder


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # model download/load failure -- not a code bug
        pytest.skip(f"could not load the local embedding model: {exc}")


def test_embed_documents_returns_expected_shape(embedder: SentenceTransformerEmbedder) -> None:
    embeddings = embedder.embed_documents(["glaciers store fresh water", "coral reefs bleach"])
    assert embeddings.shape == (2, embedder.dimension)
    assert embeddings.dtype == np.float32


def test_embed_documents_empty_list_returns_empty_array(
    embedder: SentenceTransformerEmbedder,
) -> None:
    embeddings = embedder.embed_documents([])
    assert embeddings.shape == (0, embedder.dimension)


def test_embed_query_returns_expected_shape(embedder: SentenceTransformerEmbedder) -> None:
    embedding = embedder.embed_query("how do glaciers store water?")
    assert embedding.shape == (embedder.dimension,)


def test_similar_texts_are_closer_than_unrelated_texts(
    embedder: SentenceTransformerEmbedder,
) -> None:
    query = embedder.embed_query("glacier ice melt water supply")
    on_topic = embedder.embed_documents(["Glaciers release meltwater that feeds rivers."])[0]
    off_topic = embedder.embed_documents(["The stock market closed higher on Tuesday."])[0]

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine(query, on_topic) > cosine(query, off_topic)
