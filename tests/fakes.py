"""Test doubles shared across the suite.

These exist so tests never need to load the real (heavy) embedding model or
call a real LLM API -- both would make the suite slow, network-dependent,
and in the generator's case, cost money.
"""

from __future__ import annotations

import hashlib

import numpy as np

from research_agent.models import RetrievedChunk

_DIM = 16


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for a real embedder.

    Hashes each word into one of `_DIM` buckets and sums the (signed) hits,
    so texts sharing more words end up with higher cosine similarity --
    enough to test ranking behavior without loading a real model.
    """

    model_name = "fake-hashing-embedder"

    @property
    def dimension(self) -> int:
        return _DIM

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(_DIM, dtype=np.float32)
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = digest[0] % _DIM
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign
        if not np.any(vector):
            vector[0] = 1.0  # avoid an all-zero vector for empty/unknown-word text
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, _DIM), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


class FakeGenerator:
    """Records the question/chunks it was called with; returns a fixed answer."""

    model_name = "fake-generator"

    def __init__(self, answer: str = "fake answer") -> None:
        self._answer = answer
        self.last_question: str | None = None
        self.last_chunks: list[RetrievedChunk] | None = None

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        self.last_question = question
        self.last_chunks = chunks
        return self._answer


class FailingGenerator:
    """Raises to simulate a genuine runtime API failure (not a config error)."""

    model_name = "failing-generator"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        raise RuntimeError("simulated API failure")
