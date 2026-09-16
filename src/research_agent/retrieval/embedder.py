"""Embedding interface and a local sentence-transformers implementation.

Kept behind a small `Protocol` so the embedding model can be swapped (a
different local model, or an API-based embedder) without touching retrieval
or pipeline code. Tests can implement `Embedder` with a cheap fake instead
of loading the real model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Turns text into fixed-size dense vectors."""

    model_name: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns an (n, dimension) float32 array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns a (dimension,) float32 array."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding model via `sentence-transformers`.

    Loads the model once at construction; embedding calls run on CPU by
    default, so retrieval experiments are reproducible without depending on
    an embedding API being reachable on every run.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Imported lazily so importing this module doesn't require torch to
        # be installed unless a real embedder is actually constructed.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return embedding.astype(np.float32)
