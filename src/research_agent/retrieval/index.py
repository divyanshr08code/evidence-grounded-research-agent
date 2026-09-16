"""Brute-force cosine-similarity vector index.

The Phase 1 corpus is small and fixed, so an exact, in-memory NumPy index is
the simplest appropriate solution -- see CLAUDE.md for the rationale
against reaching for a vector database at this stage. Swappable later if a
corpus size actually demands approximate search.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research_agent.models import Chunk, RetrievedChunk


class EmbeddingModelMismatchError(Exception):
    """Raised when a saved index was built with a different embedding model
    than the one requested for querying. The two embedding spaces aren't
    comparable, so this is raised instead of silently searching with
    mismatched vectors.
    """


class VectorIndex:
    """An in-memory index of chunk embeddings, searched by cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None  # (n, dim), L2-normalized
        self._embedding_model: str | None = None

    def build(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedding_model: str | None = None,
    ) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunk count ({len(chunks)}) does not match embedding count "
                f"({embeddings.shape[0]})"
            )
        self._chunks = list(chunks)
        self._embeddings = _l2_normalize(embeddings.astype(np.float32)) if chunks else None
        self._embedding_model = embedding_model

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    def __len__(self) -> int:
        return len(self._chunks)

    def document_ids(self) -> set[str]:
        return {chunk.doc_id for chunk in self._chunks}

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self._embeddings is None or not self._chunks:
            return []

        query = _l2_normalize(query_embedding.reshape(1, -1).astype(np.float32))
        scores = (self._embeddings @ query.T).ravel()  # both sides normalized -> cosine similarity

        k = min(top_k, len(self._chunks))
        # `kind="stable"` keeps tie-breaking deterministic across runs/platforms.
        top_indices = np.argsort(-scores, kind="stable")[:k]

        return [
            RetrievedChunk(chunk=self._chunks[idx], score=float(scores[idx]), rank=rank + 1)
            for rank, idx in enumerate(top_indices)
        ]

    def save(self, directory: Path) -> None:
        """Persist chunks + embeddings so `ask` can run without re-indexing."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        embeddings = self._embeddings if self._embeddings is not None else np.empty((0, 0), dtype=np.float32)
        np.save(directory / "embeddings.npy", embeddings)
        chunks_payload = [asdict(chunk) for chunk in self._chunks]
        (directory / "chunks.json").write_text(json.dumps(chunks_payload, indent=2), encoding="utf-8")
        (directory / "metadata.json").write_text(
            json.dumps({"embedding_model": self._embedding_model}, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, embedding_model: str | None = None) -> VectorIndex:
        """Load a saved index.

        If `embedding_model` is given (the model the caller intends to query
        with) and the index's saved metadata recorded a different model,
        raises `EmbeddingModelMismatchError` rather than loading an index
        whose vectors would silently be incomparable to new query
        embeddings. An index saved without recorded metadata (or loaded
        without specifying `embedding_model`) is not checked.
        """
        directory = Path(directory)
        embeddings = np.load(directory / "embeddings.npy")
        chunks_payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in chunks_payload]

        saved_model: str | None = None
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            saved_model = json.loads(metadata_path.read_text(encoding="utf-8")).get("embedding_model")

        if embedding_model is not None and saved_model is not None and saved_model != embedding_model:
            raise EmbeddingModelMismatchError(
                f"Index at {directory} was built with embedding model {saved_model!r}, "
                f"but {embedding_model!r} was requested. Re-index with the matching "
                "model, or query with the model the index was built with."
            )

        index = cls()
        if chunks:
            index.build(chunks, embeddings, embedding_model=saved_model)
        return index


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for a zero vector
    return matrix / norms
