"""The Phase 1 baseline pipeline: retrieve top-k chunks, then generate a
grounded answer from them.

question -> retrieve top-k chunks -> construct grounded context -> generate
answer -> answer + retrieved evidence/source metadata.

`generator` is optional: if unset (or if it fails due to missing
credentials), the pipeline still returns retrieval results with
`generation_error` set, rather than fabricating an answer or crashing. This
lets retrieval be inspected and tested independently of any LLM API.
"""

from __future__ import annotations

import logging
from pathlib import Path

from research_agent.config import PipelineConfig
from research_agent.corpus.chunk import chunk_document
from research_agent.corpus.ingest import ingest_corpus
from research_agent.generation.base import GenerationConfigError, Generator
from research_agent.models import Chunk, Document, PipelineResult
from research_agent.retrieval.embedder import Embedder
from research_agent.retrieval.index import VectorIndex

logger = logging.getLogger(__name__)


class BaselineRAGPipeline:
    """Owns the index; answers questions against whatever corpus was built."""

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator | None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._config = config or PipelineConfig()
        self._index = VectorIndex()
        self._documents: list[Document] = []

    @property
    def num_documents(self) -> int:
        return len(self._index.document_ids())

    @property
    def num_chunks(self) -> int:
        return len(self._index)

    def build_index(self, corpus_dir: Path) -> None:
        self._documents = ingest_corpus(corpus_dir)
        if not self._documents:
            logger.warning("No documents ingested from %s", corpus_dir)

        chunks: list[Chunk] = []
        for document in self._documents:
            chunks.extend(chunk_document(document, self._config.chunking))
        if not chunks:
            logger.warning("No chunks produced from %s", corpus_dir)

        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        self._index.build(chunks, embeddings, embedding_model=self._embedder.model_name)

    def save_index(self, directory: Path) -> None:
        self._index.save(directory)

    def load_index(self, directory: Path) -> None:
        self._index = VectorIndex.load(directory, embedding_model=self._embedder.model_name)
        self._documents = []  # original Documents are not reconstructed from a saved index

    def answer(self, question: str) -> PipelineResult:
        query_embedding = self._embedder.embed_query(question)
        retrieved = self._index.search(query_embedding, self._config.retrieval.top_k)

        if not retrieved:
            return PipelineResult(
                question=question,
                retrieved_chunks=retrieved,
                generation_error="No chunks retrieved; index may be empty.",
            )

        if self._generator is None:
            return PipelineResult(
                question=question,
                retrieved_chunks=retrieved,
                generation_error="No generator configured.",
            )

        try:
            answer_text = self._generator.generate(question, retrieved)
        except GenerationConfigError as exc:
            return PipelineResult(
                question=question,
                retrieved_chunks=retrieved,
                generation_error=str(exc),
            )

        return PipelineResult(
            question=question,
            retrieved_chunks=retrieved,
            answer=answer_text,
            generation_model=self._generator.model_name,
        )
