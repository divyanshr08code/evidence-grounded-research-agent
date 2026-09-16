from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.config import ChunkingConfig, PipelineConfig, RetrievalConfig
from research_agent.pipelines.baseline import BaselineRAGPipeline
from research_agent.retrieval.index import EmbeddingModelMismatchError
from tests.fakes import FailingGenerator, FakeEmbedder, FakeGenerator


def _small_corpus_config() -> PipelineConfig:
    return PipelineConfig(
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=20),
        retrieval=RetrievalConfig(top_k=2),
    )


def test_build_index_ingests_and_chunks_corpus(sample_corpus_dir: Path) -> None:
    pipeline = BaselineRAGPipeline(FakeEmbedder(), FakeGenerator(), _small_corpus_config())
    pipeline.build_index(sample_corpus_dir)

    assert pipeline.num_documents == 3  # glaciers.txt, coral_reefs.txt, permafrost.pdf
    assert pipeline.num_chunks > 0


def test_build_index_on_empty_corpus_is_handled_cleanly(tmp_path: Path) -> None:
    pipeline = BaselineRAGPipeline(FakeEmbedder(), FakeGenerator(), _small_corpus_config())
    pipeline.build_index(tmp_path)

    assert pipeline.num_documents == 0
    assert pipeline.num_chunks == 0


def test_answer_returns_retrieved_chunks_and_generated_answer(sample_corpus_dir: Path) -> None:
    generator = FakeGenerator(answer="Glaciers store water as ice. [1]")
    pipeline = BaselineRAGPipeline(FakeEmbedder(), generator, _small_corpus_config())
    pipeline.build_index(sample_corpus_dir)

    result = pipeline.answer("How do glaciers store water?")

    assert result.question == "How do glaciers store water?"
    assert len(result.retrieved_chunks) == 2  # top_k=2
    assert result.answer == "Glaciers store water as ice. [1]"
    assert result.generation_model == "fake-generator"
    assert result.generation_error is None
    assert generator.last_question == "How do glaciers store water?"
    assert generator.last_chunks == result.retrieved_chunks


def test_answer_ranks_are_1_indexed_and_ordered(sample_corpus_dir: Path) -> None:
    pipeline = BaselineRAGPipeline(FakeEmbedder(), FakeGenerator(), _small_corpus_config())
    pipeline.build_index(sample_corpus_dir)

    result = pipeline.answer("coral bleaching")

    ranks = [rc.rank for rc in result.retrieved_chunks]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_answer_with_no_generator_returns_retrieval_only(sample_corpus_dir: Path) -> None:
    pipeline = BaselineRAGPipeline(FakeEmbedder(), None, _small_corpus_config())
    pipeline.build_index(sample_corpus_dir)

    result = pipeline.answer("How do glaciers store water?")

    assert result.answer is None
    assert result.generation_error == "No generator configured."
    assert len(result.retrieved_chunks) > 0


def test_answer_on_empty_index_reports_no_chunks_retrieved(tmp_path: Path) -> None:
    pipeline = BaselineRAGPipeline(FakeEmbedder(), FakeGenerator(), _small_corpus_config())
    pipeline.build_index(tmp_path)  # empty corpus

    result = pipeline.answer("anything")

    assert result.retrieved_chunks == []
    assert result.generation_error == "No chunks retrieved; index may be empty."
    assert result.answer is None


def test_answer_propagates_genuine_generator_failures(sample_corpus_dir: Path) -> None:
    # A real (non-config) API failure should not be silently swallowed --
    # only GenerationConfigError is treated as a recoverable "skip generation" case.
    pipeline = BaselineRAGPipeline(FakeEmbedder(), FailingGenerator(), _small_corpus_config())
    pipeline.build_index(sample_corpus_dir)

    with pytest.raises(RuntimeError):
        pipeline.answer("How do glaciers store water?")


def test_save_and_load_index_round_trip(sample_corpus_dir: Path, tmp_path: Path) -> None:
    original = BaselineRAGPipeline(FakeEmbedder(), FakeGenerator(), _small_corpus_config())
    original.build_index(sample_corpus_dir)
    original.save_index(tmp_path)

    reloaded = BaselineRAGPipeline(FakeEmbedder(), FakeGenerator(), _small_corpus_config())
    reloaded.load_index(tmp_path)

    assert reloaded.num_chunks == original.num_chunks
    assert reloaded.num_documents == original.num_documents

    result = reloaded.answer("How do glaciers store water?")
    assert len(result.retrieved_chunks) == 2


def test_load_index_raises_on_embedding_model_mismatch(
    sample_corpus_dir: Path, tmp_path: Path
) -> None:
    original = BaselineRAGPipeline(FakeEmbedder(), FakeGenerator(), _small_corpus_config())
    original.build_index(sample_corpus_dir)
    original.save_index(tmp_path)

    mismatched_embedder = FakeEmbedder()
    mismatched_embedder.model_name = "a-different-embedding-model"
    reloaded = BaselineRAGPipeline(mismatched_embedder, FakeGenerator(), _small_corpus_config())

    with pytest.raises(EmbeddingModelMismatchError):
        reloaded.load_index(tmp_path)
