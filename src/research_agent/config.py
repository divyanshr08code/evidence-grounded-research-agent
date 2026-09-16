"""Pipeline configuration.

All Phase 1 tunables live here as plain dataclasses with explicit defaults,
so a run's configuration can be logged/reproduced and swept for later
experiments (e.g. chunk size, top-k) without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 800  # characters per chunk
    chunk_overlap: int = 150  # characters of overlap between consecutive chunks

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    max_tokens: int = 1024


@dataclass(frozen=True)
class PipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
