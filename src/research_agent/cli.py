"""Command-line interface for the Phase 1 baseline.

    python -m research_agent index --corpus data/corpus --index-dir runs/index
    python -m research_agent ask --index-dir runs/index "What is ...?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_agent.config import ChunkingConfig, GenerationConfig, PipelineConfig, RetrievalConfig
from research_agent.generation.anthropic_client import AnthropicGenerator
from research_agent.generation.base import GenerationConfigError
from research_agent.models import PipelineResult
from research_agent.pipelines.baseline import BaselineRAGPipeline
from research_agent.retrieval.embedder import SentenceTransformerEmbedder
from research_agent.retrieval.index import EmbeddingModelMismatchError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research_agent", description="Evidence-grounded research agent (Phase 1 baseline)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Ingest and index a local corpus.")
    index_parser.add_argument(
        "--corpus", type=Path, required=True, help="Directory of .txt/.pdf source documents."
    )
    index_parser.add_argument(
        "--index-dir", type=Path, default=Path("runs/index"), help="Where to write the built index."
    )
    index_parser.add_argument("--chunk-size", type=int, default=ChunkingConfig().chunk_size)
    index_parser.add_argument("--chunk-overlap", type=int, default=ChunkingConfig().chunk_overlap)
    index_parser.add_argument("--embedding-model", type=str, default=RetrievalConfig().embedding_model)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against a previously built index.")
    ask_parser.add_argument("question", type=str)
    ask_parser.add_argument("--index-dir", type=Path, default=Path("runs/index"))
    ask_parser.add_argument("--top-k", type=int, default=RetrievalConfig().top_k)
    ask_parser.add_argument("--embedding-model", type=str, default=RetrievalConfig().embedding_model)
    ask_parser.add_argument("--generation-model", type=str, default=GenerationConfig().model)
    ask_parser.add_argument(
        "--no-generate", action="store_true", help="Skip generation; show retrieval only."
    )

    return parser


def _cmd_index(args: argparse.Namespace) -> int:
    config = PipelineConfig(
        chunking=ChunkingConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        retrieval=RetrievalConfig(embedding_model=args.embedding_model),
    )
    embedder = SentenceTransformerEmbedder(model_name=config.retrieval.embedding_model)
    pipeline = BaselineRAGPipeline(embedder=embedder, generator=None, config=config)

    print(f"Ingesting and indexing {args.corpus} ...")
    pipeline.build_index(args.corpus)
    pipeline.save_index(args.index_dir)

    print(f"Indexed {pipeline.num_documents} document(s), {pipeline.num_chunks} chunk(s).")
    print(f"Index written to {args.index_dir}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    if not (args.index_dir / "chunks.json").exists():
        print(f"No index found at {args.index_dir}. Run `index` first.", file=sys.stderr)
        return 1

    config = PipelineConfig(
        retrieval=RetrievalConfig(top_k=args.top_k, embedding_model=args.embedding_model)
    )

    generator = None
    if not args.no_generate:
        try:
            generator = AnthropicGenerator(model_name=args.generation_model)
        except GenerationConfigError as exc:
            print(f"[generation skipped] {exc}", file=sys.stderr)

    embedder = SentenceTransformerEmbedder(model_name=config.retrieval.embedding_model)
    pipeline = BaselineRAGPipeline(embedder=embedder, generator=generator, config=config)
    try:
        pipeline.load_index(args.index_dir)
    except EmbeddingModelMismatchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    result = pipeline.answer(args.question)
    _print_result(result)
    return 0


def _print_result(result: PipelineResult) -> None:
    print(f"\nQuestion: {result.question}\n")
    print(f"Retrieved {len(result.retrieved_chunks)} chunk(s):")
    for retrieved in result.retrieved_chunks:
        chunk = retrieved.chunk
        location = f"p.{chunk.page}" if chunk.page is not None else "n/a"
        snippet = chunk.text[:160].replace("\n", " ")
        ellipsis = "..." if len(chunk.text) > 160 else ""
        print(
            f"  [{retrieved.rank}] score={retrieved.score:.4f}  "
            f"{chunk.filename} ({location})  chunk_id={chunk.chunk_id}"
        )
        print(f"      {snippet}{ellipsis}")

    print()
    if result.answer is not None:
        print(f"Answer (model={result.generation_model}):\n{result.answer}")
    else:
        print(f"No answer generated: {result.generation_error}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return _cmd_index(args)
    if args.command == "ask":
        return _cmd_ask(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
