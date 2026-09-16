# CLAUDE.md

Technical context and conventions for working on this project. This file
covers what's stable across sessions; day-to-day task state belongs in the
conversation, not here.

## Project

An AI research agent that answers questions by retrieving and citing
verifiable evidence (an evidence-grounded RAG system). Currently at
**Phase 1: fixed-corpus RAG baseline** — ingestion, chunking, dense
retrieval, and grounded generation over a small local corpus. Claim
extraction, claim-evidence verification, conflict detection, and
uncertainty modeling are future phases and not implemented. Full research
framing lives in the Phase 0 design discussion (not persisted as a file
yet); this document tracks what's actually built.

## Environment

- Python 3.12, via the `.venv` virtual environment at the project root.
- Activate with `.venv\Scripts\activate` (PowerShell/cmd) before running
  anything.
- Install with `pip install -e ".[dev]"`. Check `pyproject.toml` for the
  current dependency list before assuming a package is available.

## Project layout

- `src/research_agent/` — the package (`src`-layout; install with
  `pip install -e .` for imports to resolve outside packaging tools).
  - `config.py` — `ChunkingConfig`/`RetrievalConfig`/`GenerationConfig` dataclasses.
  - `models/` — shared dataclasses: `Document`, `Chunk`, `RetrievedChunk`, `PipelineResult`.
  - `corpus/` — extraction (`.txt`/`.pdf` → `Document`) and chunking (`Document` → `Chunk`s). Kept separate on purpose.
  - `retrieval/` — `Embedder` protocol + local `sentence-transformers` implementation; `VectorIndex` (brute-force NumPy cosine similarity, with save/load).
  - `generation/` — `Generator` protocol + `AnthropicGenerator`; prompt construction is a separate module.
  - `pipelines/` — `BaselineRAGPipeline` wiring retrieval → generation.
  - `cli.py` / `__main__.py` — `python -m research_agent index|ask`.
- `tests/` — mirrors the package structure. `tests/fakes.py` holds shared
  test doubles (`FakeEmbedder`, `FakeGenerator`) so most tests don't need
  the real model or a live API. `tests/fixtures/sample_corpus/` is a small
  committed demo/test corpus.
- `data/` — local corpus input, gitignored (except `.gitkeep`).
- `runs/` — generated index artifacts, gitignored (except `.gitkeep`).
- `pyproject.toml` — single source of project metadata and dependencies
  (PEP 621 + hatchling). No `setup.py`/`setup.cfg`.
- `.env.example` — documents required environment variables. Keep it in
  sync with actual usage; never put real secrets in it or in git.

## Conventions

- Prefer the standard library and a small, deliberate dependency set over
  pulling in frameworks. Every new dependency should be a conscious choice,
  not a default.
- Modular design: retrieval, evidence-handling, and generation are distinct
  concerns and should stay separable, even before all of them exist.
- No placeholder/scaffold code for components that haven't been designed
  yet — an empty, well-structured repo is preferable to speculative
  abstractions.
- Type hints on new code; keep functions small and single-purpose.

## Workflow

- Do not install dependencies or scaffold new components without
  confirming the approach first — this project is still in the design
  phase for its core architecture.
- Keep commits scoped and don't commit unless explicitly asked to.
- When adding a dependency, add it to `pyproject.toml` `dependencies`
  (or an optional group) rather than installing ad hoc.

## Decisions made so far (Phase 1)

- **LLM provider:** Anthropic (`ANTHROPIC_API_KEY`), via a `Generator`
  protocol — an assumed default, not a settled commitment. Swapping
  providers means adding a new class, not touching pipeline code.
- **Embeddings:** local `sentence-transformers` (`all-MiniLM-L6-v2`) —
  reproducible without an embedding API, standard RAG baseline choice.
- **Vector index:** brute-force NumPy cosine similarity, in-memory with
  save/load to disk. Appropriate only because the corpus is small and
  fixed; revisit if that stops being true.
- **Testing:** `pytest`, installed as a `dev` extra.

## Open design decisions (confirm with the user before building further)

These are unresolved and materially affect later phases, so don't assume an
answer when implementing:

- What "evidence-grounded" means operationally beyond Phase 1 (claim
  granularity, verification mechanism, conflict handling, abstention
  policy).
- Interface beyond the CLI: API/service, library, or none.
- Linting/formatting toolchain (e.g. ruff) — not installed; don't add
  config for tools that aren't in use.
- Corpus/domain for later evaluation phases.
