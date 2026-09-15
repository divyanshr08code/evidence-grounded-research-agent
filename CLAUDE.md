# CLAUDE.md

Technical context and conventions for working on this project. This file
covers what's stable across sessions; day-to-day task state belongs in the
conversation, not here.

## Project

An AI research agent that answers questions by retrieving and citing
verifiable evidence (an evidence-grounded RAG system). Architecture is not
yet designed — see "Open design decisions" below.

## Environment

- Python 3.12, via the `.venv` virtual environment at the project root.
- Activate with `.venv\Scripts\activate` (PowerShell/cmd) before running
  anything.
- No dependencies are installed yet. Do not assume any package is
  available — check `pyproject.toml` first.

## Project layout

- `src/research_agent/` — the package. `src`-layout (not flat), so the
  package must be installed (`pip install -e .`) or run via a tool that
  respects `src/` for imports to resolve outside of packaging tools.
- `tests/` — mirrors the package structure as it grows.
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

## Open design decisions (confirm with the user before building)

These are unresolved and materially affect structure, so don't assume an
answer when implementing:

- LLM provider(s) to target.
- Retrieval/evidence store: vector DB vs. other indexing, and where
  evidence documents come from.
- What "evidence-grounded" means operationally here (e.g., citation
  format, verification step, refusal behavior when evidence is
  insufficient).
- Interface: CLI, API/service, library, or some combination.
- Testing/linting/formatting toolchain (e.g., pytest, ruff) — none is
  installed yet; don't add config for tools that aren't in use.
