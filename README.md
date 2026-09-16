# Evidence-Grounded Research Agent

An AI research agent designed to answer questions by retrieving and citing
verifiable evidence, rather than generating unsupported claims. The goal is
a system whose outputs are traceable back to their sources.

## Status

**Phase 1: fixed-corpus RAG baseline.** A minimal, modular retrieval +
generation pipeline over a local, fixed document corpus: ingest documents
(`.txt`/`.pdf`) → chunk → embed → dense retrieval → grounded generation with
source citations. This is a controlled baseline, not the full
evidence-grounded design — claim extraction, claim-evidence verification,
conflict detection, and uncertainty modeling are future phases, not yet
implemented. See [CLAUDE.md](CLAUDE.md) for the full architecture context
and open design decisions.

## Requirements

- Python 3.12
- An Anthropic API key, only if you want to run generation (retrieval works
  without one)

## Setup

Activate the existing `.venv` (Python 3.12) and install the project with
its dev dependencies:

```sh
.venv\Scripts\activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` if you want to
run generation:

```sh
copy .env.example .env
```

## Usage

Place a corpus of `.txt`/`.pdf` files somewhere (e.g. `data/corpus/`, which
is gitignored), then:

```sh
# Ingest and index the corpus
python -m research_agent index --corpus data/corpus --index-dir runs/index

# Ask a question against the built index
python -m research_agent ask --index-dir runs/index "your question here"
```

`ask` shows the retrieved chunks (rank, similarity score, source filename,
page/chunk ID) and, if `ANTHROPIC_API_KEY` is set, a generated answer
grounded in those chunks. Without a key, it reports that generation was
skipped and still shows retrieval results. Run `python -m research_agent
--help` for all options (chunk size/overlap, top-k, model).

## Tests

```sh
pytest
```

Tests run entirely offline against fixtures in `tests/fixtures/` and use
fakes for embedding/generation — no API calls or model downloads required,
except one embedder integration test that self-skips if the local model
isn't available.

## Development

See [CLAUDE.md](CLAUDE.md) for technical context and development conventions.
