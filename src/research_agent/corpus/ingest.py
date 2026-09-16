"""Corpus-level ingestion: walk a directory and extract every supported file."""

from __future__ import annotations

import logging
from pathlib import Path

from research_agent.corpus.extract import SUPPORTED_SUFFIXES, ExtractionError, extract_document
from research_agent.models import Document

logger = logging.getLogger(__name__)


def ingest_corpus(corpus_dir: Path) -> list[Document]:
    """Extract every supported file under `corpus_dir` (recursively) into a
    `Document`. Files that fail to extract are logged and skipped rather
    than aborting the whole ingestion run.

    Results are sorted by path for deterministic downstream chunk IDs.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"{corpus_dir} is not a directory")

    paths = sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )

    documents: list[Document] = []
    for path in paths:
        try:
            documents.append(extract_document(path))
        except ExtractionError as exc:
            logger.warning("Skipping %s: %s", path, exc)

    return documents
