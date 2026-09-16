from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CORPUS_DIR = FIXTURES_DIR / "sample_corpus"


@pytest.fixture
def sample_corpus_dir() -> Path:
    return SAMPLE_CORPUS_DIR


@pytest.fixture
def sample_pdf_path() -> Path:
    return SAMPLE_CORPUS_DIR / "permafrost.pdf"
