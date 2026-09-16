"""Text extraction from source documents.

Extraction is deliberately separate from chunking: this module only turns a
file on disk into a `Document` with its per-page text. It has no knowledge
of chunk sizes, overlap, or retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from research_agent.models import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".txt", ".pdf")


class ExtractionError(Exception):
    """Raised when a document cannot be extracted at all."""


def make_doc_id(path: Path) -> str:
    """Derive a stable document ID from a file's stem.

    Phase 1 simplification: this assumes filenames are unique within a
    corpus directory. Not collision-proof across subdirectories with the
    same filename.
    """
    return path.stem


def extract_text_file(path: Path) -> Document:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{path}: not valid UTF-8 text") from exc

    if not text.strip():
        logger.warning("%s: empty or whitespace-only text file", path)

    return Document(
        doc_id=make_doc_id(path),
        filename=path.name,
        path=str(path),
        doc_type="txt",
        pages=(text,),
    )


def extract_pdf_file(path: Path) -> Document:
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        raise ExtractionError(f"{path}: could not open PDF") from exc

    if reader.is_encrypted:
        raise ExtractionError(f"{path}: encrypted PDFs are not supported")

    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pypdf's parser can raise a range of error types
            logger.warning("%s: failed to extract page %d (%s)", path, page_number, exc)
            page_text = ""
        pages.append(page_text)

    if not pages:
        raise ExtractionError(f"{path}: PDF has no pages")

    if not any(p.strip() for p in pages):
        logger.warning("%s: no extractable text on any page", path)

    return Document(
        doc_id=make_doc_id(path),
        filename=path.name,
        path=str(path),
        doc_type="pdf",
        pages=tuple(pages),
    )


def extract_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return extract_text_file(path)
    if suffix == ".pdf":
        return extract_pdf_file(path)
    raise ExtractionError(f"{path}: unsupported file type {suffix!r}")
