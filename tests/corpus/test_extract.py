from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.corpus.extract import (
    ExtractionError,
    extract_document,
    extract_pdf_file,
    extract_text_file,
    make_doc_id,
)


def test_make_doc_id_uses_filename_stem() -> None:
    assert make_doc_id(Path("some/dir/report-2024.txt")) == "report-2024"


def test_extract_text_file_reads_content_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")

    document = extract_text_file(path)

    assert document.doc_id == "notes"
    assert document.filename == "notes.txt"
    assert document.doc_type == "txt"
    assert document.pages == ("hello world",)


def test_extract_text_file_empty_file_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    document = extract_text_file(path)

    assert document.pages == ("",)


def test_extract_text_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00\x01invalid")

    with pytest.raises(ExtractionError):
        extract_text_file(path)


def test_extract_pdf_file_returns_per_page_text(sample_pdf_path: Path) -> None:
    document = extract_pdf_file(sample_pdf_path)

    assert document.doc_id == "permafrost"
    assert document.doc_type == "pdf"
    assert len(document.pages) == 2
    assert "Permafrost" in document.pages[0]
    assert "thaws" in document.pages[1]


def test_extract_pdf_file_rejects_malformed_pdf(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"not a real pdf")

    with pytest.raises(ExtractionError):
        extract_pdf_file(path)


def test_extract_document_dispatches_by_suffix(tmp_path: Path, sample_pdf_path: Path) -> None:
    txt_path = tmp_path / "a.txt"
    txt_path.write_text("content", encoding="utf-8")

    assert extract_document(txt_path).doc_type == "txt"
    assert extract_document(sample_pdf_path).doc_type == "pdf"


def test_extract_document_rejects_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ExtractionError):
        extract_document(path)
