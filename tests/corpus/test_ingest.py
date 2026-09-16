from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.corpus.ingest import ingest_corpus


def test_ingest_corpus_extracts_txt_and_pdf(sample_corpus_dir: Path) -> None:
    documents = ingest_corpus(sample_corpus_dir)
    doc_ids = {d.doc_id for d in documents}

    assert "glaciers" in doc_ids
    assert "coral_reefs" in doc_ids
    assert "permafrost" in doc_ids


def test_ingest_corpus_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("real content", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# not ingested", encoding="utf-8")

    documents = ingest_corpus(tmp_path)

    assert [d.doc_id for d in documents] == ["notes"]


def test_ingest_corpus_skips_malformed_files_without_aborting(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("fine", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a real pdf")

    documents = ingest_corpus(tmp_path)

    assert [d.doc_id for d in documents] == ["good"]


def test_ingest_corpus_is_deterministically_ordered(tmp_path: Path) -> None:
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")

    documents = ingest_corpus(tmp_path)

    assert [d.doc_id for d in documents] == ["a", "b", "c"]


def test_ingest_corpus_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        ingest_corpus(file_path)


def test_ingest_corpus_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert ingest_corpus(tmp_path) == []
