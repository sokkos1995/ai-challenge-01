import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homeworks.src.day_21_indexing import (  # noqa: E402
    Document,
    _build_index,
    _collect_documents,
    _chunk_fixed,
    _chunk_structured,
    _demo_search,
)


def test_fixed_chunking_contains_required_metadata() -> None:
    doc = Document(
        source="docs/example.md",
        title="example.md",
        text=("word " * 500).strip(),
    )
    chunks = _chunk_fixed(doc, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    first = chunks[0]
    assert first.metadata["source"] == "docs/example.md"
    assert first.metadata["title"] == "example.md"
    assert first.metadata["chunk_id"] == first.chunk_id
    assert first.metadata["strategy"] == "fixed"
    assert first.metadata["section"].startswith("words:")


def test_structured_chunking_uses_markdown_headers() -> None:
    doc = Document(
        source="notes/topic.md",
        title="topic.md",
        text="# Intro\nhello\n\n## Deep dive\nmore text",
    )
    chunks = _chunk_structured(doc, section_max_words=100)
    assert len(chunks) >= 2
    sections = {chunk.metadata["section"] for chunk in chunks}
    assert "Intro" in sections
    assert "Deep dive" in sections


def test_index_and_demo_search_return_expected_rows() -> None:
    doc = Document(
        source="src/sample.py",
        title="sample.py",
        text="def build_index():\n    return 'index'\n",
    )
    chunks = _chunk_structured(doc, section_max_words=50)
    records = _build_index(chunks, dim=64)
    assert records
    assert len(records[0]["embedding"]) == 64

    results = _demo_search(records, query="build index", top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "src/sample.py"


def test_collect_documents_indexes_text_repo_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Main.java").write_text("class Main { }", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM eclipse-temurin:21", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]", encoding="utf-8")

    docs = _collect_documents(tmp_path, max_files=100, include_patterns=["**/*"])
    sources = {doc.source for doc in docs}

    assert "src/Main.java" in sources
    assert "Dockerfile" in sources
    assert "binary.bin" not in sources
    assert ".git/config" not in sources
