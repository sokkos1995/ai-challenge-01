from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "target",
    "build",
    "dist",
}


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str]


def _is_probably_text_file(path: Path, sample_size: int = 4096) -> bool:
    try:
        sample = path.read_bytes()[:sample_size]
    except OSError:
        return False
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _collect_documents(
    root: Path,
    max_files: int,
    include_patterns: list[str],
    excluded_roots: list[Path] | None = None,
) -> list[Document]:
    documents: list[Document] = []
    normalized_excluded = [path.resolve() for path in (excluded_roots or [])]

    for pattern in include_patterns:
        for path in sorted(root.glob(pattern)):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            resolved = path.resolve()
            if any(excluded == resolved or excluded in resolved.parents for excluded in normalized_excluded):
                continue
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if not _is_probably_text_file(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            stripped = text.strip()
            if not stripped:
                continue
            rel = path.relative_to(root).as_posix()
            documents.append(Document(source=rel, title=path.name, text=stripped))
            if len(documents) >= max_files:
                return documents
    return documents


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9_]+", text.lower())


def _chunk_fixed(doc: Document, chunk_size: int = 220, overlap: int = 40) -> list[Chunk]:
    words = doc.text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        window = words[start : start + chunk_size]
        if not window:
            break
        text = " ".join(window).strip()
        section = f"words:{start}-{start + len(window)}"
        chunk_id = f"{doc.source}::fixed::{idx:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    "source": doc.source,
                    "title": doc.title,
                    "file": doc.source,
                    "section": section,
                    "chunk_id": chunk_id,
                    "strategy": "fixed",
                },
            )
        )
        idx += 1
        start += step
    return chunks


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "intro"
    current_lines: list[str] = []

    for line in lines:
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip() or "untitled"
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(block).strip()) for title, block in sections if "\n".join(block).strip()]


def _split_python_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "module"
    current_lines: list[str] = []

    for line in lines:
        marker = re.match(r"^\s*(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if marker:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = f"{marker.group(1)} {marker.group(2)}"
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(block).strip()) for title, block in sections if "\n".join(block).strip()]


def _split_plain_sections(text: str) -> list[tuple[str, str]]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []
    return [(f"paragraph_{idx:03d}", paragraph) for idx, paragraph in enumerate(paragraphs)]


def _chunk_structured(doc: Document, section_max_words: int = 280) -> list[Chunk]:
    if doc.source.endswith(".md"):
        sections = _split_markdown_sections(doc.text)
    elif doc.source.endswith(".py") or doc.source.endswith(".java"):
        sections = _split_python_sections(doc.text)
    else:
        sections = _split_plain_sections(doc.text)

    chunks: list[Chunk] = []
    idx = 0
    for section_title, section_text in sections:
        words = section_text.split()
        if not words:
            continue
        for start in range(0, len(words), section_max_words):
            window = words[start : start + section_max_words]
            text = " ".join(window).strip()
            if not text:
                continue
            section = section_title if start == 0 else f"{section_title} (part {start // section_max_words + 1})"
            chunk_id = f"{doc.source}::structured::{idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata={
                        "source": doc.source,
                        "title": doc.title,
                        "file": doc.source,
                        "section": section,
                        "chunk_id": chunk_id,
                        "strategy": "structured",
                    },
                )
            )
            idx += 1
    return chunks


def _vectorize(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vec[idx] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector dimensions do not match.")
    return sum(x * y for x, y in zip(a, b))


def _build_index(chunks: Iterable[Chunk], dim: int = 128) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": _vectorize(chunk.text, dim=dim),
                "metadata": chunk.metadata,
            }
        )
    return records


def _save_json(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"count": len(records), "records": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_sqlite(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_index (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                file TEXT NOT NULL,
                section TEXT NOT NULL,
                strategy TEXT NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM chunk_index")
        conn.executemany(
            """
            INSERT INTO chunk_index (
                chunk_id, text, embedding_json, source, title, file, section, strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(item["chunk_id"]),
                    str(item["text"]),
                    json.dumps(item["embedding"]),
                    str(item["metadata"]["source"]),
                    str(item["metadata"]["title"]),
                    str(item["metadata"]["file"]),
                    str(item["metadata"]["section"]),
                    str(item["metadata"]["strategy"]),
                )
                for item in records
            ],
        )
        conn.commit()


def _avg_words(records: list[dict[str, object]]) -> float:
    if not records:
        return 0.0
    lengths = [len(str(record["text"]).split()) for record in records]
    return sum(lengths) / len(lengths)


def _summarize_strategy(name: str, records: list[dict[str, object]]) -> str:
    sources = {str(record["metadata"]["source"]) for record in records}
    return (
        f"{name}: chunks={len(records)}, files={len(sources)}, "
        f"avg_words_per_chunk={_avg_words(records):.1f}"
    )


def _demo_search(records: list[dict[str, object]], query: str, top_k: int = 3) -> list[dict[str, object]]:
    if not records:
        return []
    first_embedding = records[0].get("embedding")
    dim = len(first_embedding) if isinstance(first_embedding, list) and first_embedding else 128
    query_vec = _vectorize(query, dim=dim)
    scored: list[tuple[float, dict[str, object]]] = []
    for record in records:
        score = _cosine_similarity(query_vec, list(record["embedding"]))
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[dict[str, object]] = []
    for score, record in scored[:top_k]:
        results.append(
            {
                "score": round(score, 4),
                "chunk_id": record["chunk_id"],
                "source": record["metadata"]["source"],
                "section": record["metadata"]["section"],
                "preview": str(record["text"])[:180].replace("\n", " ") + "...",
            }
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 21 document indexing demo.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Root directory with source documents.")
    parser.add_argument(
        "--include",
        action="append",
        default=["**/*"],
        help="Glob pattern(s) for documents. Defaults to indexing all files under --root.",
    )
    parser.add_argument("--max-files", type=int, default=250, help="Maximum number of files to index.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding vector size.")
    parser.add_argument(
        "--out-dir",
        default="homeworks/artifacts/day_21",
        help="Output directory for generated indexes and demo report.",
    )
    parser.add_argument(
        "--query",
        default="как в проекте реализованы mcp инструменты и оркестрация",
        help="Demo query for top-k retrieval.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top K search results in demo.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    documents = _collect_documents(
        root,
        max_files=max(1, args.max_files),
        include_patterns=args.include,
        excluded_roots=[out_dir],
    )
    if not documents:
        raise RuntimeError("No documents were collected. Check --root and --include patterns.")

    fixed_chunks = [chunk for doc in documents for chunk in _chunk_fixed(doc)]
    structured_chunks = [chunk for doc in documents for chunk in _chunk_structured(doc)]

    fixed_records = _build_index(fixed_chunks, dim=max(16, args.embedding_dim))
    structured_records = _build_index(structured_chunks, dim=max(16, args.embedding_dim))

    _save_json(out_dir / "index_fixed.json", fixed_records)
    _save_json(out_dir / "index_structured.json", structured_records)
    _save_sqlite(out_dir / "index_fixed.sqlite3", fixed_records)
    _save_sqlite(out_dir / "index_structured.sqlite3", structured_records)

    fixed_demo = _demo_search(fixed_records, args.query, top_k=max(1, args.top_k))
    structured_demo = _demo_search(structured_records, args.query, top_k=max(1, args.top_k))

    report = {
        "documents_indexed": len(documents),
        "query": args.query,
        "comparison": [
            _summarize_strategy("fixed", fixed_records),
            _summarize_strategy("structured", structured_records),
        ],
        "fixed_top_k": fixed_demo,
        "structured_top_k": structured_demo,
    }
    _save_json(out_dir / "comparison_report.json", [report])

    print("Day 21 indexing completed.")
    print(f"Documents indexed: {len(documents)}")
    print(_summarize_strategy("fixed", fixed_records))
    print(_summarize_strategy("structured", structured_records))
    print(f"Output directory: {out_dir}")
    print("Top results (fixed):")
    for row in fixed_demo:
        print(f"  - score={row['score']}, source={row['source']}, section={row['section']}")
    print("Top results (structured):")
    for row in structured_demo:
        print(f"  - score={row['score']}, source={row['source']}, section={row['section']}")


if __name__ == "__main__":
    main()
