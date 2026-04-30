from __future__ import annotations

from pathlib import Path


class SimpleFaqRetriever:
    """
    Minimal retriever for template purposes.
    Replace with embeddings + vector search for production.
    """

    def __init__(self, kb_path: Path) -> None:
        self._chunks = self._load_chunks(kb_path)

    @staticmethod
    def _load_chunks(kb_path: Path) -> list[str]:
        text = kb_path.read_text(encoding="utf-8")
        return [part.strip() for part in text.split("\n\n") if part.strip()]

    def retrieve(self, question: str, k: int = 2) -> list[str]:
        tokens = set(question.lower().split())
        scored: list[tuple[int, str]] = []
        for chunk in self._chunks:
            overlap = sum(1 for token in tokens if token in chunk.lower())
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored if score > 0][:k]
