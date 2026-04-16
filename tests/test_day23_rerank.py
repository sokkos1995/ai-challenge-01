import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homeworks.src.day_21_indexing import _vectorize  # noqa: E402
from homeworks.src.day_23_rerank import retrieve_then_rerank, rewrite_query  # noqa: E402


def _record(chunk_id: str, text: str, source: str = "a.py", section: str = "main") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding": _vectorize(text, dim=64),
        "metadata": {"source": source, "section": section},
    }


def test_rewrite_query_adds_expansion() -> None:
    original = "Как работает оркестрация MCP?"
    rewritten = rewrite_query(original)
    assert "orchestration" in rewritten.lower()
    assert rewritten != original


def test_retrieve_then_rerank_honors_threshold_and_top_k() -> None:
    records = [
        _record("1", "mcp orchestration registry list_tasks add create_task"),
        _record("2", "docker image layers and build"),
        _record("3", "pytest tests assertions for pipeline"),
    ]
    before, after = retrieve_then_rerank(
        records=records,
        question="как устроена оркестрация list_tasks",
        top_k_before=3,
        top_k_after=1,
        similarity_threshold=0.15,
    )
    assert len(before) == 3
    assert len(after) <= 1
    if after:
        assert after[0].rerank_score >= 0.15

