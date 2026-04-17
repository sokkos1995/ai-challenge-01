import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homeworks.src.day_21_indexing import _vectorize  # noqa: E402
from homeworks.src.day_24_grounded_rag import (  # noqa: E402
    _has_quotes,
    _has_sources,
    _quote_alignment,
    answer_with_grounded_rag,
)


def _record(chunk_id: str, text: str, source: str = "a.py", section: str = "main") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding": _vectorize(text, dim=64),
        "metadata": {"source": source, "section": section},
    }


def test_low_relevance_forces_dont_know() -> None:
    records = [
        _record("1", "docker image layers cache build"),
        _record("2", "nginx reverse proxy config"),
    ]
    result = answer_with_grounded_rag(
        question="Как работает MCP orchestration list_tasks?",
        records=records,
        top_k_before=5,
        top_k_after=3,
        similarity_threshold=0.2,
        min_context_score=0.9,
        dry_run=True,
    )
    assert result["policy"] == "low_relevance_fallback"
    assert str(result["answer"]).startswith("Не знаю")


def test_grounded_answer_contains_sources_and_quotes() -> None:
    records = [
        _record("1", "mcp orchestration registry list_tasks add create_task", "homeworks/src/day_20_orchestration.py"),
        _record("2", "pytest assertions duplicate tool registry", "tests/test_day20_orchestration.py"),
    ]
    result = answer_with_grounded_rag(
        question="Как работает orchestration list_tasks?",
        records=records,
        top_k_before=5,
        top_k_after=2,
        similarity_threshold=0.1,
        min_context_score=0.05,
        dry_run=True,
    )
    assert result["policy"] == "grounded_answer"
    assert _has_sources(result)
    assert _has_quotes(result)


def test_quote_alignment_helper() -> None:
    aligned = _quote_alignment(
        "В orchestration есть registry и list_tasks.",
        ["registry list_tasks add create_task"],
    )
    not_aligned = _quote_alignment("Совсем другой ответ.", ["docker image layers"])
    assert aligned
    assert not not_aligned


def test_source_and_quote_check_helpers() -> None:
    payload = {
        "sources": [{"source": "a.py", "section": "main", "chunk_id": "1"}],
        "quotes": ["list_tasks -> add -> create_task"],
    }
    assert _has_sources(payload)
    assert _has_quotes(payload)
