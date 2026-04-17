import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.rag_service import RagService


def _record(chunk_id: str, text: str, source: str = "a.py", section: str = "main") -> dict:
    # Precomputed style vector is not required for tests;
    # RagService computes similarity with any numeric embedding shape.
    fake_embedding = [0.1] * 64
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding": fake_embedding,
        "metadata": {"source": source, "section": section},
    }


def test_from_json_index_loads_records(tmp_path: Path) -> None:
    payload = {"count": 1, "records": [_record("1", "mcp orchestration list_tasks add create_task")]}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    rag = RagService.from_json_index(str(index_path))
    prompt, contexts, low_relevance = rag.build_prompt("Как работает list_tasks orchestration?")
    assert "Верни строго JSON" in prompt
    assert isinstance(contexts, list)
    assert isinstance(low_relevance, bool)


def test_low_relevance_answer_policy() -> None:
    rag = RagService(records=[], min_context_score=0.9)
    answer = rag.low_relevance_answer([])
    assert answer.policy == "low_relevance_fallback"
    assert answer.answer.startswith("Не знаю")


def test_parse_answer_uses_fallback_sources_quotes() -> None:
    rag = RagService(records=[])
    _, contexts, _ = RagService(
        records=[_record("1", "registry list_tasks add create_task", "homeworks/src/day_20_orchestration.py")],
        similarity_threshold=-1.0,
    ).build_prompt("Как работает оркестрация?")
    parsed = rag.parse_answer("not-a-json", contexts)
    assert parsed.policy == "grounded_answer"
    assert parsed.sources
    assert parsed.quotes


def test_parse_answer_handles_fenced_json_block() -> None:
    rag = RagService(records=[])
    _, contexts, _ = RagService(
        records=[_record("1", "registry list_tasks add create_task", "homeworks/src/day_20_orchestration.py")],
        similarity_threshold=-1.0,
    ).build_prompt("Как работает оркестрация?")
    raw = """Вот структурированный ответ:
```json
{
  "answer": "Оркестрация идет через registry и цепочку list_tasks -> add -> create_task.",
  "sources": [{"source":"homeworks/src/day_20_orchestration.py","section":"main","chunk_id":"1"}],
  "quotes": ["list_tasks add create_task"]
}
```"""
    parsed = rag.parse_answer(raw, contexts)
    assert parsed.answer.startswith("Оркестрация")
    assert parsed.sources[0]["source"] == "homeworks/src/day_20_orchestration.py"
    assert parsed.quotes


def test_parse_answer_handles_json_with_prefix_suffix() -> None:
    rag = RagService(records=[])
    _, contexts, _ = RagService(
        records=[_record("1", "registry list_tasks add create_task", "homeworks/src/day_20_orchestration.py")],
        similarity_threshold=-1.0,
    ).build_prompt("Как работает оркестрация?")
    raw = (
        "Ниже итог:\n"
        '{"answer":"Короткий ответ.","sources":[{"source":"a.py","section":"main","chunk_id":"1"}],'
        '"quotes":["list_tasks"]}\n'
        "Спасибо!"
    )
    parsed = rag.parse_answer(raw, contexts)
    assert parsed.answer == "Короткий ответ."
    assert parsed.sources[0]["source"] == "a.py"
