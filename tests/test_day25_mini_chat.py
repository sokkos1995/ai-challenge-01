import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.rag_service import RagService
from homeworks.src.day_21_indexing import _vectorize
from homeworks.src.day_25_mini_chat import MiniRagChat, run_scenarios


def _record(chunk_id: str, text: str, source: str = "a.py", section: str = "main") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding": _vectorize(text, dim=64),
        "metadata": {"source": source, "section": section},
    }


def _chat(records: list[dict]) -> MiniRagChat:
    rag = RagService(
        records=records,
        top_k_before=6,
        top_k_after=3,
        similarity_threshold=0.0,
        min_context_score=-1.0,
    )
    return MiniRagChat(rag)


def test_chat_keeps_history_and_always_returns_sources() -> None:
    chat = _chat(
        [
            _record("1", "day21 indexer saves json and sqlite artifacts", "homeworks/src/day_21_indexing.py"),
            _record("2", "day23 retrieve_then_rerank filters by threshold", "homeworks/src/day_23_rerank.py"),
        ]
    )

    first = chat.ask("Цель: сделать обзор индексатора")
    second = chat.ask("Какие артефакты сохраняются?")

    assert first["sources"]
    assert second["sources"]
    assert second["history_size"] == 4
    assert chat.task_memory.goal


def test_chat_tracks_constraints_and_terms() -> None:
    chat = _chat([_record("1", "memory service stores task state and notes", "app/services/memory_service.py")])

    chat.ask("Цель: описать memory service")
    chat.ask("Ограничение: не предлагай внешние библиотеки")
    reply = chat.ask("Уточню: называй этапы как stage")

    assert "не предлагай внешние библиотеки" in reply["task_memory"]["constraints"][-1]
    assert "stage" in " ".join(reply["task_memory"]["terms"]).lower()


def test_run_scenarios_passes_with_relevant_records() -> None:
    records = [
        _record("1", "day21 indexer parameters root include embedding out dir json sqlite", "homeworks/src/day_21_indexing.py"),
        _record("2", "day23 retrieve_then_rerank rewrite query similarity threshold contexts", "homeworks/src/day_23_rerank.py"),
        _record("3", "day24 grounded rag sources quotes anti hallucination min context score", "homeworks/src/day_24_grounded_rag.py"),
        _record("4", "task state has planning execution validation done rejected plan status", "app/models.py"),
        _record("5", "memory service updates working memory and lifecycle transitions", "app/services/memory_service.py"),
    ]
    report = run_scenarios(_chat(records))

    assert report["scenarios_count"] == 2
    assert report["all_passed"]
