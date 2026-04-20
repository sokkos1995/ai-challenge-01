import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models import AgentResponse
from app.services.rag_chat_service import RagChatService
from app.services.rag_service import RagService


def _record(chunk_id: str, text: str, source: str = "a.py", section: str = "main") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding": [0.1] * 64,
        "metadata": {"source": source, "section": section},
    }


def _response(answer: str) -> AgentResponse:
    return AgentResponse(
        answer=answer,
        raw_data={"choices": [{"message": {"content": answer}}]},
        provider="test",
        model="test",
        latency_sec=0.01,
    )


def test_updates_task_memory_and_history() -> None:
    rag = RagService(records=[_record("1", "retrieval pipeline and source citations")], similarity_threshold=-1.0)
    service = RagChatService(rag)
    answer_json = (
        '{"answer":"ok","sources":[{"source":"a.py","section":"main","chunk_id":"1"}],'
        '"quotes":["retrieval pipeline"]}'
    )

    service.ask("Цель: подготовить обзор", lambda _: _response(answer_json))
    service.ask("Ограничение: только факты", lambda _: _response(answer_json))
    service.ask("Уточню: используй термин retrieval pipeline", lambda _: _response(answer_json))
    snapshot = service.memory_snapshot()

    assert snapshot["goal"] == "подготовить обзор"
    assert "только факты" in " ".join(snapshot["constraints"])  # type: ignore[arg-type]
    assert "retrieval pipeline" in " ".join(snapshot["terms"]).lower()  # type: ignore[arg-type]
    assert snapshot["history_size"] == 6


def test_low_relevance_skips_llm_call() -> None:
    service = RagChatService(RagService(records=[], min_context_score=0.9))
    called = {"llm": False}

    def _ask(_: str) -> AgentResponse:
        called["llm"] = True
        return _response('{"answer":"x","sources":[],"quotes":[]}')

    answer = service.ask("Как работает это?", _ask)
    assert answer.policy == "low_relevance_fallback"
    assert not called["llm"]
