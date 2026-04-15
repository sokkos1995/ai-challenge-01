import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homeworks.src.day_21_indexing import _vectorize  # noqa: E402
from homeworks.src.day_22_rag import (  # noqa: E402
    _build_rag_user_prompt,
    _default_control_questions,
    _evaluate_question,
    _fastapi_sbx_control_questions,
    retrieve_relevant_chunks,
)


def _record(chunk_id: str, text: str, source: str, section: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding": _vectorize(text, dim=64),
        "metadata": {
            "source": source,
            "section": section,
        },
    }


def test_retrieve_relevant_chunks_returns_top_items() -> None:
    records = [
        _record("1", "mcp orchestration list_tasks add create_task", "a.py", "main"),
        _record("2", "docker build runtime image", "Dockerfile", "main"),
    ]
    found = retrieve_relevant_chunks(records, "как работает list_tasks orchestration", top_k=1)
    assert len(found) == 1
    assert found[0]["source"] == "a.py"


def test_rag_prompt_contains_sources_and_context() -> None:
    contexts = [
        {
            "score": 0.9,
            "source": "homeworks/src/day_20_orchestration.py",
            "section": "run",
            "text": "list_tasks -> add -> create_task",
        }
    ]
    prompt = _build_rag_user_prompt("Какой flow?", contexts)
    assert "Context 1" in prompt
    assert "homeworks/src/day_20_orchestration.py" in prompt
    assert "Sources:" in prompt


def test_evaluate_question_computes_keyword_delta() -> None:
    control = _default_control_questions()[0]
    result = _evaluate_question(
        control,
        no_rag={"answer": "есть registry"},
        with_rag={
            "answer": "registry list_tasks add create_task",
            "contexts": [{"source": "homeworks/src/day_20_orchestration.py"}],
        },
    )
    assert result["with_rag"]["keyword_coverage"] >= result["without_rag"]["keyword_coverage"]
    assert result["with_rag"]["source_recall"] > 0


def test_fastapi_profile_contains_10_questions() -> None:
    questions = _fastapi_sbx_control_questions()
    assert len(questions) == 10
