import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag_output import format_rag_answer  # noqa: E402
from app.services.rag_service import RagAnswer  # noqa: E402


def test_format_rag_answer_contains_answer_sources_quotes() -> None:
    payload = RagAnswer(
        answer="Короткий ответ.",
        sources=[{"source": "a.py", "section": "main", "chunk_id": "1"}],
        quotes=["list_tasks -> add -> create_task"],
        policy="grounded_answer",
        contexts=[],
    )
    rendered = format_rag_answer(payload, prefix="agent> ")
    assert "agent> Короткий ответ." in rendered
    assert "agent> Sources:" in rendered
    assert "a.py :: main (1)" in rendered
    assert 'agent>   - "list_tasks -> add -> create_task"' in rendered
