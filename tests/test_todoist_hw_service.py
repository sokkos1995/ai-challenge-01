"""Unit/integration tests for homeworks/todoist business logic (offline)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from homeworks.todoist.ai import AiPlan, fallback_plan, _extract_json
from homeworks.todoist.models import Task
from homeworks.todoist.service import TaskTrackerService
from homeworks.todoist.storage import JsonTaskStorage


def test_task_create_and_with_status_roundtrip() -> None:
    task = Task.create("  Buy milk  ", description="  2L  ", priority="high", tags=["errands"])
    assert task.title == "Buy milk"
    assert task.description == "2L"
    assert task.status == "todo"
    assert task.id.startswith("tsk_")
    assert task.priority == "high"
    assert task.tags == ["errands"]

    done = task.with_status("done")
    assert done.status == "done"
    assert done.id == task.id
    assert done.updated_at >= task.updated_at
    assert task.status == "todo"

    restored = Task.from_dict(done.to_dict())
    assert restored == done


def test_json_storage_empty_and_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "tasks.json"
    storage = JsonTaskStorage(db)
    assert storage.load_tasks() == []

    task = Task.create("Write tests")
    storage.save_tasks([task])
    loaded = storage.load_tasks()
    assert len(loaded) == 1
    assert loaded[0].title == "Write tests"
    assert loaded[0].id == task.id


def test_service_add_list_complete_delete(tmp_path: Path) -> None:
    service = TaskTrackerService(tmp_path / "tasks.json")
    created = service.add_task("Ship day_38", priority="high", tags=["hw"])
    listed = service.list_tasks()
    assert len(listed) == 1
    assert listed[0].id == created.id

    done = service.complete_task(created.id)
    assert done.status == "done"
    assert service.list_tasks()[0].status == "done"

    other = service.add_task("To remove")
    removed = service.delete_task(other.id)
    assert removed.id == other.id
    assert all(t.id != other.id for t in service.list_tasks())

    with pytest.raises(RuntimeError, match="Task not found"):
        service.complete_task("tsk_missing")
    with pytest.raises(RuntimeError, match="Task not found"):
        service.delete_task("tsk_missing")


def test_plan_goal_uses_fallback_when_llm_fails(tmp_path: Path) -> None:
    service = TaskTrackerService(tmp_path / "tasks.json")
    with patch("homeworks.todoist.service.generate_plan", side_effect=RuntimeError("no network")):
        created = service.plan_goal_with_ai("Выпустить pet-проект")

    assert len(created) >= 2
    assert created[0].source == "ai_fallback"
    assert any("subtask" in (t.tags or []) for t in created[1:])
    titles = {t.title for t in created}
    assert any("Выпустить pet-проект" in title or title.startswith("Уточнить") for title in titles)


def test_fallback_plan_and_extract_json() -> None:
    plan = fallback_plan("  Сделать релиз.  ")
    assert isinstance(plan, AiPlan)
    assert plan.title == "Сделать релиз"
    assert plan.priority == "medium"
    assert len(plan.subtasks) == 3
    assert "fallback" in (plan.tags or [])

    raw = _extract_json('```json\n{"title": "x", "subtasks": ["a"]}\n```')
    assert raw.startswith("{")
    assert '"title"' in raw
