import pytest

from app.cli import _handle_task_command
from app.services.memory_service import MemoryService
from app.task_state_machine import (
    allowed_task_stage_transitions,
    normalize_task_stage,
    validate_task_stage_transition,
)


def test_allows_only_declared_transitions() -> None:
    assert validate_task_stage_transition("PLANNING", "EXECUTION") == "EXECUTION"
    assert validate_task_stage_transition("EXECUTION", "VALIDATION") == "VALIDATION"
    assert validate_task_stage_transition("VALIDATION", "DONE") == "DONE"
    assert allowed_task_stage_transitions("DONE") == ()

    with pytest.raises(ValueError):
        validate_task_stage_transition("PLANNING", "DONE")


def test_normalizes_legacy_values() -> None:
    assert normalize_task_stage("new") == "PLANNING"
    assert normalize_task_stage("in_progress") == "EXECUTION"
    assert normalize_task_stage("validation") == "VALIDATION"


def test_persists_task_state_pause_and_prompt_context(tmp_path) -> None:
    base_path = str(tmp_path / "memory")
    service = MemoryService(memory_base_path=base_path, chat_keep_last_n=5)
    service.update_working_task_field("task", "Implement task state machine")
    service.update_working_task_field("expected_action", "Approve the implementation plan")
    service.update_working_task_field("plan+", "Define stages and transitions")
    service.update_working_task_field("note+", "Use SQLite for persistence")
    service.pause_current_task()
    service.resume_current_task()
    service.transition_task_state("EXECUTION")

    restored = MemoryService(memory_base_path=base_path, chat_keep_last_n=5)
    snapshot = restored.memory_snapshot()["working"]
    prompt = restored.memory_layers_system_message()["content"]

    assert snapshot["state"] == "EXECUTION"
    assert snapshot["paused"] is False
    assert snapshot["expected_action"] == "Approve the implementation plan"
    assert snapshot["allowed_transitions"] == ["VALIDATION", "PLANNING"]
    assert "Current stage: EXECUTION" in prompt
    assert "Allowed next stages from current stage: VALIDATION, PLANNING" in prompt
    assert "Task: Implement task state machine" in prompt


class _TaskCommandAgentStub:
    def __init__(self) -> None:
        self.updated_fields: list[tuple[str, str]] = []

    def memory_snapshot(self) -> dict:
        return {"working": {"task": "demo"}}

    def pause_current_task(self) -> None:
        pass

    def resume_current_task(self) -> None:
        pass

    def transition_task_state(self, new_state: str) -> str:
        return new_state

    def update_working_task_field(self, field_name: str, value: str) -> None:
        self.updated_fields.append((field_name, value))


def test_task_shortcuts_map_to_working_memory_updates(capsys) -> None:
    agent = _TaskCommandAgentStub()

    assert _handle_task_command("@task plan+ Define API contract", agent) is True
    assert _handle_task_command("@task done+ Implement storage migration", agent) is True
    assert _handle_task_command("@task expected Run validation checks", agent) is True

    captured = capsys.readouterr()
    assert "agent> task plan updated." in captured.out
    assert "agent> task done updated." in captured.out
    assert "agent> task expected action updated." in captured.out
    assert agent.updated_fields == [
        ("plan+", "Define API contract"),
        ("done+", "Implement storage migration"),
        ("expected_action", "Run validation checks"),
    ]
