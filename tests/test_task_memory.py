import pytest

from app.cli import _handle_task_command
from app.models import TaskState
from app.services.memory_service import MemoryService
from app.services.task_lifecycle_guard_service import TaskLifecycleGuardService
from app.task_state_machine import (
    allowed_task_stage_transitions,
    normalize_plan_status,
    normalize_task_stage,
    normalize_validation_status,
    validate_task_stage_transition,
)


def test_allows_only_declared_transitions() -> None:
    assert validate_task_stage_transition("PLANNING", "EXECUTION") == "EXECUTION"
    assert validate_task_stage_transition("EXECUTION", "VALIDATION") == "VALIDATION"
    assert validate_task_stage_transition("VALIDATION", "DONE") == "DONE"
    assert validate_task_stage_transition("PLANNING", "REJECTED") == "REJECTED"
    assert allowed_task_stage_transitions("DONE") == ("REJECTED",)
    assert allowed_task_stage_transitions("REJECTED") == ()

    with pytest.raises(ValueError):
        validate_task_stage_transition("PLANNING", "DONE")


def test_normalizes_legacy_values() -> None:
    assert normalize_task_stage("new") == "PLANNING"
    assert normalize_task_stage("in_progress") == "EXECUTION"
    assert normalize_task_stage("validation") == "VALIDATION"
    assert normalize_task_stage("cancelled") == "REJECTED"
    assert normalize_plan_status("approve") == "APPROVED"
    assert normalize_validation_status("fail") == "FAILED"


def test_blocks_execution_until_plan_is_approved() -> None:
    task = TaskState(
        task="Implement task state machine",
        state="PLANNING",
        plan=["Define stages and transitions"],
        plan_status="DRAFT",
    )

    with pytest.raises(ValueError, match="plan is not approved"):
        validate_task_stage_transition("PLANNING", "EXECUTION", task)


def test_blocks_done_until_validation_passes() -> None:
    task = TaskState(
        task="Implement task state machine",
        state="VALIDATION",
        plan=["Define stages and transitions"],
        plan_status="APPROVED",
        validation_status="FAILED",
    )

    with pytest.raises(ValueError, match="validation has not passed"):
        validate_task_stage_transition("VALIDATION", "DONE", task)


def test_can_reject_task_from_any_stage_including_pause() -> None:
    paused_task = TaskState(
        task="Implement task state machine",
        state="EXECUTION",
        plan=["Define stages and transitions"],
        plan_status="APPROVED",
        paused=True,
    )
    done_task = TaskState(
        task="Implement task state machine",
        state="DONE",
        plan=["Define stages and transitions"],
        plan_status="APPROVED",
        validation_status="PASSED",
    )

    assert validate_task_stage_transition("EXECUTION", "REJECTED", paused_task) == "REJECTED"
    assert validate_task_stage_transition("DONE", "REJECTED", done_task) == "REJECTED"


def test_low_level_state_update_uses_same_transition_side_effects(tmp_path) -> None:
    base_path = str(tmp_path / "memory")
    service = MemoryService(memory_base_path=base_path, chat_keep_last_n=5)
    service.update_working_task_field("task", "Implement task state machine")
    service.update_working_task_field("plan+", "Define stages and transitions")
    service.update_working_task_field("plan_status", "APPROVED")
    service.update_working_task_field("state", "EXECUTION")
    service.update_working_task_field("state", "VALIDATION")
    service.update_working_task_field("validation_status", "PASSED")
    service.update_working_task_field("state", "DONE")

    snapshot = service.memory_snapshot()["working"]

    assert snapshot["state"] == "DONE"
    assert snapshot["expected_action"] == "Summarize the completed result."
    assert snapshot["paused"] is False


def test_persists_task_state_pause_and_prompt_context(tmp_path) -> None:
    base_path = str(tmp_path / "memory")
    service = MemoryService(memory_base_path=base_path, chat_keep_last_n=5)
    service.update_working_task_field("task", "Implement task state machine")
    service.update_working_task_field("expected_action", "Approve the implementation plan")
    service.update_working_task_field("plan+", "Define stages and transitions")
    service.update_working_task_field("plan_status", "APPROVED")
    service.update_working_task_field("note+", "Use SQLite for persistence")
    service.pause_current_task()
    service.resume_current_task()
    service.transition_task_state("EXECUTION")
    service.transition_task_state("VALIDATION")
    service.update_working_task_field("validation_status", "PASSED")

    restored = MemoryService(memory_base_path=base_path, chat_keep_last_n=5)
    snapshot = restored.memory_snapshot()["working"]
    prompt = restored.memory_layers_system_message()["content"]

    assert snapshot["state"] == "VALIDATION"
    assert snapshot["plan_status"] == "APPROVED"
    assert snapshot["validation_status"] == "PASSED"
    assert snapshot["paused"] is False
    assert snapshot["expected_action"] == "Approve the implementation plan"
    assert snapshot["allowed_transitions"] == ["DONE", "EXECUTION", "REJECTED"]
    assert "Current stage: VALIDATION" in prompt
    assert "Plan status: APPROVED" in prompt
    assert "Validation status: PASSED" in prompt
    assert "Allowed next stages from current stage: DONE, EXECUTION, REJECTED" in prompt
    assert "Task: Implement task state machine" in prompt


class _TaskCommandAgentStub:
    def __init__(self) -> None:
        self.updated_fields: list[tuple[str, str]] = []
        self.transitioned_states: list[str] = []

    def memory_snapshot(self) -> dict:
        return {"working": {"task": "demo"}}

    def pause_current_task(self) -> None:
        pass

    def resume_current_task(self) -> None:
        pass

    def transition_task_state(self, new_state: str) -> str:
        self.transitioned_states.append(new_state)
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


def test_task_shortcuts_cover_explicit_lifecycle_controls(capsys) -> None:
    agent = _TaskCommandAgentStub()

    assert _handle_task_command("@task reject", agent) is True
    assert _handle_task_command("@task approve-plan", agent) is True
    assert _handle_task_command("@task validate pass", agent) is True
    assert _handle_task_command("@task validate fail", agent) is True
    assert _handle_task_command("@task reject-plan", agent) is True

    captured = capsys.readouterr()
    assert "agent> task state updated: REJECTED" in captured.out
    assert "agent> task plan approved." in captured.out
    assert "agent> task validation marked as passed." in captured.out
    assert "agent> task validation marked as failed." in captured.out
    assert "agent> task plan moved back to draft." in captured.out
    assert agent.transitioned_states == ["REJECTED"]
    assert agent.updated_fields == [
        ("plan_status", "APPROVED"),
        ("validation_status", "PASSED"),
        ("validation_status", "FAILED"),
        ("plan_status", "DRAFT"),
    ]


def test_lifecycle_guard_refuses_skipped_implementation_step() -> None:
    guard = TaskLifecycleGuardService()
    task = TaskState(
        task="Implement task state machine",
        state="PLANNING",
        plan=["Define stages and transitions"],
        plan_status="DRAFT",
    )

    conflict = guard.check_request("Сделай реализацию прямо сейчас", task)

    assert conflict is not None
    assert "реализации" in conflict.explanation


def test_lifecycle_guard_refuses_finalization_before_done() -> None:
    guard = TaskLifecycleGuardService()
    task = TaskState(
        task="Implement task state machine",
        state="VALIDATION",
        plan=["Define stages and transitions"],
        plan_status="APPROVED",
        validation_status="PASSED",
    )

    conflict = guard.check_request("Сделай финальный ответ", task)

    assert conflict is not None
    assert "DONE" in conflict.safe_alternative


def test_lifecycle_guard_allows_finalization_after_rejection() -> None:
    guard = TaskLifecycleGuardService()
    task = TaskState(
        task="Implement task state machine",
        state="REJECTED",
        plan=["Define stages and transitions"],
        plan_status="APPROVED",
    )

    conflict = guard.check_request("Сделай финальный ответ", task)

    assert conflict is None


def test_memory_context_drops_stale_finalization_refusals_after_done(tmp_path) -> None:
    base_path = str(tmp_path / "memory")
    service = MemoryService(memory_base_path=base_path, chat_keep_last_n=10)
    service.update_working_task_field("task", "Implement task state machine")
    service.update_working_task_field("plan+", "Define stages and transitions")
    service.update_working_task_field("plan_status", "APPROVED")
    service.update_working_task_field("state", "EXECUTION")
    service.update_working_task_field("state", "VALIDATION")
    service.update_working_task_field("validation_status", "PASSED")
    service.update_working_task_field("state", "DONE")

    refusal = (
        "Не могу выполнить этот шаг: он нарушает жизненный цикл задачи.\n"
        "Нельзя делать финальный ответ до завершения обязательной валидации и перехода в DONE.\n"
        "Следующий корректный шаг: Сначала зафиксируйте переход VALIDATION -> DONE, затем делайте финальный ответ."
    )
    service.memory_update_after_turn(
        {"role": "user", "content": "Сделай финальный ответ"},
        {"role": "assistant", "content": refusal},
    )
    service.memory_update_after_turn(
        {"role": "user", "content": "Сделай финальный ответ"},
        {"role": "assistant", "content": refusal},
    )

    context = service.memory_build_context_with_user(
        {"role": "user", "content": "Сделай финальный ответ"}
    )

    assert len(context) == 1
    assert context[0]["role"] == "user"
    assert context[0]["content"] == "Сделай финальный ответ"
