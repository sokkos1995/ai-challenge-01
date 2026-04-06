from typing import Final, Optional

from app.models import TaskState


TASK_STAGE_PLANNING: Final[str] = "PLANNING"
TASK_STAGE_EXECUTION: Final[str] = "EXECUTION"
TASK_STAGE_VALIDATION: Final[str] = "VALIDATION"
TASK_STAGE_DONE: Final[str] = "DONE"
TASK_STAGE_REJECTED: Final[str] = "REJECTED"

TASK_STAGES: Final[tuple[str, ...]] = (
    TASK_STAGE_PLANNING,
    TASK_STAGE_EXECUTION,
    TASK_STAGE_VALIDATION,
    TASK_STAGE_DONE,
    TASK_STAGE_REJECTED,
)

TASK_PLAN_STATUS_DRAFT: Final[str] = "DRAFT"
TASK_PLAN_STATUS_APPROVED: Final[str] = "APPROVED"
TASK_PLAN_STATUSES: Final[tuple[str, ...]] = (
    TASK_PLAN_STATUS_DRAFT,
    TASK_PLAN_STATUS_APPROVED,
)

TASK_VALIDATION_STATUS_PENDING: Final[str] = "PENDING"
TASK_VALIDATION_STATUS_PASSED: Final[str] = "PASSED"
TASK_VALIDATION_STATUS_FAILED: Final[str] = "FAILED"
TASK_VALIDATION_STATUSES: Final[tuple[str, ...]] = (
    TASK_VALIDATION_STATUS_PENDING,
    TASK_VALIDATION_STATUS_PASSED,
    TASK_VALIDATION_STATUS_FAILED,
)

ALLOWED_TASK_STAGE_TRANSITIONS: Final[dict[str, tuple[str, ...]]] = {
    TASK_STAGE_PLANNING: (TASK_STAGE_EXECUTION, TASK_STAGE_REJECTED),
    TASK_STAGE_EXECUTION: (TASK_STAGE_VALIDATION, TASK_STAGE_PLANNING, TASK_STAGE_REJECTED),
    TASK_STAGE_VALIDATION: (TASK_STAGE_DONE, TASK_STAGE_EXECUTION, TASK_STAGE_REJECTED),
    TASK_STAGE_DONE: (TASK_STAGE_REJECTED,),
    TASK_STAGE_REJECTED: (),
}

_LEGACY_TASK_STAGE_ALIASES: Final[dict[str, str]] = {
    "": TASK_STAGE_PLANNING,
    "new": TASK_STAGE_PLANNING,
    "planning": TASK_STAGE_PLANNING,
    "in_progress": TASK_STAGE_EXECUTION,
    "execution": TASK_STAGE_EXECUTION,
    "validate": TASK_STAGE_VALIDATION,
    "validation": TASK_STAGE_VALIDATION,
    "done": TASK_STAGE_DONE,
    "rejected": TASK_STAGE_REJECTED,
    "reject": TASK_STAGE_REJECTED,
    "cancelled": TASK_STAGE_REJECTED,
    "canceled": TASK_STAGE_REJECTED,
}

_LEGACY_PLAN_STATUS_ALIASES: Final[dict[str, str]] = {
    "": TASK_PLAN_STATUS_DRAFT,
    "draft": TASK_PLAN_STATUS_DRAFT,
    "approved": TASK_PLAN_STATUS_APPROVED,
    "approve": TASK_PLAN_STATUS_APPROVED,
}

_LEGACY_VALIDATION_STATUS_ALIASES: Final[dict[str, str]] = {
    "": TASK_VALIDATION_STATUS_PENDING,
    "pending": TASK_VALIDATION_STATUS_PENDING,
    "pass": TASK_VALIDATION_STATUS_PASSED,
    "passed": TASK_VALIDATION_STATUS_PASSED,
    "ok": TASK_VALIDATION_STATUS_PASSED,
    "fail": TASK_VALIDATION_STATUS_FAILED,
    "failed": TASK_VALIDATION_STATUS_FAILED,
}


def normalize_task_stage(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in TASK_STAGES:
        return normalized

    alias = _LEGACY_TASK_STAGE_ALIASES.get(value.strip().lower())
    if alias:
        return alias

    raise ValueError(f"Unknown task stage: {value}. Allowed: {list(TASK_STAGES)}")


def allowed_task_stage_transitions(stage: str) -> tuple[str, ...]:
    normalized_stage = normalize_task_stage(stage)
    return ALLOWED_TASK_STAGE_TRANSITIONS[normalized_stage]


def normalize_plan_status(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in TASK_PLAN_STATUSES:
        return normalized

    alias = _LEGACY_PLAN_STATUS_ALIASES.get(value.strip().lower())
    if alias:
        return alias

    raise ValueError(f"Unknown plan status: {value}. Allowed: {list(TASK_PLAN_STATUSES)}")


def normalize_validation_status(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in TASK_VALIDATION_STATUSES:
        return normalized

    alias = _LEGACY_VALIDATION_STATUS_ALIASES.get(value.strip().lower())
    if alias:
        return alias

    raise ValueError(
        f"Unknown validation status: {value}. Allowed: {list(TASK_VALIDATION_STATUSES)}"
    )


def transition_readiness_error(task: TaskState, next_stage: str) -> Optional[str]:
    normalized_next = normalize_task_stage(next_stage)
    if task.paused and normalized_next != TASK_STAGE_REJECTED:
        return "Cannot change task stage while the task is paused. Resume it first."

    if task.state == TASK_STAGE_PLANNING and normalized_next == TASK_STAGE_EXECUTION:
        if not [item.strip() for item in task.plan if item.strip()]:
            return "Cannot start execution: the plan is empty."
        if normalize_plan_status(task.plan_status) != TASK_PLAN_STATUS_APPROVED:
            return "Cannot start execution: the plan is not approved yet."

    if task.state == TASK_STAGE_VALIDATION and normalized_next == TASK_STAGE_DONE:
        if normalize_validation_status(task.validation_status) != TASK_VALIDATION_STATUS_PASSED:
            return "Cannot finalize the task: validation has not passed yet."

    return None


def validate_task_stage_transition(
    current_stage: str, next_stage: str, task: Optional[TaskState] = None
) -> str:
    normalized_current = normalize_task_stage(current_stage)
    normalized_next = normalize_task_stage(next_stage)
    if normalized_current == normalized_next:
        if task is not None:
            readiness_error = transition_readiness_error(task, normalized_next)
            if readiness_error is not None:
                raise ValueError(readiness_error)
        return normalized_next

    allowed = allowed_task_stage_transitions(normalized_current)
    if normalized_next not in allowed:
        raise ValueError(
            f"Invalid task stage transition: {normalized_current} -> {normalized_next}. "
            f"Allowed: {list(allowed)}"
        )
    if task is not None:
        readiness_error = transition_readiness_error(task, normalized_next)
        if readiness_error is not None:
            raise ValueError(readiness_error)
    return normalized_next
