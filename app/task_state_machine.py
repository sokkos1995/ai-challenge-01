from typing import Final


TASK_STAGE_PLANNING: Final[str] = "PLANNING"
TASK_STAGE_EXECUTION: Final[str] = "EXECUTION"
TASK_STAGE_VALIDATION: Final[str] = "VALIDATION"
TASK_STAGE_DONE: Final[str] = "DONE"

TASK_STAGES: Final[tuple[str, ...]] = (
    TASK_STAGE_PLANNING,
    TASK_STAGE_EXECUTION,
    TASK_STAGE_VALIDATION,
    TASK_STAGE_DONE,
)

ALLOWED_TASK_STAGE_TRANSITIONS: Final[dict[str, tuple[str, ...]]] = {
    TASK_STAGE_PLANNING: (TASK_STAGE_EXECUTION,),
    TASK_STAGE_EXECUTION: (TASK_STAGE_VALIDATION, TASK_STAGE_PLANNING),
    TASK_STAGE_VALIDATION: (TASK_STAGE_DONE, TASK_STAGE_EXECUTION),
    TASK_STAGE_DONE: (),
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


def validate_task_stage_transition(current_stage: str, next_stage: str) -> str:
    normalized_current = normalize_task_stage(current_stage)
    normalized_next = normalize_task_stage(next_stage)
    if normalized_current == normalized_next:
        return normalized_next

    allowed = allowed_task_stage_transitions(normalized_current)
    if normalized_next not in allowed:
        raise ValueError(
            f"Invalid task stage transition: {normalized_current} -> {normalized_next}. "
            f"Allowed: {list(allowed)}"
        )
    return normalized_next
