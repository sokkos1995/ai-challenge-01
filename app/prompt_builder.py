from app.models import LongTermMemory, ShortTermMemory, TaskState
from app.task_state_machine import ALLOWED_TASK_STAGE_TRANSITIONS


def _bullet_list(items: list[str], empty_placeholder: str) -> str:
    clean_items = [item.strip() for item in items if item.strip()]
    if not clean_items:
        return f"- {empty_placeholder}"
    return "\n".join(f"- {item}" for item in clean_items)


def _current_stage_guidance(task: TaskState) -> str:
    if task.paused:
        return (
            "The task is paused. Do not continue implementation autonomously. "
            "Help the user resume from the saved state without asking them to repeat prior context."
        )
    if task.state == "PLANNING":
        return "Focus on gathering requirements, clarifying gaps, and getting the plan ready for execution."
    if task.state == "EXECUTION":
        return "Focus on implementing the approved plan and producing concrete artifacts."
    if task.state == "VALIDATION":
        return "Focus on tests, review, verification against the plan, and fixing found gaps."
    return "The task is complete. Summarize the result and avoid reopening work unless the user asks."


def build_memory_prompt(
    short_memory: ShortTermMemory,
    task: TaskState,
    long_memory: LongTermMemory,
) -> str:
    allowed_transitions = ", ".join(ALLOWED_TASK_STAGE_TRANSITIONS[task.state]) or "(terminal)"
    expected_action = task.expected_action.strip() or "not set"
    current_step = f"{task.step}/{task.total}" if task.total > 0 else str(task.step)
    plan_status = task.plan_status.strip() or "DRAFT"
    validation_status = task.validation_status.strip() or "PENDING"

    return "\n\n".join(
        [
            "You are operating with persistent task memory and a formal task state machine.",
            "\n".join(
                [
                    "Task state machine:",
                    "- PLANNING -> EXECUTION",
                    "- EXECUTION -> VALIDATION, PLANNING",
                    "- VALIDATION -> DONE, EXECUTION",
                    "- DONE -> terminal",
                    f"- Current stage: {task.state}",
                    f"- Paused: {'yes' if task.paused else 'no'}",
                    f"- Plan status: {plan_status}",
                    f"- Validation status: {validation_status}",
                    f"- Allowed next stages from current stage: {allowed_transitions}",
                    f"- Current step: {current_step}",
                    f"- Expected action: {expected_action}",
                    _current_stage_guidance(task),
                    "Strict lifecycle rules:",
                    "- Do not implement if the current stage is not EXECUTION.",
                    "- Do not implement if the plan status is not APPROVED.",
                    "- Do not mark the task as final or complete unless validation status is PASSED and the stage becomes DONE.",
                    "- If the user asks to skip a stage, explain the required next valid step instead of complying.",
                ]
            ),
            "\n".join(
                [
                    "Task context:",
                    f"Task: {task.task.strip() or 'not set'}",
                    "Plan:",
                    _bullet_list(task.plan, "plan is empty"),
                    "Done:",
                    _bullet_list(task.done, "nothing completed yet"),
                    "Task notes:",
                    _bullet_list(task.notes, "no task notes"),
                ]
            ),
            "\n".join(
                [
                    "Supporting memory:",
                    "Long-term invariants:",
                    _bullet_list(long_memory.invariants, "no invariants"),
                    "Short-term notes:",
                    _bullet_list(short_memory.notes, "no short-term notes"),
                    "Long-term decisions:",
                    _bullet_list(long_memory.decisions, "no decisions"),
                    f"Long-term profile: {long_memory.profile}",
                    f"Long-term knowledge: {long_memory.knowledge}",
                ]
            ),
            (
                "Use this memory as the source of truth for continuing the task. "
                "Respect the long-term invariants above as non-negotiable constraints. "
                "When the user asks to move to another stage, respect the allowed transitions above. "
                "If a requested transition is invalid, explain the valid next stages."
            ),
        ]
    )
