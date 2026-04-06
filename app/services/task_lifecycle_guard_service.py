import re
from dataclasses import dataclass
from typing import Optional

from app.models import AgentResponse, TaskState
from app.task_state_machine import (
    TASK_PLAN_STATUS_APPROVED,
    TASK_STAGE_DONE,
    TASK_STAGE_EXECUTION,
    TASK_STAGE_REJECTED,
    TASK_STAGE_VALIDATION,
    TASK_VALIDATION_STATUS_PASSED,
)


@dataclass
class TaskLifecycleConflict:
    explanation: str
    safe_alternative: str
    raw_data: dict


class TaskLifecycleGuardService:
    LIFECYCLE_REFUSAL_PREFIX = "Не могу выполнить этот шаг: он нарушает жизненный цикл задачи."
    _IMPLEMENTATION_PATTERNS = (
        r"\bimplement\b",
        r"\bimplementation\b",
        r"\bwrite code\b",
        r"\bcode it\b",
        r"\bbuild it\b",
        r"\bship it\b",
        r"\bfix it\b",
        r"\brefactor\b",
        r"реализ",
        r"напиши код",
        r"сделай код",
        r"исправь",
        r"отрефактор",
    )
    _VALIDATION_PATTERNS = (
        r"\bvalidate\b",
        r"\bvalidation\b",
        r"\btest\b",
        r"\btests\b",
        r"\breview\b",
        r"\bqa\b",
        r"проверь",
        r"протест",
        r"валидац",
        r"ревью",
    )
    _FINALIZATION_PATTERNS = (
        r"\bfinal\b",
        r"\bfinalize\b",
        r"\bfinish\b",
        r"\bcomplete\b",
        r"\bwrap up\b",
        r"\bdeliver\b",
        r"финал",
        r"завер",
        r"закончи",
        r"готово",
        r"подведи итог",
    )

    @staticmethod
    def _has_signal(user_request: str, patterns: tuple[str, ...]) -> bool:
        text = user_request.strip().lower()
        if not text:
            return False
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    @classmethod
    def classify_request(cls, user_request: str) -> Optional[str]:
        if cls._has_signal(user_request, cls._FINALIZATION_PATTERNS):
            return "finalize"
        if cls._has_signal(user_request, cls._VALIDATION_PATTERNS):
            return "validate"
        if cls._has_signal(user_request, cls._IMPLEMENTATION_PATTERNS):
            return "implement"
        return None

    @classmethod
    def is_lifecycle_refusal_message(cls, assistant_message: str) -> bool:
        return assistant_message.strip().startswith(cls.LIFECYCLE_REFUSAL_PREFIX)

    @staticmethod
    def _is_task_tracked(task: TaskState) -> bool:
        return bool(
            task.task.strip()
            or task.plan
            or task.done
            or task.notes
            or task.expected_action.strip()
        )

    def check_request(self, user_request: str, task: TaskState) -> Optional[TaskLifecycleConflict]:
        if not self._is_task_tracked(task):
            return None

        requested_action = self.classify_request(user_request)
        if requested_action is None:
            return None

        if task.paused:
            return TaskLifecycleConflict(
                explanation="Задача сейчас на паузе, поэтому нельзя продолжать следующий этап.",
                safe_alternative="Сначала возобновите задачу, затем выполните следующий допустимый переход.",
                raw_data={"reason": "paused", "requested_action": requested_action},
            )

        if requested_action == "implement":
            if task.state != TASK_STAGE_EXECUTION:
                return TaskLifecycleConflict(
                    explanation=(
                        f"Нельзя переходить к реализации на стадии {task.state}. "
                        "Сначала выполните допустимый переход в EXECUTION."
                    ),
                    safe_alternative="Сначала подготовьте и утвердите план, затем переведите задачу в EXECUTION.",
                    raw_data={"reason": "not_in_execution", "requested_action": requested_action},
                )
            if task.plan_status != TASK_PLAN_STATUS_APPROVED:
                return TaskLifecycleConflict(
                    explanation="Нельзя переходить к реализации, пока план не утвержден.",
                    safe_alternative="Сначала утвердите план, затем продолжайте реализацию.",
                    raw_data={"reason": "plan_not_approved", "requested_action": requested_action},
                )

        if requested_action == "validate" and task.state != TASK_STAGE_VALIDATION:
            return TaskLifecycleConflict(
                explanation=(
                    f"Нельзя делать валидацию на стадии {task.state}. "
                    "Сначала выполните переход в VALIDATION."
                ),
                safe_alternative="Когда реализация будет готова, переведите задачу в VALIDATION и только потом запускайте проверку.",
                raw_data={"reason": "not_in_validation", "requested_action": requested_action},
            )

        if requested_action == "finalize":
            if task.state not in {TASK_STAGE_DONE, TASK_STAGE_REJECTED}:
                if task.state == TASK_STAGE_VALIDATION and task.validation_status == TASK_VALIDATION_STATUS_PASSED:
                    safe_alternative = "Сначала зафиксируйте переход VALIDATION -> DONE, затем делайте финальный ответ."
                else:
                    safe_alternative = "Сначала завершите валидацию и только после этого переходите к финалу."
                return TaskLifecycleConflict(
                    explanation="Нельзя делать финальный ответ до завершения обязательной валидации и перехода в DONE.",
                    safe_alternative=safe_alternative,
                    raw_data={"reason": "not_done", "requested_action": requested_action},
                )

        return None

    def build_refusal_response(self, conflict: TaskLifecycleConflict) -> AgentResponse:
        answer = "\n".join(
            [
                "Не могу выполнить этот шаг: он нарушает жизненный цикл задачи.",
                # Keep the refusal prefix stable so short-term memory can filter stale refusal loops.
                conflict.explanation,
                f"Следующий корректный шаг: {conflict.safe_alternative}",
            ]
        )
        return AgentResponse(
            answer=answer,
            raw_data=conflict.raw_data,
            model="task-lifecycle-guard",
            latency_sec=0.0,
            provider="local",
        )
