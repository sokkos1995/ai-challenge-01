from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.models import AgentResponse
from app.services.rag_service import RagAnswer, RagService


@dataclass
class TaskMemoryState:
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    clarified: list[str] = field(default_factory=list)


class RagChatService:
    def __init__(self, rag_service: RagService) -> None:
        self._rag = rag_service
        self._task_memory = TaskMemoryState()
        self._history: list[dict[str, str]] = []

    @property
    def history_size(self) -> int:
        return len(self._history)

    def memory_snapshot(self) -> dict[str, object]:
        return {
            "goal": self._task_memory.goal,
            "constraints": list(self._task_memory.constraints),
            "terms": list(self._task_memory.terms),
            "clarified": list(self._task_memory.clarified),
            "history_size": self.history_size,
        }

    def ask(self, user_message: str, ask_llm: Callable[[str], AgentResponse]) -> RagAnswer:
        self._update_task_memory(user_message)
        query = self._compose_augmented_query(user_message)
        rag_prompt, contexts, low_relevance = self._rag.build_prompt(query)
        if low_relevance:
            answer = self._rag.low_relevance_answer(contexts)
        else:
            response = ask_llm(rag_prompt)
            answer = self._rag.parse_answer(response.answer, contexts)

        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": answer.answer})
        return answer

    @staticmethod
    def _extract_phrase_after_marker(text: str, marker: str) -> str:
        lowered = text.lower()
        marker_idx = lowered.find(marker.lower())
        if marker_idx < 0:
            return ""
        return text[marker_idx + len(marker) :].strip(" :,-").strip()

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        clean = " ".join(value.strip().split())
        if clean and clean not in items:
            items.append(clean)

    def _update_task_memory(self, user_message: str) -> None:
        normalized = " ".join(user_message.strip().split())
        if not normalized:
            return

        goal = self._extract_phrase_after_marker(normalized, "цель")
        if goal:
            self._task_memory.goal = goal

        for marker in ("ограничение", "ограничения", "нельзя", "только"):
            value = self._extract_phrase_after_marker(normalized, marker)
            if value:
                self._append_unique(self._task_memory.constraints, value)

        for marker in ("термин", "называй", "используй формулировку"):
            value = self._extract_phrase_after_marker(normalized, marker)
            if value:
                self._append_unique(self._task_memory.terms, value)

        if re.search(r"\bуточню\b|\bуточнение\b|\bдобавлю\b|\bважно\b", normalized.lower()):
            self._append_unique(self._task_memory.clarified, normalized)

    def _compose_augmented_query(self, question: str) -> str:
        parts = [question.strip()]
        if self._task_memory.goal:
            parts.append(f"Цель диалога: {self._task_memory.goal}")
        if self._task_memory.constraints:
            parts.append("Ограничения: " + "; ".join(self._task_memory.constraints[-3:]))
        if self._task_memory.terms:
            parts.append("Зафиксированные термины: " + "; ".join(self._task_memory.terms[-3:]))
        if self._task_memory.clarified:
            parts.append("Последние уточнения: " + "; ".join(self._task_memory.clarified[-2:]))
        return "\n".join(part for part in parts if part)
