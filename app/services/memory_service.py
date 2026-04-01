import json
from typing import Optional

from app.messages import facts_system_message
from app.models import (
    BranchState,
    FactsState,
    LongTermMemory,
    ShortTermMemory,
    WorkingMemory,
)
from app.storage import (
    load_long_term_memory,
    load_short_term_memory,
    load_working_memory,
    save_long_term_memory,
    save_short_term_memory,
    save_working_memory,
)


class MemoryService:
    def __init__(self, memory_base_path: str, chat_keep_last_n: int) -> None:
        self._memory_base_path = memory_base_path
        self._chat_keep_last_n = max(1, chat_keep_last_n)

        self._facts_state: FactsState = FactsState()
        self._branch_state: BranchState = BranchState()

        self._short_memory: ShortTermMemory = ShortTermMemory()
        self._working_memory: WorkingMemory = WorkingMemory()
        self._long_memory: LongTermMemory = LongTermMemory()

        self._memory_loaded = False

    def _ensure_memory_loaded(self) -> None:
        if self._memory_loaded:
            return
        self._short_memory = load_short_term_memory(self._memory_base_path)
        self._working_memory = load_working_memory(self._memory_base_path)
        self._long_memory = load_long_term_memory(self._memory_base_path)

        if len(self._short_memory.dialog_tail) > self._chat_keep_last_n:
            self._short_memory.dialog_tail = self._short_memory.dialog_tail[-self._chat_keep_last_n :]

        self._memory_loaded = True

    def _save_all_memory_layers(self) -> None:
        save_short_term_memory(self._memory_base_path, self._short_memory)
        save_working_memory(self._memory_base_path, self._working_memory)
        save_long_term_memory(self._memory_base_path, self._long_memory)

    @staticmethod
    def _extract_sticky_fact_update(user_message: str) -> Optional[tuple[str, str]]:
        raw = user_message.strip()
        if not raw:
            return None

        lower = raw.lower()

        def value_after_first_colon(text: str) -> str:
            if ":" in text:
                return text.split(":", 1)[1].strip()
            return text.strip()

        value = value_after_first_colon(raw) if "запомни" in lower else raw

        if "стек" in lower:
            return ("stack", value)
        if "дедлайн" in lower:
            return ("deadline", value)
        if "ограничен" in lower:
            return ("constraints", value)
        if "предпочит" in lower or "коротк" in lower or "списком" in lower:
            return ("preference", value)
        if "формат" in lower:
            return ("output_format", value)
        if "цель" in lower:
            return ("goal", value)

        return None

    def update_facts_from_user_message(self, user_message: str) -> None:
        update = self._extract_sticky_fact_update(user_message)
        if update is None:
            return
        key, value = update
        cleaned_value = value.strip()
        if cleaned_value:
            self._facts_state.facts[key] = cleaned_value

    def _branch_base_messages(self) -> list[dict[str, str]]:
        bs = self._branch_state
        if bs.active_branch is None:
            return list(bs.root_messages)
        return list(bs.root_messages) + list(bs.branches.get(bs.active_branch, []))

    def branch_checkpoint(self) -> None:
        self._branch_state.checkpoint_messages = self._branch_base_messages()

    def branch_fork(self) -> None:
        bs = self._branch_state
        checkpoint = bs.checkpoint_messages if bs.checkpoint_messages is not None else self._branch_base_messages()
        bs.root_messages = list(checkpoint)
        bs.branches = {"1": [], "2": []}
        bs.active_branch = "1"
        bs.checkpoint_messages = None

    def branch_switch(self, branch_id: str) -> None:
        normalized = branch_id.strip()
        if normalized not in {"1", "2"}:
            raise ValueError("branch_id must be '1' or '2'")
        bs = self._branch_state
        if not bs.branches:
            raise RuntimeError("No branches yet. Run branch_fork() first.")
        bs.active_branch = normalized

    def get_branch_info(self) -> str:
        bs = self._branch_state
        checkpoint_set = bs.checkpoint_messages is not None
        active = bs.active_branch or "(pre-fork)"
        b1_len = len(bs.branches.get("1", []))
        b2_len = len(bs.branches.get("2", []))
        return (
            f"checkpoint_set={checkpoint_set}, active_branch={active}, "
            f"branch1_messages={b1_len}, branch2_messages={b2_len}, "
            f"root_messages={len(bs.root_messages)}"
        )

    def _memory_layers_system_message(self) -> dict[str, str]:
        task = self._working_memory.current_task
        payload = {
            "short_term_notes": self._short_memory.notes,
            "working_task": {
                "task": task.task,
                "state": task.state,
                "step": task.step,
                "total": task.total,
                "plan": task.plan,
                "done": task.done,
                "notes": task.notes,
            },
            "long_term_profile": self._long_memory.profile,
            "long_term_decisions": self._long_memory.decisions,
            "long_term_knowledge": self._long_memory.knowledge,
        }
        return {
            "role": "system",
            "content": (
                "Memory layers for this assistant (explicitly separated):\n"
                f"{json.dumps(payload, ensure_ascii=True)}\n\n"
                "Use memory as supporting context. If memory conflicts with new user input, ask clarification."
            ),
        }

    def memory_layers_system_message(self) -> dict[str, str]:
        self._ensure_memory_loaded()
        return self._memory_layers_system_message()

    def add_short_term_note(self, note: str) -> None:
        self._ensure_memory_loaded()
        clean = note.strip()
        if not clean:
            return
        self._short_memory.notes.append(clean)
        self._save_all_memory_layers()

    def update_working_task_field(self, field_name: str, value: str) -> None:
        self._ensure_memory_loaded()
        task = self._working_memory.current_task
        clean = value.strip()
        if field_name == "task":
            task.task = clean
        elif field_name == "state":
            task.state = clean or "new"
        elif field_name == "step":
            task.step = int(clean) if clean else 0
        elif field_name == "total":
            task.total = int(clean) if clean else 0
        elif field_name == "plan+":
            if clean:
                task.plan.append(clean)
        elif field_name == "done+":
            if clean:
                task.done.append(clean)
        elif field_name == "note+":
            if clean:
                task.notes.append(clean)
        else:
            raise ValueError("Unknown work field. Use: task, state, step, total, plan+, done+, note+.")
        self._save_all_memory_layers()

    def update_long_term_memory(self, bucket: str, key: str, value: str) -> None:
        self._ensure_memory_loaded()
        clean_key = key.strip()
        clean_value = value.strip()
        if bucket == "profile":
            if not clean_key:
                raise ValueError("profile key is empty")
            self._long_memory.profile[clean_key] = clean_value
        elif bucket == "knowledge":
            if not clean_key:
                raise ValueError("knowledge key is empty")
            self._long_memory.knowledge[clean_key] = clean_value
        else:
            raise ValueError("Unknown long-term bucket. Use profile or knowledge.")
        self._save_all_memory_layers()

    def add_long_term_decision(self, decision: str) -> None:
        self._ensure_memory_loaded()
        clean = decision.strip()
        if not clean:
            return
        self._long_memory.decisions.append(clean)
        self._save_all_memory_layers()

    def clear_memory_layer(self, layer: str) -> None:
        self._ensure_memory_loaded()
        normalized = layer.strip().lower()
        if normalized == "short":
            self._short_memory = ShortTermMemory()
        elif normalized == "work":
            self._working_memory = WorkingMemory()
        elif normalized == "long":
            self._long_memory = LongTermMemory()
        elif normalized == "all":
            self._short_memory = ShortTermMemory()
            self._working_memory = WorkingMemory()
            self._long_memory = LongTermMemory()
        else:
            raise ValueError("Unknown layer. Use: short, work, long, all.")
        self._save_all_memory_layers()

    def memory_snapshot(self) -> dict:
        self._ensure_memory_loaded()
        task = self._working_memory.current_task
        return {
            "short_term": {
                "dialog_tail_size": len(self._short_memory.dialog_tail),
                "notes": list(self._short_memory.notes),
            },
            "working": {
                "task": task.task,
                "state": task.state,
                "step": task.step,
                "total": task.total,
                "plan": list(task.plan),
                "done": list(task.done),
                "notes": list(task.notes),
            },
            "long_term": {
                "profile": dict(self._long_memory.profile),
                "decisions": list(self._long_memory.decisions),
                "knowledge": dict(self._long_memory.knowledge),
            },
        }

    def facts_context_messages(self) -> list[dict[str, str]]:
        return [facts_system_message(self._facts_state.facts)]

    def facts_build_context_with_user(self, user_msg: dict[str, str]) -> list[dict[str, str]]:
        combined = list(self._facts_state.history) + [user_msg]
        return combined[-self._chat_keep_last_n :]

    def facts_history_without_current(self, context_with_current: list[dict[str, str]]) -> list[dict[str, str]]:
        return context_with_current[:-1]

    def facts_update_after_turn(self, user_msg: dict[str, str], assistant_msg: dict[str, str]) -> None:
        combined_after = list(self._facts_state.history) + [user_msg, assistant_msg]
        self._facts_state.history = combined_after[-self._chat_keep_last_n :]

    def branching_history_without_current(self) -> list[dict[str, str]]:
        return self._branch_base_messages()

    def branching_update_after_turn(self, user_msg: dict[str, str], assistant_msg: dict[str, str]) -> None:
        bs = self._branch_state
        if bs.active_branch is None:
            bs.root_messages.extend([user_msg, assistant_msg])
        else:
            bs.branches.setdefault(bs.active_branch, []).extend([user_msg, assistant_msg])

    def memory_context_messages(self) -> list[dict[str, str]]:
        return [self._memory_layers_system_message()]

    def memory_build_context_with_user(self, user_msg: dict[str, str]) -> list[dict[str, str]]:
        self._ensure_memory_loaded()
        combined = list(self._short_memory.dialog_tail) + [user_msg]
        return combined[-self._chat_keep_last_n :]

    def memory_update_after_turn(self, user_msg: dict[str, str], assistant_msg: dict[str, str]) -> None:
        self._ensure_memory_loaded()
        combined_after = list(self._short_memory.dialog_tail) + [user_msg, assistant_msg]
        self._short_memory.dialog_tail = combined_after[-self._chat_keep_last_n :]
        self._save_all_memory_layers()
