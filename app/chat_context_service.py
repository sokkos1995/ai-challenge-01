from dataclasses import dataclass
from typing import Optional

from app.messages import chat_session_system_message
from app.models import AgentRequestOptions
from app.chat_history_service import ChatHistoryService
from app.memory_service import MemoryService


@dataclass
class ChatPayload:
    system_msg: dict[str, str]
    context_messages: list[dict[str, str]]
    history_without_current: list[dict[str, str]]
    payload_messages: list[dict[str, str]]


class ChatContextService:
    def __init__(
        self,
        chat_keep_last_n: int,
        chat_history_service: ChatHistoryService,
        memory_service: MemoryService,
    ) -> None:
        self._chat_keep_last_n = max(1, chat_keep_last_n)
        self._chat_history = chat_history_service
        self._memory = memory_service
        self._sliding_history: list[dict[str, str]] = []

    def build(self, strategy: str, prompt: str, options: AgentRequestOptions) -> ChatPayload:
        user_msg: dict[str, str] = {"role": "user", "content": prompt}
        system_msg = chat_session_system_message(options)

        if strategy in {"full", "summary"}:
            history_without_current = self._chat_history.get_history_without_current()
            context_messages: list[dict[str, str]] = []
            summary_msg = self._chat_history.maybe_build_summary_system_message()
            if summary_msg:
                context_messages.append(summary_msg)
            payload_messages = [system_msg] + context_messages + history_without_current + [user_msg]
            return ChatPayload(
                system_msg=system_msg,
                context_messages=context_messages,
                history_without_current=history_without_current,
                payload_messages=payload_messages,
            )

        if strategy == "sliding":
            combined = list(self._sliding_history) + [user_msg]
            context_with_current = combined[-self._chat_keep_last_n :]
            payload_messages = [system_msg] + context_with_current
            history_without_current = context_with_current[:-1]
            return ChatPayload(
                system_msg=system_msg,
                context_messages=[],
                history_without_current=history_without_current,
                payload_messages=payload_messages,
            )

        if strategy == "facts":
            self._memory.update_facts_from_user_message(prompt)
            context_messages = self._memory.facts_context_messages()
            context_with_current = self._memory.facts_build_context_with_user(user_msg)
            payload_messages = [system_msg] + context_messages + context_with_current
            history_without_current = self._memory.facts_history_without_current(context_with_current)
            return ChatPayload(
                system_msg=system_msg,
                context_messages=context_messages,
                history_without_current=history_without_current,
                payload_messages=payload_messages,
            )

        if strategy == "branching":
            history_without_current = self._memory.branching_history_without_current()
            payload_messages = [system_msg] + history_without_current + [user_msg]
            return ChatPayload(
                system_msg=system_msg,
                context_messages=[],
                history_without_current=history_without_current,
                payload_messages=payload_messages,
            )

        if strategy == "memory":
            # Memory layers system message is inserted as explicit context.
            context_messages = [self._memory.memory_layers_system_message()]
            context_with_current = self._memory.memory_build_context_with_user(user_msg)
            payload_messages = [system_msg] + context_messages + context_with_current
            history_without_current = context_with_current[:-1]
            return ChatPayload(
                system_msg=system_msg,
                context_messages=context_messages,
                history_without_current=history_without_current,
                payload_messages=payload_messages,
            )

        raise RuntimeError(f"Unsupported context strategy: {strategy}")

    def apply_after_turn(
        self,
        strategy: str,
        user_msg: dict[str, str],
        assistant_msg: dict[str, str],
        options: AgentRequestOptions,
    ) -> None:
        if strategy in {"full", "summary"}:
            self._chat_history.append_turn_and_maybe_compress(user_msg, assistant_msg, options)
            return

        if strategy == "sliding":
            combined_after = list(self._sliding_history) + [user_msg, assistant_msg]
            self._sliding_history = combined_after[-self._chat_keep_last_n :]
            return

        if strategy == "facts":
            self._memory.facts_update_after_turn(user_msg, assistant_msg)
            return

        if strategy == "branching":
            self._memory.branching_update_after_turn(user_msg, assistant_msg)
            return

        if strategy == "memory":
            self._memory.memory_update_after_turn(user_msg, assistant_msg)
            return

        raise RuntimeError(f"Unsupported context strategy: {strategy}")

