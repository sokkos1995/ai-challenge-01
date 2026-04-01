import sys
from typing import Optional

from app.messages import chat_summary_system_message, chat_summary_update_messages
from app.models import AgentRequestOptions
from app.response_parser import parse_agent_response
from app.services.provider_service import ProviderService
from app.storage import load_chat_messages, load_chat_summary, save_chat_state


class ChatHistoryService:
    def __init__(
        self,
        chat_history_path: Optional[str],
        chat_keep_last_n: int,
        chat_summary_batch_size: int,
        summary_enabled: bool,
        provider_service: ProviderService,
    ) -> None:
        self._chat_history_path = chat_history_path
        self._chat_keep_last_n = max(1, chat_keep_last_n)
        self._chat_summary_batch_size = max(1, chat_summary_batch_size)
        self._summary_enabled = summary_enabled
        self._provider = provider_service

        self._chat_history: list[dict[str, str]] = []
        self._chat_summary: str = ""
        self._loaded = False

    def set_summary_enabled(self, enabled: bool) -> None:
        self._summary_enabled = enabled

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._chat_history_path:
            self._chat_history = load_chat_messages(self._chat_history_path)
            self._chat_summary = load_chat_summary(self._chat_history_path)
        else:
            self._chat_history = []
            self._chat_summary = ""
        self._loaded = True

    def get_history_without_current(self) -> list[dict[str, str]]:
        self.ensure_loaded()
        return list(self._chat_history)

    def get_summary(self) -> str:
        self.ensure_loaded()
        return self._chat_summary

    def maybe_build_summary_system_message(self) -> Optional[dict[str, str]]:
        if not self._summary_enabled:
            return None
        summary_msg = chat_summary_system_message(self.get_summary())
        return summary_msg

    def _summarize_history_chunk(
        self, chunk: list[dict[str, str]], current_summary: str, options: AgentRequestOptions
    ) -> str:
        summary_options = AgentRequestOptions(
            temperature=min(options.temperature, 0.3),
            top_p=options.top_p,
            top_k=options.top_k,
            response_format=None,
            max_output_tokens=500,
            stop_sequences=[],
            finish_instruction=None,
            count_tokens=False,
        )
        messages = chat_summary_update_messages(current_summary, chunk)
        data, tried_model, elapsed = self._provider.complete(messages, summary_options)
        response = parse_agent_response(data, tried_model, elapsed, self._provider.provider)
        return response.answer.strip()

    def _compress_chat_history_if_needed(self, options: AgentRequestOptions) -> None:
        if not self._summary_enabled:
            return
        if self._chat_summary_batch_size <= 0:
            return

        while len(self._chat_history) - self._chat_keep_last_n >= self._chat_summary_batch_size:
            chunk = self._chat_history[: self._chat_summary_batch_size]
            self._chat_summary = self._summarize_history_chunk(chunk, self._chat_summary, options)
            self._chat_history = self._chat_history[self._chat_summary_batch_size :]

    def append_turn_and_maybe_compress(
        self,
        user_msg: dict[str, str],
        assistant_msg: dict[str, str],
        options: AgentRequestOptions,
    ) -> None:
        self.ensure_loaded()
        self._chat_history.append(user_msg)
        self._chat_history.append(assistant_msg)

        try:
            self._compress_chat_history_if_needed(options)
        except Exception:
            print(f"Warning: could not compress chat history: {sys.exc_info()[1]}", file=sys.stderr)

        if self._chat_history_path:
            try:
                save_chat_state(self._chat_history_path, self._chat_history, self._chat_summary)
            except OSError:
                pass
