from typing import Optional

from app.config import (
    build_ssl_context,
    chat_history_path_from_env,
    get_provider_config,
    load_env_file,
    memory_base_path_from_env,
    positive_int_from_env,
    user_scoped_chat_history_path,
    user_scoped_memory_base_path,
    users_base_path_from_env,
)
from app.messages import constraint_system_message, merge_system_messages
from app.models import AgentRequestOptions, AgentResponse, AgentTokenStats
from app.response_parser import parse_agent_response
from app.services.chat_context_service import ChatContextService
from app.services.chat_history_service import ChatHistoryService
from app.services.invariant_guard_service import InvariantGuardService
from app.services.memory_service import MemoryService
from app.services.personalization_service import PersonalizationService
from app.services.provider_service import ProviderService
from app.services.token_service import TokenAccountingService


class SimpleLLMAgent:
    """
    Public agent API preserved, but internal responsibilities are delegated
    to smaller services (provider/token/context/memory/history).
    """

    def __init__(
        self,
        provider: str,
        api_url: str,
        api_key: str,
        model_candidates: list[str],
        chat_history_path: Optional[str] = None,
        chat_keep_last_n: int = 10,
        chat_summary_batch_size: int = 10,
        chat_summary_enabled: bool = False,
        memory_base_path: str = ".llm_memory",
        users_base_path: str = ".llm_users",
        user_id: Optional[str] = None,
    ) -> None:
        # Public attrs used by CLI (verbose printing, context prints).
        self.user_id = user_id.strip() if user_id and user_id.strip() else None
        self.users_base_path = users_base_path
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.model_candidates = model_candidates
        if self.user_id:
            self.chat_history_path = user_scoped_chat_history_path(
                chat_history_path,
                users_base_path,
                self.user_id,
            )
            resolved_memory_base_path = user_scoped_memory_base_path(
                memory_base_path,
                users_base_path,
                self.user_id,
            )
        else:
            self.chat_history_path = chat_history_path
            resolved_memory_base_path = memory_base_path

        # These are needed for consistency with older code.
        self.chat_keep_last_n = max(1, chat_keep_last_n)
        self.chat_summary_batch_size = max(1, chat_summary_batch_size)

        self.chat_summary_enabled = chat_summary_enabled
        self.context_strategy: str = "full"

        self._provider_service = ProviderService(
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            model_candidates=model_candidates,
            ssl_context=build_ssl_context(),
        )
        self._token_service = TokenAccountingService(self._provider_service)
        self._invariant_guard_service = InvariantGuardService(self._provider_service, provider)
        self._personalization_service = PersonalizationService(
            users_base_path=users_base_path,
            user_id=self.user_id,
        )

        self._chat_history_service = ChatHistoryService(
            chat_history_path=self.chat_history_path,
            chat_keep_last_n=self.chat_keep_last_n,
            chat_summary_batch_size=self.chat_summary_batch_size,
            summary_enabled=chat_summary_enabled,
            provider_service=self._provider_service,
        )
        self._memory_service = MemoryService(
            memory_base_path=resolved_memory_base_path,
            chat_keep_last_n=self.chat_keep_last_n,
        )
        self._context_service = ChatContextService(
            chat_keep_last_n=self.chat_keep_last_n,
            chat_history_service=self._chat_history_service,
            memory_service=self._memory_service,
            personalization_service=self._personalization_service,
        )

    @classmethod
    def from_env(cls, user_id: Optional[str] = None) -> "SimpleLLMAgent":
        provider, api_url, api_key, model_candidates = get_provider_config()
        return cls(
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            model_candidates=model_candidates,
            chat_history_path=chat_history_path_from_env(),
            chat_keep_last_n=positive_int_from_env("LLM_CHAT_KEEP_LAST_N", 10),
            chat_summary_batch_size=positive_int_from_env("LLM_CHAT_SUMMARY_BATCH_SIZE", 10),
            chat_summary_enabled=False,
            memory_base_path=memory_base_path_from_env(),
            users_base_path=users_base_path_from_env(),
            user_id=user_id,
        )

    def set_summary_mode(self, enabled: bool) -> None:
        # Kept for API compatibility.
        self.chat_summary_enabled = enabled
        self._chat_history_service.set_summary_enabled(enabled)

    def set_context_strategy(self, strategy: str) -> None:
        normalized = strategy.lower().strip()
        allowed = {"full", "summary", "sliding", "facts", "branching", "memory"}
        if normalized not in allowed:
            raise ValueError(f"Unknown context strategy: {strategy}. Allowed: {sorted(allowed)}")

        self.context_strategy = normalized
        # Legacy behavior: "--summary" is treated as context_strategy="summary".
        self.chat_summary_enabled = normalized == "summary"
        self._chat_history_service.set_summary_enabled(self.chat_summary_enabled)

    # ---- Memory commands (CLI @mem ...) ----
    def add_short_term_note(self, note: str) -> None:
        self._memory_service.add_short_term_note(note)

    def update_working_task_field(self, field_name: str, value: str) -> None:
        self._memory_service.update_working_task_field(field_name, value)

    def transition_task_state(self, new_state: str) -> str:
        return self._memory_service.transition_task_state(new_state)

    def pause_current_task(self) -> None:
        self._memory_service.pause_current_task()

    def resume_current_task(self) -> None:
        self._memory_service.resume_current_task()

    def update_long_term_memory(self, bucket: str, key: str, value: str) -> None:
        self._memory_service.update_long_term_memory(bucket, key, value)

    def add_long_term_decision(self, decision: str) -> None:
        self._memory_service.add_long_term_decision(decision)

    def add_invariant(self, invariant: str) -> None:
        self._memory_service.add_invariant(invariant)

    def clear_invariants(self) -> None:
        self._memory_service.clear_invariants()

    def clear_memory_layer(self, layer: str) -> None:
        self._memory_service.clear_memory_layer(layer)

    def memory_snapshot(self) -> dict:
        return self._memory_service.memory_snapshot()

    # ---- Personalization (CLI --user-id / @personalization) ----
    def ensure_user_profile(self) -> bool:
        return self._personalization_service.ensure_user_exists()

    def user_profile_needs_interview(self) -> bool:
        return self._personalization_service.needs_interview()

    def save_user_profile_interview(self, answers: dict[str, str]) -> None:
        self._personalization_service.save_interview_answers(answers)

    def update_personalization(self, key: str, value: str) -> None:
        self._personalization_service.update_profile_entries({key: value})

    def personalization_snapshot(self) -> dict:
        return self._personalization_service.snapshot()

    # ---- Branching commands (CLI @checkpoint/@fork/@switch) ----
    def branch_checkpoint(self) -> None:
        if self.context_strategy != "branching":
            raise RuntimeError("branch_checkpoint() is only available in context_strategy='branching'")
        self._memory_service.branch_checkpoint()

    def branch_fork(self) -> None:
        if self.context_strategy != "branching":
            raise RuntimeError("branch_fork() is only available in context_strategy='branching'")
        self._memory_service.branch_fork()

    def branch_switch(self, branch_id: str) -> None:
        if self.context_strategy != "branching":
            raise RuntimeError("branch_switch() is only available in context_strategy='branching'")
        self._memory_service.branch_switch(branch_id)

    def get_branch_info(self) -> str:
        if self.context_strategy != "branching":
            return "Branching is not active."
        return self._memory_service.get_branch_info()

    # ---- Summary ----
    def get_chat_summary(self) -> str:
        # Summary is stored only when using chat history (full/summary strategies).
        # Even if chat history is disabled for the current run, this returns the
        # current in-memory value.
        return self._chat_history_service.get_summary()

    # ---- LLM calls ----
    def ask(self, prompt: str, options: AgentRequestOptions) -> AgentResponse:
        conflict = self._invariant_guard_service.check_request(prompt, self._memory_service.get_invariants())
        if conflict is not None:
            return self._invariant_guard_service.build_refusal_response(conflict)

        messages: list[dict[str, str]] = []
        system_msg = merge_system_messages(
            constraint_system_message(options),
            self._personalization_service.system_message(),
            self._memory_service.invariants_system_message(),
        )
        if system_msg:
            messages.append(system_msg)
        messages.append({"role": "user", "content": prompt})

        data, tried_model, elapsed = self._provider_service.complete(messages, options)
        response = parse_agent_response(data, tried_model, elapsed, self.provider)

        if options.count_tokens:
            token_stats: AgentTokenStats = self._token_service.compute_for_ask(
                data=data,
                tried_model=tried_model,
                system_msg=system_msg,
                options=options,
            )
            response.token_stats = token_stats.__dict__

        return response

    def ask_chat(self, prompt: str, options: AgentRequestOptions) -> AgentResponse:
        strategy = self.context_strategy
        conflict = self._invariant_guard_service.check_request(prompt, self._memory_service.get_invariants())
        if conflict is not None:
            response = self._invariant_guard_service.build_refusal_response(conflict)
            user_msg: dict[str, str] = {"role": "user", "content": prompt}
            assistant_msg: dict[str, str] = {"role": "assistant", "content": response.answer}
            self._context_service.apply_after_turn(strategy, user_msg, assistant_msg, options)
            return response

        payload = self._context_service.build(strategy, prompt, options)

        data, tried_model, elapsed = self._provider_service.complete(payload.payload_messages, options)
        response = parse_agent_response(data, tried_model, elapsed, self.provider)

        if options.count_tokens:
            token_stats = self._token_service.compute_for_chat(
                data=data,
                tried_model=tried_model,
                system_msg=payload.system_msg,
                context_messages=payload.context_messages,
                history_without_current=payload.history_without_current,
                options=options,
            )
            response.token_stats = token_stats.__dict__

        user_msg: dict[str, str] = {"role": "user", "content": prompt}
        assistant_msg: dict[str, str] = {"role": "assistant", "content": response.answer}
        self._context_service.apply_after_turn(strategy, user_msg, assistant_msg, options)

        return response


__all__ = [
    "AgentRequestOptions",
    "SimpleLLMAgent",
    "load_env_file",
]

