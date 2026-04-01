import ssl
from typing import Optional

from app.models import AgentRequestOptions, AgentTokenStats
from app.provider_client import post_chat_completion
from app.services.provider_service import ProviderService


class TokenAccountingService:
    def __init__(self, provider_service: ProviderService) -> None:
        self._provider = provider_service
        self._ssl_context: ssl.SSLContext = provider_service.ssl_context
        self._system_only_prompt_tokens_cache: dict[tuple[str, str], int] = {}

    @staticmethod
    def _extract_prompt_completion_total_tokens(
        data: dict,
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        usage = data.get("usage", {}) or {}
        return usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")

    def _dry_run_prompt_tokens(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
    ) -> Optional[int]:
        dry_options = AgentRequestOptions(
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            response_format=options.response_format,
            max_output_tokens=1,
            stop_sequences=[],
            finish_instruction=options.finish_instruction,
            count_tokens=False,
        )
        try:
            data = post_chat_completion(
                api_url=self._provider.api_url,
                api_key=self._provider.api_key,
                model=model,
                messages=messages,
                ssl_context=self._ssl_context,
                options=dry_options,
            )
        except Exception:
            return None

        prompt_tokens, _, _ = self._extract_prompt_completion_total_tokens(data)
        return prompt_tokens

    def _system_only_prompt_tokens(
        self,
        model: str,
        system_msg: dict[str, str],
        options: AgentRequestOptions,
    ) -> Optional[int]:
        cache_key = (model, system_msg.get("content", ""))
        cached = self._system_only_prompt_tokens_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt_tokens = self._dry_run_prompt_tokens(
            model=model,
            messages=[system_msg],
            options=options,
        )
        if prompt_tokens is None:
            return None

        self._system_only_prompt_tokens_cache[cache_key] = prompt_tokens
        return prompt_tokens

    def compute_for_ask(
        self,
        data: dict,
        tried_model: str,
        system_msg: Optional[dict[str, str]],
        options: AgentRequestOptions,
    ) -> AgentTokenStats:
        prompt_tokens_full, completion_tokens, total_tokens = self._extract_prompt_completion_total_tokens(data)

        system_only_tokens: Optional[int] = None
        if system_msg:
            system_only_tokens = self._system_only_prompt_tokens(tried_model, system_msg, options)

        dialog_history_tokens: Optional[int] = 0
        if prompt_tokens_full is None:
            current_request_tokens: Optional[int] = None
        elif system_only_tokens is None:
            current_request_tokens = prompt_tokens_full
        else:
            current_request_tokens = max(0, prompt_tokens_full - system_only_tokens)

        return AgentTokenStats(
            current_request_tokens=current_request_tokens,
            dialog_history_tokens=dialog_history_tokens,
            response_model_tokens=completion_tokens,
            prompt_tokens_total=prompt_tokens_full,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def compute_for_chat(
        self,
        data: dict,
        tried_model: str,
        system_msg: dict[str, str],
        context_messages: list[dict[str, str]],
        history_without_current: list[dict[str, str]],
        options: AgentRequestOptions,
    ) -> AgentTokenStats:
        prompt_tokens_full, completion_tokens, total_tokens = self._extract_prompt_completion_total_tokens(data)

        prompt_tokens_history_with_system = self._dry_run_prompt_tokens(
            model=tried_model,
            messages=[system_msg] + context_messages + history_without_current,
            options=options,
        )
        system_only_tokens = self._system_only_prompt_tokens(tried_model, system_msg, options)

        if prompt_tokens_full is None or prompt_tokens_history_with_system is None:
            current_request_tokens: Optional[int] = None
        else:
            current_request_tokens = max(0, prompt_tokens_full - prompt_tokens_history_with_system)

        if prompt_tokens_history_with_system is None or system_only_tokens is None:
            dialog_history_tokens: Optional[int] = None
        else:
            dialog_history_tokens = max(0, prompt_tokens_history_with_system - system_only_tokens)

        return AgentTokenStats(
            current_request_tokens=current_request_tokens,
            dialog_history_tokens=dialog_history_tokens,
            response_model_tokens=completion_tokens,
            prompt_tokens_total=prompt_tokens_full,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
