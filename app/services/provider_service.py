from __future__ import annotations

import ssl
import sys
import time
import urllib.error
from typing import Optional

from app.config import cursor_cwd_from_env
from app.models import AgentRequestOptions
from app.provider_client import post_chat_completion, post_cursor_completion
from app.services.llm_guard_service import LlmGuardService


class ProviderService:
    """
    Provider communication with fallback models and consistent error messages.

    This class returns raw provider payloads; parsing is handled elsewhere.
    All completions pass through in-process LLM Input/Output Guard (day_48)
    unless disabled via ``LLM_INPUT_GUARD`` / ``LLM_OUTPUT_GUARD``.
    """

    def __init__(
        self,
        provider: str,
        api_url: str,
        api_key: str,
        model_candidates: list[str],
        ssl_context: ssl.SSLContext,
        llm_guard: LlmGuardService | None = None,
    ) -> None:
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.model_candidates = model_candidates
        self.ssl_context = ssl_context
        self._llm_guard = llm_guard if llm_guard is not None else LlmGuardService.from_env()

    def complete(
        self,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
        *,
        model_candidates: list[str] | None = None,
    ) -> tuple[dict, str, float]:
        candidates = model_candidates if model_candidates is not None else self.model_candidates
        if not candidates:
            raise ValueError("model_candidates must not be empty")

        guarded_messages = self._llm_guard.prepare_messages(messages)

        data: Optional[dict] = None
        tried_model = candidates[0]
        request_started = time.perf_counter()
        response_elapsed_sec = 0.0

        try:
            for current_model in candidates:
                tried_model = current_model
                try:
                    if self.provider == "cursor":
                        data = post_cursor_completion(
                            api_key=self.api_key,
                            model=current_model,
                            messages=guarded_messages,
                            cwd=cursor_cwd_from_env(),
                        )
                    else:
                        data = post_chat_completion(
                            self.api_url,
                            self.api_key,
                            current_model,
                            guarded_messages,
                            self.ssl_context,
                            options,
                        )
                    response_elapsed_sec = time.perf_counter() - request_started
                    if current_model != candidates[0]:
                        print(
                            f"Info: primary model unavailable, used fallback: {current_model}",
                            file=sys.stderr,
                        )
                    break
                except urllib.error.HTTPError as exc:
                    error_text = exc.read().decode("utf-8", errors="replace")
                    no_endpoints = exc.code == 404 and (
                        "No endpoints found for" in error_text
                        or "model_not_found" in error_text
                        or "does not exist" in error_text
                    )
                    if no_endpoints and current_model != candidates[-1]:
                        continue
                    if exc.code == 403 and "1010" in error_text:
                        raise RuntimeError(
                            "HTTP 403 (Cloudflare 1010): provider blocked your request by policy/region.\n"
                            "Try VPN or another provider endpoint (via LLM_PROVIDER / LLM_API_URL)."
                        ) from exc
                    raise RuntimeError(f"HTTP error {exc.code}: {error_text}") from exc
                except RuntimeError as exc:
                    if (
                        self.provider == "cursor"
                        and current_model != candidates[-1]
                        and _is_cursor_model_unavailable(str(exc))
                    ):
                        continue
                    raise
        except ssl.SSLCertVerificationError as exc:
            raise RuntimeError(
                "SSL certificate verification failed.\n"
                "Try one of:\n"
                "1) pip install certifi\n"
                "2) export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')\n"
                f"Details: {exc}"
            ) from exc

        if not data:
            if self.provider == "openrouter":
                raise RuntimeError(
                    f"Request failed: no response from OpenRouter (last model: {tried_model}).\n"
                    "Try free Groq provider:\n"
                    "export LLM_PROVIDER=groq\n"
                    "export GROQ_API_KEY=your_key"
                )
            if self.provider == "cursor":
                raise RuntimeError(
                    f"Request failed: no response from Cursor (last model: {tried_model}).\n"
                    "Install cursor-sdk and set CURSOR_API_KEY:\n"
                    "pip install cursor-sdk\n"
                    "export LLM_PROVIDER=cursor\n"
                    "export CURSOR_API_KEY=your_key"
                )
            raise RuntimeError("Request failed: no response from provider.")

        data = self._llm_guard.filter_response_payload(data)
        return data, tried_model, response_elapsed_sec


def _is_cursor_model_unavailable(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "model_not_found",
            "does not exist",
            "unknown model",
            "invalid model",
            "model not available",
            "model unavailable",
        )
    )
