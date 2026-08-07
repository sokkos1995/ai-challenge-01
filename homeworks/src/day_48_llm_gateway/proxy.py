"""LLM upstream proxy: mock (default) or OpenAI-compatible live call."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

from .cost import rough_token_count
from .output_guard import KNOWN_SYSTEM_SNIPPETS


@dataclass(frozen=True)
class ProxyUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ProxyResult:
    answer: str
    model: str
    usage: ProxyUsage
    live: bool


class Completer(Protocol):
    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> ProxyResult: ...


DEFAULT_SYSTEM = KNOWN_SYSTEM_SNIPPETS[0] + " " + KNOWN_SYSTEM_SNIPPETS[1]


def mock_complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> ProxyResult:
    """Deterministic echo for tests / offline demo."""
    user_bits = [m.get("content", "") for m in messages if m.get("role") == "user"]
    prompt = "\n".join(user_bits).strip() or "(empty)"
    answer = f"[mock] Echo: {prompt[:200]}"
    used_model = model or "mock-gateway"
    p_tok = rough_token_count("\n".join(m.get("content", "") for m in messages))
    c_tok = rough_token_count(answer)
    return ProxyResult(
        answer=answer,
        model=used_model,
        usage=ProxyUsage(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
        ),
        live=False,
    )


def live_complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> ProxyResult:
    """Call OpenAI-compatible chat/completions via app.provider_client."""
    from app.config import build_ssl_context, get_provider_config, load_env_file
    from app.models import AgentRequestOptions
    from app.provider_client import post_chat_completion

    load_env_file()
    provider, api_url, api_key, model_candidates = get_provider_config()
    if provider == "cursor":
        raise RuntimeError("GATEWAY live mode does not support LLM_PROVIDER=cursor; use openrouter/groq")
    if not api_key:
        raise RuntimeError("No LLM API key configured for live gateway proxy")

    used_model = model or model_candidates[0]
    # Prepend gateway system if missing
    has_system = any(m.get("role") == "system" for m in messages)
    payload_messages = list(messages)
    if not has_system:
        payload_messages = [{"role": "system", "content": DEFAULT_SYSTEM}, *payload_messages]

    data = post_chat_completion(
        api_url,
        api_key,
        used_model,
        payload_messages,
        build_ssl_context(),
        AgentRequestOptions(),
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Empty choices from upstream LLM")
    message = choices[0].get("message") or {}
    answer = str(message.get("content") or "")
    usage_raw = data.get("usage") or {}
    p_tok = int(usage_raw.get("prompt_tokens") or rough_token_count(str(payload_messages)))
    c_tok = int(usage_raw.get("completion_tokens") or rough_token_count(answer))
    total = int(usage_raw.get("total_tokens") or (p_tok + c_tok))
    return ProxyResult(
        answer=answer,
        model=str(data.get("model") or used_model),
        usage=ProxyUsage(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=total),
        live=True,
    )


def resolve_completer(*, live: bool | None = None) -> Completer:
    if live is None:
        live = os.getenv("GATEWAY_LIVE", "").strip().lower() in {"1", "true", "yes"}
    return live_complete if live else mock_complete


# Mutable hook for tests
_active_completer: Optional[Completer] = None


def set_completer(completer: Completer | None) -> None:
    global _active_completer
    _active_completer = completer


def get_completer() -> Completer:
    if _active_completer is not None:
        return _active_completer
    return resolve_completer()
