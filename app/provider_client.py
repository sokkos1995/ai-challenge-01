from __future__ import annotations

import json
import ssl
import urllib.request
from typing import Any

from app.models import AgentRequestOptions


def post_chat_completion(
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    ssl_context: ssl.SSLContext,
    options: AgentRequestOptions,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": options.temperature,
    }
    if options.top_p is not None:
        payload["top_p"] = options.top_p
    if options.top_k is not None:
        payload["top_k"] = options.top_k
    if options.max_output_tokens is not None:
        payload["max_tokens"] = options.max_output_tokens
    if options.stop_sequences:
        payload["stop"] = options.stop_sequences

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = (message.get("role") or "user").strip() or "user"
        content = (message.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _usage_from_cursor_result(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    reasoning_tokens = getattr(usage, "reasoning_tokens", None)
    if reasoning_tokens is not None:
        result["completion_tokens_details"] = {"reasoning_tokens": int(reasoning_tokens)}
    return result


def post_cursor_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    cwd: str,
) -> dict:
    """
    Call Cursor local agent via official cursor-sdk and return an OpenAI-like payload.
    """
    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError as exc:
        raise RuntimeError(
            "Cursor provider requires the optional package cursor-sdk (Python 3.10+).\n"
            "Install: pip install cursor-sdk"
        ) from exc

    prompt = _messages_to_prompt(messages)
    if not prompt:
        raise ValueError("messages must contain at least one non-empty content")

    try:
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=api_key,
                model=model,
                local=LocalAgentOptions(cwd=cwd),
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Cursor agent request failed: {exc}") from exc

    status = getattr(result, "status", None)
    answer = getattr(result, "result", None)
    if status != "finished":
        raise RuntimeError(
            f"Cursor agent run did not finish successfully (status={status!r})."
        )
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Cursor agent returned empty assistant content.")

    resolved_model = model
    result_model = getattr(result, "model", None)
    if result_model is not None:
        model_id = getattr(result_model, "id", None)
        if isinstance(model_id, str) and model_id.strip():
            resolved_model = model_id.strip()

    return {
        "id": getattr(result, "id", None),
        "object": "chat.completion",
        "model": resolved_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": answer},
            }
        ],
        "usage": _usage_from_cursor_result(getattr(result, "usage", None)),
    }
