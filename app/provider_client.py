import json
import ssl
import urllib.request

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
