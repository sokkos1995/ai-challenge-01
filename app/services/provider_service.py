import ssl
import sys
import time
import urllib.error
from typing import Optional

from app.models import AgentRequestOptions
from app.provider_client import post_chat_completion


class ProviderService:
    """
    Provider communication with fallback models and consistent error messages.

    This class returns raw provider payloads; parsing is handled elsewhere.
    """

    def __init__(
        self,
        provider: str,
        api_url: str,
        api_key: str,
        model_candidates: list[str],
        ssl_context: ssl.SSLContext,
    ) -> None:
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.model_candidates = model_candidates
        self.ssl_context = ssl_context

    def complete(
        self, messages: list[dict[str, str]], options: AgentRequestOptions
    ) -> tuple[dict, str, float]:
        data: Optional[dict] = None
        tried_model = self.model_candidates[0]
        request_started = time.perf_counter()
        response_elapsed_sec = 0.0

        try:
            for current_model in self.model_candidates:
                tried_model = current_model
                try:
                    data = post_chat_completion(
                        self.api_url,
                        self.api_key,
                        current_model,
                        messages,
                        self.ssl_context,
                        options,
                    )
                    response_elapsed_sec = time.perf_counter() - request_started
                    if current_model != self.model_candidates[0]:
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
                    if no_endpoints and current_model != self.model_candidates[-1]:
                        continue
                    if exc.code == 403 and "1010" in error_text:
                        raise RuntimeError(
                            "HTTP 403 (Cloudflare 1010): provider blocked your request by policy/region.\n"
                            "Try VPN or another provider endpoint (via LLM_PROVIDER / LLM_API_URL)."
                        ) from exc
                    raise RuntimeError(f"HTTP error {exc.code}: {error_text}") from exc
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
            raise RuntimeError("Request failed: no response from provider.")

        return data, tried_model, response_elapsed_sec
