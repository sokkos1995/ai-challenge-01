from dataclasses import dataclass
import json
import os
import sqlite3
import ssl
import sys
import time
from typing import Optional, Tuple
import urllib.error
import urllib.request


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
OPENROUTER_FALLBACK_MODELS = [
    "qwen/qwen-2.5-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"
GROQ_FALLBACK_MODELS = [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
]


@dataclass
class AgentRequestOptions:
    temperature: float
    top_p: Optional[float]
    top_k: Optional[int]
    response_format: Optional[str]
    max_output_tokens: Optional[int]
    stop_sequences: list[str]
    finish_instruction: Optional[str]
    count_tokens: bool = False


@dataclass
class AgentTokenStats:
    # tokens for current user message contribution in the current request (excludes system prompt)
    current_request_tokens: Optional[int]
    # tokens for previous chat messages contribution in the current request (excludes system prompt)
    dialog_history_tokens: Optional[int]
    # tokens produced by the model (completion tokens)
    response_model_tokens: Optional[int]
    # prompt_tokens as returned by the provider for the actual request
    prompt_tokens_total: Optional[int]
    # completion_tokens as returned by the provider for the actual request
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass
class AgentResponse:
    answer: str
    raw_data: dict
    model: str
    latency_sec: float
    provider: str
    token_stats: Optional[dict] = None


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def build_ssl_context() -> ssl.SSLContext:
    cert_file = os.getenv("SSL_CERT_FILE")
    if cert_file:
        return ssl.create_default_context(cafile=cert_file)

    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def get_provider_config() -> tuple[str, str, str, list[str]]:
    provider = os.getenv("LLM_PROVIDER", "auto").lower().strip()
    llm_api_key = os.getenv("LLM_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if provider == "auto":
        if llm_api_key:
            provider = "openrouter"
        elif groq_key:
            provider = "groq"
        elif openrouter_key:
            provider = "openrouter"
        else:
            raise RuntimeError(
                "Error: set one of API keys: LLM_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY.\n"
                "Tip: if OpenRouter returns 404 for free models, create free GROQ_API_KEY and run again."
            )

    if provider == "groq":
        api_key = llm_api_key or groq_key
        if not api_key:
            raise RuntimeError("Error: set GROQ_API_KEY (or LLM_API_KEY with LLM_PROVIDER=groq).")
        api_url = os.getenv("LLM_API_URL", GROQ_API_URL)
        model = os.getenv("LLM_MODEL", GROQ_DEFAULT_MODEL)
        fallback_raw = os.getenv("LLM_FALLBACK_MODELS", ",".join(GROQ_FALLBACK_MODELS))
    elif provider == "openrouter":
        api_key = llm_api_key or openrouter_key
        if not api_key:
            raise RuntimeError("Error: set OPENROUTER_API_KEY (or LLM_API_KEY with LLM_PROVIDER=openrouter).")
        api_url = os.getenv("LLM_API_URL", OPENROUTER_API_URL)
        model = os.getenv("LLM_MODEL", OPENROUTER_DEFAULT_MODEL)
        fallback_raw = os.getenv("LLM_FALLBACK_MODELS", ",".join(OPENROUTER_FALLBACK_MODELS))
    else:
        raise RuntimeError("Error: LLM_PROVIDER must be auto, openrouter, or groq.")

    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]
    model_candidates = [model] + [m for m in fallback_models if m != model]
    return provider, api_url, api_key, model_candidates


def _constraint_system_message(options: AgentRequestOptions) -> Optional[dict[str, str]]:
    constraints = []
    if options.response_format:
        constraints.append(f"Output format requirement: {options.response_format}")
    if options.max_output_tokens is not None:
        constraints.append(
            f"Length limit: keep the answer within approximately {options.max_output_tokens} output tokens."
        )
    if options.finish_instruction:
        constraints.append(f"Finish condition: {options.finish_instruction}")
    if not constraints:
        return None
    return {"role": "system", "content": "\n".join(constraints)}


def _chat_session_system_message(options: AgentRequestOptions) -> dict[str, str]:
    """Tells the model that prior user/assistant turns in the payload are real persisted chat."""
    parts = [
        "This request includes the full conversation so far with this user: every user and assistant "
        "message before the final user message is prior dialogue from the same session (restored across "
        "restarts). Use that history as ground truth. Do not claim you cannot see, remember, or access "
        "those earlier messages."
    ]
    extra = _constraint_system_message(options)
    if extra:
        parts.append(extra["content"])
    return {"role": "system", "content": "\n\n".join(parts)}


def _ensure_chat_db_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _init_chat_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL
        )
        """
    )


def load_chat_messages(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    try:
        _init_chat_db(conn)
        rows = conn.execute(
            "SELECT role, content FROM chat_messages ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    return [{"role": str(role), "content": str(content)} for role, content in rows]


def save_chat_messages(path: str, messages: list[dict[str, str]]) -> None:
    _ensure_chat_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_chat_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM chat_messages")
        conn.executemany(
            "INSERT INTO chat_messages (role, content) VALUES (?, ?)",
            [(m["role"], m["content"]) for m in messages],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _post_chat_completion(
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


def _chat_history_path_from_env() -> Optional[str]:
    raw = os.getenv("LLM_CHAT_HISTORY_PATH", ".llm_chat_history.db")
    stripped = raw.strip() if raw else ""
    return stripped if stripped else None


class SimpleLLMAgent:
    """LLM agent that encapsulates provider communication and retries."""

    def __init__(
        self,
        provider: str,
        api_url: str,
        api_key: str,
        model_candidates: list[str],
        chat_history_path: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.model_candidates = model_candidates
        self.chat_history_path = chat_history_path
        self.ssl_context = build_ssl_context()
        self._chat_history: list[dict[str, str]] = []
        self._chat_history_loaded = False
        # Cache system-only prompt token counts to reduce extra dry-run calls.
        # Key: (model, system_message_content)
        self._system_only_prompt_tokens_cache: dict[tuple[str, str], int] = {}

    @classmethod
    def from_env(cls) -> "SimpleLLMAgent":
        provider, api_url, api_key, model_candidates = get_provider_config()
        return cls(provider, api_url, api_key, model_candidates, _chat_history_path_from_env())

    def _ensure_chat_history_loaded(self) -> None:
        if self._chat_history_loaded:
            return
        if self.chat_history_path:
            self._chat_history = load_chat_messages(self.chat_history_path)
        else:
            self._chat_history = []
        self._chat_history_loaded = True

    @staticmethod
    def _extract_prompt_completion_total_tokens(data: dict) -> tuple[Optional[int], Optional[int], Optional[int]]:
        usage = data.get("usage", {}) or {}
        return usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")

    def _dry_run_prompt_tokens(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
    ) -> Optional[int]:
        """
        Cheap call to estimate prompt tokens for provided messages.
        Completion is limited to keep cost low; output is ignored.
        """

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
            data = _post_chat_completion(
                api_url=self.api_url,
                api_key=self.api_key,
                model=model,
                messages=messages,
                ssl_context=self.ssl_context,
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

    def _complete(
        self, messages: list[dict[str, str]], options: AgentRequestOptions
    ) -> Tuple[dict, str, float]:
        data: Optional[dict] = None
        tried_model = self.model_candidates[0]
        request_started = time.perf_counter()
        response_elapsed_sec = 0.0

        try:
            for current_model in self.model_candidates:
                tried_model = current_model
                try:
                    data = _post_chat_completion(
                        self.api_url,
                        self.api_key,
                        current_model,
                        messages,
                        self.ssl_context,
                        options,
                    )
                    response_elapsed_sec = time.perf_counter() - request_started
                    if current_model != self.model_candidates[0]:
                        print(f"Info: primary model unavailable, used fallback: {current_model}", file=sys.stderr)
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

    @staticmethod
    def _response_from_raw(
        data: dict, tried_model: str, response_elapsed_sec: float, provider: str
    ) -> AgentResponse:
        choice0 = data.get("choices", [{}])[0]
        message = choice0.get("message", {})
        answer = message.get("content")
        if isinstance(answer, str) and answer.strip():
            return AgentResponse(
                answer=answer,
                raw_data=data,
                model=tried_model,
                latency_sec=response_elapsed_sec,
                provider=provider,
            )

        finish_reason = choice0.get("finish_reason")
        usage = data.get("usage", {})
        completion_details = usage.get("completion_tokens_details", {})
        reasoning_tokens = completion_details.get("reasoning_tokens")
        raise RuntimeError(
            "Received empty assistant content from provider (message.content is null/empty).\n"
            f"finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}.\n"
            "This often happens when strict limits are used and the model spends output tokens on reasoning.\n"
            "Try one of:\n"
            "1) increase --max-output-tokens (or LLM_MAX_OUTPUT_TOKENS),\n"
            "2) remove/relax stop sequences and finish instruction,\n"
            "3) try another model/provider."
        )

    def ask(self, prompt: str, options: AgentRequestOptions) -> AgentResponse:
        messages: list[dict[str, str]] = []
        system_msg = _constraint_system_message(options)
        if system_msg:
            messages.append(system_msg)
        messages.append({"role": "user", "content": prompt})
        data, tried_model, elapsed = self._complete(messages, options)
        response = self._response_from_raw(data, tried_model, elapsed, self.provider)

        if options.count_tokens:
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

            response.token_stats = AgentTokenStats(
                current_request_tokens=current_request_tokens,
                dialog_history_tokens=dialog_history_tokens,
                response_model_tokens=completion_tokens,
                prompt_tokens_total=prompt_tokens_full,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ).__dict__

        return response

    def ask_chat(self, prompt: str, options: AgentRequestOptions) -> AgentResponse:
        self._ensure_chat_history_loaded()
        messages: list[dict[str, str]] = []
        system_msg = _chat_session_system_message(options)
        messages.append(system_msg)
        messages.extend(self._chat_history)
        messages.append({"role": "user", "content": prompt})
        data, tried_model, elapsed = self._complete(messages, options)
        response = self._response_from_raw(data, tried_model, elapsed, self.provider)

        if options.count_tokens:
            prompt_tokens_full, completion_tokens, total_tokens = self._extract_prompt_completion_total_tokens(data)

            # "dialog history" tokens: all previous chat messages (excluding system message).
            # We'll estimate it as:
            #   (system + history) prompt tokens - (system only) prompt tokens
            history_messages = [system_msg] + self._chat_history
            prompt_tokens_history_with_system = self._dry_run_prompt_tokens(
                model=tried_model,
                messages=history_messages,
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

            response.token_stats = AgentTokenStats(
                current_request_tokens=current_request_tokens,
                dialog_history_tokens=dialog_history_tokens,
                response_model_tokens=completion_tokens,
                prompt_tokens_total=prompt_tokens_full,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ).__dict__

        self._chat_history.append({"role": "user", "content": prompt})
        self._chat_history.append({"role": "assistant", "content": response.answer})
        if self.chat_history_path:
            try:
                save_chat_messages(self.chat_history_path, self._chat_history)
            except OSError as exc:
                print(f"Warning: could not save chat history: {exc}", file=sys.stderr)
        return response
