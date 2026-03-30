from dataclasses import dataclass, field
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


@dataclass
class FactsState:
    # Key-value "sticky facts" extracted from user messages.
    facts: dict[str, str] = field(default_factory=dict)
    # Recent messages used as additional context (truncated by strategy).
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class BranchState:
    # Shared prefix before branching (checkpoint).
    root_messages: list[dict[str, str]] = field(default_factory=list)
    # Snapshot of conversation at the moment of @checkpoint (optional).
    checkpoint_messages: Optional[list[dict[str, str]]] = None
    # Branch histories after fork: keys are "1" and "2".
    branches: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    # Active branch id ("1" or "2") or None if we are still before the first fork.
    active_branch: Optional[str] = None


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


def _chat_summary_system_message(summary: str) -> Optional[dict[str, str]]:
    clean_summary = summary.strip()
    if not clean_summary:
        return None
    return {
        "role": "system",
        "content": (
            "Conversation memory summary for older turns (this replaces full old history):\n"
            f"{clean_summary}"
        ),
    }


def _chat_message_to_line(msg: dict[str, str]) -> str:
    role = str(msg.get("role", "user")).strip() or "user"
    content = str(msg.get("content", "")).strip().replace("\n", "\\n")
    return f"{role}: {content}"


def _chat_summary_update_messages(previous_summary: str, chunk: list[dict[str, str]]) -> list[dict[str, str]]:
    chunk_text = "\n".join(_chat_message_to_line(msg) for msg in chunk)
    prior = previous_summary.strip() or "(empty)"
    prompt = (
        "Update the running conversation summary.\n"
        "Requirements:\n"
        "- Keep key user facts, constraints, preferences, and unresolved tasks.\n"
        "- Keep concrete decisions and important outputs.\n"
        "- Remove redundancy and small talk.\n"
        "- Keep it concise (5-12 bullet points max).\n"
        "- Output plain text only.\n\n"
        "Previous summary:\n"
        f"{prior}\n\n"
        "New messages to merge:\n"
        f"{chunk_text}"
    )
    return [
        {"role": "system", "content": "You are a chat memory compressor."},
        {"role": "user", "content": prompt},
    ]


def _positive_int_from_env(var_name: str, default: int) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    stripped = raw.strip()
    if not stripped:
        return default
    try:
        value = int(stripped)
    except ValueError:
        return default
    return value if value > 0 else default


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_summary (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            summary TEXT NOT NULL
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


def load_chat_summary(path: str) -> str:
    if not os.path.exists(path):
        return ""
    conn = sqlite3.connect(path)
    try:
        _init_chat_db(conn)
        row = conn.execute("SELECT summary FROM chat_summary WHERE id = 1").fetchone()
    finally:
        conn.close()
    return str(row[0]) if row and row[0] is not None else ""


def save_chat_state(path: str, messages: list[dict[str, str]], summary: str) -> None:
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
        conn.execute(
            """
            INSERT INTO chat_summary (id, summary) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET summary=excluded.summary
            """,
            (summary,),
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
        chat_keep_last_n: int = 10,
        chat_summary_batch_size: int = 10,
        chat_summary_enabled: bool = False,
    ) -> None:
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.model_candidates = model_candidates
        self.chat_history_path = chat_history_path
        self.chat_keep_last_n = max(1, chat_keep_last_n)
        self.chat_summary_batch_size = max(1, chat_summary_batch_size)
        self.chat_summary_enabled = chat_summary_enabled
        self.ssl_context = build_ssl_context()
        self._chat_history: list[dict[str, str]] = []
        self._chat_summary: str = ""
        self._chat_history_loaded = False
        # Cache system-only prompt token counts to reduce extra dry-run calls.
        # Key: (model, system_message_content)
        self._system_only_prompt_tokens_cache: dict[tuple[str, str], int] = {}
        # Context strategy for chat mode.
        # - full: full history
        # - summary: summary + last N (legacy day_09)
        # - sliding: last N only
        # - facts: sticky key-value facts + last N messages
        # - branching: two independent branches created from a checkpoint
        self.context_strategy: str = "full"

        # Strategy-local state (kept in-memory during this process).
        self._sliding_history: list[dict[str, str]] = []
        self._facts_state: FactsState = FactsState()
        self._branch_state: BranchState = BranchState()

    @classmethod
    def from_env(cls) -> "SimpleLLMAgent":
        provider, api_url, api_key, model_candidates = get_provider_config()
        return cls(
            provider,
            api_url,
            api_key,
            model_candidates,
            _chat_history_path_from_env(),
            _positive_int_from_env("LLM_CHAT_KEEP_LAST_N", 10),
            _positive_int_from_env("LLM_CHAT_SUMMARY_BATCH_SIZE", 10),
            False,
        )

    def _ensure_chat_history_loaded(self) -> None:
        if self._chat_history_loaded:
            return
        if self.chat_history_path:
            self._chat_history = load_chat_messages(self.chat_history_path)
            self._chat_summary = load_chat_summary(self.chat_history_path)
        else:
            self._chat_history = []
            self._chat_summary = ""
        self._chat_history_loaded = True

    def set_summary_mode(self, enabled: bool) -> None:
        self.chat_summary_enabled = enabled

    def set_context_strategy(self, strategy: str) -> None:
        normalized = strategy.lower().strip()
        allowed = {"full", "summary", "sliding", "facts", "branching"}
        if normalized not in allowed:
            raise ValueError(f"Unknown context strategy: {strategy}. Allowed: {sorted(allowed)}")
        self.context_strategy = normalized
        # Keep legacy day_09 "summary mode" in sync with the chosen strategy.
        self.chat_summary_enabled = normalized == "summary"

    def _branch_base_messages(self) -> list[dict[str, str]]:
        bs = self._branch_state
        if bs.active_branch is None:
            return list(bs.root_messages)
        return list(bs.root_messages) + list(bs.branches.get(bs.active_branch, []))

    def branch_checkpoint(self) -> None:
        if self.context_strategy != "branching":
            raise RuntimeError("branch_checkpoint() is only available in context_strategy='branching'")
        self._branch_state.checkpoint_messages = self._branch_base_messages()

    def branch_fork(self) -> None:
        if self.context_strategy != "branching":
            raise RuntimeError("branch_fork() is only available in context_strategy='branching'")
        bs = self._branch_state
        checkpoint = bs.checkpoint_messages if bs.checkpoint_messages is not None else self._branch_base_messages()
        bs.root_messages = list(checkpoint)
        bs.branches = {"1": [], "2": []}
        bs.active_branch = "1"
        bs.checkpoint_messages = None

    def branch_switch(self, branch_id: str) -> None:
        if self.context_strategy != "branching":
            raise RuntimeError("branch_switch() is only available in context_strategy='branching'")
        normalized = branch_id.strip()
        if normalized not in {"1", "2"}:
            raise ValueError("branch_id must be '1' or '2'")
        bs = self._branch_state
        if not bs.branches:
            raise RuntimeError("No branches yet. Run branch_fork() first.")
        bs.active_branch = normalized

    def get_branch_info(self) -> str:
        if self.context_strategy != "branching":
            return "Branching is not active."
        bs = self._branch_state
        checkpoint_set = bs.checkpoint_messages is not None
        active = bs.active_branch or "(pre-fork)"
        b1_len = len(bs.branches.get("1", []))
        b2_len = len(bs.branches.get("2", []))
        return (
            f"checkpoint_set={checkpoint_set}, active_branch={active}, "
            f"branch1_messages={b1_len}, branch2_messages={b2_len}, "
            f"root_messages={len(bs.root_messages)}"
        )

    @staticmethod
    def _extract_sticky_fact_update(user_message: str) -> Optional[tuple[str, str]]:
        """
        Lightweight heuristic for extracting "sticky facts" from user messages.
        This avoids extra LLM calls and fits the day_10 requirement: update after each user message.
        """
        raw = user_message.strip()
        if not raw:
            return None

        lower = raw.lower()

        def value_after_first_colon(text: str) -> str:
            if ":" in text:
                return text.split(":", 1)[1].strip()
            return text.strip()

        # Common labels in the provided scenario (day_09/day_10).
        if "запомни" in lower:
            value = value_after_first_colon(raw)
        else:
            value = raw

        # Choose a key by keyword.
        if "стек" in lower:
            return ("stack", value)
        if "дедлайн" in lower:
            return ("deadline", value)
        if "ограничен" in lower:
            return ("constraints", value)
        if "предпочит" in lower or "коротк" in lower or "списком" in lower:
            return ("preference", value)
        if "формат" in lower:
            return ("output_format", value)
        if "цель" in lower:
            return ("goal", value)

        # If there's no recognizable label, don't change facts.
        return None

    def _update_facts_from_user_message(self, user_message: str) -> None:
        if self.context_strategy != "facts":
            return
        update = self._extract_sticky_fact_update(user_message)
        if update is None:
            return
        key, value = update
        cleaned_value = value.strip()
        if cleaned_value:
            self._facts_state.facts[key] = cleaned_value

    def _facts_system_message(self) -> dict[str, str]:
        facts_json = json.dumps(self._facts_state.facts, ensure_ascii=True)
        content = (
            "facts (key-value memory extracted from the user):\n"
            f"{facts_json}\n\n"
            "Use these facts as ground truth. If you need missing details, ask the user."
        )
        return {"role": "system", "content": content}

    def get_chat_summary(self) -> str:
        self._ensure_chat_history_loaded()
        return self._chat_summary

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
        data, tried_model, elapsed = self._complete(
            _chat_summary_update_messages(current_summary, chunk), summary_options
        )
        response = self._response_from_raw(data, tried_model, elapsed, self.provider)
        return response.answer.strip()

    def _compress_chat_history_if_needed(self, options: AgentRequestOptions) -> None:
        if self.chat_summary_batch_size <= 0:
            return
        while len(self._chat_history) - self.chat_keep_last_n >= self.chat_summary_batch_size:
            chunk = self._chat_history[: self.chat_summary_batch_size]
            new_summary = self._summarize_history_chunk(chunk, self._chat_summary, options)
            self._chat_summary = new_summary
            self._chat_history = self._chat_history[self.chat_summary_batch_size :]

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
        user_msg: dict[str, str] = {"role": "user", "content": prompt}
        system_msg = _chat_session_system_message(options)

        context_messages: list[dict[str, str]] = []
        history_without_current: list[dict[str, str]] = []
        payload_messages: list[dict[str, str]]

        strategy = self.context_strategy
        if strategy in {"full", "summary"}:
            self._ensure_chat_history_loaded()
            history_without_current = list(self._chat_history)
            if self.chat_summary_enabled:
                summary_msg = _chat_summary_system_message(self._chat_summary)
                if summary_msg:
                    context_messages.append(summary_msg)
            payload_messages = [system_msg] + context_messages + history_without_current + [user_msg]

        elif strategy == "sliding":
            combined = list(self._sliding_history) + [user_msg]
            context_with_current = combined[-self.chat_keep_last_n :]
            payload_messages = [system_msg] + context_with_current
            history_without_current = context_with_current[:-1]

        elif strategy == "facts":
            self._update_facts_from_user_message(prompt)
            context_messages.append(self._facts_system_message())
            combined = list(self._facts_state.history) + [user_msg]
            context_with_current = combined[-self.chat_keep_last_n :]
            payload_messages = [system_msg] + context_messages + context_with_current
            history_without_current = context_with_current[:-1]

        elif strategy == "branching":
            history_without_current = self._branch_base_messages()
            payload_messages = [system_msg] + history_without_current + [user_msg]

        else:
            raise RuntimeError(f"Unsupported context strategy: {strategy}")

        data, tried_model, elapsed = self._complete(payload_messages, options)
        response = self._response_from_raw(data, tried_model, elapsed, self.provider)

        if options.count_tokens:
            prompt_tokens_full, completion_tokens, total_tokens = self._extract_prompt_completion_total_tokens(data)

            # History tokens: prompt_tokens(system + context + history) - prompt_tokens(system only)
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

            response.token_stats = AgentTokenStats(
                current_request_tokens=current_request_tokens,
                dialog_history_tokens=dialog_history_tokens,
                response_model_tokens=completion_tokens,
                prompt_tokens_total=prompt_tokens_full,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ).__dict__

        assistant_msg: dict[str, str] = {"role": "assistant", "content": response.answer}
        if strategy in {"full", "summary"}:
            self._chat_history.append(user_msg)
            self._chat_history.append(assistant_msg)
            if self.chat_summary_enabled:
                try:
                    self._compress_chat_history_if_needed(options)
                except Exception as exc:
                    print(f"Warning: could not compress chat history: {exc}", file=sys.stderr)
            if self.chat_history_path:
                try:
                    save_chat_state(self.chat_history_path, self._chat_history, self._chat_summary)
                except OSError as exc:
                    print(f"Warning: could not save chat history: {exc}", file=sys.stderr)

        elif strategy == "sliding":
            combined_after = list(self._sliding_history) + [user_msg, assistant_msg]
            self._sliding_history = combined_after[-self.chat_keep_last_n :]

        elif strategy == "facts":
            combined_after = list(self._facts_state.history) + [user_msg, assistant_msg]
            self._facts_state.history = combined_after[-self.chat_keep_last_n :]

        elif strategy == "branching":
            bs = self._branch_state
            if bs.active_branch is None:
                bs.root_messages.extend([user_msg, assistant_msg])
            else:
                bs.branches.setdefault(bs.active_branch, []).extend([user_msg, assistant_msg])

        return response
