import os
import ssl


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


def positive_int_from_env(var_name: str, default: int) -> int:
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


def chat_history_path_from_env() -> str | None:
    raw = os.getenv("LLM_CHAT_HISTORY_PATH", ".llm_chat_history.db")
    stripped = raw.strip() if raw else ""
    return stripped if stripped else None


def memory_base_path_from_env() -> str:
    raw = os.getenv("LLM_MEMORY_BASE_PATH", ".llm_memory")
    stripped = raw.strip() if raw else ""
    return stripped or ".llm_memory"
