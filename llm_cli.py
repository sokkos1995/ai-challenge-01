import json
import os
import ssl
import sys
import urllib.error
import urllib.request


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "openrouter/auto"
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
            print(
                "Error: set one of API keys: LLM_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY.\n"
                "Tip: if OpenRouter returns 404 for free models, create free GROQ_API_KEY and run again."
            )
            sys.exit(1)

    if provider == "groq":
        api_key = llm_api_key or groq_key
        if not api_key:
            print("Error: set GROQ_API_KEY (or LLM_API_KEY with LLM_PROVIDER=groq).")
            sys.exit(1)
        api_url = os.getenv("LLM_API_URL", GROQ_API_URL)
        model = os.getenv("LLM_MODEL", GROQ_DEFAULT_MODEL)
        fallback_raw = os.getenv("LLM_FALLBACK_MODELS", ",".join(GROQ_FALLBACK_MODELS))
    elif provider == "openrouter":
        api_key = llm_api_key or openrouter_key
        if not api_key:
            print("Error: set OPENROUTER_API_KEY (or LLM_API_KEY with LLM_PROVIDER=openrouter).")
            sys.exit(1)
        api_url = os.getenv("LLM_API_URL", OPENROUTER_API_URL)
        model = os.getenv("LLM_MODEL", OPENROUTER_DEFAULT_MODEL)
        fallback_raw = os.getenv("LLM_FALLBACK_MODELS", ",".join(OPENROUTER_FALLBACK_MODELS))
    else:
        print("Error: LLM_PROVIDER must be auto, openrouter, or groq.")
        sys.exit(1)

    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]
    model_candidates = [model] + [m for m in fallback_models if m != model]
    return provider, api_url, api_key, model_candidates


def send_request(
    api_url: str, api_key: str, model: str, prompt: str, ssl_context: ssl.SSLContext
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
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


def main() -> None:
    load_env_file()
    provider, api_url, api_key, model_candidates = get_provider_config()
    prompt = " ".join(sys.argv[1:]).strip() or "Привет! Скажи коротко, что такое LLM?"

    ssl_context = build_ssl_context()
    data = None
    tried_model = model_candidates[0]

    try:
        for current_model in model_candidates:
            tried_model = current_model
            try:
                data = send_request(api_url, api_key, current_model, prompt, ssl_context)
                if current_model != model_candidates[0]:
                    print(f"Info: primary model unavailable, used fallback: {current_model}", file=sys.stderr)
                break
            except urllib.error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="replace")
                no_endpoints = exc.code == 404 and (
                    "No endpoints found for" in error_text
                    or "model_not_found" in error_text
                    or "does not exist" in error_text
                )
                if no_endpoints and current_model != model_candidates[-1]:
                    continue
                if exc.code == 403 and "1010" in error_text:
                    print(
                        "HTTP 403 (Cloudflare 1010): provider blocked your request by policy/region.\n"
                        "Try VPN or another provider endpoint (via LLM_PROVIDER / LLM_API_URL)."
                    )
                    sys.exit(1)
                print(f"HTTP error {exc.code}: {error_text}")
                sys.exit(1)
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {error_text}")
        sys.exit(1)
    except ssl.SSLCertVerificationError as exc:
        print(
            "SSL certificate verification failed.\n"
            "Try one of:\n"
            "1) pip install certifi\n"
            "2) export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')\n"
            f"Details: {exc}"
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Request failed: {exc}")
        sys.exit(1)

    if not data:
        if provider == "openrouter":
            print(
                f"Request failed: no response from OpenRouter (last model: {tried_model}).\n"
                "Try free Groq provider:\n"
                "export LLM_PROVIDER=groq\n"
                "export GROQ_API_KEY=your_key"
            )
        else:
            print("Request failed: no response from provider.")
        sys.exit(1)

    answer = data["choices"][0]["message"]["content"]
    print(answer)


if __name__ == "__main__":
    main()
