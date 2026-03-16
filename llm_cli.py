import argparse
import json
import os
import ssl
import sys
from typing import Optional
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send prompt to LLM API and print answer.")
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        help="Sampling temperature (0..2). Default: 0.7",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=float(os.getenv("LLM_TOP_P")) if os.getenv("LLM_TOP_P") else None,
        help="Nucleus sampling top_p (0..1). Optional.",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=int(os.getenv("LLM_TOP_K")) if os.getenv("LLM_TOP_K") else None,
        help="Top-k sampling (>=1). Optional.",
    )
    args = parser.parse_args()

    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be in range [0, 2]")
    if args.top_p is not None and not 0 <= args.top_p <= 1:
        parser.error("--top-p must be in range [0, 1]")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1")
    return args


def send_request(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    ssl_context: ssl.SSLContext,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
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
    args = parse_args()
    provider, api_url, api_key, model_candidates = get_provider_config()
    prompt = " ".join(args.prompt).strip() or "Привет! Скажи коротко, что такое LLM?"

    ssl_context = build_ssl_context()
    data = None
    tried_model = model_candidates[0]

    try:
        for current_model in model_candidates:
            tried_model = current_model
            try:
                data = send_request(
                    api_url,
                    api_key,
                    current_model,
                    prompt,
                    ssl_context,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                )
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
