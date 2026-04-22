from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_local_llm(prompt: str, model: str, ollama_url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=ollama_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to local Ollama API. Make sure `ollama serve` is running."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from Ollama: {raw[:300]}") from exc

    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError(f"Ollama response does not contain `response`: {raw[:300]}")
    return answer


def run_interactive_chat(model: str, ollama_url: str) -> None:
    print(f"Local LLM chat started. Model: {model}")
    print("Type your question and press Enter. Type 'exit' to stop.")
    while True:
        user_prompt = input("\nYou: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            print("Chat stopped.")
            return
        try:
            answer = ask_local_llm(prompt=user_prompt, model=model, ollama_url=ollama_url)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            continue
        print(f"LLM: {answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day 27: small app that talks to local LLM via Ollama API."
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Single prompt for one-shot mode. If omitted, starts interactive chat.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama generate endpoint (default: {DEFAULT_OLLAMA_URL})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prompt.strip():
        try:
            answer = ask_local_llm(
                prompt=args.prompt.strip(),
                model=args.model.strip(),
                ollama_url=args.ollama_url.strip(),
            )
        except RuntimeError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        print(answer)
        return

    run_interactive_chat(model=args.model.strip(), ollama_url=args.ollama_url.strip())


if __name__ == "__main__":
    main()
