#!/usr/bin/env python3
"""Run day_41 baseline on eval examples (no fine-tune).

Providers (no OpenRouter): Cursor API (CURSOR_API_KEY) or local Ollama.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_env_file  # noqa: E402

DEFAULT_EVAL = ROOT / "homeworks" / "artifacts" / "day_41" / "eval.jsonl"
DEFAULT_OUT = ROOT / "homeworks" / "artifacts" / "day_41" / "baseline_responses.json"
OLLAMA_URL = os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434/api/chat"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "qwen2.5-coder:7b"


def ollama_available() -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def resolve_endpoint(preferred: str | None = None) -> tuple[str, str, str]:
    """Return (provider, api_key_or_empty, model). Prefer local Ollama, then Cursor."""
    load_env_file()
    preferred = (preferred or os.getenv("DAY41_BASELINE_PROVIDER") or "auto").strip().lower()
    cursor_key = (os.getenv("CURSOR_API_KEY") or "").strip()
    ollama_model = os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    cursor_model = os.getenv("LLM_MODEL") or "composer-2.5"
    # BASELINE_MODEL overrides whichever provider is chosen
    baseline_override = (os.getenv("BASELINE_MODEL") or "").strip()

    def ollama() -> tuple[str, str, str]:
        return ("ollama", "", baseline_override or ollama_model)

    def cursor() -> tuple[str, str, str]:
        if not cursor_key or cursor_key == "...":
            raise RuntimeError("CURSOR_API_KEY required for provider=cursor")
        return ("cursor", cursor_key, baseline_override or cursor_model)

    if preferred == "ollama":
        if not ollama_available():
            raise RuntimeError("Ollama is not reachable at http://127.0.0.1:11434")
        return ollama()
    if preferred == "cursor":
        return cursor()

    # auto
    if ollama_available():
        return ollama()
    if cursor_key and cursor_key != "...":
        return cursor()
    raise RuntimeError(
        "Need local Ollama (http://127.0.0.1:11434) or CURSOR_API_KEY for day_41 baseline"
    )


def chat_ollama(model: str, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data.get("message") or {}
    content = str(msg.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"empty Ollama response: {data!r}")
    return content


def chat_cursor_batch(api_key: str, model: str, examples: list[dict]) -> list[str]:
    from app.config import cursor_cwd_from_env
    from app.provider_client import post_cursor_completion

    system = examples[0]["messages"][0]["content"]
    parts: list[str] = [
        "You are measuring a BASELINE before fine-tuning an aviation instructor model.",
        "Answer each question independently as a helpful assistant.",
        "Do NOT use tools, do NOT edit files, do NOT search the web.",
        "Return STRICT JSON only: {\"answers\": [\"...\", ...]} with exactly "
        f"{len(examples)} strings in the same order.",
        "",
    ]
    for i, ex in enumerate(examples):
        parts.append(f"### Q{i}\n{ex['messages'][1]['content']}")
    prompt_msgs = [
        {
            "role": "system",
            "content": system + "\nReply with JSON only. Do not use tools or edit files.",
        },
        {"role": "user", "content": "\n".join(parts)},
    ]
    raw = post_cursor_completion(api_key, model, prompt_msgs, cursor_cwd_from_env())
    text = str(raw["choices"][0]["message"]["content"]).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    answers = obj.get("answers")
    if not isinstance(answers, list) or len(answers) != len(examples):
        raise RuntimeError(f"batch baseline expected {len(examples)} answers, got {type(answers)}")
    return [str(a).strip() for a in answers]


def chat_ollama_batch(model: str, examples: list[dict]) -> list[str]:
    system = examples[0]["messages"][0]["content"]
    parts: list[str] = [
        "Answer each aviation question independently.",
        "Return STRICT JSON only: {\"answers\": [\"...\", ...]} with exactly "
        f"{len(examples)} strings in the same order.",
        "",
    ]
    for i, ex in enumerate(examples):
        parts.append(f"### Q{i}\n{ex['messages'][1]['content']}")
    text = chat_ollama(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(parts)},
        ],
    )
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    answers = obj.get("answers")
    if not isinstance(answers, list) or len(answers) != len(examples):
        # Fallback: one-by-one if batch JSON fails
        out: list[str] = []
        for ex in examples:
            msgs = ex["messages"]
            out.append(
                chat_ollama(
                    model,
                    [
                        {"role": "system", "content": msgs[0]["content"]},
                        {"role": "user", "content": msgs[1]["content"]},
                    ],
                )
            )
        return out
    return [str(a).strip() for a in answers]


def load_eval(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise RuntimeError(f"Need {n} eval examples, found {len(rows)} in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline on eval sample (Ollama local or Cursor API)"
    )
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument(
        "--provider",
        choices=("auto", "ollama", "cursor"),
        default="auto",
        help="auto: Ollama if up, else Cursor",
    )
    parser.add_argument(
        "--offline-stub",
        action="store_true",
        help="Write placeholder baselines without calling a model",
    )
    args = parser.parse_args()

    examples = load_eval(args.eval, args.n)
    results: list[dict] = []
    model = "offline-stub"
    provider = "offline"
    note = (
        "day_41 baseline uses local Ollama or Cursor API "
        "(OpenRouter disabled)."
    )

    if args.offline_stub:
        for i, ex in enumerate(examples):
            msgs = ex["messages"]
            results.append(
                {
                    "index": i,
                    "user": msgs[1]["content"],
                    "gold_assistant": msgs[2]["content"],
                    "baseline_assistant": "[offline stub — run without --offline-stub]",
                    "model": model,
                }
            )
    else:
        provider, api_key, model = resolve_endpoint(args.provider)
        print(f"Baseline via provider={provider} model={model}")
        try:
            if provider == "cursor":
                answers = chat_cursor_batch(api_key, model, examples)
            else:
                answers = chat_ollama_batch(model, examples)
        except (RuntimeError, json.JSONDecodeError, KeyError, urllib.error.URLError) as exc:
            print(f"baseline failed: {exc}", file=sys.stderr)
            return 1
        for i, ex in enumerate(examples):
            msgs = ex["messages"]
            results.append(
                {
                    "index": i,
                    "user": msgs[1]["content"],
                    "gold_assistant": msgs[2]["content"],
                    "baseline_assistant": answers[i],
                    "model": model,
                }
            )
            print(f"[{i + 1}/{args.n}] ok")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_path": str(args.eval),
        "provider": provider,
        "model": model,
        "note": note,
        "n": args.n,
        "responses": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
