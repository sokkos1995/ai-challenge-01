#!/usr/bin/env python3
"""Day 43: cheap→strong model routing benchmark (offline / ollama / cursor)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    build_ssl_context,
    float_from_env,
    get_provider_config,
    get_routing_models,
    load_env_file,
    positive_int_from_env,
)
from app.models import AgentRequestOptions  # noqa: E402
from app.services.model_routing_service import (  # noqa: E402
    DEFAULT_MIN_ANSWER_CHARS,
    DEFAULT_MIN_CONFIDENCE,
    ModelRoutingService,
    RoutingResult,
)
from app.services.provider_service import ProviderService  # noqa: E402

ART = ROOT / "homeworks" / "artifacts" / "day_43"
CASES_PATH = ART / "cases.json"


class ScriptedRoutingProvider:
    """Offline deterministic cheap/strong answers keyed by case text."""

    def __init__(self, case_by_text: dict[str, dict[str, Any]]) -> None:
        self.provider = "offline"
        self._case_by_text = case_by_text
        self.cheap_model = "offline-cheap"
        self.strong_model = "offline-strong"

    def complete(
        self,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
        *,
        model_candidates: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], str, float]:
        if not model_candidates:
            raise RuntimeError("model_candidates required")
        model = model_candidates[0]
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = (message.get("content") or "").strip()
                break
        # Strong path embeds original question after "Original question:"
        if "Original question:" in user_text:
            user_text = user_text.split("Original question:", 1)[1].strip()
            user_text = user_text.split("\n\n", 1)[0].strip()

        case = self._case_by_text.get(user_text)
        if case is None:
            # fuzzy: match by id prefix in text list
            for key, value in self._case_by_text.items():
                if key in user_text or user_text in key:
                    case = value
                    break
        if case is None:
            case = {"category": "hard", "id": "unknown"}

        category = str(case.get("category") or "easy")
        if model == self.cheap_model or model.endswith("cheap") or "1.5b" in model or model == "composer-2":
            if category == "easy":
                payload = {
                    "answer": f"Clear offline answer for {case.get('id')}.",
                    "confidence": 0.95,
                    "status": "OK",
                }
            elif category == "ambiguous":
                payload = {
                    "answer": "I don't know — the request is underspecified.",
                    "confidence": 0.35,
                    "status": "UNSURE",
                }
            else:
                payload = {
                    "answer": "Maybe something.",
                    "confidence": 0.25,
                    "status": "UNSURE",
                }
            content = json.dumps(payload, ensure_ascii=False)
        else:
            content = f"Strong offline answer for {case.get('id')} with more detail and caveats."

        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        return data, model, 0.01


def ollama_available() -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return int(getattr(resp, "status", 200) or 200) == 200
    except Exception:
        return False


class OllamaNativeProvider:
    """OpenAI-shaped complete() via Ollama /api/chat (longer timeout than urllib client)."""

    def __init__(self, *, cheap_model: str, strong_model: str) -> None:
        self.provider = "ollama"
        self.model_candidates = [cheap_model, strong_model]
        self._url = os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434/api/chat"

    def complete(
        self,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
        *,
        model_candidates: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], str, float]:
        import time

        candidates = model_candidates or self.model_candidates
        model = candidates[0]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(options.temperature)},
        }
        if options.max_output_tokens is not None:
            payload["options"]["num_predict"] = int(options.max_output_tokens)
        started = time.perf_counter()
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed for model={model}: {exc}") from exc
        content = str((raw.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError(f"empty Ollama response for model={model}: {raw!r}")
        elapsed = time.perf_counter() - started
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": int(raw.get("prompt_eval_count") or 0),
                "completion_tokens": int(raw.get("eval_count") or 0),
            },
            "model": model,
        }
        return data, model, elapsed


def build_ollama_provider() -> OllamaNativeProvider:
    cheap = os.getenv("LLM_CHEAP_MODEL") or "qwen2.5-coder:1.5b"
    strong = os.getenv("LLM_STRONG_MODEL") or "qwen2.5-coder:7b"
    return OllamaNativeProvider(cheap_model=cheap, strong_model=strong)


def build_cursor_provider() -> ProviderService:
    provider, api_url, api_key, model_candidates = get_provider_config()
    if provider != "cursor":
        # Force cursor path using env key even if LLM_PROVIDER points elsewhere.
        key = os.getenv("CURSOR_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            raise RuntimeError("CURSOR_API_KEY required for provider=cursor")
        from app.config import CURSOR_API_URL, CURSOR_DEFAULT_MODEL, CURSOR_FALLBACK_MODELS

        model = os.getenv("LLM_MODEL", CURSOR_DEFAULT_MODEL)
        fallback_raw = os.getenv("LLM_FALLBACK_MODELS", ",".join(CURSOR_FALLBACK_MODELS))
        fallbacks = [m.strip() for m in fallback_raw.split(",") if m.strip()]
        model_candidates = [model] + [m for m in fallbacks if m != model]
        api_url = os.getenv("LLM_API_URL", CURSOR_API_URL)
        api_key = key
        provider = "cursor"
    return ProviderService(
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        model_candidates=model_candidates,
        ssl_context=build_ssl_context(),
    )


def resolve_live_backend(preferred: str) -> tuple[str, Any, str, str]:
    """Return (name, provider_service, cheap, strong). Prefer cursor, else ollama."""
    load_env_file()
    preferred = preferred.strip().lower()

    if preferred == "cursor":
        service = build_cursor_provider()
        cheap, strong = get_routing_models("cursor", service.model_candidates)
        return "cursor", service, cheap, strong

    if preferred == "ollama":
        if not ollama_available():
            raise RuntimeError("Ollama is not reachable at http://127.0.0.1:11434")
        service = build_ollama_provider()
        cheap, strong = get_routing_models("ollama", service.model_candidates)
        return "ollama", service, cheap, strong

    # auto: cursor first, then ollama
    cursor_key = (os.getenv("CURSOR_API_KEY") or "").strip()
    if cursor_key and cursor_key != "...":
        try:
            service = build_cursor_provider()
            cheap, strong = get_routing_models("cursor", service.model_candidates)
            # Probe one tiny call? Too expensive. Return and let run fail over.
            return "cursor", service, cheap, strong
        except Exception as exc:
            print(f"Info: cursor backend unavailable ({exc}); trying ollama", file=sys.stderr)

    if ollama_available():
        service = build_ollama_provider()
        cheap, strong = get_routing_models("ollama", service.model_candidates)
        return "ollama", service, cheap, strong

    raise RuntimeError(
        "Need CURSOR_API_KEY (cursor-sdk) or local Ollama at http://127.0.0.1:11434"
    )


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise RuntimeError(f"cases.json must be a list: {CASES_PATH}")
    return data


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    stayed = [r for r in results if not r.get("escalated")]
    escalated = [r for r in results if r.get("escalated")]
    reason_counts: dict[str, int] = {}
    for row in results:
        for reason in row.get("escalate_reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "total": len(results),
        "stayed_cheap": len(stayed),
        "escalated_strong": len(escalated),
        "stayed_ids": [r["id"] for r in stayed],
        "escalated_ids": [r["id"] for r in escalated],
        "reason_counts": reason_counts,
        "avg_llm_calls": round(
            sum(int(r.get("llm_calls") or 0) for r in results) / max(1, len(results)), 3
        ),
    }


def run_case(service: ModelRoutingService, case: dict[str, Any]) -> dict[str, Any]:
    text = str(case["text"])
    result: RoutingResult = service.route(text)
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "expect_tier": case.get("expect_tier"),
        "text": text,
        "tier": result.tier,
        "escalated": result.escalated,
        "model_used": result.model_used,
        "cheap_model": result.cheap_model,
        "strong_model": result.strong_model,
        "confidence": result.confidence,
        "status": result.status,
        "escalate_reasons": result.metrics.escalate_reasons,
        "llm_calls": result.metrics.llm_calls,
        "cheap_latency_sec": result.metrics.cheap_latency_sec,
        "strong_latency_sec": result.metrics.strong_latency_sec,
        "answer_preview": result.answer[:240],
        "cheap_answer_preview": result.cheap_answer[:240],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 43 model routing benchmark")
    parser.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
        help="offline scripted provider or live LLM",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cursor", "ollama"),
        default="auto",
        help="live backend (cursor preferred in auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional max number of cases (0 = all)",
    )
    args = parser.parse_args()

    load_env_file()
    ART.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    min_chars = positive_int_from_env("LLM_ROUTE_MIN_ANSWER_CHARS", DEFAULT_MIN_ANSWER_CHARS)
    min_conf = float_from_env("LLM_ROUTE_CONFIDENCE_MIN", DEFAULT_MIN_CONFIDENCE)

    if args.mode == "offline":
        case_by_text = {str(c["text"]): c for c in cases}
        provider = ScriptedRoutingProvider(case_by_text)
        cheap, strong = provider.cheap_model, provider.strong_model
        backend = "offline"
    else:
        backend, provider, cheap, strong = resolve_live_backend(args.provider)

    router = ModelRoutingService(
        provider,
        cheap,
        strong,
        min_answer_chars=min_chars,
        min_confidence=min_conf,
    )

    print(
        f"day_43 routing backend={backend} cheap={cheap!r} strong={strong!r} "
        f"min_chars={min_chars} min_confidence={min_conf} cases={len(cases)}",
        file=sys.stderr,
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            row = run_case(router, case)
            results.append(row)
            flag = "STRONG" if row["escalated"] else "cheap"
            print(
                f"[{flag}] {row['id']}: model={row['model_used']} "
                f"reasons={row['escalate_reasons']} conf={row['confidence']}",
                file=sys.stderr,
            )
        except Exception as exc:
            # Live cursor may fail mid-run — record and continue for partial metrics.
            results.append(
                {
                    "id": case.get("id"),
                    "category": case.get("category"),
                    "expect_tier": case.get("expect_tier"),
                    "text": case.get("text"),
                    "error": str(exc),
                    "escalated": None,
                    "tier": "error",
                }
            )
            print(f"[error] {case.get('id')}: {exc}", file=sys.stderr)
            if backend == "cursor" and args.provider == "auto":
                print(
                    "Info: cursor failed; re-run with --provider ollama",
                    file=sys.stderr,
                )

    metrics = summarize([r for r in results if r.get("tier") != "error"])
    metrics["backend"] = backend
    metrics["cheap_model"] = cheap
    metrics["strong_model"] = strong
    metrics["errors"] = [r for r in results if r.get("tier") == "error"]

    out_results = ART / f"results_{backend}.json"
    out_metrics = ART / f"metrics_{backend}.json"
    out_results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote {out_results}", file=sys.stderr)
    print(f"Wrote {out_metrics}", file=sys.stderr)
    return 0 if not metrics.get("errors") else 2


if __name__ == "__main__":
    raise SystemExit(main())
