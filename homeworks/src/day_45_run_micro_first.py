#!/usr/bin/env python3
"""Day 45: micro-model first classification benchmark (offline / cursor / ollama)."""
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
)
from app.models import AgentRequestOptions  # noqa: E402
from app.services.micro_model_first_service import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    MicroFirstResult,
    MicroModelFirstService,
)
from app.services.provider_service import ProviderService  # noqa: E402

ART = ROOT / "homeworks" / "artifacts" / "day_45"
CASES_PATH = ART / "cases.json"


class ScriptedMicroProvider:
    """Offline deterministic micro/fallback answers keyed by case text."""

    def __init__(self, case_by_text: dict[str, dict[str, Any]]) -> None:
        self.provider = "offline"
        self._case_by_text = case_by_text
        self.micro_model = "offline-micro"
        self.fallback_model = "offline-fallback"

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

        # Fallback prompt embeds "User query:" first line after that marker.
        query = user_text
        if "User query:" in user_text:
            query = user_text.split("User query:", 1)[1].strip()
            query = query.split("\n\n", 1)[0].strip()

        case = self._case_by_text.get(query)
        if case is None:
            for key, value in self._case_by_text.items():
                if key in query or query in key:
                    case = value
                    break
        if case is None:
            case = {"category": "complex", "expect_label": "other", "id": "unknown"}

        category = str(case.get("category") or "simple")
        expect_label = str(case.get("expect_label") or "other")
        is_micro = model == self.micro_model or "micro" in model or model == "composer-2"

        if is_micro:
            if category == "simple":
                payload = {
                    "label": expect_label,
                    "confidence": 0.92,
                    "status": "OK",
                }
            elif category == "borderline":
                payload = {
                    "label": expect_label,
                    "confidence": 0.42,
                    "status": "UNSURE",
                }
            else:
                # Invalid / noisy micro output for complex → forces fallback.
                payload = {"label": "???", "confidence": "n/a", "status": "maybe"}
            content = json.dumps(payload, ensure_ascii=False)
        else:
            payload = {
                "label": expect_label,
                "confidence": 0.88,
                "status": "OK",
            }
            content = json.dumps(payload, ensure_ascii=False)

        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 18},
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
    def __init__(self, *, micro_model: str, fallback_model: str) -> None:
        self.provider = "ollama"
        self.model_candidates = [micro_model, fallback_model]
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
    micro = os.getenv("LLM_CHEAP_MODEL") or "qwen2.5-coder:1.5b"
    fallback = os.getenv("LLM_STRONG_MODEL") or "qwen2.5-coder:7b"
    return OllamaNativeProvider(micro_model=micro, fallback_model=fallback)


def build_cursor_provider() -> ProviderService:
    provider, api_url, api_key, model_candidates = get_provider_config()
    if provider != "cursor":
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
    load_env_file()
    preferred = preferred.strip().lower()

    if preferred == "cursor":
        service = build_cursor_provider()
        micro, fallback = get_routing_models("cursor", service.model_candidates)
        return "cursor", service, micro, fallback

    if preferred == "ollama":
        if not ollama_available():
            raise RuntimeError("Ollama is not reachable at http://127.0.0.1:11434")
        service = build_ollama_provider()
        micro, fallback = get_routing_models("ollama", service.model_candidates)
        return "ollama", service, micro, fallback

    # Hybrid: micro = Ollama (small), fallback = Cursor (large)
    if preferred == "hybrid":
        if not ollama_available():
            raise RuntimeError(
                "hybrid mode needs Ollama at http://127.0.0.1:11434 "
                "(see homeworks/day_45.md for setup)"
            )
        cursor = build_cursor_provider()
        ollama = build_ollama_provider()
        micro, _ = get_routing_models("ollama", ollama.model_candidates)
        _, fallback = get_routing_models("cursor", cursor.model_candidates)
        return "hybrid", HybridProvider(ollama=ollama, cursor=cursor), micro, fallback

    cursor_key = (os.getenv("CURSOR_API_KEY") or "").strip()
    if cursor_key and cursor_key != "...":
        service = build_cursor_provider()
        micro, fallback = get_routing_models("cursor", service.model_candidates)
        return "cursor", service, micro, fallback

    if ollama_available():
        service = build_ollama_provider()
        micro, fallback = get_routing_models("ollama", service.model_candidates)
        return "ollama", service, micro, fallback

    raise RuntimeError(
        "Need CURSOR_API_KEY (cursor-sdk) or local Ollama at http://127.0.0.1:11434"
    )


class HybridProvider:
    """Micro calls → Ollama; fallback calls → Cursor."""

    def __init__(self, *, ollama: OllamaNativeProvider, cursor: ProviderService) -> None:
        self.provider = "hybrid"
        self._ollama = ollama
        self._cursor = cursor
        self.model_candidates = list(ollama.model_candidates) + list(cursor.model_candidates)

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
        # Heuristic: ollama tags contain ':' (qwen2.5-coder:1.5b); cursor models do not.
        if ":" in model or model in self._ollama.model_candidates:
            return self._ollama.complete(messages, options, model_candidates=[model])
        return self._cursor.complete(messages, options, model_candidates=[model])


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise RuntimeError(f"cases.json must be a list: {CASES_PATH}")
    return data


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    micro = [r for r in results if not r.get("escalated")]
    fallback = [r for r in results if r.get("escalated")]
    reason_counts: dict[str, int] = {}
    for row in results:
        for reason in row.get("fallback_reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    latencies = [
        float(r.get("micro_latency_sec") or 0) + float(r.get("fallback_latency_sec") or 0)
        for r in results
    ]
    label_hits = sum(
        1
        for r in results
        if r.get("expect_label") and r.get("label") == r.get("expect_label")
    )
    return {
        "total": len(results),
        "handled_by_micro": len(micro),
        "escalated_fallback": len(fallback),
        "micro_ids": [r["id"] for r in micro],
        "fallback_ids": [r["id"] for r in fallback],
        "reason_counts": reason_counts,
        "avg_llm_calls": round(
            sum(int(r.get("llm_calls") or 0) for r in results) / max(1, len(results)), 3
        ),
        "fallback_llm_calls": sum(int(r.get("fallback_calls") or 0) for r in results),
        "avg_latency_sec": round(sum(latencies) / max(1, len(latencies)), 4),
        "label_accuracy": round(label_hits / max(1, len(results)), 3),
    }


def run_case(service: MicroModelFirstService, case: dict[str, Any]) -> dict[str, Any]:
    text = str(case["text"])
    result: MicroFirstResult = service.classify(text)
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "expect_label": case.get("expect_label"),
        "expect_tier": case.get("expect_tier"),
        "text": text,
        "label": result.label,
        "confidence": result.confidence,
        "status": result.status,
        "tier": result.tier,
        "escalated": result.escalated,
        "model_used": result.model_used,
        "micro_model": result.micro_model,
        "fallback_model": result.fallback_model,
        "micro_label": result.micro_label,
        "micro_confidence": result.micro_confidence,
        "micro_status": result.micro_status,
        "fallback_reasons": result.metrics.fallback_reasons,
        "llm_calls": result.metrics.llm_calls,
        "micro_calls": result.metrics.micro_calls,
        "fallback_calls": result.metrics.fallback_calls,
        "micro_latency_sec": result.metrics.micro_latency_sec,
        "fallback_latency_sec": result.metrics.fallback_latency_sec,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 45 micro-model first benchmark")
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument(
        "--provider",
        choices=("auto", "cursor", "ollama", "hybrid"),
        default="auto",
        help="live backend: cursor (both tiers), ollama, or hybrid (ollama micro + cursor fallback)",
    )
    parser.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    args = parser.parse_args()

    load_env_file()
    ART.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    min_conf = float_from_env("LLM_ROUTE_CONFIDENCE_MIN", DEFAULT_MIN_CONFIDENCE)

    if args.mode == "offline":
        case_by_text = {str(c["text"]): c for c in cases}
        provider = ScriptedMicroProvider(case_by_text)
        micro, fallback = provider.micro_model, provider.fallback_model
        backend = "offline"
    else:
        backend, provider, micro, fallback = resolve_live_backend(args.provider)

    service = MicroModelFirstService(
        provider,
        micro,
        fallback,
        min_confidence=min_conf,
    )

    print(
        f"day_45 micro-first backend={backend} micro={micro!r} fallback={fallback!r} "
        f"min_confidence={min_conf} cases={len(cases)}",
        file=sys.stderr,
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            row = run_case(service, case)
            results.append(row)
            flag = "FALLBACK" if row["escalated"] else "micro"
            print(
                f"[{flag}] {row['id']}: label={row['label']} "
                f"reasons={row['fallback_reasons']} conf={row['confidence']}",
                file=sys.stderr,
            )
        except Exception as exc:
            results.append(
                {
                    "id": case.get("id"),
                    "category": case.get("category"),
                    "expect_label": case.get("expect_label"),
                    "expect_tier": case.get("expect_tier"),
                    "text": case.get("text"),
                    "error": str(exc),
                    "escalated": None,
                    "tier": "error",
                }
            )
            print(f"[error] {case.get('id')}: {exc}", file=sys.stderr)

    metrics = summarize([r for r in results if r.get("tier") != "error"])
    metrics["backend"] = backend
    metrics["micro_model"] = micro
    metrics["fallback_model"] = fallback
    metrics["min_confidence"] = min_conf
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
