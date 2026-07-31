#!/usr/bin/env python3
"""Day 44: monolithic vs multi-stage ticket triage (offline / live)."""
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
    get_provider_config,
    get_routing_models,
    load_env_file,
)
from app.models import AgentRequestOptions  # noqa: E402
from app.services.multistage_inference_service import (  # noqa: E402
    MultistageInferenceService,
    TicketTriageResult,
    decide_by_rules,
)
from app.services.provider_service import ProviderService  # noqa: E402

ART = ROOT / "homeworks" / "artifacts" / "day_44"
CASES_PATH = ART / "cases.json"


class ScriptedTriageProvider:
    """Offline deterministic answers keyed by case id / expect fields."""

    def __init__(self, cases: list[dict[str, Any]]) -> None:
        self.provider = "offline"
        self.cheap_model = "offline-cheap"
        self.strong_model = "offline-strong"
        self._by_text = {str(c["text"]).strip(): c for c in cases}
        self._by_id = {str(c["id"]): c for c in cases}

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
        system = ""
        user = ""
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                system = content
            elif role == "user":
                user = content

        case = self._resolve_case(system, user)
        expect = dict(case.get("expect") or {})
        intent = str(expect.get("intent") or "other")
        urgency = str(expect.get("urgency") or "low")
        product = str(expect.get("product") or "unknown")
        decision = str(expect.get("decision") or decide_by_rules(intent, urgency))
        summary = f"Offline triage for {case.get('id')}"

        if "single JSON object" in system or '"intent"' in system and "decision" in system:
            # Monolithic
            content = json.dumps(
                {
                    "intent": intent,
                    "urgency": urgency,
                    "product": product,
                    "decision": decision,
                    "summary": summary,
                },
                ensure_ascii=False,
            )
        elif "Normalize a support ticket" in system:
            noise = "1" if case.get("category") in {"noisy", "spam", "ambiguous"} else "0"
            lang = "mixed" if case.get("category") == "mixed" else (
                "ru" if any("\u0400" <= ch <= "\u04FF" for ch in str(case.get("text") or "")) else "en"
            )
            core = str(case.get("text") or "")[:80].replace(";", ",")
            content = f"lang={lang};noise={noise};product_hint={product};core={core}"
        elif "Classify a normalized ticket" in system:
            content = f"intent={intent};urgency={urgency};product={product}"
        elif "Decide triage action" in system:
            content = f"decision={decision};summary={summary}"
        else:
            content = json.dumps(
                {
                    "intent": intent,
                    "urgency": urgency,
                    "product": product,
                    "decision": decision,
                    "summary": summary,
                },
                ensure_ascii=False,
            )

        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 18},
        }
        return data, model, 0.01

    def _resolve_case(self, system: str, user: str) -> dict[str, Any]:
        user_stripped = user.strip()
        if user_stripped in self._by_text:
            return self._by_text[user_stripped]

        # Prefer longest core/text overlap (stage2/3 embed stage1 core = text[:80]).
        best: Optional[dict[str, Any]] = None
        best_score = 0
        for case in self._by_text.values():
            text = str(case.get("text") or "")
            core = text[:80].replace(";", ",")
            score = 0
            if text and text in user_stripped:
                score = len(text)
            elif core and core in user_stripped:
                score = len(core)
            else:
                for n in (60, 40, 24):
                    snip = core[:n] if core else ""
                    if snip and snip in user_stripped:
                        score = n
                        break
            cid = str(case.get("id") or "")
            if cid and f"Offline triage for {cid}" in user_stripped:
                score = max(score, 1000)
            if score > best_score:
                best_score = score
                best = case
        if best is not None and best_score > 0:
            return best

        # Fallback: unique intent+urgency+product triple from stage3 user.
        for case in self._by_text.values():
            expect = case.get("expect") or {}
            intent = str(expect.get("intent") or "")
            urgency = str(expect.get("urgency") or "")
            product = str(expect.get("product") or "")
            if (
                intent
                and urgency
                and product
                and f"intent={intent}" in user_stripped
                and f"urgency={urgency}" in user_stripped
                and f"product={product}" in user_stripped
            ):
                return case

        return {
            "id": "unknown",
            "expect": {
                "intent": "other",
                "urgency": "low",
                "product": "unknown",
                "decision": "queue",
            },
        }


def ollama_available() -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return int(getattr(resp, "status", 200) or 200) == 200
    except Exception:
        return False


class OllamaNativeProvider:
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
        payload: dict[str, Any] = {
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
        cheap, strong = get_routing_models("cursor", service.model_candidates)
        return "cursor", service, cheap, strong

    if preferred == "ollama":
        if not ollama_available():
            raise RuntimeError("Ollama is not reachable at http://127.0.0.1:11434")
        service = build_ollama_provider()
        cheap, strong = get_routing_models("ollama", service.model_candidates)
        return "ollama", service, cheap, strong

    cursor_key = (os.getenv("CURSOR_API_KEY") or "").strip()
    if cursor_key and cursor_key != "...":
        try:
            service = build_cursor_provider()
            cheap, strong = get_routing_models("cursor", service.model_candidates)
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


def result_row(
    case: dict[str, Any],
    result: TicketTriageResult,
) -> dict[str, Any]:
    expect = case.get("expect") or {}
    fields = result.fields
    row: dict[str, Any] = {
        "id": case.get("id"),
        "category": case.get("category"),
        "mode": result.mode,
        "ok": result.ok,
        "error": result.error,
        "expect": expect,
        "llm_calls": result.metrics.llm_calls,
        "latency_sec_total": result.metrics.latency_sec_total,
        "models_used": list(result.metrics.models_used),
        "stages": [s.name for s in result.metrics.stages],
    }
    if fields is not None:
        row["intent"] = fields.intent
        row["urgency"] = fields.urgency
        row["product"] = fields.product
        row["decision"] = fields.decision
        row["summary"] = fields.summary
        row["match_intent"] = fields.intent == expect.get("intent")
        row["match_decision"] = fields.decision == expect.get("decision")
    return row


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("error") and not row.get("ok") and row.get("mode") is None:
            continue
        mode = str(row.get("mode") or "unknown")
        by_mode.setdefault(mode, []).append(row)

    summary: dict[str, Any] = {"total_rows": len(results), "modes": {}}
    for mode, rows in by_mode.items():
        ok_rows = [r for r in rows if r.get("ok")]
        summary["modes"][mode] = {
            "count": len(rows),
            "ok": len(ok_rows),
            "errors": len(rows) - len(ok_rows),
            "avg_llm_calls": round(
                sum(int(r.get("llm_calls") or 0) for r in rows) / max(1, len(rows)), 3
            ),
            "avg_latency_sec": round(
                sum(float(r.get("latency_sec_total") or 0.0) for r in rows) / max(1, len(rows)),
                4,
            ),
            "intent_match_rate": round(
                sum(1 for r in ok_rows if r.get("match_intent")) / max(1, len(ok_rows)), 3
            ),
            "decision_match_rate": round(
                sum(1 for r in ok_rows if r.get("match_decision")) / max(1, len(ok_rows)), 3
            ),
        }

    # Pairwise agreement when both modes present for same id
    mono = {r["id"]: r for r in results if r.get("mode") == "monolithic" and r.get("ok")}
    multi = {r["id"]: r for r in results if r.get("mode") == "multistage" and r.get("ok")}
    shared = sorted(set(mono) & set(multi))
    if shared:
        agree_decision = sum(
            1 for cid in shared if mono[cid].get("decision") == multi[cid].get("decision")
        )
        agree_intent = sum(
            1 for cid in shared if mono[cid].get("intent") == multi[cid].get("intent")
        )
        summary["agreement"] = {
            "paired_cases": len(shared),
            "intent_agree_rate": round(agree_intent / len(shared), 3),
            "decision_agree_rate": round(agree_decision / len(shared), 3),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 44 monolithic vs multi-stage triage")
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument(
        "--variant",
        choices=("both", "monolithic", "multistage"),
        default="both",
        help="which inference variant(s) to run",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cursor", "ollama"),
        default="auto",
        help="live backend",
    )
    parser.add_argument("--case-id", default="", help="run a single case by id")
    parser.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    args = parser.parse_args()

    load_env_file()
    ART.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if args.case_id.strip():
        cases = [c for c in cases if c.get("id") == args.case_id.strip()]
        if not cases:
            raise SystemExit(f"unknown case-id: {args.case_id}")
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    if args.mode == "offline":
        provider = ScriptedTriageProvider(cases)
        cheap, strong = provider.cheap_model, provider.strong_model
        backend = "offline"
    else:
        backend, provider, cheap, strong = resolve_live_backend(args.provider)

    service = MultistageInferenceService(provider, cheap, strong)
    variants: list[str]
    if args.variant == "both":
        variants = ["monolithic", "multistage"]
    else:
        variants = [args.variant]

    print(
        f"day_44 triage backend={backend} cheap={cheap!r} strong={strong!r} "
        f"variants={variants} cases={len(cases)}",
        file=sys.stderr,
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        text = str(case["text"])
        for variant in variants:
            try:
                if variant == "monolithic":
                    result = service.run_monolithic(text)
                else:
                    result = service.run_multistage(text)
                row = result_row(case, result)
                results.append(row)
                status = "ok" if row["ok"] else f"ERR:{row.get('error')}"
                print(
                    f"[{variant}] {row['id']}: {status} "
                    f"decision={row.get('decision')} calls={row['llm_calls']}",
                    file=sys.stderr,
                )
            except Exception as exc:
                results.append(
                    {
                        "id": case.get("id"),
                        "category": case.get("category"),
                        "mode": variant,
                        "ok": False,
                        "error": str(exc),
                        "llm_calls": 0,
                    }
                )
                print(f"[error] {variant} {case.get('id')}: {exc}", file=sys.stderr)

    metrics = summarize(results)
    metrics["backend"] = backend
    metrics["cheap_model"] = cheap
    metrics["strong_model"] = strong
    metrics["variants"] = variants

    out_results = ART / f"results_{backend}.json"
    out_metrics = ART / f"metrics_{backend}.json"
    out_results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote {out_results}", file=sys.stderr)
    print(f"Wrote {out_metrics}", file=sys.stderr)

    hard_errors = [r for r in results if not r.get("ok") and args.mode == "offline"]
    return 0 if not hard_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
