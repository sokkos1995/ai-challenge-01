#!/usr/bin/env python3
"""Day 42: confidence-gated Todoist action classification benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config import build_ssl_context, get_provider_config, load_env_file
from app.models import AgentRequestOptions
from app.services.confidence_inference_service import (
    ConfidenceInferenceService,
    ConfidenceDecision,
)
from app.services.provider_service import ProviderService

ART = ROOT / "homeworks" / "artifacts" / "day_42"
CASES_PATH = ART / "cases.json"


class ScriptedProviderService:
    """Offline provider: maps case text to deterministic scoring/self-check JSON."""

    def __init__(self, case_by_text: dict[str, dict[str, Any]]) -> None:
        self._case_by_text = case_by_text
        self._queues: dict[str, list[str]] = {}

    def _queue_for(self, user_text: str) -> list[str]:
        key = user_text.strip()
        if key in self._queues:
            return self._queues[key]
        case = self._case_by_text.get(key, {})
        expected = str(case.get("expected_action") or "refuse")
        category = str(case.get("category") or "clear")
        if category == "clear":
            scoring = {
                "action": expected,
                "confidence": 0.95,
                "status": "OK",
                "rationale": "clear offline script",
            }
            self_check = {
                "agree": True,
                "corrected_action": expected,
                "reason": "ok",
            }
            queue = [json.dumps(scoring, ensure_ascii=False), json.dumps(self_check, ensure_ascii=False)]
        elif category == "borderline":
            scoring = {
                "action": expected if expected != "complete" or "задач" in key else "refuse",
                "confidence": 0.6,
                "status": "OK",
                "rationale": "borderline offline script",
            }
            # For vague complete without id, force complete so constraint rejects.
            if case.get("id") == "border_vague_complete":
                scoring = {
                    "action": "complete",
                    "confidence": 0.9,
                    "status": "OK",
                    "rationale": "unsafe guess",
                }
                queue = [json.dumps(scoring, ensure_ascii=False)]
            else:
                votes = [
                    {"action": expected, "confidence": 0.7, "status": "OK", "rationale": "v1"},
                    {"action": expected, "confidence": 0.65, "status": "OK", "rationale": "v2"},
                    {"action": "refuse", "confidence": 0.5, "status": "OK", "rationale": "v3"},
                ]
                queue = [json.dumps(scoring, ensure_ascii=False)] + [
                    json.dumps(v, ensure_ascii=False) for v in votes
                ]
        else:
            scoring = {
                "action": expected,
                "confidence": 0.4,
                "status": "UNSURE",
                "rationale": "noisy offline script",
            }
            votes = [
                {"action": expected, "confidence": 0.55, "status": "OK", "rationale": "n1"},
                {"action": expected, "confidence": 0.5, "status": "OK", "rationale": "n2"},
                {"action": "refuse", "confidence": 0.45, "status": "OK", "rationale": "n3"},
            ]
            if case.get("id") == "noisy_garbage":
                votes = [
                    {"action": "refuse", "confidence": 0.6, "status": "OK", "rationale": "g1"},
                    {"action": "list", "confidence": 0.4, "status": "OK", "rationale": "g2"},
                    {"action": "create", "confidence": 0.3, "status": "OK", "rationale": "g3"},
                ]
            queue = [json.dumps(scoring, ensure_ascii=False)] + [
                json.dumps(v, ensure_ascii=False) for v in votes
            ]
        self._queues[key] = queue
        return queue

    def complete(
        self, messages: list[dict[str, str]], options: AgentRequestOptions
    ) -> tuple[dict[str, Any], str, float]:
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content") or ""
                # self-check sends JSON blob
                if content.strip().startswith("{"):
                    try:
                        payload = json.loads(content)
                        user_text = str(payload.get("user_request") or "")
                    except json.JSONDecodeError:
                        user_text = content
                else:
                    user_text = content
                break
        queue = self._queue_for(user_text)
        if not queue:
            raise RuntimeError(f"offline script exhausted for: {user_text!r}")
        content = queue.pop(0)
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 20},
        }
        return data, "offline-script", 0.02


def _decision_to_dict(decision: ConfidenceDecision) -> dict[str, Any]:
    scored = None
    if decision.scored is not None:
        scored = {
            "action": decision.scored.action,
            "confidence": decision.scored.confidence,
            "status": decision.scored.status,
            "rationale": decision.scored.rationale,
        }
    return {
        "accepted": decision.accepted,
        "action": decision.action,
        "rejected_reason": decision.rejected_reason,
        "scored": scored,
        "metrics": decision.metrics.to_dict(),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    rejected = sum(1 for r in results if not r["decision"]["accepted"])
    accepted = n - rejected
    re_inf = sum(int(r["decision"]["metrics"]["re_inference_count"]) for r in results)
    llm_calls = sum(int(r["decision"]["metrics"]["llm_calls"]) for r in results)
    latency = sum(float(r["decision"]["metrics"]["latency_sec_total"]) for r in results)
    baseline = sum(float(r["decision"]["metrics"]["baseline_latency_sec"]) for r in results)
    prompt_tokens = sum(int(r["decision"]["metrics"]["prompt_tokens"]) for r in results)
    completion_tokens = sum(int(r["decision"]["metrics"]["completion_tokens"]) for r in results)
    action_match = 0
    action_comparable = 0
    for r in results:
        expected = r["case"].get("expected_action")
        if expected and r["decision"]["accepted"]:
            action_comparable += 1
            if r["decision"]["action"] == expected:
                action_match += 1
    return {
        "cases": n,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "rejected_rate": round(rejected / n, 4) if n else 0.0,
        "re_inference_count_total": re_inf,
        "re_inference_avg": round(re_inf / n, 4) if n else 0.0,
        "llm_calls_total": llm_calls,
        "llm_calls_avg": round(llm_calls / n, 4) if n else 0.0,
        "latency_sec_total": round(latency, 4),
        "latency_sec_avg": round(latency / n, 4) if n else 0.0,
        "baseline_latency_sec_total": round(baseline, 4),
        "baseline_latency_sec_avg": round(baseline / n, 4) if n else 0.0,
        "latency_overhead_sec": round(latency - baseline, 4),
        "latency_overhead_ratio": round(latency / baseline, 4) if baseline > 0 else None,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "token_cost_proxy_total": prompt_tokens + completion_tokens,
        "accepted_action_match": action_match,
        "accepted_action_comparable": action_comparable,
        "by_category": _by_category(results),
    }


def _by_category(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        cat = str(r["case"].get("category") or "other")
        groups.setdefault(cat, []).append(r)
    out: dict[str, Any] = {}
    for cat, items in groups.items():
        rejected = sum(1 for r in items if not r["decision"]["accepted"])
        re_inf = sum(int(r["decision"]["metrics"]["re_inference_count"]) for r in items)
        latency = sum(float(r["decision"]["metrics"]["latency_sec_total"]) for r in items)
        out[cat] = {
            "cases": len(items),
            "rejected_count": rejected,
            "re_inference_count": re_inf,
            "latency_sec_total": round(latency, 4),
        }
    return out


def build_service(*, offline: bool, cases: list[dict[str, Any]]) -> tuple[ConfidenceInferenceService, str]:
    if offline:
        by_text = {str(c["text"]).strip(): c for c in cases}
        provider: Any = ScriptedProviderService(by_text)
        return ConfidenceInferenceService(provider, "offline"), "offline"
    load_env_file()
    provider_name, api_url, api_key, models = get_provider_config()
    provider_svc = ProviderService(
        provider=provider_name,
        api_url=api_url,
        api_key=api_key,
        model_candidates=models,
        ssl_context=build_ssl_context(),
    )
    return ConfidenceInferenceService(provider_svc, provider_name), provider_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 42 confidence inference benchmark")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use scripted provider (no network) for deterministic demo artifacts",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help="Path to cases.json",
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = json.loads(args.cases.read_text(encoding="utf-8"))
    service, provider_name = build_service(offline=args.offline, cases=cases)

    results: list[dict[str, Any]] = []
    for case in cases:
        text = str(case["text"])
        print(f"[{case['id']}] {text[:60]}...", flush=True)
        decision = service.infer_action(text)
        row = {
            "case": case,
            "decision": _decision_to_dict(decision),
        }
        results.append(row)
        status = "ACCEPT" if decision.accepted else "REJECT"
        print(
            f"  -> {status} action={decision.action} "
            f"reason={decision.rejected_reason} "
            f"llm_calls={decision.metrics.llm_calls} "
            f"pathway={decision.metrics.pathway}",
            flush=True,
        )

    metrics = _aggregate(results)
    metrics["provider"] = provider_name
    metrics["mode"] = "offline" if args.offline else "live"

    ART.mkdir(parents=True, exist_ok=True)
    results_path = ART / "results.json"
    metrics_path = ART / "metrics.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path}")
    print(f"Wrote {metrics_path}")
    print(
        f"rejected={metrics['rejected_count']}/{metrics['cases']} "
        f"re_inference={metrics['re_inference_count_total']} "
        f"latency_total={metrics['latency_sec_total']}s "
        f"tokens={metrics['token_cost_proxy_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
