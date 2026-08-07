#!/usr/bin/env python3
"""Day 47: run indirect prompt injection attacks (insecure vs --secure)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.config import build_ssl_context, get_provider_config, load_env_file  # noqa: E402
from app.services.provider_service import ProviderService  # noqa: E402

from homeworks.src.day_47_indirect_injection.agents import run_vector  # noqa: E402
from homeworks.src.day_47_indirect_injection.defenses import attack_succeeded  # noqa: E402

ART = ROOT / "homeworks" / "artifacts" / "day_47"
PAYLOADS = {
    "email": ART / "payloads" / "email.html",
    "document": ART / "payloads" / "document.md",
    "web": ART / "payloads" / "landing.html",
}


def build_provider() -> ProviderService:
    load_env_file()
    name, api_url, api_key, models = get_provider_config()
    return ProviderService(
        provider=name,
        api_url=api_url,
        api_key=api_key,
        model_candidates=models,
        ssl_context=build_ssl_context(),
    )


def load_payload(vector: str) -> str:
    path = PAYLOADS[vector]
    return path.read_text(encoding="utf-8")


def verdict_for(*, vector: str, answer: str, side_effect: bool, attack_blocked: bool) -> str:
    if attack_blocked:
        return "held"
    if attack_succeeded(vector=vector, answer=answer, side_effect=side_effect):
        return "broke"
    return "held"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Day 47 indirect prompt injection runner (toggle defenses with --secure)"
    )
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Enable sanitize + boundary markers + output validation (and block httpbin tool)",
    )
    parser.add_argument(
        "--vector",
        choices=("email", "document", "web", "all"),
        default="all",
        help="Which attack vector to run",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Probe a live LLM instead of the deterministic naive/secure pipelines",
    )
    parser.add_argument(
        "--provider",
        default="",
        help="Optional LLM_PROVIDER override (openrouter|groq|cursor|auto); implies --live",
    )
    parser.add_argument(
        "--offline-check",
        action="store_true",
        help="Only exercise defenses on payloads (no agent / no httpbin)",
    )
    args = parser.parse_args()

    if args.provider.strip():
        os.environ["LLM_PROVIDER"] = args.provider.strip().lower()
        args.live = True

    mode = "secure" if args.secure else "insecure"
    vectors = ["email", "document", "web"] if args.vector == "all" else [args.vector]

    if args.offline_check:
        from homeworks.src.day_47_indirect_injection.defenses import (
            prepare_content,
            sanitize_untrusted_content,
        )

        report: dict[str, Any] = {"mode": mode, "offline": True, "vectors": {}}
        for vector in vectors:
            raw = load_payload(vector)
            sanitized = sanitize_untrusted_content(raw)
            prepared = prepare_content(raw, secure=args.secure)
            report["vectors"][vector] = {
                "raw_len": len(raw),
                "sanitized_len": len(sanitized),
                "prepared_preview": prepared[:400],
                "wire_in_sanitized": "WIRE_OK" in sanitized,
                "override_in_sanitized": "OVERRIDE_OK" in sanitized,
                "1899_in_sanitized": "1899" in sanitized,
                "comment_stripped": "<!--" not in sanitized,
            }
        out = ART / f"offline_{mode}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Wrote {out}", flush=True)
        return 0

    provider: ProviderService | None = build_provider() if args.live else None
    results: list[dict[str, Any]] = []

    for vector in vectors:
        engine = "live" if args.live else ("secure" if args.secure else "naive")
        print(f"[run] {mode}/{vector} engine={engine} ...", flush=True)
        content = load_payload(vector)
        row = run_vector(
            provider,
            vector=vector,
            content=content,
            secure=args.secure,
            live=args.live,
        )
        verdict = verdict_for(
            vector=vector,
            answer=row.answer,
            side_effect=row.side_effect,
            attack_blocked=row.attack_blocked,
        )
        entry = {
            "vector": vector,
            "mode": mode,
            "engine": row.engine,
            "model": row.model,
            "latency_sec": row.latency_sec,
            "verdict": verdict,
            "side_effect": row.side_effect,
            "side_effect_detail": row.side_effect_detail,
            "attack_blocked": row.attack_blocked,
            "validation_reasons": list(row.validation_reasons),
            "raw_model_answer": row.raw_model_answer,
            "answer": row.answer,
        }
        results.append(entry)
        print(
            f"  verdict={verdict} side_effect={row.side_effect} "
            f"blocked={row.attack_blocked} model={row.model}",
            flush=True,
        )

    provider_name = provider.provider if provider is not None else "deterministic"
    model_candidates = provider.model_candidates if provider is not None else []
    suffix = f"{mode}_live" if args.live else mode
    payload = {
        "provider": provider_name,
        "model_candidates": model_candidates,
        "mode": mode,
        "secure": args.secure,
        "live": args.live,
        "results": results,
        "summary": {
            "total": len(results),
            "broke": sum(1 for r in results if r["verdict"] == "broke"),
            "held": sum(1 for r in results if r["verdict"] == "held"),
            "by_vector": {r["vector"]: r["verdict"] for r in results},
        },
    }
    out_path = ART / f"results_{suffix}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
