#!/usr/bin/env python3
"""Side-by-side demo: base Ollama model vs fine-tuned model on fixed prompts.

Example:
  source .venv/bin/activate
  python homeworks/src/day_41_demo_compare.py \\
    --base qwen2.5:7b --tuned aviation-faa:latest
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "homeworks" / "artifacts" / "day_41" / "demo_compare.json"
OLLAMA = "http://127.0.0.1:11434/api/chat"

SYSTEM = (
    "You are an aviation expert and FAA-certified flight instructor. "
    "Answer questions accurately based on Federal Aviation Regulations, "
    "the Aeronautical Information Manual, and standard aviation knowledge."
)

# Demo prompts: mix of eval traps + private-pilot oral style.
DEMO_PROMPTS: list[str] = [
    "What should a pilot know about this topic from 14 CFR 93.93? Topic: 14 CFR 93.93: Description of area",
    "What specific requirements are stated in 14 CFR 107.77?",
    "According to 14 CFR 91.103, what preflight action is required of the PIC?",
    "Explain VFR weather minimums in Class G airspace below 1,200 feet AGL (day).",
    "What is the right-of-way rule when two aircraft are converging at approximately the same altitude?",
    "What fuel reserves are required for IFR flight under 14 CFR 91.167 (airplane)?",
]


def chat(model: str, user: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str((data.get("message") or {}).get("content") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare base vs tuned Ollama models")
    parser.add_argument("--base", default="qwen2.5:7b", help="untuned base model tag")
    parser.add_argument("--tuned", default="aviation-faa:latest", help="fine-tuned model tag")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-tuned", action="store_true", help="only run base (pre-train)")
    args = parser.parse_args()

    rows: list[dict] = []
    for i, prompt in enumerate(DEMO_PROMPTS):
        print(f"[{i + 1}/{len(DEMO_PROMPTS)}] base…", flush=True)
        try:
            base_ans = chat(args.base, prompt)
        except Exception as exc:
            print(f"base failed: {exc}", file=sys.stderr)
            return 1
        tuned_ans = None
        if not args.skip_tuned:
            print(f"[{i + 1}/{len(DEMO_PROMPTS)}] tuned…", flush=True)
            try:
                tuned_ans = chat(args.tuned, prompt)
            except Exception as exc:
                print(f"tuned failed: {exc}", file=sys.stderr)
                return 1
        rows.append(
            {
                "index": i,
                "user": prompt,
                "base_model": args.base,
                "base_assistant": base_ans,
                "tuned_model": None if args.skip_tuned else args.tuned,
                "tuned_assistant": tuned_ans,
                "score_base_0_to_2": None,
                "score_tuned_0_to_2": None,
                "winner": None,
                "notes": "",
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rubric": "0=wrong/hallucination, 1=partial, 2=accurate+instructor style",
        "how_to_score": (
            "Fill score_base_0_to_2 / score_tuned_0_to_2 and winner "
            "(base|tuned|tie) for each row. See evaluation_criteria.md."
        ),
        "comparisons": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print("Open the file and score each pair manually for the demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
