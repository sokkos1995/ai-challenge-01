#!/usr/bin/env python3
"""Build day_41 aviation chat JSONL from FAA training data (+ optional synth)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_env_file  # noqa: E402

DEFAULT_OUT = ROOT / "homeworks" / "artifacts" / "day_41"
HF_DATASET = "ziksy/faa-aviation-training"
SYSTEM_PROMPT = (
    "You are an aviation expert and FAA-certified flight instructor. "
    "Answer questions accurately based on Federal Aviation Regulations, "
    "the Aeronautical Information Manual, and standard aviation knowledge."
)
MIN_USER = 20
MIN_ASSISTANT = 20
MAX_ASSISTANT = 4000
SEED = 41
TOTAL_TARGET = 100
REAL_TARGET = 30


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _user_key(user: str) -> str:
    return hashlib.sha256(_norm(user).encode("utf-8")).hexdigest()


def _ok_lengths(user: str, assistant: str) -> bool:
    if len(user.strip()) < MIN_USER:
        return False
    a = assistant.strip()
    if len(a) < MIN_ASSISTANT or len(a) > MAX_ASSISTANT:
        return False
    return True


def to_example(
    user: str,
    assistant: str,
    *,
    provenance: str,
    source: str = "",
    category: str = "",
) -> dict[str, Any] | None:
    user_s = user.strip()
    asst_s = assistant.strip()
    if not _ok_lengths(user_s, asst_s):
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_s},
            {"role": "assistant", "content": asst_s},
        ],
        "_meta": {
            "provenance": provenance,
            "source": source,
            "category": category,
        },
    }


def load_faa_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset

    # Hub card sometimes mismatches recorded parquet row counts.
    ds = load_dataset(HF_DATASET, split="train", verification_mode="no_checks")
    return [dict(row) for row in ds]


def collect_real(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if (row.get("format") or "").strip() != "instruction":
            continue
        user = (row.get("instruction") or "").strip()
        assistant = (row.get("output") or "").strip()
        ex = to_example(
            user,
            assistant,
            provenance="real",
            source=str(row.get("source") or ""),
            category=str(row.get("category") or ""),
        )
        if ex is None:
            continue
        key = _user_key(user)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(ex)

    rng.shuffle(candidates)
    return candidates[:n]


def _knowledge_heading(text: str, source: str) -> str:
    first = (text or "").strip().splitlines()[0] if text else ""
    first = re.sub(r"^#+\s*", "", first).strip()
    if first and len(first) <= 180:
        return first
    return source or "FAA publication"


def collect_synthetic_from_knowledge(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    used_users: set[str],
) -> list[dict[str, Any]]:
    """Deterministic synthetic Q&A from FAA knowledge chunks (no API required)."""
    templates = [
        "Explain the following aviation regulation or procedure ({source}): {heading}",
        "As a flight instructor, summarize what {source} requires regarding: {heading}",
        "What should a pilot know about this topic from {source}? Topic: {heading}",
        "Provide a concise instructor-style answer covering {source}: {heading}",
    ]
    pool: list[dict[str, Any]] = []
    for row in rows:
        if (row.get("format") or "").strip() != "knowledge":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        source = str(row.get("source") or "FAA")
        heading = _knowledge_heading(text, source)
        # Keep answers bounded for chat SFT
        assistant = text if len(text) <= MAX_ASSISTANT else text[: MAX_ASSISTANT - 20].rsplit(" ", 1)[0] + "…"
        tmpl = templates[len(pool) % len(templates)]
        user = tmpl.format(source=source, heading=heading)
        if _user_key(user) in used_users:
            continue
        ex = to_example(
            user,
            assistant,
            provenance="synthetic",
            source=source,
            category=str(row.get("category") or ""),
        )
        if ex is None:
            continue
        pool.append(ex)

    rng.shuffle(pool)
    out: list[dict[str, Any]] = []
    for ex in pool:
        user = ex["messages"][1]["content"]
        key = _user_key(user)
        if key in used_users:
            continue
        used_users.add(key)
        out.append(ex)
        if len(out) >= n:
            break
    return out


def collect_synthetic_via_llm(
    real_examples: list[dict[str, Any]],
    n: int,
    used_users: set[str],
) -> list[dict[str, Any]]:
    """Optional synth via Cursor API or local Ollama (OpenRouter not used)."""
    load_env_file()
    cursor_key = (os.getenv("CURSOR_API_KEY") or "").strip()
    use_ollama = False
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            use_ollama = resp.status == 200
    except Exception:
        use_ollama = False

    provider = (os.getenv("DAY41_SYNTH_PROVIDER") or "auto").strip().lower()
    if provider == "ollama" and not use_ollama:
        print("Ollama unavailable for synth", file=sys.stderr)
        return []
    if provider == "cursor" and (not cursor_key or cursor_key == "..."):
        print("CURSOR_API_KEY missing for synth", file=sys.stderr)
        return []
    if provider == "auto":
        if use_ollama:
            provider = "ollama"
        elif cursor_key and cursor_key != "...":
            provider = "cursor"
        else:
            return []

    ollama_model = os.getenv("OLLAMA_MODEL") or "qwen2.5-coder:7b"
    cursor_model = os.getenv("LLM_MODEL") or "composer-2.5"

    few_shot = real_examples[:5]
    shot_lines: list[str] = []
    for ex in few_shot:
        u = ex["messages"][1]["content"]
        a = ex["messages"][2]["content"]
        shot_lines.append(f"Q: {u}\nA: {a[:500]}")

    topics = [
        "VFR weather minimums in Class G airspace",
        "right-of-way rules when converging aircraft",
        "preflight action required by 14 CFR 91.103",
        "fuel requirements for IFR flight",
        "meaning of MEA and MOCA on IFR charts",
        "pilot responsibilities under see-and-avoid",
        "when a Mode C transponder is required",
        "special VFR clearance basics",
        "hypoxia symptoms and prevention",
        "weight and balance CG limits importance",
        "runway incursion prevention best practices",
        "wake turbulence avoidance for light aircraft",
        "lost communications IFR procedures",
        "density altitude effects on takeoff",
        "sterile cockpit concept",
        "NOTAM types a private pilot should check",
        "airspace classes A through E overview",
        "night currency requirements for carrying passengers",
        "ADS-B Out requirements overview",
        "go-around decision making",
    ]

    def ask(messages: list[dict[str, str]]) -> str:
        if provider == "ollama":
            payload = {
                "model": ollama_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7},
            }
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str((data.get("message") or {}).get("content") or "").strip()
        from app.config import cursor_cwd_from_env
        from app.provider_client import post_cursor_completion

        data = post_cursor_completion(
            cursor_key,
            cursor_model,
            messages,
            cursor_cwd_from_env(),
        )
        return str(data["choices"][0]["message"]["content"]).strip()

    out: list[dict[str, Any]] = []
    for topic in topics:
        if len(out) >= n:
            break
        prompt = (
            "Generate ONE aviation training Q&A pair for a flight-instructor chatbot.\n"
            f"Topic: {topic}\n"
            "Return STRICT JSON only: {\"user\": \"...\", \"assistant\": \"...\"}\n"
            "Match the style of these examples (accurate, cites FAR/AIM when relevant, concise):\n\n"
            + "\n\n".join(shot_lines)
        )
        try:
            raw = ask(
                [
                    {
                        "role": "system",
                        "content": "You create high-quality aviation SFT examples. Output JSON only. No tools.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError, RuntimeError) as exc:
            print(f"LLM synth skip ({topic}): {exc}", file=sys.stderr)
            continue

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            print(f"LLM synth bad JSON for topic={topic}", file=sys.stderr)
            continue

        user = str(obj.get("user") or "").strip()
        assistant = str(obj.get("assistant") or "").strip()
        key = _user_key(user)
        if key in used_users:
            continue
        ex = to_example(
            user,
            assistant,
            provenance="synthetic",
            source=f"llm:{provider}",
            category="generated",
        )
        if ex is None:
            continue
        used_users.add(key)
        out.append(ex)
    return out


def strip_meta(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": ex["messages"]} for ex in examples]


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build day_41 aviation JSONL dataset")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--total", type=int, default=TOTAL_TARGET)
    parser.add_argument("--real", type=int, default=REAL_TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM synth; use knowledge templates only")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading {HF_DATASET}…")
    rows = load_faa_rows()
    print(f"Loaded {len(rows)} rows")

    real_n = max(args.real, int(args.total * 0.20 + 0.999))
    real = collect_real(rows, real_n, rng)
    if len(real) < max(10, int(args.total * 0.20)):
        raise RuntimeError(f"Not enough real instruction examples: got {len(real)}")

    used = {_user_key(ex["messages"][1]["content"]) for ex in real}
    need_synth = max(0, args.total - len(real))
    synth: list[dict[str, Any]] = []
    if need_synth and not args.no_llm:
        print(f"Generating up to {need_synth} LLM synthetic examples…")
        synth = collect_synthetic_via_llm(real, need_synth, used)
        print(f"LLM synthetic: {len(synth)}")
    if len(synth) < need_synth:
        more = need_synth - len(synth)
        print(f"Filling {more} synthetic from FAA knowledge templates…")
        synth.extend(collect_synthetic_from_knowledge(rows, more, rng, used))

    all_ex = real + synth
    if len(all_ex) < 50:
        raise RuntimeError(f"Need ≥50 examples, got {len(all_ex)}")

    rng.shuffle(all_ex)
    # Cap at total if we overshot
    all_ex = all_ex[: max(args.total, 50)]
    n_eval = max(1, int(round(len(all_ex) * 0.20)))
    n_train = len(all_ex) - n_eval
    train, eval_set = all_ex[:n_train], all_ex[n_train:]

    out_dir: Path = args.out_dir
    write_jsonl(out_dir / "train.jsonl", strip_meta(train))
    write_jsonl(out_dir / "eval.jsonl", strip_meta(eval_set))

    real_count = sum(1 for ex in all_ex if ex.get("_meta", {}).get("provenance") == "real")
    synth_count = len(all_ex) - real_count
    manifest = {
        "task": "generation",
        "domain": "aviation_qa",
        "source": HF_DATASET,
        "seed": args.seed,
        "total": len(all_ex),
        "train": n_train,
        "eval": n_eval,
        "real_count": real_count,
        "synthetic_count": synth_count,
        "real_ratio": round(real_count / len(all_ex), 3),
        "system_prompt": SYSTEM_PROMPT,
        "filters": {
            "min_user_chars": MIN_USER,
            "min_assistant_chars": MIN_ASSISTANT,
            "max_assistant_chars": MAX_ASSISTANT,
        },
        "files": {
            "train": "train.jsonl",
            "eval": "eval.jsonl",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir / 'train.jsonl'} and {out_dir / 'eval.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
