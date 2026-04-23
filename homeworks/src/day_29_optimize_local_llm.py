from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homeworks.src.day_22_rag import _load_index_records, retrieve_relevant_chunks

DEFAULT_INDEX_PATH = "homeworks/artifacts/day_21/index_structured.json"
DEFAULT_OUT_DIR = "homeworks/artifacts/day_29"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    expected_keywords: list[str]


def _default_questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(
            question="Как устроен retrieval pipeline в проекте?",
            expected_keywords=["retrieval", "chunks", "index", "context", "sources"],
        ),
        EvalQuestion(
            question="Какие параметры индексатора day_21 самые важные?",
            expected_keywords=["root", "include", "embedding", "out-dir", "sqlite"],
        ),
        EvalQuestion(
            question="Какие anti-hallucination механики используются в RAG?",
            expected_keywords=["grounded", "source", "context", "relevance", "не хватает"],
        ),
        EvalQuestion(
            question="Как в проекте обрабатывается fallback модели провайдера?",
            expected_keywords=["fallback", "model_candidates", "404", "provider", "error"],
        ),
        EvalQuestion(
            question="Что делает mini chat из day_25 по памяти задачи?",
            expected_keywords=["task_memory", "goal", "constraints", "history", "terms"],
        ),
    ]


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lowered = answer.lower()
    hits = sum(1 for item in keywords if item.lower() in lowered)
    return round(hits / len(keywords), 3)


def _build_baseline_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_block = "\n\n".join(item.get("text", "") for item in contexts)
    return (
        "Ответь на вопрос по контексту ниже.\n\n"
        f"Контекст:\n{context_block}\n\n"
        f"Вопрос: {question}"
    )


def _build_optimized_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_lines: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        context_lines.extend(
            [
                f"[Chunk {idx}]",
                f"source: {item.get('source', 'unknown')}",
                f"section: {item.get('section', 'unknown')}",
                f"score: {item.get('score', 0.0)}",
                f"text: {item.get('text', '')}",
                "",
            ]
        )
    context_block = "\n".join(context_lines).strip()
    return (
        "Ты технический ассистент по этому репозиторию.\n"
        "Правила:\n"
        "1) Отвечай только на основе контекста.\n"
        "2) Если данных недостаточно, явно напиши: 'Недостаточно данных в контексте'.\n"
        "3) Дай ответ в 3-5 коротких пунктах.\n"
        "4) В конце добавь блок Sources: со списком source.\n\n"
        f"{context_block}\n\n"
        f"Вопрос: {question}"
    )


def _call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float,
    max_tokens: int,
    context_window: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": context_window,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=ollama_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Cannot connect to local Ollama. Start `ollama serve`.") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from Ollama: {raw[:300]}") from exc

    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError(f"No `response` field in Ollama output: {raw[:300]}")

    total_duration_ns = int(data.get("total_duration", 0))
    prompt_eval_count = int(data.get("prompt_eval_count", 0))
    eval_count = int(data.get("eval_count", 0))
    # token/sec proxy from Ollama counters (if available)
    tps = 0.0
    if total_duration_ns > 0 and eval_count > 0:
        tps = eval_count / (total_duration_ns / 1_000_000_000)

    return {
        "answer": answer,
        "latency_ms": round(elapsed_ms, 1),
        "total_duration_ms": round(total_duration_ns / 1_000_000, 1) if total_duration_ns else None,
        "prompt_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "tokens_per_second": round(tps, 2) if tps else None,
    }


def _evaluate_profile(
    records: list[dict[str, Any]],
    questions: list[EvalQuestion],
    *,
    model: str,
    ollama_url: str,
    temperature: float,
    max_tokens: int,
    context_window: int,
    top_k: int,
    optimized_prompt: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    quality_scores: list[float] = []
    latencies: list[float] = []
    total_duration_list: list[float] = []
    prompt_tokens_list: list[int] = []
    generated_tokens_list: list[int] = []
    tps_list: list[float] = []
    errors: list[str] = []

    for item in questions:
        contexts = retrieve_relevant_chunks(records, item.question, top_k=max(1, top_k))
        prompt = (
            _build_optimized_prompt(item.question, contexts)
            if optimized_prompt
            else _build_baseline_prompt(item.question, contexts)
        )
        try:
            result = _call_ollama(
                prompt=prompt,
                model=model,
                ollama_url=ollama_url,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=context_window,
            )
            quality = _keyword_coverage(result["answer"], item.expected_keywords)
            quality_scores.append(quality)
            latencies.append(float(result["latency_ms"]))
            prompt_tokens_list.append(int(result["prompt_tokens"]))
            generated_tokens_list.append(int(result["generated_tokens"]))
            if result["total_duration_ms"] is not None:
                total_duration_list.append(float(result["total_duration_ms"]))
            if result["tokens_per_second"] is not None:
                tps_list.append(float(result["tokens_per_second"]))
            rows.append(
                {
                    "question": item.question,
                    "expected_keywords": item.expected_keywords,
                    "contexts": contexts,
                    "prompt": prompt,
                    "answer": result["answer"],
                    "quality_keyword_coverage": quality,
                    "latency_ms": result["latency_ms"],
                    "total_duration_ms": result["total_duration_ms"],
                    "prompt_tokens": result["prompt_tokens"],
                    "generated_tokens": result["generated_tokens"],
                    "tokens_per_second": result["tokens_per_second"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.question}: {exc}")
            rows.append(
                {
                    "question": item.question,
                    "expected_keywords": item.expected_keywords,
                    "contexts": contexts,
                    "prompt": prompt,
                    "error": str(exc),
                }
            )

    success_rate = (len(questions) - len(errors)) / max(1, len(questions))
    return {
        "config": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "context_window": context_window,
            "prompt_template": "optimized" if optimized_prompt else "baseline",
            "top_k": max(1, top_k),
        },
        "summary": {
            "questions_count": len(questions),
            "success_rate": round(success_rate, 3),
            "avg_quality_keyword_coverage": round(statistics.mean(quality_scores), 3) if quality_scores else None,
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
            "avg_total_duration_ms": round(statistics.mean(total_duration_list), 1)
            if total_duration_list
            else None,
            "avg_prompt_tokens": round(statistics.mean(prompt_tokens_list), 1) if prompt_tokens_list else None,
            "avg_generated_tokens": round(statistics.mean(generated_tokens_list), 1)
            if generated_tokens_list
            else None,
            "avg_tokens_per_second": round(statistics.mean(tps_list), 2) if tps_list else None,
            "errors_count": len(errors),
        },
        "errors": errors,
        "questions": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 29: optimize local LLM for repository RAG tasks.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--top-k", type=int, default=4)

    parser.add_argument("--baseline-model", default="qwen2.5:3b")
    parser.add_argument("--baseline-temperature", type=float, default=0.7)
    parser.add_argument("--baseline-max-tokens", type=int, default=700)
    parser.add_argument("--baseline-context-window", type=int, default=2048)

    parser.add_argument(
        "--optimized-model",
        default="qwen2.5:3b",
        help="Set quantized model tag if available (example: qwen2.5:3b-instruct-q4_K_M).",
    )
    parser.add_argument("--optimized-temperature", type=float, default=0.2)
    parser.add_argument("--optimized-max-tokens", type=int, default=450)
    parser.add_argument("--optimized-context-window", type=int, default=4096)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    records = _load_index_records(Path(args.index_path).expanduser().resolve())
    questions = _default_questions()

    baseline = _evaluate_profile(
        records=records,
        questions=questions,
        model=args.baseline_model,
        ollama_url=args.ollama_url,
        temperature=args.baseline_temperature,
        max_tokens=max(1, args.baseline_max_tokens),
        context_window=max(256, args.baseline_context_window),
        top_k=max(1, args.top_k),
        optimized_prompt=False,
    )
    optimized = _evaluate_profile(
        records=records,
        questions=questions,
        model=args.optimized_model,
        ollama_url=args.ollama_url,
        temperature=args.optimized_temperature,
        max_tokens=max(1, args.optimized_max_tokens),
        context_window=max(256, args.optimized_context_window),
        top_k=max(1, args.top_k),
        optimized_prompt=True,
    )

    b = baseline["summary"]
    o = optimized["summary"]
    comparison = {
        "quality_delta_after_minus_before": (
            round(float(o["avg_quality_keyword_coverage"]) - float(b["avg_quality_keyword_coverage"]), 3)
            if o["avg_quality_keyword_coverage"] is not None and b["avg_quality_keyword_coverage"] is not None
            else None
        ),
        "latency_delta_ms_after_minus_before": (
            round(float(o["avg_latency_ms"]) - float(b["avg_latency_ms"]), 1)
            if o["avg_latency_ms"] is not None and b["avg_latency_ms"] is not None
            else None
        ),
        "tokens_per_second_delta_after_minus_before": (
            round(float(o["avg_tokens_per_second"]) - float(b["avg_tokens_per_second"]), 2)
            if o["avg_tokens_per_second"] is not None and b["avg_tokens_per_second"] is not None
            else None
        ),
        "prompt_tokens_delta_after_minus_before": (
            round(float(o["avg_prompt_tokens"]) - float(b["avg_prompt_tokens"]), 1)
            if o["avg_prompt_tokens"] is not None and b["avg_prompt_tokens"] is not None
            else None
        ),
        "generated_tokens_delta_after_minus_before": (
            round(float(o["avg_generated_tokens"]) - float(b["avg_generated_tokens"]), 1)
            if o["avg_generated_tokens"] is not None and b["avg_generated_tokens"] is not None
            else None
        ),
        "success_rate_delta_after_minus_before": round(float(o["success_rate"]) - float(b["success_rate"]), 3),
    }

    payload = {
        "task": "day_29_local_llm_optimization",
        "before_optimization": baseline,
        "after_optimization": optimized,
        "comparison": comparison,
        "notes": [
            "Resource usage is estimated by Ollama counters: prompt/generated tokens and throughput.",
            "For exact CPU/GPU/RAM profiling use external system monitoring during the same run.",
        ],
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optimization_report.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Day 29 optimization completed.")
    print(f"Report: {out_path}")
    print(f"Before quality: {b['avg_quality_keyword_coverage']} | After quality: {o['avg_quality_keyword_coverage']}")
    print(f"Before latency ms: {b['avg_latency_ms']} | After latency ms: {o['avg_latency_ms']}")
    print(f"Before tps: {b['avg_tokens_per_second']} | After tps: {o['avg_tokens_per_second']}")


if __name__ == "__main__":
    main()
