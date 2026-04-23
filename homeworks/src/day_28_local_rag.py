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
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent import AgentRequestOptions, SimpleLLMAgent, load_env_file
from homeworks.src.day_22_rag import _load_index_records, retrieve_relevant_chunks

DEFAULT_INDEX_PATH = "homeworks/artifacts/day_21/index_structured.json"
DEFAULT_OUT_DIR = "homeworks/artifacts/day_28"
DEFAULT_LOCAL_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    expected_keywords: list[str]


def _default_questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(
            question="Как в проекте устроена индексация документов в day_21?",
            expected_keywords=["chunk", "embedding", "json", "sqlite", "index"],
        ),
        EvalQuestion(
            question="Как в day_22 выполняется ответ с RAG?",
            expected_keywords=["retrieval", "context", "sources", "without_rag", "with_rag"],
        ),
        EvalQuestion(
            question="Что делает day_25 mini chat с точки зрения памяти диалога?",
            expected_keywords=["history", "task_memory", "constraints", "goal", "sources"],
        ),
        EvalQuestion(
            question="Как в проекте выбирается LLM provider?",
            expected_keywords=["auto", "openrouter", "groq", "fallback", "api key"],
        ),
        EvalQuestion(
            question="Какие anti-hallucination механики есть в RAG-части проекта?",
            expected_keywords=["context", "source", "не хватает", "grounded", "relevance"],
        ),
    ]


def _build_rag_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_lines: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        context_lines.extend(
            [
                f"[Context {idx}]",
                f"source: {item['source']}",
                f"section: {item['section']}",
                f"score: {item['score']}",
                f"text: {item['text']}",
                "",
            ]
        )
    context_block = "\n".join(context_lines).strip()
    return (
        "Ты отвечаешь только по контексту из проекта.\n"
        "Если данных недостаточно, явно скажи об этом.\n\n"
        f"{context_block}\n\n"
        f"Вопрос: {question}\n\n"
        "Формат ответа:\n"
        "1) Короткий ответ по сути\n"
        "2) Sources: список source из контекста"
    )


def _local_generate(prompt: str, model: str, ollama_url: str) -> str:
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
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local LLM HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Start local server with `ollama serve`."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from Ollama: {raw[:300]}") from exc
    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError(f"Local LLM response has no `response` field: {raw[:300]}")
    return answer


def _cloud_generate(prompt: str) -> str:
    agent = SimpleLLMAgent.from_env(user_id="day28-rag-cloud")
    answer = agent.ask(prompt, _request_options())
    return answer.answer


def _request_options() -> AgentRequestOptions:
    return AgentRequestOptions(
        temperature=0.2,
        top_p=None,
        top_k=None,
        response_format=None,
        max_output_tokens=700,
        stop_sequences=[],
        finish_instruction=None,
        count_tokens=False,
    )


def _keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    lowered = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return round(hits / len(expected_keywords), 3)


def _token_jaccard(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", a.lower()))
    b_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", b.lower()))
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _evaluate_stability(answers: list[str], errors: int) -> dict[str, Any]:
    if len(answers) <= 1:
        return {
            "runs": len(answers) + errors,
            "success_rate": round(len(answers) / max(1, len(answers) + errors), 3),
            "avg_pairwise_similarity": 1.0 if answers else 0.0,
        }
    sims: list[float] = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            sims.append(_token_jaccard(answers[i], answers[j]))
    return {
        "runs": len(answers) + errors,
        "success_rate": round(len(answers) / max(1, len(answers) + errors), 3),
        "avg_pairwise_similarity": round(sum(sims) / max(1, len(sims)), 3),
    }


def _run_model(
    generator: Callable[[str], str],
    prompt: str,
    repeats: int,
) -> dict[str, Any]:
    answers: list[str] = []
    durations_ms: list[float] = []
    errors: list[str] = []
    for _ in range(repeats):
        started = time.perf_counter()
        try:
            answer = generator(prompt)
            duration = (time.perf_counter() - started) * 1000
            answers.append(answer)
            durations_ms.append(duration)
        except Exception as exc:  # noqa: BLE001 - want stable report
            errors.append(str(exc))

    stability = _evaluate_stability(answers, errors=len(errors))
    return {
        "answers": answers,
        "durations_ms": [round(item, 1) for item in durations_ms],
        "avg_latency_ms": round(statistics.mean(durations_ms), 1) if durations_ms else None,
        "errors": errors,
        "stability": stability,
    }


def _load_questions(path: str) -> list[EvalQuestion]:
    if not path.strip():
        return _default_questions()
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("questions", [])
    if not isinstance(rows, list):
        raise RuntimeError("questions file must contain `questions` list")
    result: list[EvalQuestion] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        keywords = item.get("expected_keywords", [])
        if not q or not isinstance(keywords, list):
            continue
        result.append(EvalQuestion(question=q, expected_keywords=[str(k) for k in keywords]))
    if not result:
        raise RuntimeError("No valid questions found in questions file")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 28: Local RAG + local/cloud model comparison.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--questions-file", default="")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument(
        "--with-cloud",
        action="store_true",
        help="Also run cloud provider via SimpleLLMAgent.from_env().",
    )
    return parser


def main() -> None:
    load_env_file()
    args = _build_parser().parse_args()

    index_path = Path(args.index_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_index_records(index_path)
    questions = _load_questions(args.questions_file)

    report_rows: list[dict[str, Any]] = []
    local_quality: list[float] = []
    cloud_quality: list[float] = []
    local_latency: list[float] = []
    cloud_latency: list[float] = []
    local_stability: list[float] = []
    cloud_stability: list[float] = []

    for item in questions:
        contexts = retrieve_relevant_chunks(records, item.question, top_k=max(1, args.top_k))
        prompt = _build_rag_prompt(item.question, contexts)

        local_result = _run_model(
            generator=lambda x: _local_generate(x, model=args.local_model, ollama_url=args.ollama_url),
            prompt=prompt,
            repeats=max(1, args.repeats),
        )
        local_answer = local_result["answers"][0] if local_result["answers"] else ""
        local_cov = _keyword_coverage(local_answer, item.expected_keywords)
        local_quality.append(local_cov)
        if local_result["avg_latency_ms"] is not None:
            local_latency.append(float(local_result["avg_latency_ms"]))
        local_stability.append(float(local_result["stability"]["success_rate"]))

        cloud_result: dict[str, Any] | None = None
        cloud_cov: float | None = None
        if args.with_cloud:
            cloud_result = _run_model(
                generator=_cloud_generate,
                prompt=prompt,
                repeats=max(1, args.repeats),
            )
            cloud_answer = cloud_result["answers"][0] if cloud_result["answers"] else ""
            cloud_cov = _keyword_coverage(cloud_answer, item.expected_keywords)
            cloud_quality.append(cloud_cov)
            if cloud_result["avg_latency_ms"] is not None:
                cloud_latency.append(float(cloud_result["avg_latency_ms"]))
            cloud_stability.append(float(cloud_result["stability"]["success_rate"]))

        report_rows.append(
            {
                "question": item.question,
                "expected_keywords": item.expected_keywords,
                "contexts": contexts,
                "local_model": {
                    **local_result,
                    "quality_keyword_coverage": local_cov,
                },
                "cloud_model": (
                    {
                        **cloud_result,
                        "quality_keyword_coverage": cloud_cov,
                    }
                    if cloud_result is not None
                    else None
                ),
            }
        )

    summary = {
        "questions_count": len(report_rows),
        "retrieval": {
            "index_path": str(index_path),
            "top_k": max(1, args.top_k),
            "mode": "local",
        },
        "local_generation": {
            "model": args.local_model,
            "ollama_url": args.ollama_url,
            "avg_quality_keyword_coverage": round(statistics.mean(local_quality), 3) if local_quality else None,
            "avg_latency_ms": round(statistics.mean(local_latency), 1) if local_latency else None,
            "avg_success_rate": round(statistics.mean(local_stability), 3) if local_stability else None,
        },
        "cloud_generation": (
            {
                "enabled": True,
                "avg_quality_keyword_coverage": round(statistics.mean(cloud_quality), 3)
                if cloud_quality
                else None,
                "avg_latency_ms": round(statistics.mean(cloud_latency), 1) if cloud_latency else None,
                "avg_success_rate": round(statistics.mean(cloud_stability), 3) if cloud_stability else None,
            }
            if args.with_cloud
            else {"enabled": False}
        ),
        "comparison": (
            {
                "quality_delta_local_minus_cloud": round(
                    (statistics.mean(local_quality) if local_quality else 0.0)
                    - (statistics.mean(cloud_quality) if cloud_quality else 0.0),
                    3,
                ),
                "latency_delta_ms_local_minus_cloud": round(
                    (statistics.mean(local_latency) if local_latency else 0.0)
                    - (statistics.mean(cloud_latency) if cloud_latency else 0.0),
                    1,
                ),
                "stability_delta_local_minus_cloud": round(
                    (statistics.mean(local_stability) if local_stability else 0.0)
                    - (statistics.mean(cloud_stability) if cloud_stability else 0.0),
                    3,
                ),
            }
            if args.with_cloud
            else None
        ),
    }

    payload = {"summary": summary, "questions": report_rows}
    out_path = out_dir / "local_rag_report.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Day 28 completed: local RAG pipeline is configured.")
    print(f"Report: {out_path}")
    print(f"Local avg quality(keyword coverage): {summary['local_generation']['avg_quality_keyword_coverage']}")
    print(f"Local avg latency ms: {summary['local_generation']['avg_latency_ms']}")
    print(f"Local avg success rate: {summary['local_generation']['avg_success_rate']}")
    if args.with_cloud:
        cloud = summary["cloud_generation"]
        print(f"Cloud avg quality(keyword coverage): {cloud['avg_quality_keyword_coverage']}")
        print(f"Cloud avg latency ms: {cloud['avg_latency_ms']}")
        print(f"Cloud avg success rate: {cloud['avg_success_rate']}")


if __name__ == "__main__":
    main()
