from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent import AgentRequestOptions, SimpleLLMAgent, load_env_file
from homeworks.src.day_21_indexing import _cosine_similarity, _vectorize
from homeworks.src.day_22_rag import _default_control_questions


@dataclass(frozen=True)
class RankedChunk:
    score: float
    rerank_score: float
    source: str
    section: str
    text: str
    chunk_id: str


def _load_index_records(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("Index payload does not contain a valid records list.")
    return [item for item in records if isinstance(item, dict)]


def rewrite_query(question: str) -> str:
    replacements = {
        "оркестрация": "orchestration маршрутизация tool registry",
        "индекс": "index embedding retrieval chunk",
        "провайдер": "provider model fallback api key",
        "тесты": "tests pytest assertions",
    }
    rewritten = question
    lowered = question.lower()
    for key, expansion in replacements.items():
        if key in lowered and expansion not in lowered:
            rewritten = f"{rewritten}. {expansion}"
    return rewritten


def _heuristic_rerank_score(question: str, record: dict[str, Any], dense_score: float) -> float:
    text = str(record.get("text", ""))
    metadata = record.get("metadata", {})
    source = str(metadata.get("source", "")) if isinstance(metadata, dict) else ""
    section = str(metadata.get("section", "")) if isinstance(metadata, dict) else ""

    q_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", question.lower()))
    d_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", f"{text} {source} {section}".lower()))

    overlap = 0.0
    if q_tokens and d_tokens:
        overlap = len(q_tokens & d_tokens) / len(q_tokens)
    source_bonus = 0.08 if source.endswith(".py") or source.endswith(".md") else 0.0
    return dense_score + 0.45 * overlap + source_bonus


def retrieve_then_rerank(
    records: list[dict[str, Any]],
    question: str,
    top_k_before: int,
    top_k_after: int,
    similarity_threshold: float,
) -> tuple[list[RankedChunk], list[RankedChunk]]:
    if not records:
        return [], []
    first_embedding = records[0].get("embedding")
    dim = len(first_embedding) if isinstance(first_embedding, list) and first_embedding else 128
    query_vector = _vectorize(question, dim=dim)

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            continue
        dense_score = _cosine_similarity(query_vector, [float(x) for x in embedding])
        scored.append((dense_score, record))
    scored.sort(key=lambda item: item[0], reverse=True)

    pre_ranked: list[RankedChunk] = []
    for dense_score, record in scored[: max(1, top_k_before)]:
        metadata = record.get("metadata", {})
        source = str(metadata.get("source", "unknown")) if isinstance(metadata, dict) else "unknown"
        section = str(metadata.get("section", "unknown")) if isinstance(metadata, dict) else "unknown"
        rerank_score = _heuristic_rerank_score(question, record, dense_score)
        pre_ranked.append(
            RankedChunk(
                score=round(dense_score, 4),
                rerank_score=round(rerank_score, 4),
                source=source,
                section=section,
                text=str(record.get("text", "")),
                chunk_id=str(record.get("chunk_id", "")),
            )
        )

    filtered = [item for item in pre_ranked if item.rerank_score >= similarity_threshold]
    filtered.sort(key=lambda item: item.rerank_score, reverse=True)
    return pre_ranked, filtered[: max(1, top_k_after)]


def _build_rag_prompt(question: str, contexts: list[RankedChunk]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        lines.extend(
            [
                f"[Context {idx}]",
                f"source: {item.source}",
                f"section: {item.section}",
                f"dense_score: {item.score}",
                f"rerank_score: {item.rerank_score}",
                f"text: {item.text}",
                "",
            ]
        )
    context_block = "\n".join(lines).strip()
    return (
        "Ответь на вопрос по контексту. Если данных не хватает, скажи об этом прямо.\n\n"
        f"{context_block}\n\n"
        f"Вопрос: {question}\n\n"
        "Добавь блок `Sources:` со списком source."
    )


def _default_request_options() -> AgentRequestOptions:
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


def _ask_model(prompt: str, dry_run: bool) -> str:
    if dry_run:
        return "[dry-run] LLM call skipped."
    agent = SimpleLLMAgent.from_env(user_id="day23-rerank")
    answer = agent.ask(prompt, _default_request_options())
    return answer.answer


def _keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return round(hits / len(expected_keywords), 3)


def _run_control_set(
    records: list[dict[str, Any]],
    out_dir: Path,
    top_k_before: int,
    top_k_after: int,
    similarity_threshold: float,
    dry_run: bool,
) -> None:
    questions = _default_control_questions()
    report_questions: list[dict[str, Any]] = []
    for item in questions:
        pre_raw, raw_final = retrieve_then_rerank(
            records=records,
            question=item.question,
            top_k_before=top_k_before,
            top_k_after=top_k_before,
            similarity_threshold=-1.0,
        )
        pre_rw, filtered = retrieve_then_rerank(
            records=records,
            question=rewrite_query(item.question),
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            similarity_threshold=similarity_threshold,
        )
        raw_answer = _ask_model(_build_rag_prompt(item.question, raw_final), dry_run=dry_run)
        filtered_answer = _ask_model(_build_rag_prompt(item.question, filtered), dry_run=dry_run)

        report_questions.append(
            {
                "question": item.question,
                "expectation": item.expectation,
                "expected_sources": item.expected_sources,
                "without_filter_or_rewrite": {
                    "answer": raw_answer,
                    "top_k_before": top_k_before,
                    "top_k_after": len(raw_final),
                    "avg_rerank_score": round(
                        sum(chunk.rerank_score for chunk in raw_final) / len(raw_final), 4
                    )
                    if raw_final
                    else 0.0,
                    "keyword_coverage": _keyword_coverage(raw_answer, item.expected_keywords),
                    "contexts": [chunk.__dict__ for chunk in pre_raw],
                },
                "with_filter_and_rewrite": {
                    "rewritten_question": rewrite_query(item.question),
                    "answer": filtered_answer,
                    "top_k_before": top_k_before,
                    "top_k_after": len(filtered),
                    "similarity_threshold": similarity_threshold,
                    "avg_rerank_score": round(
                        sum(chunk.rerank_score for chunk in filtered) / len(filtered), 4
                    )
                    if filtered
                    else 0.0,
                    "keyword_coverage": _keyword_coverage(filtered_answer, item.expected_keywords),
                    "contexts": [chunk.__dict__ for chunk in pre_rw],
                    "selected_contexts": [chunk.__dict__ for chunk in filtered],
                },
            }
        )

    avg_no_filter = round(
        sum(item["without_filter_or_rewrite"]["keyword_coverage"] for item in report_questions) / len(report_questions),
        3,
    )
    avg_with_filter = round(
        sum(item["with_filter_and_rewrite"]["keyword_coverage"] for item in report_questions) / len(report_questions),
        3,
    )
    output = {
        "questions_count": len(report_questions),
        "settings": {
            "top_k_before": top_k_before,
            "top_k_after": top_k_after,
            "similarity_threshold": similarity_threshold,
        },
        "avg_keyword_coverage_without_filter_or_rewrite": avg_no_filter,
        "avg_keyword_coverage_with_filter_and_rewrite": avg_with_filter,
        "delta_keyword_coverage": round(avg_with_filter - avg_no_filter, 3),
        "dry_run": dry_run,
        "questions": report_questions,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {out_dir / 'comparison_report.json'}")
    print(f"Avg keyword coverage (no filter/rewrite): {avg_no_filter}")
    print(f"Avg keyword coverage (with filter/rewrite): {avg_with_filter}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 23 reranking and filtering for RAG.")
    parser.add_argument("--index-path", default="homeworks/artifacts/day_21/index_structured.json")
    parser.add_argument("--out-dir", default="homeworks/artifacts/day_23")
    parser.add_argument("--top-k-before", type=int, default=8)
    parser.add_argument("--top-k-after", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.2)
    parser.add_argument("--run-control-set", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    load_env_file()
    parser = _build_parser()
    args = parser.parse_args()
    records = _load_index_records(Path(args.index_path).expanduser().resolve())
    if args.run_control_set:
        _run_control_set(
            records=records,
            out_dir=Path(args.out_dir).expanduser().resolve(),
            top_k_before=max(1, args.top_k_before),
            top_k_after=max(1, args.top_k_after),
            similarity_threshold=args.similarity_threshold,
            dry_run=args.dry_run,
        )
    else:
        print("Nothing to run. Use --run-control-set.")


if __name__ == "__main__":
    main()
