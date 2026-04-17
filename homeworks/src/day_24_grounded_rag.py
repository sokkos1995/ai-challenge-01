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
from homeworks.src.day_22_rag import ControlQuestion, _default_control_questions
from homeworks.src.day_23_rerank import retrieve_then_rerank, rewrite_query


@dataclass(frozen=True)
class GroundedContext:
    chunk_id: str
    source: str
    section: str
    score: float
    quote: str


def _load_index_records(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("Index payload does not contain a valid records list.")
    return [item for item in records if isinstance(item, dict)]


def _default_request_options() -> AgentRequestOptions:
    return AgentRequestOptions(
        temperature=0.1,
        top_p=None,
        top_k=None,
        response_format=None,
        max_output_tokens=700,
        stop_sequences=[],
        finish_instruction=None,
        count_tokens=False,
    )


def _extract_quote(text: str, query: str, max_chars: int = 220) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    query_tokens = [
        token for token in re.findall(r"[A-Za-zА-Яа-я0-9_]+", query.lower()) if len(token) >= 4
    ]
    lowered = cleaned.lower()
    hit_index = -1
    for token in query_tokens:
        idx = lowered.find(token)
        if idx >= 0:
            hit_index = idx
            break
    if hit_index < 0:
        return cleaned[:max_chars].strip()
    start = max(0, hit_index - max_chars // 3)
    return cleaned[start : start + max_chars].strip()


def _select_grounded_contexts(
    records: list[dict[str, Any]],
    question: str,
    top_k_before: int,
    top_k_after: int,
    similarity_threshold: float,
) -> list[GroundedContext]:
    _, selected = retrieve_then_rerank(
        records=records,
        question=rewrite_query(question),
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        similarity_threshold=similarity_threshold,
    )
    contexts: list[GroundedContext] = []
    for chunk in selected:
        contexts.append(
            GroundedContext(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                section=chunk.section,
                score=chunk.rerank_score,
                quote=_extract_quote(chunk.text, question),
            )
        )
    return contexts


def _build_grounded_prompt(question: str, contexts: list[GroundedContext]) -> str:
    context_lines: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        context_lines.extend(
            [
                f"[Context {idx}]",
                f"source: {item.source}",
                f"section: {item.section}",
                f"chunk_id: {item.chunk_id}",
                f"score: {item.score}",
                f"quote: {item.quote}",
                "",
            ]
        )
    context_block = "\n".join(context_lines).strip()
    return (
        "Ты отвечаешь только по контексту ниже.\n"
        "Верни строго JSON с ключами answer, sources, quotes.\n"
        "sources: массив объектов {source, section, chunk_id}.\n"
        "quotes: массив коротких цитат из контекста (дословно или максимально близко).\n"
        "Если данных недостаточно, в answer напиши: "
        '"Не знаю по текущему контексту. Уточните вопрос." и все равно заполни sources/quotes по найденным чанкам.\n\n'
        f"{context_block}\n\n"
        f"Вопрос: {question}"
    )


def _safe_json_payload(raw_answer: str, fallback_contexts: list[GroundedContext]) -> dict[str, Any]:
    default = {
        "answer": "Не удалось распарсить ответ модели.",
        "sources": [
            {"source": ctx.source, "section": ctx.section, "chunk_id": ctx.chunk_id} for ctx in fallback_contexts
        ],
        "quotes": [ctx.quote for ctx in fallback_contexts if ctx.quote],
    }
    try:
        payload = json.loads(raw_answer)
    except json.JSONDecodeError:
        return default
    if not isinstance(payload, dict):
        return default
    answer = str(payload.get("answer", "")).strip() or default["answer"]
    raw_sources = payload.get("sources", [])
    raw_quotes = payload.get("quotes", [])
    sources: list[dict[str, str]] = []
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            sources.append(
                {
                    "source": str(item.get("source", "")),
                    "section": str(item.get("section", "")),
                    "chunk_id": str(item.get("chunk_id", "")),
                }
            )
    quotes = [str(item).strip() for item in raw_quotes] if isinstance(raw_quotes, list) else []
    quotes = [item for item in quotes if item]
    if not sources:
        sources = default["sources"]
    if not quotes:
        quotes = default["quotes"]
    return {"answer": answer, "sources": sources, "quotes": quotes}


def _ask_model(prompt: str, dry_run: bool, fallback_contexts: list[GroundedContext]) -> dict[str, Any]:
    if dry_run:
        return {
            "answer": "[dry-run] LLM call skipped.",
            "sources": [
                {"source": ctx.source, "section": ctx.section, "chunk_id": ctx.chunk_id} for ctx in fallback_contexts
            ],
            "quotes": [ctx.quote for ctx in fallback_contexts if ctx.quote],
        }
    agent = SimpleLLMAgent.from_env(user_id="day24-grounded-rag")
    response = agent.ask(prompt, _default_request_options())
    return _safe_json_payload(response.answer, fallback_contexts=fallback_contexts)


def answer_with_grounded_rag(
    question: str,
    records: list[dict[str, Any]],
    top_k_before: int,
    top_k_after: int,
    similarity_threshold: float,
    min_context_score: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    contexts = _select_grounded_contexts(
        records=records,
        question=question,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        similarity_threshold=similarity_threshold,
    )
    low_relevance = not contexts or max(ctx.score for ctx in contexts) < min_context_score
    if low_relevance:
        return {
            "question": question,
            "answer": "Не знаю по текущему контексту. Уточните вопрос.",
            "sources": [
                {"source": ctx.source, "section": ctx.section, "chunk_id": ctx.chunk_id} for ctx in contexts
            ],
            "quotes": [ctx.quote for ctx in contexts if ctx.quote],
            "contexts": [ctx.__dict__ for ctx in contexts],
            "policy": "low_relevance_fallback",
        }

    prompt = _build_grounded_prompt(question, contexts)
    grounded = _ask_model(prompt, dry_run=dry_run, fallback_contexts=contexts)
    return {
        "question": question,
        "answer": grounded["answer"],
        "sources": grounded["sources"],
        "quotes": grounded["quotes"],
        "contexts": [ctx.__dict__ for ctx in contexts],
        "policy": "grounded_answer",
    }


def _has_sources(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("sources"), list) and len(payload["sources"]) > 0


def _has_quotes(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("quotes"), list) and len(payload["quotes"]) > 0


def _quote_alignment(answer: str, quotes: list[str]) -> bool:
    if not answer.strip():
        return False
    if answer.startswith("[dry-run]"):
        return True
    if answer.startswith("Не знаю"):
        return True
    answer_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", answer.lower()))
    if not answer_tokens:
        return False
    quote_tokens = set(
        token
        for quote in quotes
        for token in re.findall(r"[A-Za-zА-Яа-я0-9_]+", quote.lower())
        if len(token) >= 4
    )
    if not quote_tokens:
        return False
    overlap = len(answer_tokens & quote_tokens) / max(1, len(answer_tokens))
    return overlap >= 0.08


def _run_control_set(
    records: list[dict[str, Any]],
    out_dir: Path,
    top_k_before: int,
    top_k_after: int,
    similarity_threshold: float,
    min_context_score: float,
    dry_run: bool,
) -> None:
    questions: list[ControlQuestion] = _default_control_questions()
    per_question: list[dict[str, Any]] = []
    for item in questions:
        result = answer_with_grounded_rag(
            question=item.question,
            records=records,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            similarity_threshold=similarity_threshold,
            min_context_score=min_context_score,
            dry_run=dry_run,
        )
        has_sources = _has_sources(result)
        has_quotes = _has_quotes(result)
        aligned = _quote_alignment(str(result["answer"]), list(result["quotes"]))
        per_question.append(
            {
                "question": item.question,
                "expectation": item.expectation,
                "answer": result["answer"],
                "sources": result["sources"],
                "quotes": result["quotes"],
                "policy": result["policy"],
                "checks": {
                    "has_sources": has_sources,
                    "has_quotes": has_quotes,
                    "answer_quote_alignment": aligned,
                },
                "contexts": result["contexts"],
            }
        )

    sources_coverage = round(sum(1 for q in per_question if q["checks"]["has_sources"]) / len(per_question), 3)
    quotes_coverage = round(sum(1 for q in per_question if q["checks"]["has_quotes"]) / len(per_question), 3)
    alignment_rate = round(
        sum(1 for q in per_question if q["checks"]["answer_quote_alignment"]) / len(per_question),
        3,
    )
    dont_know_count = sum(1 for q in per_question if str(q["answer"]).startswith("Не знаю"))

    report = {
        "questions_count": len(per_question),
        "settings": {
            "top_k_before": top_k_before,
            "top_k_after": top_k_after,
            "similarity_threshold": similarity_threshold,
            "min_context_score": min_context_score,
        },
        "metrics": {
            "sources_coverage": sources_coverage,
            "quotes_coverage": quotes_coverage,
            "answer_quote_alignment_rate": alignment_rate,
            "dont_know_count": dont_know_count,
        },
        "dry_run": dry_run,
        "questions": per_question,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "grounded_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"sources coverage: {sources_coverage}")
    print(f"quotes coverage: {quotes_coverage}")
    print(f"answer/quote alignment rate: {alignment_rate}")
    print(f"'не знаю' answers: {dont_know_count}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 24 grounded RAG with sources and quotes.")
    parser.add_argument("--index-path", default="homeworks/artifacts/day_21/index_structured.json")
    parser.add_argument("--out-dir", default="homeworks/artifacts/day_24")
    parser.add_argument("--top-k-before", type=int, default=8)
    parser.add_argument("--top-k-after", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.2)
    parser.add_argument("--min-context-score", type=float, default=0.24)
    parser.add_argument("--question", default="")
    parser.add_argument("--run-control-set", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    load_env_file()
    parser = _build_parser()
    args = parser.parse_args()

    records = _load_index_records(Path(args.index_path).expanduser().resolve())
    top_k_before = max(1, args.top_k_before)
    top_k_after = max(1, args.top_k_after)
    if args.question.strip():
        payload = answer_with_grounded_rag(
            question=args.question.strip(),
            records=records,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            similarity_threshold=args.similarity_threshold,
            min_context_score=args.min_context_score,
            dry_run=args.dry_run,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.run_control_set:
        _run_control_set(
            records=records,
            out_dir=Path(args.out_dir).expanduser().resolve(),
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            similarity_threshold=args.similarity_threshold,
            min_context_score=args.min_context_score,
            dry_run=args.dry_run,
        )
    elif not args.question.strip():
        print("Nothing to run. Use --question and/or --run-control-set.")


if __name__ == "__main__":
    main()
