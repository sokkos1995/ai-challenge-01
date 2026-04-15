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


@dataclass(frozen=True)
class ControlQuestion:
    question: str
    expectation: str
    expected_sources: list[str]
    expected_keywords: list[str]


def _default_control_questions() -> list[ControlQuestion]:
    return [
        ControlQuestion(
            question="Как в проекте выполняется MCP orchestration между серверами?",
            expectation="Должно быть описание registry tool->server и flow list_tasks -> add -> create_task.",
            expected_sources=["homeworks/src/day_20_orchestration.py", "tests/test_day20_orchestration.py"],
            expected_keywords=["registry", "list_tasks", "add", "create_task"],
        ),
        ControlQuestion(
            question="Что делает day_19 pipeline?",
            expectation="Цепочка list_tasks -> summarize_workload -> save_summary_to_file и сохранение markdown.",
            expected_sources=["homeworks/src/day_19_pipeline.py", "homeworks/src/day_19_pipeline_tools_server.py"],
            expected_keywords=["list_tasks", "summarize_workload", "save_summary_to_file", "markdown"],
        ),
        ControlQuestion(
            question="Какие параметры у day_21 indexer и что он сохраняет?",
            expectation="Параметры root/include/max-files/embedding-dim/out-dir и JSON+SQLite артефакты.",
            expected_sources=["homeworks/src/day_21_indexing.py", "homeworks/day_21.md"],
            expected_keywords=["root", "include", "embedding", "json", "sqlite"],
        ),
        ControlQuestion(
            question="Как выбирается LLM provider в конфиге?",
            expectation="auto/openrouter/groq, API key checks и fallback моделей.",
            expected_sources=["app/config.py"],
            expected_keywords=["auto", "openrouter", "groq", "fallback"],
        ),
        ControlQuestion(
            question="Что делает ProviderService.complete?",
            expectation="Перебор model_candidates, fallback при 404 model not found, обработка SSL/HTTP ошибок.",
            expected_sources=["app/services/provider_service.py"],
            expected_keywords=["model_candidates", "fallback", "404", "ssl"],
        ),
        ControlQuestion(
            question="Какие возможности есть у SimpleLLMAgent.ask_chat?",
            expectation="Проверки инвариантов, сбор контекста, вызов провайдера и запись истории.",
            expected_sources=["app/agent.py"],
            expected_keywords=["invariant", "context", "provider", "history"],
        ),
        ControlQuestion(
            question="Как устроен parse_agent_response?",
            expectation="Берется choices[0].message.content, иначе RuntimeError с пояснением.",
            expected_sources=["app/response_parser.py"],
            expected_keywords=["choices", "message", "content", "runtimeerror"],
        ),
        ControlQuestion(
            question="Какие данные хранит модель AgentRequestOptions?",
            expectation="temperature, top_p/top_k, response_format, max_output_tokens, stop_sequences.",
            expected_sources=["app/models.py"],
            expected_keywords=["temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences"],
        ),
        ControlQuestion(
            question="Какой результат ожидается в day_22 домашке?",
            expectation="Агент с режимами с RAG и без RAG, плюс 10 контрольных вопросов и сравнение качества.",
            expected_sources=["homeworks/day_22.md"],
            expected_keywords=["rag", "без rag", "10", "сравнение"],
        ),
        ControlQuestion(
            question="Что проверяют тесты day20 orchestration?",
            expectation="Маршрутизацию по серверам, порядок вызовов и защиту от duplicate tools.",
            expected_sources=["tests/test_day20_orchestration.py"],
            expected_keywords=["routes", "order", "duplicate", "tool"],
        ),
    ]


def _fastapi_sbx_control_questions() -> list[ControlQuestion]:
    return [
        ControlQuestion(
            question="Как в fastapi_ecommerce подключаются роутеры категорий и продуктов?",
            expectation="В app/main.py используется app.include_router для categories.router и products.router.",
            expected_sources=[
                "beginner/06_store/fastapi_ecommerce/app/main.py",
                "beginner/06_store/fastapi_ecommerce/app/routers/categories.py",
                "beginner/06_store/fastapi_ecommerce/app/routers/products.py",
            ],
            expected_keywords=["include_router", "categories", "products", "app/main.py"],
        ),
        ControlQuestion(
            question="Какие эндпоинты есть у categories router?",
            expectation="GET /, POST /, PUT /{category_id}, DELETE /{category_id} с prefix /categories.",
            expected_sources=["beginner/06_store/fastapi_ecommerce/app/routers/categories.py"],
            expected_keywords=["/categories", "get", "post", "put", "delete"],
        ),
        ControlQuestion(
            question="Какие эндпоинты есть у products router?",
            expectation="GET /products/, POST /products/, а также заглушки /category/{category_id}, /{product_id}, PUT/DELETE.",
            expected_sources=["beginner/06_store/fastapi_ecommerce/app/routers/products.py"],
            expected_keywords=["/products", "post", "get", "put", "delete"],
        ),
        ControlQuestion(
            question="Как организована зависимость get_db в проекте магазина?",
            expectation="get_db создает SessionLocal, yield сессию и закрывает ее в finally.",
            expected_sources=["beginner/06_store/fastapi_ecommerce/app/db_depends.py"],
            expected_keywords=["get_db", "SessionLocal", "yield", "close"],
        ),
        ControlQuestion(
            question="Какая база данных и engine используются в fastapi_ecommerce?",
            expectation="SQLite sqlite:///ecommerce.db, create_engine, sessionmaker, DeclarativeBase.",
            expected_sources=["beginner/06_store/fastapi_ecommerce/app/database.py"],
            expected_keywords=["sqlite", "ecommerce.db", "create_engine", "sessionmaker"],
        ),
        ControlQuestion(
            question="Какие поля у модели Category и как устроена иерархия parent/children?",
            expectation="id, name, parent_id, is_active и self-relation parent/children + products relationship.",
            expected_sources=["beginner/06_store/fastapi_ecommerce/app/models/categories.py"],
            expected_keywords=["parent_id", "children", "parent", "relationship", "is_active"],
        ),
        ControlQuestion(
            question="Какие поля у модели Product и связь с Category?",
            expectation="name, description, price, image_url, stock, is_active, category_id FK и relationship category.",
            expected_sources=["beginner/06_store/fastapi_ecommerce/app/models/products.py"],
            expected_keywords=["price", "stock", "category_id", "foreignkey", "relationship"],
        ),
        ControlQuestion(
            question="Какие Pydantic схемы определены для категорий и товаров?",
            expectation="CategoryCreate/Category и ProductCreate/Product с валидацией Field и ConfigDict(from_attributes=True).",
            expected_sources=["beginner/06_store/fastapi_ecommerce/app/schemas.py"],
            expected_keywords=["CategoryCreate", "ProductCreate", "Field", "ConfigDict", "from_attributes"],
        ),
        ControlQuestion(
            question="Что делает миграция 29f296437750?",
            expectation="Создает таблицы categories и products и FK products.category_id -> categories.id.",
            expected_sources=[
                "beginner/06_store/fastapi_ecommerce/app/migrations/versions/29f296437750_create_categories_and_products_tables.py"
            ],
            expected_keywords=["create_table", "categories", "products", "foreignkey", "upgrade"],
        ),
        ControlQuestion(
            question="О чем раздел beginner/05_depencency и какая проверка авторизации показана?",
            expectation="Пример Depends(check_auth), токен secret и ошибка 401 Unauthorized.",
            expected_sources=[
                "beginner/05_depencency/main.py",
                "beginner/05_depencency/readme.md",
            ],
            expected_keywords=["Depends", "check_auth", "secret", "401", "Unauthorized"],
        ),
    ]


def _load_index_records(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("Index payload does not contain a valid records list.")
    return [item for item in records if isinstance(item, dict)]


def retrieve_relevant_chunks(records: list[dict[str, Any]], question: str, top_k: int = 4) -> list[dict[str, Any]]:
    if not records:
        return []
    first_embedding = records[0].get("embedding")
    dim = len(first_embedding) if isinstance(first_embedding, list) and first_embedding else 128
    query_vector = _vectorize(question, dim=dim)

    query_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", question.lower()))
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            continue
        dense_score = _cosine_similarity(query_vector, [float(x) for x in embedding])
        text = str(record.get("text", ""))
        metadata = record.get("metadata", {})
        source = str(metadata.get("source", "")).lower() if isinstance(metadata, dict) else ""
        lexical_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", f"{text} {source}".lower()))
        overlap_score = 0.0
        if query_tokens and lexical_tokens:
            overlap_score = len(query_tokens & lexical_tokens) / len(query_tokens)
        score = dense_score + 0.45 * overlap_score
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    result: list[dict[str, Any]] = []
    for score, record in scored[: max(1, top_k)]:
        metadata = record.get("metadata", {})
        source = str(metadata.get("source", "unknown")) if isinstance(metadata, dict) else "unknown"
        section = str(metadata.get("section", "unknown")) if isinstance(metadata, dict) else "unknown"
        result.append(
            {
                "score": round(score, 4),
                "chunk_id": str(record.get("chunk_id", "")),
                "source": source,
                "section": section,
                "text": str(record.get("text", "")),
            }
        )
    return result


def _build_rag_user_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
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
        "Ответь на вопрос пользователя, опираясь только на контекст ниже. "
        "Если контекста не хватает, явно напиши об этом.\n\n"
        f"{context_block}\n\n"
        f"Вопрос: {question}\n\n"
        "В ответе добавь блок `Sources:` со списком source из контекста."
    )


def _build_plain_prompt(question: str) -> str:
    return (
        "Ответь на вопрос пользователя по доступным знаниям. "
        "Если не уверен, укажи это явно.\n\n"
        f"Вопрос: {question}"
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
    agent = SimpleLLMAgent.from_env(user_id="day22-rag")
    answer = agent.ask(prompt, _default_request_options())
    return answer.answer


def answer_without_rag(question: str, dry_run: bool = False) -> dict[str, Any]:
    prompt = _build_plain_prompt(question)
    answer = _ask_model(prompt, dry_run=dry_run)
    return {
        "mode": "without_rag",
        "prompt": prompt,
        "answer": answer,
        "contexts": [],
    }


def answer_with_rag(
    question: str,
    records: list[dict[str, Any]],
    top_k: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    contexts = retrieve_relevant_chunks(records, question, top_k=top_k)
    prompt = _build_rag_user_prompt(question, contexts)
    answer = _ask_model(prompt, dry_run=dry_run)
    return {
        "mode": "with_rag",
        "prompt": prompt,
        "answer": answer,
        "contexts": contexts,
    }


def _keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return round(hits / len(expected_keywords), 3)


def _source_recall(contexts: list[dict[str, Any]], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    used_sources = {str(item.get("source", "")) for item in contexts}
    hits = sum(1 for source in expected_sources if source in used_sources)
    return round(hits / len(expected_sources), 3)


def _evaluate_question(
    question: ControlQuestion,
    no_rag: dict[str, Any],
    with_rag: dict[str, Any],
) -> dict[str, Any]:
    no_rag_cov = _keyword_coverage(str(no_rag.get("answer", "")), question.expected_keywords)
    rag_cov = _keyword_coverage(str(with_rag.get("answer", "")), question.expected_keywords)
    rag_source_recall = _source_recall(list(with_rag.get("contexts", [])), question.expected_sources)
    return {
        "question": question.question,
        "expectation": question.expectation,
        "expected_sources": question.expected_sources,
        "without_rag": {
            "answer": no_rag.get("answer", ""),
            "keyword_coverage": no_rag_cov,
        },
        "with_rag": {
            "answer": with_rag.get("answer", ""),
            "keyword_coverage": rag_cov,
            "source_recall": rag_source_recall,
            "contexts": with_rag.get("contexts", []),
        },
        "delta_keyword_coverage": round(rag_cov - no_rag_cov, 3),
    }


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_control_set(
    records: list[dict[str, Any]],
    records_fixed: list[dict[str, Any]] | None,
    out_dir: Path,
    top_k: int,
    dry_run: bool,
    control_set_profile: str,
    retriever_index: str,
) -> None:
    if control_set_profile == "fastapi_sbx":
        control_questions = _fastapi_sbx_control_questions()
    else:
        control_questions = _default_control_questions()
    active_records = records
    selected_index = "structured"
    if records_fixed is not None and retriever_index in {"fixed", "auto"}:
        if retriever_index == "fixed":
            active_records = records_fixed
            selected_index = "fixed"
        else:
            structured_recall = sum(
                _source_recall(retrieve_relevant_chunks(records, item.question, top_k=top_k), item.expected_sources)
                for item in control_questions
            ) / len(control_questions)
            fixed_recall = sum(
                _source_recall(
                    retrieve_relevant_chunks(records_fixed, item.question, top_k=top_k), item.expected_sources
                )
                for item in control_questions
            ) / len(control_questions)
            if fixed_recall > structured_recall:
                active_records = records_fixed
                selected_index = "fixed"
    _save_json(
        out_dir / "control_questions.json",
        [
            {
                "question": item.question,
                "expectation": item.expectation,
                "expected_sources": item.expected_sources,
                "expected_keywords": item.expected_keywords,
            }
            for item in control_questions
        ],
    )

    per_question: list[dict[str, Any]] = []
    retriever_diagnostics: list[dict[str, Any]] = []
    for item in control_questions:
        no_rag = answer_without_rag(item.question, dry_run=dry_run)
        with_rag = answer_with_rag(item.question, active_records, top_k=top_k, dry_run=dry_run)
        per_question.append(_evaluate_question(item, no_rag, with_rag))
        active_recall = _source_recall(list(with_rag.get("contexts", [])), item.expected_sources)
        structured_contexts = retrieve_relevant_chunks(records, item.question, top_k=top_k)
        structured_recall = _source_recall(structured_contexts, item.expected_sources)
        diagnostics_item = {
            "question": item.question,
            "active_source_recall": active_recall,
            "structured_source_recall": structured_recall,
        }
        if records_fixed is not None:
            fixed_contexts = retrieve_relevant_chunks(records_fixed, item.question, top_k=top_k)
            diagnostics_item["fixed_source_recall"] = _source_recall(fixed_contexts, item.expected_sources)
        retriever_diagnostics.append(diagnostics_item)

    avg_no_rag = round(
        sum(q["without_rag"]["keyword_coverage"] for q in per_question) / len(per_question), 3
    )
    avg_with_rag = round(sum(q["with_rag"]["keyword_coverage"] for q in per_question) / len(per_question), 3)
    avg_source_recall = round(sum(q["with_rag"]["source_recall"] for q in per_question) / len(per_question), 3)
    avg_structured_recall = round(
        sum(float(item["structured_source_recall"]) for item in retriever_diagnostics) / len(retriever_diagnostics), 3
    )
    avg_fixed_recall = None
    if records_fixed is not None:
        avg_fixed_recall = round(
            sum(float(item.get("fixed_source_recall", 0.0)) for item in retriever_diagnostics) / len(retriever_diagnostics),
            3,
        )

    report = {
        "control_set_profile": control_set_profile,
        "selected_retriever_index": selected_index,
        "questions_count": len(per_question),
        "avg_keyword_coverage_without_rag": avg_no_rag,
        "avg_keyword_coverage_with_rag": avg_with_rag,
        "avg_source_recall_with_rag": avg_source_recall,
        "embedding_retrieval_diagnostics": {
            "avg_structured_source_recall_at_k": avg_structured_recall,
            "avg_fixed_source_recall_at_k": avg_fixed_recall,
            "per_question": retriever_diagnostics,
        },
        "dry_run": dry_run,
        "questions": per_question,
    }
    _save_json(out_dir / "comparison_report.json", report)

    print(f"Control set processed: {len(per_question)} questions")
    print(f"retriever index selected: {selected_index}")
    print(f"avg keyword coverage without RAG: {avg_no_rag}")
    print(f"avg keyword coverage with RAG: {avg_with_rag}")
    print(f"avg source recall with RAG: {avg_source_recall}")
    print(f"avg structured source recall@{top_k}: {avg_structured_recall}")
    if avg_fixed_recall is not None:
        print(f"avg fixed source recall@{top_k}: {avg_fixed_recall}")
    print(f"Report: {out_dir / 'comparison_report.json'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 22 first RAG request.")
    parser.add_argument(
        "--index-path",
        default="homeworks/artifacts/day_21/index_structured.json",
        help="Path to JSON index from day 21.",
    )
    parser.add_argument(
        "--fixed-index-path",
        default="",
        help="Optional path to fixed JSON index for retrieval diagnostics comparison.",
    )
    parser.add_argument(
        "--out-dir",
        default="homeworks/artifacts/day_22",
        help="Directory for control questions and comparison report.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="How many chunks to retrieve for RAG context.")
    parser.add_argument("--question", default="", help="Optional single question run.")
    parser.add_argument(
        "--mode",
        choices=["without_rag", "with_rag", "both"],
        default="both",
        help="Mode for single question run.",
    )
    parser.add_argument(
        "--run-control-set",
        action="store_true",
        help="Run built-in set of 10 control questions.",
    )
    parser.add_argument(
        "--control-set-profile",
        choices=["default", "fastapi_sbx"],
        default="default",
        help="Select control questions profile.",
    )
    parser.add_argument(
        "--retriever-index",
        choices=["structured", "fixed", "auto"],
        default="structured",
        help="Which index to use for RAG contexts in control-set mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call provider API; return placeholder answers.",
    )
    return parser


def main() -> None:
    load_env_file()
    parser = _build_parser()
    args = parser.parse_args()

    index_path = Path(args.index_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    records = _load_index_records(index_path)
    fixed_records: list[dict[str, Any]] | None = None
    if args.fixed_index_path.strip():
        fixed_records = _load_index_records(Path(args.fixed_index_path).expanduser().resolve())

    if args.question.strip():
        question = args.question.strip()
        if args.mode in {"without_rag", "both"}:
            no_rag = answer_without_rag(question, dry_run=args.dry_run)
            print("--- WITHOUT RAG ---")
            print(no_rag["answer"])
            print()
        if args.mode in {"with_rag", "both"}:
            with_rag = answer_with_rag(question, records, top_k=max(1, args.top_k), dry_run=args.dry_run)
            print("--- WITH RAG ---")
            print(with_rag["answer"])
            print("Sources:")
            for ctx in with_rag["contexts"]:
                print(f"  - {ctx['source']} :: {ctx['section']} (score={ctx['score']})")
            print()

    if args.run_control_set:
        _run_control_set(
            records=records,
            records_fixed=fixed_records,
            out_dir=out_dir,
            top_k=max(1, args.top_k),
            dry_run=args.dry_run,
            control_set_profile=args.control_set_profile,
            retriever_index=args.retriever_index,
        )
    elif not args.question.strip():
        print("Nothing to run. Pass --question and/or --run-control-set.")


if __name__ == "__main__":
    main()
