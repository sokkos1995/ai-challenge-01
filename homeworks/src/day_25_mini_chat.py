from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.rag_service import RagContext, RagService


@dataclass
class TaskMemoryState:
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    clarified: list[str] = field(default_factory=list)


def _load_index_records(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("Index payload does not contain a valid records list.")
    return [item for item in records if isinstance(item, dict)]


def _extract_phrase_after_marker(text: str, marker: str) -> str:
    lowered = text.lower()
    marker_idx = lowered.find(marker.lower())
    if marker_idx < 0:
        return ""
    candidate = text[marker_idx + len(marker) :].strip(" :,-")
    return candidate.strip()


def _append_unique(items: list[str], value: str) -> None:
    clean = value.strip()
    if clean and clean not in items:
        items.append(clean)


def _update_task_memory(task_memory: TaskMemoryState, user_message: str) -> None:
    normalized = " ".join(user_message.strip().split())
    if not normalized:
        return

    goal = _extract_phrase_after_marker(normalized, "цель")
    if goal:
        task_memory.goal = goal

    for marker in ("ограничение", "ограничения", "нельзя", "только"):
        value = _extract_phrase_after_marker(normalized, marker)
        if value:
            _append_unique(task_memory.constraints, value)

    for marker in ("термин", "называй", "используй формулировку"):
        value = _extract_phrase_after_marker(normalized, marker)
        if value:
            _append_unique(task_memory.terms, value)

    if re.search(r"\bуточню\b|\bуточнение\b|\bдобавлю\b|\bважно\b", normalized.lower()):
        _append_unique(task_memory.clarified, normalized)


def _compose_augmented_query(question: str, task_memory: TaskMemoryState) -> str:
    parts = [question.strip()]
    if task_memory.goal:
        parts.append(f"Цель диалога: {task_memory.goal}")
    if task_memory.constraints:
        parts.append("Ограничения: " + "; ".join(task_memory.constraints[-3:]))
    if task_memory.terms:
        parts.append("Зафиксированные термины: " + "; ".join(task_memory.terms[-3:]))
    if task_memory.clarified:
        parts.append("Последние уточнения: " + "; ".join(task_memory.clarified[-2:]))
    return "\n".join(part for part in parts if part)


def _render_answer_from_context(question: str, contexts: list[RagContext], task_memory: TaskMemoryState) -> str:
    if not contexts:
        return "Не знаю по текущему контексту. Уточните вопрос."
    key_quote = contexts[0].quote or contexts[0].text[:180]
    answer_parts = [f"По найденному контексту: {key_quote}"]
    if task_memory.goal:
        answer_parts.append(f"Цель диалога: {task_memory.goal}.")
    if task_memory.constraints:
        answer_parts.append(f"Учитываю ограничения: {'; '.join(task_memory.constraints[-2:])}.")
    if task_memory.terms:
        answer_parts.append(f"Фиксирую термины: {'; '.join(task_memory.terms[-2:])}.")
    answer_parts.append(f"Ответ дан на вопрос: {question.strip()}")
    return " ".join(answer_parts).strip()


def _normalize_sources(contexts: list[RagContext]) -> list[dict[str, str]]:
    if not contexts:
        return [{"source": "no_context", "section": "n/a", "chunk_id": "n/a"}]
    return [{"source": item.source, "section": item.section, "chunk_id": item.chunk_id} for item in contexts]


def _normalize_quotes(contexts: list[RagContext]) -> list[str]:
    quotes = [item.quote.strip() for item in contexts if item.quote.strip()]
    if quotes:
        return quotes
    return [item.text[:200].strip() for item in contexts if item.text.strip()]


class MiniRagChat:
    def __init__(self, rag_service: RagService) -> None:
        self._rag = rag_service
        self._history: list[dict[str, str]] = []
        self._task_memory = TaskMemoryState()

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    @property
    def task_memory(self) -> TaskMemoryState:
        return self._task_memory

    def ask(self, user_message: str) -> dict[str, Any]:
        _update_task_memory(self._task_memory, user_message)
        augmented_query = _compose_augmented_query(user_message, self._task_memory)
        _, contexts, low_relevance = self._rag.build_prompt(augmented_query)

        if low_relevance:
            answer = "Не знаю по текущему контексту. Уточните вопрос."
            policy = "low_relevance_fallback"
        else:
            answer = _render_answer_from_context(user_message, contexts, self._task_memory)
            policy = "grounded_answer"

        response_payload = {
            "answer": answer,
            "sources": _normalize_sources(contexts),
            "quotes": _normalize_quotes(contexts),
            "policy": policy,
            "task_memory": {
                "goal": self._task_memory.goal,
                "constraints": list(self._task_memory.constraints),
                "terms": list(self._task_memory.terms),
                "clarified": list(self._task_memory.clarified),
            },
        }
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": answer})
        response_payload["history_size"] = len(self._history)
        return response_payload


def _scenario_messages() -> list[list[str]]:
    return [
        [
            "Цель: подготовить короткий технический обзор RAG в проекте.",
            "Ограничение: только факты из исходников.",
            "Какие параметры есть у day_21 indexer?",
            "Уточню: в ответах используй термин retrieval pipeline.",
            "Как работает retrieve_then_rerank в day_23?",
            "Какие anti-hallucination правила есть в day_24?",
            "Важно: перечисляй шаги лаконично.",
            "Как это перенесено в app/services/rag_service.py?",
            "Добавлю ограничение: не уходи в общую теорию.",
            "Собери итог по цели диалога.",
        ],
        [
            "Цель: описать memory + task lifecycle в чате.",
            "Ограничения: не предлагай внешние библиотеки.",
            "Что хранится в TaskState?",
            "Уточнение: называй этапы как stage.",
            "Как MemoryService обновляет рабочую память?",
            "Какие проверки делает task state machine?",
            "Важно: учитывай, что нужен production-like поток.",
            "Как lifecycle guard блокирует некорректные шаги?",
            "Уточню: в финале напомни цель диалога.",
            "Сформируй итог с учетом ограничений и терминов.",
        ],
    ]


def _evaluate_scenario(turns: list[dict[str, Any]]) -> dict[str, Any]:
    sources_ok = all(bool(turn.get("sources")) for turn in turns)
    goal = str(turns[-1].get("task_memory", {}).get("goal", "")).strip() if turns else ""
    keeps_goal = True
    if goal:
        goal_tokens = [token for token in re.findall(r"[A-Za-zА-Яа-я0-9_]+", goal.lower()) if len(token) >= 4]
        for turn in turns:
            answer_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", str(turn.get("answer", "")).lower()))
            if goal_tokens and not any(token in answer_tokens for token in goal_tokens):
                keeps_goal = False
                break
    return {
        "messages_count": len(turns),
        "sources_on_each_turn": sources_ok,
        "goal_preserved": keeps_goal,
        "passed": sources_ok and keeps_goal,
    }


def run_scenarios(chat: MiniRagChat) -> dict[str, Any]:
    scenario_reports: list[dict[str, Any]] = []
    for idx, messages in enumerate(_scenario_messages(), start=1):
        turns: list[dict[str, Any]] = []
        for message in messages:
            turns.append({"user": message, "assistant": chat.ask(message)})
        evaluation = _evaluate_scenario([item["assistant"] for item in turns])
        scenario_reports.append(
            {
                "scenario_id": idx,
                "messages": messages,
                "turns": turns,
                "evaluation": evaluation,
            }
        )
    return {
        "scenarios_count": len(scenario_reports),
        "all_passed": all(item["evaluation"]["passed"] for item in scenario_reports),
        "scenarios": scenario_reports,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 25 mini chat: RAG + sources + task memory.")
    parser.add_argument("--index-path", default="homeworks/artifacts/day_21/index_structured.json")
    parser.add_argument("--question", default="")
    parser.add_argument("--run-scenarios", action="store_true")
    parser.add_argument("--out-dir", default="homeworks/artifacts/day_25")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    records = _load_index_records(Path(args.index_path).expanduser().resolve())
    chat = MiniRagChat(
        RagService(
            records=records,
            top_k_before=8,
            top_k_after=4,
            similarity_threshold=0.2,
            min_context_score=0.24,
        )
    )

    if args.question.strip():
        payload = chat.ask(args.question.strip())
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.run_scenarios:
        report = run_scenarios(chat)
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "mini_chat_report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {out_path}")
        print(f"all_passed: {report['all_passed']}")
    elif not args.question.strip():
        print("Nothing to run. Use --question and/or --run-scenarios.")


if __name__ == "__main__":
    main()
