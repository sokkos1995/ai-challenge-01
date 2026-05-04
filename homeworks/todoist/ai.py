from __future__ import annotations

import json
from dataclasses import dataclass

from app.config import build_ssl_context, get_provider_config, load_env_file
from app.models import AgentRequestOptions
from app.provider_client import post_chat_completion


@dataclass
class AiPlan:
    title: str
    priority: str
    subtasks: list[str]
    due_date: str | None = None
    tags: list[str] | None = None


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def generate_plan(goal: str) -> AiPlan:
    load_env_file()
    provider, api_url, api_key, models = get_provider_config()
    model = models[0]
    options = AgentRequestOptions(
        temperature=0.2,
        top_p=None,
        top_k=None,
        response_format={"type": "json_object"},
        max_output_tokens=500,
        stop_sequences=[],
        finish_instruction=None,
        count_tokens=False,
    )
    prompt = (
        "Ты помощник для task tracker сервиса. "
        "Преобразуй цель пользователя в JSON со схемой: "
        '{"title": "string", "priority": "low|medium|high", '
        '"due_date": "YYYY-MM-DD|null", "tags": ["string"], '
        '"subtasks": ["string", "..."]}. '
        "Верни только JSON без пояснений. "
        f"Цель пользователя: {goal}"
    )
    data = post_chat_completion(
        api_url=api_url,
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        ssl_context=build_ssl_context(),
        options=options,
    )
    content = data["choices"][0]["message"]["content"]
    payload = json.loads(_extract_json(content))
    subtasks = payload.get("subtasks") or []
    if not isinstance(subtasks, list) or not subtasks:
        raise RuntimeError("AI returned empty subtasks list.")
    return AiPlan(
        title=str(payload.get("title") or goal).strip(),
        priority=str(payload.get("priority") or "medium").strip().lower(),
        due_date=payload.get("due_date"),
        tags=[str(item) for item in (payload.get("tags") or [])],
        subtasks=[str(item).strip() for item in subtasks if str(item).strip()],
    )


def fallback_plan(goal: str) -> AiPlan:
    base = goal.strip().rstrip(".")
    title = base[:120] or "Новая задача"
    return AiPlan(
        title=title,
        priority="medium",
        due_date=None,
        tags=["auto", "fallback"],
        subtasks=[
            f"Уточнить требования: {title}",
            "Подготовить первый рабочий вариант",
            "Проверить результат и закрыть задачу",
        ],
    )
