# AGENTS.md — homeworks/todoist (локальный конфиг)

Локальные правила для мини task-tracker (day 35). Дополняют корневой `AGENTS.md`.

## Границы пакета

- `models.py` — `Task` dataclass + `create` / `with_status` / `to_dict` / `from_dict`
- `storage.py` — только JSON I/O (`JsonTaskStorage`)
- `service.py` — бизнес-логика (`TaskTrackerService`)
- `ai.py` — LLM-план + обязательный `fallback_plan`
- `main.py` — CLI / demo

## Правила

- Не тащить SQLite из `app/storage.py` сюда без явной задачи миграции.
- AI-декомпозиция должна переживать отсутствие API-ключа (fallback).
- Данные по умолчанию: `homeworks/todoist/data/tasks.json`.
- Новые команды CLI добавлять в `main.py`, логику — в `service.py`.
