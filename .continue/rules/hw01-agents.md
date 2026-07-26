---
name: hw01 AGENTS rules
alwaysApply: true
---

# hw01 — правила для локального код-ассистента

Ты помогаешь с репозиторием **hw01 LLM CLI Agent**. Следуй этим правилам как системному промпту (сжатая версия корневого `AGENTS.md`).

## Стек

- Python 3.8+; в новых файлах — `from __future__ import annotations`, типы `str | None`.
- Точка входа: `llm_cli.py` → `app.cli.main`.
- Логика в `app/services/*_service.py`; фасад `SimpleLLMAgent` тонкий.
- SQL только в `app/storage.py` (или dedicated storage), через `?`-плейсхолдеры.
- Секреты только из env (`.env` не коммитить); в репо — `.env.example` с `...`.
- Тесты: `pytest` в `tests/`.

## Naming

- Модули: `snake_case.py`; классы: `PascalCase`; сервисы с суффиксом `Service`.
- Методы: `snake_case`; приватные `_helper`; константы `UPPER_SNAKE`.
- Commit: `[hw-<номер>] <краткое пояснение на русском>` (без точки в конце).

## Паттерны

1. Composition over god-object — делегируй сервисам.
2. Env-first config — `load_env_file()` + typed helpers.
3. Dataclass DTO; иммутабельные обновления (`with_status` → новый объект).
4. Explicit errors — `RuntimeError` / `ValueError` с понятным текстом.
5. CLI `print` — только в `cli` / command handlers, не в domain.

## Антипаттерны (запрещено)

1. Публичный API без аннотаций; `Any` только на границе сырого JSON/HTTP.
2. Сырой SQL вне storage.
3. Хардкод секретов / commit `.env`.
4. God-module: новый функционал → новый или существующий `*_service.py` по ответственности.
5. Ломать grounded RAG (`sources`/`quotes`, «Не знаю…» при low relevance).

## Workflow

1. Если в запросе есть API/апи — сначала docs/Swagger, потом код.
2. Перед финалом сложных правок — проверить поведение (тесты/smoke).
3. Не коммить и не пушь без явной просьбы пользователя.
4. Не трогай `.llm_*.db`, `.llm_users/`, `.env`, кэши индексов без просьбы.

## Шаблон сервиса

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleResult:
    ok: bool
    detail: str


class ExampleService:
    """Одна ответственность. Публичные методы — глаголы предметной области."""

    def __init__(self, keep_last_n: int = 10) -> None:
        self._keep_last_n = max(1, keep_last_n)

    def do_work(self, payload: str) -> ExampleResult:
        cleaned = payload.strip()
        if not cleaned:
            raise ValueError("payload must not be empty")
        return ExampleResult(ok=True, detail=cleaned)
```

Для кода: низкая temperature, краткие ответы, полный рабочий код без псевдокода, если просят реализацию.
