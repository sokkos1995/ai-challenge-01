# AGENTS.md — hw01 LLM CLI Agent

Инструкции для AI-агентов, работающих с этим репозиторием.
Читайте этот файл перед любыми изменениями кода.

---

## Cursor config layers (обязательно использовать)

| Слой | Где лежит | Когда применять |
|------|-----------|-----------------|
| **Глобальный конфиг** | этот `AGENTS.md` (+ User Rules в Cursor, если заданы) | всегда: стек, архитектура, naming, антипаттерны |
| **Локальный конфиг** | `.cursor/rules/*.mdc`, вложенные `**/AGENTS.md` | по `globs` / каталогу (например `app/`, `tests/`, `homeworks/todoist/`) |
| **Субагенты** | `.cursor/agents/*.md` | делегировать review / тесты / исследование, не раздувая основной контекст |
| **Скиллы** | `.cursor/skills/*/SKILL.md` | повторяемые процедуры (новый сервис, день ДЗ, прогон тестов) |

Правила приоритета:

1. Ближайший вложенный `AGENTS.md` > корневой `AGENTS.md`.
2. Scoped `.mdc` с `globs` дополняет глобальный конфиг, не противоречит ему.
3. Перед типовой задачей сначала проверьте подходящий skill, затем при необходимости делегируйте субагенту.

---

## Обязательный workflow агента

### 1. Внешние API — сначала документация

Если в запросе пользователя есть слово **«API» / «апи» / `api`** (в любом регистре) **или** задача подразумевает вызов/интеграцию внешнего API:

1. **До написания кода** найдите актуальную документацию в интернете: официальный docs, OpenAPI/Swagger, reference, changelog.
2. Зафиксируйте: base URL, auth, нужные endpoints, формат request/response, известные deprecations/fallback.
3. Только после этого проектируйте клиент/MCP-tool/сервис. Не опирайтесь на память о старых путях API.
4. Если docs недоступны — явно напишите это пользователю и что взяли как fallback-источник (существующий код в репо).

### 2. Проверка результата перед ответом пользователю

Перед завершением задачи (после кода/фиксов/команд) **обязательно** делегируйте проверку субагенту `result-verifier`:

1. Основной агент собирает: что сделано, какие файлы изменены, как проверить.
2. Запускает субагент `result-verifier` (отдельный контекст).
3. Если верификатор нашёл blocker/major — чинит и повторяет проверку.
4. Только после вердикта «ок» (или явного списка оставшихся рисков, согласованного с проверкой) выдаёт итог пользователю.

Нельзя отдавать финальный ответ пользователю, минуя шаг `result-verifier`, если задача подразумевала изменения кода, интеграцию API или проверку поведения.

---

## Стек

- **Язык:** Python 3.8+ (в новых файлах предпочитайте `from __future__ import annotations` и синтаксис `str | None`).
- **Точка входа:** `llm_cli.py` → `app.cli.main`.
- **LLM API:** OpenRouter / Groq через OpenAI-compatible `chat/completions`; Cursor через Cloud Agents API (`POST /v1/agents` + poll run) — всё на `urllib`, без тяжёлых SDK.
- **Конфиг:** `.env` + `app/config.py` (`load_env_file`, `get_provider_config`).
- **Хранение:** SQLite (`app/storage.py`) для chat/memory; JSON для homework Todoist (`homeworks/todoist/`).
- **MCP:** официальный SDK `mcp` + `FastMCP` (`app/mcp_servers/`).
- **RAG:** локальный индекс JSON + heuristic retrieval/rerank (`app/services/rag_service.py`).
- **Тесты:** `pytest` в `tests/`.
- **Зависимости:** `requirements.txt` (`certifi`, `mcp`).

Секреты только из env (`.env` не коммитить): `OPENROUTER_API_KEY` / `GROQ_API_KEY` / `CURSOR_API_KEY` / `TODOIST_API_TOKEN` / `GITHUB_TOKEN`.

---

## Архитектура

```
CLI (app/cli.py, llm_cli.py)
  └─ SimpleLLMAgent (app/agent.py)          # фасад, публичный API
       ├─ ProviderService                   # HTTP к LLM / Cursor Cloud Agents + fallback
       ├─ ChatContextService                # стратегии контекста
       ├─ ChatHistoryService / MemoryService
       ├─ PersonalizationService
       ├─ RagService / RagChatService
       ├─ TaskLifecycleGuardService / InvariantGuardService
       └─ Todoist*Service                   # reminders / chat integration
  └─ MCP servers (stdio): todoist / github / git
Storage: SQLite helpers в app/storage.py (не сырой SQL из сервисов)
Models: dataclasses в app/models.py
```

Принципы:

- **Фасад + сервисы:** `SimpleLLMAgent` тонкий; логика в `app/services/*_service.py`.
- **Config из env:** фабрики `from_env` / хелперы в `app/config.py`.
- **Dataclass-модели:** состояние и DTO без ORM.
- **Guards:** инварианты и lifecycle проверяются до/после ответа модели.
- **Grounded RAG:** ответ = `answer` + `sources` + `quotes`; при слабом контексте — «Не знаю…».
- **MCP как граница внешних API:** Todoist/GitHub/Git через tools, не через ad-hoc curl в бизнес-логике.

---

## Структура папок

```
app/                      # основное приложение
  agent.py                # SimpleLLMAgent
  cli.py / cli_utils.py   # CLI
  config.py               # env / provider / paths
  cursor_cloud_client.py  # Cursor Cloud Agents API (create/poll/archive)
  models.py               # dataclasses
  storage.py              # SQLite persistence API
  task_state_machine.py   # стадии задачи + переходы
  services/               # *Service классы
  mcp_servers/            # FastMCP servers + _http.py
tests/                    # pytest
homeworks/                # дневные отчёты day_XX.md + src/
  todoist/                # мини task-tracker (day 35)
docs/                     # база знаний для RAG
scripts/                  # утилиты (PR AI review и т.п.)
.cursor/rules/            # локальные scoped-правила
.cursor/agents/           # субагенты
.cursor/skills/           # скиллы
```

Не трогать без явной просьбы: `.llm_*.db`, `.llm_users/`, `.env`, артефакты индексов в `homeworks/artifacts/` (если есть).

---

## Naming conventions

| Что | Как |
|-----|-----|
| Модули / пакеты | `snake_case.py` |
| Классы | `PascalCase`, сервисы с суффиксом `Service` (`RagService`, `ProviderService`) |
| Функции / методы | `snake_case`; приватные `_helper` |
| Константы | `UPPER_SNAKE` (`TASK_STAGE_PLANNING`) |
| Env-переменные | `LLM_*`, `CURSOR_*`, `TODOIST_*`, `GITHUB_*` |
| Chat-команды | `@name` / `/help` |
| Homework-скрипты | `homeworks/src/day_NN_*.py`, отчёты `homeworks/day_NN.md` |
| Тесты | `tests/test_<module>.py`, функции `test_<behavior>` |
| MCP tools | глагол + объект: `list_tasks`, `create_task` |
| Commit message | `[hw-<номер дз>] <краткое пояснение на русском>` |

### Commit messages

Формат обязателен:

```text
[hw-<номер дз>] <краткое пояснение на русском>
```

Примеры:

- `[hw-35] Добавить мини task-tracker с AI-декомпозицией цели`
- `[hw-24] Перенести grounded RAG в app/services`
- `[hw-16] Подключить MCP-серверы Todoist и GitHub`

Правила:

- `<номер дз>` — номер дня домашки без ведущих нулей (`35`, не `035`), если правка относится к конкретному дню.
- Пояснение — одно короткое предложение на русском, без точки в конце.
- Коммитьте только когда пользователь явно попросил создать commit.

---

## Паттерны

1. **Composition over god-object** — агент делегирует сервисам.
2. **Env-first config** — `load_env_file()` + typed helpers (`positive_int_from_env`).
3. **Parameterized SQLite** — SQL только в `app/storage.py` (или узком storage-модуле), через `?`-плейсхолдеры.
4. **Explicit errors** — `RuntimeError` / `ValueError` с понятным текстом; `raise ... from exc`.
5. **Fallback** — LLM models, Todoist API v2→v1, AI plan → `fallback_plan`.
6. **Immutable-ish updates** — `Task.with_status(...)` возвращает новый dataclass.
7. **CLI side effects отдельно** — `print` допустим в CLI/`*_command_service`, не в core domain без нужды.

---

## Примеры хорошего кода (как ХОТИМ писать)

### 1. Фасад агента + делегирование сервисам

```python
# app/agent.py — тонкий фасад, зависимости собраны в __init__
self._provider_service = ProviderService(...)
self._token_service = TokenAccountingService(self._provider_service)
self._context_service = ChatContextService(
    chat_keep_last_n=self.chat_keep_last_n,
    chat_history_service=self._chat_history_service,
    memory_service=self._memory_service,
    personalization_service=self._personalization_service,
)
```

### 2. Dataclass-модель с фабрикой и иммутабельным обновлением

```python
# homeworks/todoist/models.py
@classmethod
def create(cls, title: str, description: str = "", *, priority: str = "medium", ...) -> "Task":
    now = utc_now_iso()
    return cls(id=f"tsk_{uuid4().hex[:10]}", title=title.strip(), status="todo", ...)

def with_status(self, status: str) -> "Task":
    data = asdict(self)
    data["status"] = status
    data["updated_at"] = utc_now_iso()
    return Task(**data)
```

### 3. Storage API вместо SQL в сервисе

```python
# homeworks/todoist/service.py — бизнес-логика без знания формата файла
class TaskTrackerService:
    def __init__(self, db_path: Path) -> None:
        self.storage = JsonTaskStorage(db_path)

    def complete_task(self, task_id: str) -> Task:
        tasks = self.storage.load_tasks()
        ...
        self.storage.save_tasks(updated)
        return changed
```

### 4. State machine с явными переходами

```python
# app/task_state_machine.py
ALLOWED_TASK_STAGE_TRANSITIONS: Final[dict[str, tuple[str, ...]]] = {
    TASK_STAGE_PLANNING: (TASK_STAGE_EXECUTION, TASK_STAGE_REJECTED),
    TASK_STAGE_EXECUTION: (TASK_STAGE_VALIDATION, TASK_STAGE_PLANNING, TASK_STAGE_REJECTED),
    ...
}
```

### 5. HTTP helper с keyword-only args и понятными ошибками

```python
# app/mcp_servers/_http.py
def request_json(*, method: str, url: str, token: str, payload: dict[str, Any] | None = None, ...) -> dict[str, Any] | list[Any]:
    ...
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
```

---

## Антипаттерны (запрещено)

1. **`Any` / отсутствие типов в публичном API** — не пишите `def foo(x):` без аннотаций; `Any` только на границе сырого JSON/HTTP и по возможности сужайте сразу.
2. **`print` / debug-логи в domain-слое «для прода»** — диагностику в сервисах пишите в `sys.stderr` и только для реальных warnings; отладочный шум не оставлять. CLI `print` — только в `cli` / command handlers.
3. **Сырой SQL вне storage** — запрещены `sqlite3.connect` + строки запросов внутри `*Service` / agent. Используйте функции из `app/storage.py` (или dedicated storage-класса).
4. **Секреты в коде** — запрещены хардкод API keys, токенов, `.env` в git.
5. **God-module** — не пихайте provider + RAG + memory + CLI в один файл; новый функционал → новый `*_service.py` / расширение существующего сервиса по ответственности.
6. **Ломание grounded RAG** — нельзя убирать обязательные `sources`/`quotes` или отвечать «из головы» при `low_relevance`.

---

## Шаблон типичного файла сервиса

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models import AgentResponse  # stdlib → third-party → local app.*


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

    def _private_helper(self, value: str) -> str:
        return value.lower()


# Экспорт: класс сервиса (+ dataclass DTO при необходимости).
# Не делайте `from app.services.example_service import *`.
```

Порядок в файле:

1. `from __future__ import annotations` (если нужен)
2. stdlib → third-party → `app.*` / относительные импорты пакета
3. константы / dataclass DTO
4. класс сервиса (`__init__` → public methods → `_private`)
5. без side-effect кода на import-time (кроме MCP `server = FastMCP(...)`)

---

## Команды

```bash
# CLI
python3 llm_cli.py "вопрос"
python3 llm_cli.py --chat --context-strategy memory --user-id 123
python3 llm_cli.py --chat --rag

# MCP
python3 -m app.mcp_servers.todoist_server
python3 -m app.mcp_servers.github_server
python3 -m app.mcp_servers.git_server

# Todoist homework
python3 -m homeworks.todoist.main demo

# Тесты
python3 -m pytest tests/ -q
```

---

## Когда звать субагентов и скиллы

- **Скилл `add-app-service`** — новый сервис в `app/services/`.
- **Скилл `run-pytest`** — после изменений логики прогнать релевантные тесты.
- **Субагент `python-reviewer`** — ревью диффа на соответствие AGENTS.md.
- **Субагент `test-runner`** — изолированный прогон/починка тестов.
- **Субагент `codebase-explorer`** — широкий поиск по репо перед крупным рефакторингом.
- **Субагент `result-verifier`** — **обязателен** перед финальным ответом пользователю: проверка, что результат соответствует запросу и работает.

Не дублируйте длинные куски README в ответах пользователю — ссылайтесь на файлы и меняйте код точечно.
