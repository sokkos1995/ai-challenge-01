# 🔥 День 36. AGENTS.md, конфиг Cursor-агента и провайдер Cursor

## Какую задачу решал

Настроил «операционную систему» для AI-агента в репозитории: единые правила (`AGENTS.md`), локальные rules/skills/субагенты, безопасный git-workflow и отдельный LLM-провайдер `cursor` через `cursor-sdk`. Параллельно добавил два профиля агентного режима — **Bug Fix** и **Research**.

## Что сделано

### 1. Глобальный и локальный конфиг агента

| Слой | Где | Зачем |
|------|-----|--------|
| Глобальный | `AGENTS.md` | стек, архитектура, naming, примеры/антипаттерны, шаблон файла, commit format `[hw-N] …` |
| Локальный | `.cursor/rules/*.mdc`, `homeworks/todoist/AGENTS.md` | always-on + scoped правила |
| Субагенты | `.cursor/agents/*.md` | делегирование review / тестов / верификации / pre-push |
| Скиллы | `.cursor/skills/*/SKILL.md` | `add-app-service`, `run-pytest` |

Ключевые субагенты инфраструктуры:

- `result-verifier` — обязательная проверка результата перед ответом пользователю
- `pre-push-secrets` — обязательная проверка `git status` и отсутствия секретов перед `git push`
- `python-reviewer`, `test-runner`, `codebase-explorer`

Обязательный workflow в `AGENTS.md`:

1. Если в запросе есть «API»/«апи»/`api` — сначала docs/Swagger в интернете, потом код.
2. Перед финалом — `result-verifier`.
3. Перед `git push` — `pre-push-secrets`.

Секреты:

- реальные ключи только в игнорируемых `.env*`
- `.gitignore`: `.env`, `.env.local`, `.env.*.local`, исключение `!.env.example`
- демо: `.env.example` с плейсхолдерами `...` (можно коммитить)

### 2. Провайдер Cursor (`cursor-sdk`) — параллельная сессия по LLM

В приложение добавлен `LLM_PROVIDER=cursor`:

- `app/config.py` — выбор провайдера и `CURSOR_API_KEY`
- `app/provider_client.py` — вызов через optional `cursor-sdk` (`Agent.prompt`)
- `app/services/provider_service.py` / `token_service.py` — интеграция в общий pipeline
- тесты: `tests/test_config_cursor_provider.py`, `tests/test_provider_client_cursor.py`
- `requirements.txt` — опциональная зависимость `cursor-sdk`
- обновлён корневой `README.md`

Запуск:

```bash
pip install cursor-sdk
# в .env:
# LLM_PROVIDER=cursor
# CURSOR_API_KEY=...
python3 llm_cli.py "Объясни, что такое LLM в 2 предложениях"
```

Ключ: [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations) (`crsr_...`).

### 3. Профили Agent mode: Bug Fix и Research — параллельная сессия

Экспорт сессии: профили сохранены как **agents + rules** (вариант 3).

| Профиль | Субагент | Rule | Режим |
|---------|----------|------|--------|
| **Bug Fix** | `.cursor/agents/bug-fix.md` | `.cursor/rules/bug-fix.mdc` | правка + тесты |
| **Research** | `.cursor/agents/research.md` | `.cursor/rules/research.mdc` | readonly |

Кратко про поведение (детали и примеры промптов также в `homeworks/day_37.md`):

**Bug Fix — ДОЛЖЕН:** найти root cause, минимальный фикс, прогнать pytest, вызвать `result-verifier`.  
**Bug Fix — НЕ ДОЛЖЕН:** сдавать без тестов, drive-by рефакторинг, commit/push без просьбы, трогать `.env`.  
**Формат:** Причина → Что починил → Что проверил → Риски.

**Research — ДОЛЖЕН:** искать/читать код, прослеживать связи CLI → agent → services, опираться на факты.  
**Research — НЕ ДОЛЖЕН:** менять код/конфиг, мутировать окружение, выдумывать модули.  
**Формат:** Файлы → Связи → Выводы → Где править.

Реестр обновлён в `AGENTS.md` и `.cursor/rules/core.mdc`.

## Где смотреть файлы

```
AGENTS.md
.env.example
.gitignore
.cursor/rules/          # core, python-app, tests, bug-fix, research
.cursor/agents/         # result-verifier, pre-push-secrets, bug-fix, research, …
.cursor/skills/         # add-app-service, run-pytest
homeworks/todoist/AGENTS.md
app/config.py
app/provider_client.py
app/services/provider_service.py
tests/test_config_cursor_provider.py
tests/test_provider_client_cursor.py
homeworks/day_37.md     # короткое how-to по Bug Fix / Research
```

## Как пользоваться агентом

```text
# инфраструктура уже подхватывается из AGENTS.md + rules
Упал tests/test_rag_service.py — почини (bug-fix)

Как устроен ProviderService и fallback моделей? (research)
```

Перед push агент обязан прогнать `pre-push-secrets`; секреты в `.env` не коммитятся.

## Коммиты (метка hw-36)

- `[hw-36] Добавить AGENTS.md и конфиг Cursor-агентов`
- `[hw-36] Добавить pre-push проверку секретов и .env.example`
- `[hw-36] Добавить провайдер Cursor через cursor-sdk`
