# 🔥 День 38. Два уровня тестирования: код + UI smoke

## Цель

Полный цикл агентного тестирования:

1. **Уровень 1** — unit/integration на бизнес-логику (≥3 файла, зелёные с первого прогона).
2. **Уровень 2** — UI smoke (логин → создать → проверить → завершить → удалить) через Playwright, со скриншотами и отчётом.
3. **Flow** — после PR / после деплоя фичи агент гоняет оба уровня и собирает **единый** `qa_report`.

---

## Усиление: встроенный QA-flow

| Режим | Когда | Что делает агент |
|-------|--------|------------------|
| **A — после PR** | «прогони QA после PR» | pytest (L1) + smoke (L2) → `qa_report` |
| **B — новая фича** | «задеплоил фичу — обнови smoke и прогони всё» | обновить `scenarios.md` + `run_smoke.py` → полный прогон → `qa_report` |

Инфраструктура:

- Скилл: `.cursor/skills/post-pr-qa/SKILL.md`
- Субагент: `.cursor/agents/post-pr-qa.md`
- Rule: `.cursor/rules/post-pr-qa.mdc`
- Runner: `homeworks/src/day_38_smoke/run_qa_flow.py`

```bash
# Режим A
.venv/bin/python -m homeworks.src.day_38_smoke.run_qa_flow post-pr

# Режим B (после обновления smoke под фичу)
.venv/bin/python -m homeworks.src.day_38_smoke.run_qa_flow after-feature "Завершить задачу в UI"
```

Единый отчёт: `homeworks/artifacts/day_38/qa_report.md` (+ `qa_report.json`).

### Промпт A (после PR)

```text
После PR прогони post-pr-qa: Level-1 pytest + Level-2 smoke, собери единый qa_report.
```

### Промпт B (новая фича)

```text
Я задеплоил новую фичу — обнови smoke-сценарии и прогони всё заново (post-pr-qa, режим B).
```

### Демо режима B (фича «Завершить»)

1. В `webapp.py` добавлен POST `/tasks/complete` + кнопка `data-testid=task-complete`.
2. Обновлены `scenarios.md` (S6) и `run_smoke.py`.
3. Полный прогон `run_qa_flow after-feature "Завершить задачу в UI"` → **Overall PASS** (L1 11 passed, S1–S6 PASS).

---

## Уровень 1. Автотесты на код

### Промпт для ассистента (можно копировать)

```text
Найди в hw01 модули бизнес-логики без (или с слабым) покрытием pytest.
Приоритет: homeworks/todoist (models/storage/service/ai fallback), app/storage.py,
app/services/personalization_service.py — без живой LLM и сети.
Напиши минимум 3 файла tests/test_*.py, запускай через .venv/bin/python -m pytest.
Тесты должны пройти с первого запуска. Временные файлы — только tmp_path.
В конце дай: какие модули закрыл, команды, результат.
```

### Что нашёл агент (непокрытое)

| Модуль | Было |
|--------|------|
| `homeworks/todoist/*` | не покрыт |
| `app/storage.py` | не покрыт напрямую |
| `app/services/personalization_service.py` | не покрыт |

### Новые тесты

| Файл | Что проверяет |
|------|----------------|
| `tests/test_todoist_hw_service.py` | Task/storage CRUD, complete/delete, AI fallback без сети |
| `tests/test_personalization_service.py` | interview flow, snapshot, system_message |
| `tests/test_storage.py` | chat/memory/user/todoist-notification SQLite API |

### Прогон

```bash
.venv/bin/python -m pytest \
  tests/test_todoist_hw_service.py \
  tests/test_personalization_service.py \
  tests/test_storage.py -q
```

**Результат:** `11 passed` с первого запуска.

Дополнительно в `TaskTrackerService`: `delete_task` + UI complete через существующий `complete_task`.

---

## Уровень 2. UI smoke

### Почему не Playwright MCP / Claude in Mobile

В Cursor workspace **нет** Playwright MCP и Node/npx; Claude in Mobile недоступен.  
Эквивалент: локальный demo UI + **Playwright Python** (`homeworks/src/day_38_smoke/run_smoke.py`).

### Demo UI

```bash
.venv/bin/python -m homeworks.todoist.webapp
# http://127.0.0.1:8765/login  — demo / demo123
```

Файл: `homeworks/todoist/webapp.py` (stdlib HTTP + JSON storage).

### Сценарии (текст)

См. `homeworks/src/day_38_smoke/scenarios.md`:

| ID | Сценарий |
|----|----------|
| S1 | Login (`demo` / `demo123`) → `/tasks` |
| S2 | Создать задачу `Smoke task day_38` |
| S3 | Проверить, что задача в списке со статусом `[todo]` |
| S6 | **Завершить** задачу → `[done]`, кнопка complete скрыта |
| S4 | Удалить задачу → список пуст |
| S5 | Logout → `/tasks` снова требует логин |

### Прогон агентом

```bash
.venv/bin/pip install playwright   # один раз
.venv/bin/playwright install chromium
.venv/bin/python -m homeworks.src.day_38_smoke.run_smoke
# или сразу оба уровня:
.venv/bin/python -m homeworks.src.day_38_smoke.run_qa_flow after-feature "Завершить"
```

Артефакты: `homeworks/artifacts/day_38/`

- `qa_report.md` / `qa_report.json` — **единый** отчёт L1+L2
- `smoke_report.md` / `smoke_report.json` — детали UI
- `s1_*.png` … `s6_*.png` — скриншоты шагов

При FAIL отчёт указывает зону: pytest-файл или UI handler / `data-testid`.

---

## Итог цикла

| Этап | Статус |
|------|--------|
| Агент нашёл непокрытое → написал ≥3 test-файла | ✅ |
| pytest зелёный с первого раза | ✅ 11 passed |
| Описаны UI smoke-сценарии (+ S6 complete) | ✅ |
| Агент протыкал UI + скриншоты | ✅ |
| Flow после PR / after-feature → единый `qa_report` | ✅ |
| Сценарий «задеплоил фичу — обнови smoke» | ✅ (Завершить) |

## Где смотреть

```
.cursor/skills/post-pr-qa/SKILL.md
.cursor/agents/post-pr-qa.md
.cursor/rules/post-pr-qa.mdc
homeworks/src/day_38_smoke/run_qa_flow.py
homeworks/src/day_38_smoke/run_smoke.py
homeworks/src/day_38_smoke/scenarios.md
homeworks/todoist/webapp.py
homeworks/artifacts/day_38/qa_report.md
tests/test_todoist_hw_service.py
tests/test_personalization_service.py
tests/test_storage.py
```
