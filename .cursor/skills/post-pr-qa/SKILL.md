---
name: post-pr-qa
description: После PR или деплоя фичи прогоняет Level-1 pytest + Level-2 UI smoke и собирает единый qa_report. Используйте при «после PR проверь», «задеплоил фичу — обнови smoke и прогони всё».
---

# Post-PR / after-feature QA (day_38)

Два режима одного flow.

## Режим A — после PR (оба уровня + единый отчёт)

Триггеры: «после PR», «прогони QA», «post-pr-qa», ссылка на PR.

1. Уточните scope PR (diff / изменённые пути), но **всегда** гоняйте оба уровня day_38.
2. Запустите единый runner:
   ```bash
   .venv/bin/python -m homeworks.src.day_38_smoke.run_qa_flow post-pr
   ```
3. Прочитайте `homeworks/artifacts/day_38/qa_report.md` (+ `smoke_report.md` при UI-fail).
4. В ответе пользователю:
   - Overall PASS/FAIL
   - Level 1 краткий результат
   - Level 2 по сценариям S*
   - При FAIL — где проблема (тест-файл / UI handler / `data-testid`)
5. Перед финалом — `result-verifier`.

Не ходите в живые LLM API. Не трогайте `.env` / реальные `.llm_*.db`.

## Режим B — «я задеплоил новую фичу — обнови smoke и прогони всё заново»

Триггеры: «задеплоил фичу», «обнови smoke», «feature deployed».

1. Изучите новую фичу (diff UI/`webapp.py`/`service.py`/маршруты/`data-testid`).
2. Обновите текстовые сценарии: `homeworks/src/day_38_smoke/scenarios.md`.
3. Обновите автоматизацию: `homeworks/src/day_38_smoke/run_smoke.py` (новые шаги + скриншоты).
4. При необходимости добавьте/поправьте unit-тесты в `tests/` (Level 1).
5. Прогоните всё заново:
   ```bash
   .venv/bin/python -m homeworks.src.day_38_smoke.run_qa_flow after-feature "кратко: что за фича"
   ```
6. Зафиксируйте итог в `homeworks/day_38.md` (секция усиления / changelog smoke).
7. Ответ: что добавили в smoke → Overall → ссылка на `qa_report.md`.

## Команды по отдельности (если нужно точечно)

```bash
# Level 1
.venv/bin/python -m pytest tests/test_todoist_hw_service.py tests/test_personalization_service.py tests/test_storage.py -q

# Level 2
.venv/bin/python -m homeworks.src.day_38_smoke.run_smoke
```

Demo UI: `.venv/bin/python -m homeworks.todoist.webapp` → `http://127.0.0.1:8765/login` (`demo`/`demo123`).
