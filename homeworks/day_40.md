🔥 День 5. Execution Loop: кто дольше продержится без паузы

## Трекер: Todoist MCP (`todoist-local`)

Cursor MCP (`~/.cursor/mcp.json`):

- command: `.venv/bin/python -m app.mcp_servers.todoist_server`
- `cwd` + `PYTHONPATH` = корень hw01
- `TODOIST_API_TOKEN` в env MCP (после дебага сменить токен)

Инструменты: `list_tasks`, `create_task`, `complete_task`.

Фильтр пула: контент начинается с `[day40]`.
Манифест: `homeworks/artifacts/day_40/task_pool.json`.

## Пул задач (18) — изучение Python, микс типов

| # | Тип | id | Кратко | Done |
|---|-----|----|--------|------|
| 1 | feature | `6h8HWJq8Hg8vVP37` | FizzBuzz CLI | скрипт + usage/exit 2 |
| 2 | feature | `6h8HWJmC49qFqFg7` | wordcount | слова/строки/символы + `--help` |
| 3 | feature | `6h8HWJvmgvF6W4Mf` | json_pretty | indent=2 / exit 1 |
| 4 | feature | `6h8HWJxRXfM95g37` | temp_convert | c2f/f2c, 1 знак |
| 5 | feature | `6h8HWM66CV39CG77` | csv_to_md | MD-таблица |
| 6 | bug | `6h8HWM8vQ6GPcXqf` | broken_avg | пустой → None |
| 7 | bug | `6h8HWM79vHj29697` | broken_slice | ровно n последних |
| 8 | bug | `6h8HWM9Qc85vRC6f` | broken_append | нет shared list |
| 9 | refactor | `6h8HWMHW3MvvgMhf` | monolith_stats | mean/median/mode |
| 10 | refactor | `6h8HWMQCmxc88Mvf` | validate_email | `is_valid_email` |
| 11 | test | `6h8HWMMWJChW8WC7` | test fizzbuzz | pytest ≥3 зелёный |
| 12 | test | `6h8HWMWQH92WFww7` | test wordcount | pytest зелёный |
| 13 | test | `6h8HWMWV77h6M8r7` | test bugfixes | pytest зелёный |
| 14 | docs | `6h8HWMfwVX8p74Cf` | README day_40 | примеры запуска |
| 15 | docs | `6h8HWMpxRH298Xcf` | typing_cheatsheet | ≥8 примеров |
| 16 | feature | `6h8HWMmpgmvg5Jv7` | password_gen | length≥8 |
| 17 | feature | `6h8HWMqWGRC8QHJf` | anagrams | YES/NO |
| 18 | refactor | `6h8HWMxg3G6RQgv7` | student_grade | `@dataclass` |

Стартовые broken/refactor файлы уже лежат в `homeworks/artifacts/day_40/`.

## Протокол execution loop

1. `list_tasks` → взять следующую незакрытую с префиксом `[day40]` (порядок: bug → feature → refactor → test → docs, иначе FIFO).
2. Выбрать профиль: `[bug]` → Bug Fix; research/чтение → Research; иначе фича/docs/test.
3. Выполнить до критерия Done; прогнать ручную проверку / pytest.
4. Закоммитить: `[hw-40] <кратко>` (только по явному запросу на коммит в чате loop, либо по договорённости loop-сессии).
5. `complete_task(task_id)`.
6. Записать строку в лог → следующая задача. Без правок промпта между задачами.

Лог: `homeworks/artifacts/day_40/execution_log.md`  
Метрики: `homeworks/artifacts/day_40/metrics.json`

## Метрики (заполняется после прогонов)

| Прогон | Модель | Подряд без паузы | Сломался на | Причина | Ср. время/задачу | % с 1-го раза | Минут без пауз |
|--------|--------|------------------|-------------|---------|------------------|---------------|----------------|
| 1 | cloud (Cursor) | 18/18 | — | — | ~0.2 мин | 100% | **3** |
| 2 | cloud + `execution-loop` rule | 18/18 | — | — | ~0.1 мин | 100% | **2** |
| 3 | local LLM | — | — | — | — | — | — |

Сравнение 1→2: streak без изменения (18/18); время без пауз **3 → 2 мин** (−1 мин) за счёт правила «fallback close Todoist без остановки на MCP auto-review».

## Правило после прогона 1

Файл: `.cursor/rules/execution-loop.mdc` — коммиты в loop-сессии, порядок очереди, Done→close, fallback `complete_task` через Python.

## Статус подготовки

- [x] Todoist MCP в Cursor (`todoist-local`)
- [x] Токен в `.env` + env MCP (сменить после дебага)
- [x] Пул 18 задач в Todoist
- [x] Стартовые артефакты для bug/refactor
- [x] Прогон 1 (execution loop) — 18/18, **3 минуты** без пауз
- [x] Доработка правил / прогон 2 — 18/18, **2 минуты** без пауз
- [ ] Сравнение с локальной моделью

Формат сдачи: **количество времени без пауз (в минутах)** по прогонам.

Прогон 1 (факт): **3 минуты**.  
Прогон 2 (факт): **2 минуты**.
