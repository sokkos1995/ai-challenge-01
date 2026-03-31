🔥 День 11. Модель памяти ассистента

Реализована явная модель памяти `memory layers` в `app/agent.py` + CLI-команды в `app/cli.py`.

## 1) Слои памяти

- **Краткосрочная (`ShortTermMemory`)**:
  - `dialog_tail` (последние сообщения диалога),
  - `notes` (краткосрочные заметки через явную команду).
- **Рабочая (`WorkingMemory`)**:
  - `TaskState(task, state, step, total, plan, done, notes)`.
- **Долговременная (`LongTermMemory`)**:
  - `profile` (данные профиля),
  - `decisions` (принятые решения),
  - `knowledge` (устойчивые знания).

## 2) Раздельное хранение

Используется отдельное persistent-хранилище для каждого слоя:

- `<LLM_MEMORY_BASE_PATH>.short.db`
- `<LLM_MEMORY_BASE_PATH>.work.db`
- `<LLM_MEMORY_BASE_PATH>.long.db`

По умолчанию `LLM_MEMORY_BASE_PATH=.llm_memory`.

## 3) Явный выбор записи по слоям

Добавлены команды:

- `@mem short note <text>`
- `@mem work <field>=<value>`
- `@mem long profile <key>=<value>`
- `@mem long knowledge <key>=<value>`
- `@mem long decision <text>`
- `@mem clear short|work|long|all`
- `@mem show`

Это позволяет явно управлять тем, **что** и **в какой слой** сохраняется.

## 4) Влияние на ответы ассистента

Новая стратегия `--context-strategy memory`:

- в prompt добавляется отдельный системный блок с memory layers:
  - short-term notes,
  - working task state,
  - long-term profile/decisions/knowledge;
- плюс последние сообщения (`dialog_tail`) как оперативный контекст.

Итог: ответы учитывают не только текущий диалог, но и явно сохраненные рабочие и долговременные данные.

## 5) Демо-проверка: какие данные попадают в слои

### Шаг 1. Запуск

```bash
python3 llm_cli.py --chat --context-strategy memory
```

### Шаг 2. Явно записываем данные в разные слои

Вставьте в чат:

```text
@mem clear all
@mem short note Отвечай кратко, списком.
@mem work task=Сделать MVP ассистента
@mem work state=in_progress
@mem work total=5
@mem work step=2
@mem work plan+=Сделать auth
@mem work plan+=Сделать billing
@mem work done+=Поднят базовый FastAPI сервис
@mem long profile name=Konstantin
@mem long profile stack=Python, FastAPI, Postgres
@mem long decision Архитектура: monolith + postgres
@mem long knowledge deadline=15 мая
@mem show
```

### Шаг 3. Проверяем, что реально попало в каждый слой

После `@mem show` ожидается структура вида:

- `short_term.notes` содержит: `Отвечай кратко, списком.`
- `working.task/state/step/total` содержит состояние текущей задачи
- `working.plan` и `working.done` содержат план/выполненные шаги
- `long_term.profile` содержит профиль (`name`, `stack`)
- `long_term.decisions` содержит архитектурное решение
- `long_term.knowledge` содержит устойчивое знание (`deadline`)

## 6) Демо-проверка: как memory layers влияет на ответы

### Сравнение A: без memory layers (`full`)

Запустите отдельный сеанс:

```bash
python3 llm_cli.py --chat --context-strategy full
```

Спросите:

```text
Сделай краткий статус: кто я, мой стек, дедлайн, текущий прогресс задачи.
```

Ожидание: если данных нет в текущей истории, ответ будет более общий или с уточнениями.

### Сравнение B: с memory layers (`memory`)

Вернитесь в сеанс `memory` (после заполнения памяти выше) и задайте тот же вопрос:

```text
Сделай краткий статус: кто я, мой стек, дедлайн, текущий прогресс задачи.
```

Ожидание: ответ должен опираться на сохраненную память:

- профиль: `Konstantin`
- стек: `Python, FastAPI, Postgres`
- дедлайн: `15 мая`
- прогресс: `step=2 / total=5`, `state=in_progress`
- сжатый формат (из short-term note): коротко, списком

## 7) Что показать в видео/отчете

- Команды записи в каждый слой (`short`, `work`, `long`)
- `@mem show` с фактическим содержимым слоев
- Один и тот же вопрос в режимах `full` и `memory`
- Короткий вывод:
  - какие данные живут в каком слое;
  - какие элементы ответа улучшились благодаря memory layers.
