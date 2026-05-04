# AI Task Tracker (Todoist-like)

Мини-сервис таск-трекера с AI-декомпозицией цели на задачи и подзадачи.

## Возможности

- Добавление задач вручную.
- Список задач со статусом и приоритетом.
- Закрытие задач.
- AI-команда `ai_plan`: из реальной цели создаёт набор задач.
- Персистентность в JSON (`homeworks/todoist/data/tasks.json`).

## Быстрый старт

```bash
python3 -m homeworks.todoist.main demo
```

Или вручную:

```bash
python3 -m homeworks.todoist.main add "Собрать метрики релиза" --priority high
python3 -m homeworks.todoist.main ai_plan "Подготовить релиз мобильного приложения"
python3 -m homeworks.todoist.main list
```

## AI-режим

Для реального LLM укажите один из ключей:

- `LLM_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`

Если ключей нет, сервис использует fallback-декомпозицию, чтобы демо всё равно работало.
