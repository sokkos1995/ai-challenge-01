# Refactoring TODO

## Context

В проекте уже реализованы:
- создание задач Todoist из chat-режима;
- polling Todoist reminders через `TodoistReminderService`;
- вывод reminder-сообщений прямо в CLI чат.

Следующий этап должен делать другой агент. Ниже собраны результаты второго прохода code review и ожидаемый scope рефакторинга.

## Summary

Текущая реализация в целом рабочая, но есть три проблемы в архитектуре и поведении:

1. `TodoistChatService` может ошибочно создать задачу из обычного вопроса пользователя.
2. При недоступном Todoist chat startup может упасть еще до входа в основной цикл.
3. Reminder-сообщения и ответы на успешные Todoist-команды не попадают в chat history / memory, поэтому агент не видит их как часть диалога.

## Tasks

### 1. Tighten Todoist intent parsing

Проблема:
- `app/services/todoist_chat_service.py` слишком агрессивно распознает create intent.
- Сообщения с `todoist` / `тудуист` и словом `задач...` могут интерпретироваться как команда на создание, даже если это вопрос.

Что нужно сделать:
- сузить условия распознавания create intent;
- создавать задачу только по явным командам, например:
  - `create_task ...`
  - `создай задачу ...`
  - `добавь задачу ...`
- не создавать задачу по вопросам вроде:
  - `Какие задачи в тудуисте у меня сегодня?`
  - `Покажи задачи в Todoist`
  - `Есть ли у меня задача на сегодня?`

Где менять:
- `app/services/todoist_chat_service.py`
- при необходимости тесты: `tests/test_todoist_chat_service.py`

Критерий приемки:
- create intent срабатывает только на явные create-команды;
- read-only вопросы не приводят к вызову `create_task`.

### 2. Make reminder startup non-fatal

Проблема:
- в `app/cli.py` reminder-service стартует до входа в chat loop;
- в `TodoistReminderService.start()` выполняется синхронный `prime_existing_due_tasks()`;
- если Todoist / MCP временно недоступен, чат может вообще не стартовать.

Что нужно сделать:
- сделать запуск reminders best-effort;
- если reminder initialization падает, chat должен все равно стартовать;
- ошибку reminder можно:
  - логировать в stderr, или
  - печатать один раз как `agent> Todoist reminder error: ...`,
  но без завершения CLI.

Где менять:
- `app/cli.py`
- `app/services/todoist_reminder_service.py`

Критерий приемки:
- при ошибке Todoist chat стартует и остается usable;
- reminder subsystem деградирует отдельно от основного чата.

### 3. Persist tool/reminder outputs into chat context

Проблема:
- reminder-сообщения и ответы `TodoistChatService` печатаются в stdout, но не попадают в chat history / memory;
- из-за этого агент не может опираться на них как на часть предыдущего диалога.

Что нужно сделать:
- определить единый способ записывать system/tool-generated assistant messages в историю;
- после успешного Todoist create reply добавлять эту реплику в chat context;
- reminder-сообщения тоже по возможности сохранять как assistant messages, не ломая текущую архитектуру.

Возможный путь:
- добавить в `SimpleLLMAgent` или `ChatContextService` отдельный метод для append-only assistant/system message;
- использовать его из `app/cli.py` там, где сейчас сообщения только печатаются.

Где менять:
- `app/cli.py`
- `app/agent.py`
- `app/services/chat_context_service.py`
- возможно `app/services/chat_history_service.py` или связанные memory services

Критерий приемки:
- после reminder/tool reply эти сообщения доступны в последующем контексте;
- поведение не ломает существующие режимы `full`, `summary`, `memory`, `sliding`, `facts`, `branching`.

## Suggested Order

1. Исправить распознавание create intent.
2. Защитить startup чата от падений reminder subsystem.
3. После этого аккуратно интегрировать reminder/tool outputs в chat history.

## Test Coverage To Add

- Негативные тесты на parser:
  - вопрос про Todoist не создает задачу;
  - сообщение без явного create action не уходит в `create_task`.
- Тест на graceful startup:
  - при ошибке reminder initialization chat не падает.
- Тест на запись tool/reminder output в историю:
  - после такой реплики следующий context build видит ее в истории.

## Notes For Next Agent

- Не менять пользовательский контракт reminder MVP: reminders должны продолжать приходить автоматически в `--chat`.
- Не убирать текущую дедупликацию через SQLite.
- Не откатывать текущую оптимизацию polling через Todoist filter `today | overdue`.
