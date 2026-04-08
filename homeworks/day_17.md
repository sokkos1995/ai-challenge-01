🔥 День 17. Первый инструмент MCP

Реализуйте свой MCP-сервер вокруг любого API (например: Яндекс.Трекер, Git, CRM, mock API)

Сделайте:

👉 регистрацию инструмента
👉 описание входных параметров
👉 возврат результата

Подключите инструмент к своему агенту и:

👉 вызовите его из приложения
👉 получите и используйте результат

Результат:

Агент делает вызов к MCP-инструменту и получает результат

## Что реализовано

Реализован MCP-сервер вокруг Todoist API и интегрирован в приложение:

- сервер: `app/mcp_servers/todoist_server.py`
- демо-клиент: `app/mcp_todoist_demo.py`

Также добавлен шаблон MCP-сервера для GitHub API:

- `app/mcp_servers/github_server.py`

## Инструменты Todoist MCP

Зарегистрированы инструменты:

- `list_tasks(project_id: str = "", limit: int = 20)`
- `create_task(content: str, project_id: str = "", due_string: str = "")`
- `complete_task(task_id: str)`

Для каждого инструмента есть описание входных параметров и возврат результата (JSON).

## Как запускать демо (вызов из приложения)

Переменная окружения:

- `TODOIST_API_TOKEN=...`

Команды:

```bash
python3 -m app.mcp_todoist_demo --action list --limit 5
python3 -m app.mcp_todoist_demo --action create --content "Buy milk" --due-string "tomorrow"
python3 -m app.mcp_todoist_demo --action complete --task-id 1234567890
```

## Что показывает демо

Демо-клиент автоматически:

1. поднимает MCP-сервер Todoist через `stdio`,
2. выполняет `initialize`,
3. получает `list_tools`,
4. вызывает выбранный инструмент и выводит результат.

Результат: агент делает вызов к MCP-инструменту и получает результат.

## Как подключить MCP в Cursor (Todoist)

1. Откройте настройки MCP в Cursor и добавьте локальный сервер.
2. Укажите команду запуска через модуль Python и явно задайте `PYTHONPATH`:

```json
{
  "mcpServers": {
    "todoist-local": {
      "command": "python3",
      "args": ["-m", "app.mcp_servers.todoist_server"],
      "env": {
        "PYTHONPATH": "/Users/konstantinsokolov/dev/projects/pet_projects/ai_challenge/hw/hw01",
        "TODOIST_API_TOKEN": "your_todoist_token"
      }
    }
  }
}
```

3. Перезапустите/переподключите MCP-сервер в Cursor.
4. Проверьте, что видны инструменты:
   - `list_tasks`
   - `create_task`
   - `complete_task`
5. Сделайте тестовый вызов инструмента из чата Cursor (например: "Вызови `list_tasks` с `limit=5`").

Почему это нужно: Cursor может запускать MCP-процесс не из корня проекта, и без `PYTHONPATH` модуль `app` не импортируется (`ModuleNotFoundError: No module named 'app'`).

Если токен хранится в `.env`, можно не дублировать его в `env`, если Cursor запускается в окружении, где эта переменная уже экспортирована.