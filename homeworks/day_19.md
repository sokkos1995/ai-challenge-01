🔥 День 19. Композиция MCP-инструментов

Создайте несколько MCP-инструментов, например:

👉 search
👉 summarize
👉 saveToFile

Реализуйте пайплайн:

👉 первый инструмент получает данные
👉 второй — обрабатывает
👉 третий — сохраняет результат

Проверьте:

👉 автоматическое выполнение цепочки
👉 корректность передачи данных между инструментами

Результат:

Автоматический пайплайн из нескольких MCP-инструментов

## Реализация

Сделано 3 шага в цепочке:

1. `Todoist MCP` (`list_tasks`) получает задачи по фильтру (для команды про "сегодня" используется `filter_query="today"`).
2. Локальный MCP-инструмент `summarize_workload` строит краткую сводку загрузки в Markdown.
3. Локальный MCP-инструмент `save_summary_to_file` сохраняет сводку в `workload/*.md`.

Файлы:
- `homeworks/src/day_19_pipeline.py` — оркестратор пайплайна;
- `homeworks/src/day_19_pipeline_tools_server.py` — MCP-инструменты `summarize_workload` и `save_summary_to_file`.

## Запуск

Подготовить `.env`:

```dotenv
TODOIST_API_TOKEN=ваш_todoist_token
```

Запуск:

```bash
python3 homeworks/src/day_19_pipeline.py --command "какие у меня задачи на сегодня"
```

Опционально:

```bash
python3 homeworks/src/day_19_pipeline.py \
  --command "какие у меня задачи на сегодня" \
  --output-file "today-workload.md"
```
