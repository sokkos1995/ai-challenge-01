🔥 День 13. Состояние задачи (Task State Machine)

Реализуйте состояние задачи как конечный автомат:

👉 этап задачи
👉 текущий шаг
👉 ожидаемое действие

Пример состояний:

👉 planning → execution → validation → done

Проверьте:

👉 паузу на любом этапе
👉 продолжение без повторных объяснений

Результат:

Агент с формализованным состоянием задачи

Демо:

```text
$ python3 llm_cli.py --chat --context-strategy memory --user-id demo

you> @task show
agent> task:
{
  "task": "",
  "state": "PLANNING",
  "paused": false,
  "step": 0,
  "total": 0,
  "expected_action": "",
  "allowed_transitions": ["EXECUTION"],
  "plan": [],
  "done": [],
  "notes": []
}

you> @mem work task=Сделать task state machine
agent> working memory updated: task

you> @task plan+ Собрать требования и согласовать стадии
agent> task plan updated.

you> @task expected Утвердить план
agent> task expected action updated.

you> @task pause
agent> task paused.
```

Проверка паузы на любом этапе:
- задача поставлена на паузу в `PLANNING`;
- аналогично можно поставить на паузу в `EXECUTION` или `VALIDATION` через `@task pause`.

```text
# Перезапускаем CLI
$ python3 llm_cli.py --chat --context-strategy memory --user-id demo

you> @task show
agent> task:
{
  "task": "Сделать task state machine",
  "state": "PLANNING",
  "paused": true,
  "step": 0,
  "total": 0,
  "expected_action": "Утвердить план",
  "allowed_transitions": ["EXECUTION"],
  "plan": ["Собрать требования и согласовать стадии"],
  "done": [],
  "notes": []
}

you> @task resume
agent> task resumed.

you> Перейди к выполнению. План уже согласован, начинай реализацию storage и prompt builder.
agent> Перехожу к стадии EXECUTION. По сохраненному контексту задача уже определена:
- сделать task state machine;
- сохранить состояние в SQLite;
- подмешивать state, plan и expected action в prompt.
Начинаю с storage migration и prompt builder.
```

Проверка продолжения без повторных объяснений:
- после рестарта агент восстановил `task`, `state`, `paused`, `expected_action` и `plan` из SQLite;
- после `@task resume` можно продолжать с того же места без повторного описания задачи.

Отдельный сценарий: пауза на `EXECUTION`

```text
$ python3 llm_cli.py --chat --context-strategy memory --user-id demo-exec

you> @mem work task=Собрать task pipeline
agent> working memory updated: task

you> @task plan+ Описать state machine
agent> task plan updated.

you> @task state EXECUTION
agent> task state updated: EXECUTION

you> @task expected Реализовать migration и prompt builder
agent> task expected action updated.

you> @task done+ Описаны стадии и разрешенные переходы
agent> task done updated.

you> @task pause
agent> task paused.
```

```text
# Перезапускаем CLI
$ python3 llm_cli.py --chat --context-strategy memory --user-id demo-exec

you> @task show
agent> task:
{
  "task": "Собрать task pipeline",
  "state": "EXECUTION",
  "paused": true,
  "step": 0,
  "total": 0,
  "expected_action": "Реализовать migration и prompt builder",
  "allowed_transitions": ["VALIDATION", "PLANNING"],
  "plan": ["Описать state machine"],
  "done": ["Описаны стадии и разрешенные переходы"],
  "notes": []
}

you> @task resume
agent> task resumed.

you> Продолжай выполнение.
agent> Продолжаю стадию EXECUTION без повторного сбора требований. Уже известно, что:
- задача: собрать task pipeline;
- план содержит описание state machine;
- уже выполнено описание стадий и переходов;
- следующий ожидаемый шаг: реализовать migration и prompt builder.
```
