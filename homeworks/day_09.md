## Задание

🔥 День 9. Управление контекстом: сжатие истории

Реализуйте механизм управления контекстом:

👉 храните последние N сообщений “как есть”
👉 остальное заменяйте summary (например каждые 10 сообщений)
👉 храните summary отдельно и подставляйте его в запрос вместо полной истории

Сравните:

👉 качество ответов без сжатия
👉 качество ответов со сжатием
👉 расход токенов до/после

Результат:

Агент, который работает с компрессией истории и экономит токены

---

## Что уже реализовано в проекте

- Обычный чат: `python3 llm_cli.py --chat`
- Чат с компрессией: `python3 llm_cli.py --chat --summary`
- В summary-режиме:
  - хранится отдельное накопительное summary старых сообщений;
  - последние `N` сообщений держатся "как есть";
  - в запрос отправляется `summary + последние N + текущий user prompt`.

Параметры:

- `LLM_CHAT_KEEP_LAST_N` (по умолчанию `10`)
- `LLM_CHAT_SUMMARY_BATCH_SIZE` (по умолчанию `10`)

---

## Демо: быстрый ручной прогон

### 1) Подготовка

```bash
set -a
source .env
set +a
```

Опционально зафиксируйте параметры окна/батча:

```bash
export LLM_CHAT_KEEP_LAST_N=10
export LLM_CHAT_SUMMARY_BATCH_SIZE=10
```

### 2) Сценарий диалога (одинаковый для обоих режимов)

Используйте один и тот же набор реплик. Пример:

1. "Запомни: мой стек Python + FastAPI + Postgres."
2. "Запомни: дедлайн проекта 15 мая."
3. "Предпочитаю ответы коротко, списком."
4. "Составь план MVP на 2 недели."
5. "Добавь риски и как их снизить."
6. "Что я говорил про стек?"
7. "Какой дедлайн?"
8. "Переформулируй план в 5 пунктах."
9. "Добавь чеклист на запуск."
10. "Сделай итог: стек, дедлайн, план."

### 3) Режим A: без сжатия

```bash
python3 llm_cli.py --chat --tokens
```

Прогоните все 10 реплик, сохраните:
- качество итогового ответа (насколько точно восстановлены факты);
- `prompt_tokens_total`, `dialog_history_tokens`, `total_tokens` на последних 2-3 сообщениях.

### 4) Режим B: со сжатием

```bash
python3 llm_cli.py --chat --summary --tokens
```

Прогоните те же 10 реплик и сравните метрики.  
Команда `@summary` покажет, что именно попало в сжатую память.

---

## Мини-бенчмарк (токены + latency)

Ниже шаблон таблицы, который можно заполнить по результатам запуска:

| Режим | Качество ответа (0-5) | prompt_tokens_total (среднее) | dialog_history_tokens (среднее) | total_tokens (среднее) | latency_ms (среднее) |
|---|---:|---:|---:|---:|---:|
| Без сжатия (`--chat`) |  | 2568 | 2497 | 2662 |  |
| Со сжатием (`--chat --summary`) |  | 1661 | 1590 | 1785 |  |

Рекомендация по оценке качества:

- **5**: все ключевые факты сохранены (стек, дедлайн, формат, план).
- **4**: мелкие потери формулировки, но факты корректны.
- **3**: 1 важный факт потерян/искажен.
- **2 и ниже**: потеря нескольких ключевых фактов или логики.

---

## Пример автоматизированного прогона (код)

Если нужен повторяемый benchmark, используйте скрипт ниже: он запускает одинаковый сценарий в двух режимах и выводит агрегированные метрики.

```python
import os
import re
import subprocess
from statistics import mean

SCENARIO = [
    "Запомни: мой стек Python + FastAPI + Postgres.",
    "Запомни: дедлайн проекта 15 мая.",
    "Предпочитаю ответы коротко, списком.",
    "Составь план MVP на 2 недели.",
    "Добавь риски и как их снизить.",
    "Что я говорил про стек?",
    "Какой дедлайн?",
    "Переформулируй план в 5 пунктах.",
    "Добавь чеклист на запуск.",
    "Сделай итог: стек, дедлайн, план.",
    "Оцени себя: насколько ответ полный и точный?",
]

TOKEN_RE = re.compile(r"^(prompt_tokens_total|dialog_history_tokens|total_tokens)=(.+)$")
LAT_RE = re.compile(r"^latency_ms=(.+)$")


def run_chat(mode_name: str, with_summary: bool):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["LLM_CHAT_HISTORY_PATH"] = f".llm_chat_history_{mode_name}.db"
    env["LLM_CHAT_KEEP_LAST_N"] = "10"
    env["LLM_CHAT_SUMMARY_BATCH_SIZE"] = "10"

    cmd = ["python3", "llm_cli.py", "--chat", "--tokens", "-v"]
    if with_summary:
        cmd.append("--summary")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stdin_payload = "\n".join(SCENARIO + ["exit"]) + "\n"
    out, err = proc.communicate(stdin_payload, timeout=600)

    prompt_tokens, history_tokens, total_tokens, latency = [], [], [], []
    for line in err.splitlines():
        m = TOKEN_RE.match(line.strip())
        if m:
            key, raw_val = m.group(1), m.group(2).strip()
            if raw_val.isdigit():
                val = int(raw_val)
                if key == "prompt_tokens_total":
                    prompt_tokens.append(val)
                elif key == "dialog_history_tokens":
                    history_tokens.append(val)
                elif key == "total_tokens":
                    total_tokens.append(val)
        lm = LAT_RE.match(line.strip())
        if lm:
            raw_lat = lm.group(1).strip()
            if raw_lat.isdigit():
                latency.append(int(raw_lat))

    return {
        "prompt_tokens_avg": round(mean(prompt_tokens), 2) if prompt_tokens else None,
        "history_tokens_avg": round(mean(history_tokens), 2) if history_tokens else None,
        "total_tokens_avg": round(mean(total_tokens), 2) if total_tokens else None,
        "latency_ms_avg": round(mean(latency), 2) if latency else None,
        "stdout_last_lines": "\n".join(out.splitlines()[-8:]),
    }


if __name__ == "__main__":
    baseline = run_chat("baseline", with_summary=False)
    summary = run_chat("summary", with_summary=True)

    print("=== BASELINE (no summary) ===")
    print(baseline)
    print("\n=== SUMMARY MODE ===")
    print(summary)
```

Запуск:

```bash
python3 benchmark_day09.py
```

После запуска вручную оцените качество финального ответа (`stdout_last_lines`) и заполните таблицу выше.