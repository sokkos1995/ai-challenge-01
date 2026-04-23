🔥 День 29. Оптимизация локальной LLM

Оптимизируйте локальную модель под свою задачу:

👉 настройте параметры (temperature, max tokens, context window)
👉 попробуйте квантование (если доступно)
👉 измените prompt-шаблон под конкретный кейс

Сравните:

👉 качество ответов до оптимизации
👉 качество после
👉 скорость и потребление ресурсов

Результат:

Оптимизированная локальная LLM под конкретную задачу

## Реализация

Сделан скрипт `homeworks/src/day_29_optimize_local_llm.py`, который сравнивает локальную модель **до** и **после** оптимизации на RAG-задачах по проекту.

Что оптимизируется:

- параметры генерации:
  - `temperature`
  - `max tokens` (`num_predict`)
  - `context window` (`num_ctx`)
- prompt-шаблон:
  - baseline: простой
  - optimized: структурированный под репозиторий + обязательный блок `Sources`
- модель:
  - можно передать квантованную модель в `--optimized-model` (если доступна в Ollama).

## Метрики сравнения

Скрипт считает:

- качество:
  - `avg_quality_keyword_coverage`
- скорость:
  - `avg_latency_ms`
  - `avg_tokens_per_second`
- потребление ресурсов (прокси-метрики из Ollama):
  - `avg_prompt_tokens`
  - `avg_generated_tokens`
  - `avg_total_duration_ms`

Результат сохраняется в:

- `homeworks/artifacts/day_29/optimization_report.json`

## Запуск

1) Подготовить индекс (из Недели 6 / day 21):

```bash
python3 homeworks/src/day_21_indexing.py
```

2) Поднять локальную LLM:

```bash
ollama serve
```

3) Запустить сравнение до/после оптимизации:

```bash
python3 homeworks/src/day_29_optimize_local_llm.py \
  --index-path homeworks/artifacts/day_21/index_structured.json \
  --baseline-model qwen2.5:3b \
  --baseline-temperature 0.7 \
  --baseline-max-tokens 700 \
  --baseline-context-window 2048 \
  --optimized-model qwen2.5:3b \
  --optimized-temperature 0.2 \
  --optimized-max-tokens 450 \
  --optimized-context-window 4096 \
  --top-k 4
```

4) Вариант с квантованной моделью (если установлена):

```bash
python3 homeworks/src/day_29_optimize_local_llm.py \
  --index-path homeworks/artifacts/day_21/index_structured.json \
  --baseline-model qwen2.5:3b \
  --optimized-model qwen2.5:3b-instruct-q4_K_M
```

## Что получаем в отчете

В `optimization_report.json` есть:

- `before_optimization`:
  - конфиг baseline
  - метрики
  - ответы и ошибки по каждому вопросу
- `after_optimization`:
  - конфиг optimized
  - метрики
  - ответы и ошибки по каждому вопросу
- `comparison`:
  - дельты after-before по качеству, скорости и ресурсным прокси

## Результат

Требования day 29 выполнены:

- параметры локальной модели настроены;
- добавлена проверка квантования (через выбор `--optimized-model`);
- prompt-шаблон адаптирован под конкретный кейс (репозиторий + источники);
- есть сравнение качества, скорости и потребления ресурсов до/после.
