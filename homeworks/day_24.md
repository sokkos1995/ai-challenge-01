🔥 День 24. Цитаты, источники и анти-галлюцинации

Доработайте RAG так, чтобы модель обязательно возвращала:

👉 ответ
👉 список источников (source + section/chunk_id)
👉 цитаты (фрагменты из найденных чанков)

Проверьте на 10 вопросах:

👉 есть ли источники в каждом ответе
👉 есть ли цитаты в каждом ответе
👉 совпадает ли смысл ответа с цитатами

Усиление:

👉 добавьте правило: если релевантность ниже порога — ассистент обязан сказать “не знаю” и попросить уточнение

Результат:

Ответы с обязательными источниками и цитатами + режим “не знаю” при слабом контексте

## План

1. Взять retrieval + rerank из day 23 как основу.
2. Добавить grounded-формат ответа с обязательной структурой:
   - `answer`
   - `sources` (`source`, `section`, `chunk_id`)
   - `quotes` (цитаты из выбранных чанков)
3. Добавить anti-hallucination правило:
   - если max релевантность контекста ниже `min_context_score`, ответ строго:
     `Не знаю по текущему контексту. Уточните вопрос.`
4. Прогнать 10 контрольных вопросов и проверить:
   - наличие источников;
   - наличие цитат;
   - согласованность ответа с цитатами.

## Реализация

Добавлен скрипт `homeworks/src/day_24_grounded_rag.py`:

- Использует post-retrieval этап из day 23 (`retrieve_then_rerank` + `rewrite_query`).
- Собирает grounded-контекст в формате:
  - `source`
  - `section`
  - `chunk_id`
  - `score`
  - короткая `quote` из чанка.
- Для генерации ответа задает строгий JSON-контракт:
  - `answer`
  - `sources`
  - `quotes`
- Добавляет fallback-политику:
  - при низкой релевантности возвращает `"Не знаю..."` и просит уточнение.
- Сохраняет отчет `grounded_report.json` с метриками по 10 вопросам:
  - `sources_coverage`
  - `quotes_coverage`
  - `answer_quote_alignment_rate`
  - `dont_know_count`

Дополнительно функционал перенесен в основное приложение `app/`:

- Добавлен сервис `app/services/rag_service.py`:
  - retrieval + heuristic rerank;
  - grounded output (`answer`, `sources`, `quotes`);
  - fallback `"Не знаю..."` при низкой релевантности.
- Интеграция в CLI (`app/cli.py`, `app/cli_utils.py`):
  - флаг `--rag` для включения RAG-режима;
  - параметры `--rag-index-path`, `--rag-top-k-before`, `--rag-top-k-after`, `--rag-similarity-threshold`, `--rag-min-context-score`;
  - chat-команды `@rag`, `@rag on`, `@rag off`, `@rag status`.
- Вынесено форматирование RAG-ответа в `app/rag_output.py` (рефакторинг для упрощения `app/cli.py`).

Добавлены тесты:

- `tests/test_day24_grounded_rag.py`:
  - low-relevance сценарий принудительно возвращает `"Не знаю..."`;
  - в grounded режиме всегда есть `sources` и `quotes`;
  - helper для проверки согласованности answer/quotes работает корректно.
- `tests/test_rag_service.py`:
  - загрузка индекса и базовый RAG flow;
  - fallback-поведение и устойчивый парсинг ответа.
- `tests/test_rag_output.py`:
  - проверка формата вывода `answer/sources/quotes`.

## Запуск

```bash
# (опционально) переиндексация day 21
python3 homeworks/src/day_21_indexing.py

# day 24: 10 контрольных вопросов + отчет
python3 homeworks/src/day_24_grounded_rag.py \
  --index-path "homeworks/artifacts/day_21/index_structured.json" \
  --run-control-set \
  --top-k-before 8 \
  --top-k-after 4 \
  --similarity-threshold 0.2 \
  --min-context-score 0.24
```

Тестовый запуск без внешнего LLM API:

```bash
python3 homeworks/src/day_24_grounded_rag.py --run-control-set --dry-run
```

Запуск day 24 RAG через основное приложение:

```bash
# 1) загрузить ключи провайдера в текущую shell-сессию
set -a
source .env
set +a

# 2) проверить, что есть хотя бы один ключ
echo "$OPENROUTER_API_KEY"
echo "$GROQ_API_KEY"

# single prompt
python3 llm_cli.py --rag \
  --rag-index-path "homeworks/artifacts/day_21/index_structured.json" \
  "Как в fastapi_ecommerce подключаются роутеры categories и products?"

# chat mode
python3 llm_cli.py --chat --rag
```

Если видите `HTTP error 401` (например, `"User not found"`), это не ошибка команды RAG, а ошибка авторизации провайдера:
- проверьте актуальность `OPENROUTER_API_KEY` / `GROQ_API_KEY`;
- убедитесь, что ключ реально подхватился в текущем терминале (после `source .env`);
- при необходимости задайте провайдера явно:

```bash
LLM_PROVIDER=groq python3 llm_cli.py --rag \
  --rag-index-path "homeworks/artifacts/day_21/index_structured.json" \
  "Какие поля у модели Product и как она связана с Category в fastapi_ecommerce?"
```

## Результат

Артефакты сохраняются в `homeworks/artifacts/day_24`:

- `grounded_report.json`:
  - ответы по 10 вопросам;
  - обязательные источники и цитаты для каждого ответа;
  - признаки прохождения проверок (`has_sources`, `has_quotes`, `answer_quote_alignment`);
  - режим fallback `"Не знаю..."` при слабом контексте.

Также результат day 24 доступен как часть продуктового CLI в `app/`:
- grounded-RAG ответы можно получать напрямую через `llm_cli.py --rag`;
- в chat-режиме RAG можно динамически включать/выключать командами `@rag ...`.
