🔥 День 22. Первый RAG-запрос

Реализуйте функцию:

👉 вопрос → поиск релевантных чанков → объединение с вопросом → запрос к LLM

Сравните:

👉 ответ модели без RAG
👉 ответ модели с RAG

Усиление:

👉 составьте мини-набор из 10 контрольных вопросов по вашей базе
👉 для каждого вопроса зафиксируйте:

- ожидание (что должно быть в ответе)
- какие источники должны быть использованы (если применимо)

Результат:

Агент с двумя режимами (с RAG / без RAG) + 10 контрольных вопросов и сравнение качества

## Реализация

Сделан скрипт `homeworks/src/day_22_rag.py` с двумя режимами ответа:

1. `without_rag`:
   - вопрос отправляется в LLM напрямую.
2. `with_rag`:
   - вопрос -> поиск релевантных чанков в индексе day 21 -> сбор контекста -> запрос в LLM.

Для retrieval используется индекс из day 21 (`index_structured.json`), опционально можно подключить `index_fixed.json` и сравнить качество retriever.

Добавлен mini-набор из 10 контрольных вопросов.  
Для каждого вопроса фиксируются:
- ожидание;
- ожидаемые источники;
- ответ без RAG;
- ответ с RAG;
- метрики сравнения:
  - keyword coverage (без/с RAG),
  - source recall (для RAG).

## Запуск

Сначала подготовьте индекс документов (day 21):

```bash
python3 homeworks/src/day_21_indexing.py
```

Сравнение одного вопроса (оба режима):

```bash
python3 homeworks/src/day_22_rag.py \
  --question "Как в проекте устроен orchestration между MCP-серверами?" \
  --mode both
```

Запуск контрольного набора (10 вопросов):

```bash
python3 homeworks/src/day_22_rag.py --run-control-set
```

Профиль контрольных вопросов для базы `fastapi_sbx` + проверка качества эмбеддингов/retrieval:

```bash
# Шаг 1. Индексация fastapi_sbx (day 21)
python3 homeworks/src/day_21_indexing.py \
  --root "/Users/konstantinsokolov/dev/projects/pet_projects/fastapi_sbx" \
  --query "как реализован магазин товаров, через какие инструменты" \
  --top-k 5 \
  --out-dir "homeworks/artifacts/day_21"

# Шаг 2. Day 22: сравнение без RAG/с RAG + диагностика retriever
python3 homeworks/src/day_22_rag.py \
  --index-path "homeworks/artifacts/day_21/index_structured.json" \
  --fixed-index-path "homeworks/artifacts/day_21/index_fixed.json" \
  --control-set-profile fastapi_sbx \
  --retriever-index auto \
  --run-control-set \
  --top-k 5
```

Тестовый запуск без вызова LLM API:

```bash
python3 homeworks/src/day_22_rag.py --run-control-set --dry-run
```

## Результат

Артефакты сохраняются в `homeworks/artifacts/day_22`:
- `control_questions.json` — контрольные вопросы/ожидания/источники;
- `comparison_report.json` — ответы без RAG и с RAG + сравнение качества.

В отчете есть блок `embedding_retrieval_diagnostics`:
- `avg_structured_source_recall_at_k`;
- `avg_fixed_source_recall_at_k`;
- сравнение recall по каждому из 10 вопросов.
