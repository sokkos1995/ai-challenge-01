🔥 День 23. Реранкинг и фильтрация

Добавьте второй этап после поиска:

👉 reranker или фильтр релевантности (порог similarity / отдельная модель / heuristic)

Настройте:

👉 порог отсечения нерелевантных результатов
👉 топ-K до и после фильтрации

Сравните:

👉 качество без фильтра/rewriting
👉 качество с фильтром

Результат:

Улучшенный RAG: фильтрация/реранкинг + query rewrite + сравнение режимов

## План

1. Взять retriever из day 22 как baseline.
2. Добавить второй этап post-retrieval:
   - heuristic reranker (dense similarity + lexical overlap + source bonus);
   - фильтрацию по порогу `similarity_threshold`.
3. Добавить `query rewrite` перед retrieval во втором режиме.
4. Сделать настраиваемыми `top_k_before` и `top_k_after`.
5. Прогнать контрольный набор и сохранить сравнительный отчет:
   - без фильтра/rewriting;
   - с фильтром + rewriting.

## Реализация

Сделан скрипт `homeworks/src/day_23_rerank.py` с двумя режимами сравнения для каждого контрольного вопроса:

1. `without_filter_or_rewrite`:
   - обычный retrieval из индекса;
   - без отсечения по threshold;
   - top-k после retrieval равен `top_k_before`.
2. `with_filter_and_rewrite`:
   - сначала `query rewrite`;
   - затем retrieval;
   - затем reranking + фильтрация по `similarity_threshold`;
   - top-k после фильтрации ограничивается `top_k_after`.

Реранкер реализован как heuristic:
- dense cosine similarity по embedding;
- lexical overlap запроса и чанка;
- небольшой bonus для кодовых/markdown источников.

Добавлены тесты `tests/test_day23_rerank.py`:
- проверка query rewrite;
- проверка порога отсечения и top-K до/после фильтрации.

## Запуск

```bash
# (опционально) переиндексация day 21
python3 homeworks/src/day_21_indexing.py

# day 23: сравнение baseline vs filter+rewrite
python3 homeworks/src/day_23_rerank.py \
  --index-path "homeworks/artifacts/day_21/index_structured.json" \
  --run-control-set \
  --top-k-before 8 \
  --top-k-after 4 \
  --similarity-threshold 0.2
```

Тестовый запуск без внешнего LLM API:

```bash
python3 homeworks/src/day_23_rerank.py --run-control-set --dry-run
```

## Результат

Артефакты сохраняются в `homeworks/artifacts/day_23`:
- `comparison_report.json` — детальное сравнение по каждому вопросу:
  - качество без фильтра/rewriting;
  - качество с filter+rewrite;
  - top-K до/после, выбранные контексты, threshold и итоговая дельта качества.
