🔥 День 21. Индексация документов

Возьмите набор документов:

👉 README / статьи / код / pdf → текст
👉 минимум 20–30 страниц текста суммарно (или эквивалент в коде)

Реализуйте пайплайн индексации:

👉 разбиение на чанки (chunking)
👉 генерация эмбеддингов
👉 сохранение индекса (FAISS / SQLite / JSON)

Усиление:

👉 добавьте метаданные к каждому чанку (source, title/file, section, chunk_id)
👉 сделайте минимум 2 стратегии chunking и сравните их:

- по фиксированному размеру
- по структуре (заголовки/разделы/файлы)

Результат:

Локальный индекс документов с эмбеддингами + метаданные + сравнение 2 стратегий chunking

## Реализация

Сделан скрипт `homeworks/src/day_21_indexing.py`, который строит локальный индекс документов по двум стратегиям chunking:

1. `fixed` — фиксированный размер чанка по словам (с overlap).
2. `structured` — разбиение по структуре документа:
   - для `.md` по заголовкам;
   - для `.py` по `class/def`;
   - для остальных текстов по абзацам.

Каждый чанк содержит метаданные:
- `source`
- `title`
- `file`
- `section`
- `chunk_id`
- `strategy`

Скрипт работает от переданного корня `--root` и по умолчанию индексирует **все текстовые файлы** внутри директории (включая код, конфиги и файлы без расширения вроде `Dockerfile`).
Служебные и тяжелые директории (`.git`, `node_modules`, `target`, `build`, `dist` и т.д.) пропускаются, бинарные файлы тоже.

Эмбеддинги считаются локально (deterministic hash-embeddings), без внешних API.  
Индексы сохраняются в:
- JSON (`index_fixed.json`, `index_structured.json`);
- SQLite (`index_fixed.sqlite3`, `index_structured.sqlite3`).

Также генерируется демо-отчет `comparison_report.json`:
- количество документов и чанков;
- средний размер чанка;
- top-k результаты поиска для запроса.

## Запуск

```bash
python3 homeworks/src/day_21_indexing.py
```

Расширенный запуск:

```bash
python3 homeworks/src/day_21_indexing.py \
  --root "/Users/konstantinsokolov/dev/projects/pet_projects/fastapi_sbx" \
  --query "как реализован магазин товаров, через какие инструменты" \
  --top-k 5 \
  --out-dir "/Users/konstantinsokolov/dev/projects/pet_projects/ai_challenge/hw/hw01/homeworks/artifacts/day_21"
```

## Результат

После запуска в `homeworks/artifacts/day_21` будут:
- 2 JSON-индекса;
- 2 SQLite-индекса;
- `comparison_report.json` со сравнением `fixed` и `structured` chunking.
