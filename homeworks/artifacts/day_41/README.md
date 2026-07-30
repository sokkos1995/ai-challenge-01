# Day 41 — Aviation Dataset (краткая копия)

Основной отчёт со сдачей и демо: [`homeworks/day_41.md`](../../day_41.md).

## Демо-задача

Чат **FAA Flight Instructor** (генерация Q&A по FAR/AIM/PHAK).  
100 JSONL (80/20), real 30%, seed 41.

## Быстрый старт

```bash
source .venv/bin/activate
pip install -r homeworks/artifacts/day_41/requirements.txt

# Ollama
ollama serve
ollama pull qwen2.5:7b

python homeworks/src/day_41_baseline.py --n 10 --provider ollama
python homeworks/src/day_41_validate_jsonl.py \
  homeworks/artifacts/day_41/train.jsonl \
  homeworks/artifacts/day_41/eval.jsonl
python homeworks/src/day_41_finetune_client.py   # dry-run
```

## Обучение (кратко)

```bash
# 1) Ollama base уже запущена
# 2) Unsloth LoRA на train.jsonl → см. train_unsloth.py и day_41.md
# 3) GGUF + ollama create aviation-faa -f Modelfile.aviation-faa
# 4) Сравнение:
python homeworks/src/day_41_demo_compare.py --base qwen2.5:7b --tuned aviation-faa:latest
```

## Примеры запросов

- `What specific requirements are stated in 14 CFR 107.77?` (ловушка: не alcohol)
- `What should a pilot know about … 14 CFR 93.93 … Description of area` (ловушка: Los Angeles SFRA, не Grand Canyon)
- `According to 14 CFR 91.103, what preflight action is required of the PIC?`
- `Explain VFR weather minimums in Class G airspace below 1,200 feet AGL (day).`

## Сравнение лучше/хуже

Шкала 0–2 в `evaluation_criteria.md`. Заполнить win/tie/loss в `demo_compare.json`.  
«Лучше» = выше средний балл + нет регрессии на ловушках 93.93 / 107.77.

## Файлы

| Файл | |
|------|--|
| `train.jsonl` / `eval.jsonl` | датасет |
| `baseline_responses.json` | baseline |
| `evaluation_criteria.md` | рубрика |
| `train_unsloth.py` | локальный SFT |
| `Modelfile.aviation-faa` | Ollama после GGUF |
| `requirements.txt` | deps |
| `datasets.md` | каталог датасетов |
