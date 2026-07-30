# Day 41 — Aviation Dataset (Fine-Tuning prep)

Основной отчёт для сдачи: [`homeworks/day_41.md`](../../day_41.md). Этот README — краткая копия инструкций рядом с артефактами.

## Задача

**Генерация:** авиационный Q&A-ассистент (пилот / FAA instructor style).

Источник: [`ziksy/faa-aviation-training`](https://huggingface.co/datasets/ziksy/faa-aviation-training) — 100 примеров (train 80 / eval 20), real 30%, seed 41.

## Окружение

```bash
source .venv/bin/activate
pip install -r homeworks/artifacts/day_41/requirements.txt
```

Baseline / synth: **Ollama** (локально) или **Cursor API** (`CURSOR_API_KEY`). OpenRouter не используем.

## Команды

```bash
source .venv/bin/activate

python homeworks/src/day_41_build_dataset.py --no-llm
python homeworks/src/day_41_validate_jsonl.py \
  homeworks/artifacts/day_41/train.jsonl \
  homeworks/artifacts/day_41/eval.jsonl
python homeworks/src/day_41_baseline.py --n 10
python homeworks/src/day_41_finetune_client.py   # dry-run only
```

## Файлы

| Файл | Описание |
|------|----------|
| `train.jsonl` / `eval.jsonl` | датасет |
| `manifest.json` | provenance / split |
| `baseline_responses.json` | 10 baseline-ответов |
| `evaluation_criteria.md` | критерии оценки |
| `requirements.txt` | зависимости |
| `datasets.md` | каталог открытых датасетов |

## Unsloth (позже)

Те же JSONL → локальный LoRA/QLoRA. OpenAI FT-клиент только для dry-run усиления ДЗ.
