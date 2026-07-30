# Day 41 — Aviation Dataset (Fine-Tuning prep)

## Задача

**Генерация:** авиационный Q&A-ассистент (пилот / FAA instructor style).

Источник реальных данных: [`ziksy/faa-aviation-training`](https://huggingface.co/datasets/ziksy/faa-aviation-training) (14 CFR, AIM, PHAK).

| | |
|--|--|
| Всего | 100 (train 80 / eval 20) |
| Real | 30 (30%) |
| Synthetic | 70 (knowledge→Q&A templates; LLM optional) |
| Seed | 41 |
| Формат | `messages`: system + user + assistant |

## Артефакты

Каталог: [`homeworks/artifacts/day_41/`](artifacts/day_41/)

| Файл | Описание |
|------|----------|
| `train.jsonl` / `eval.jsonl` | датасет |
| `manifest.json` | provenance / split |
| `baseline_responses.json` | 10 ответов базовой модели |
| `evaluation_criteria.md` | критерии «стало лучше» |
| `datasets.md` | каталог открытых датасетов |
| `requirements.txt` | зависимости day_41 |
| `README.md` | копия инструкций (основной отчёт — этот файл) |

## Окружение

```bash
# из корня репозитория
source .venv/bin/activate
pip install -r homeworks/artifacts/day_41/requirements.txt

# либо одной строкой через хелпер:
# ./homeworks/src/day_41_with_venv.sh python homeworks/src/day_41_baseline.py --n 10
```

Провайдеры для baseline / опциональной LLM-синтетики: **локальная Ollama** (если запущена) или **Cursor API** (`CURSOR_API_KEY`). OpenRouter не используем.

## Скрипты

| Скрипт | |
|--------|--|
| [`homeworks/src/day_41_build_dataset.py`](src/day_41_build_dataset.py) | сборка JSONL |
| [`homeworks/src/day_41_validate_jsonl.py`](src/day_41_validate_jsonl.py) | валидация |
| [`homeworks/src/day_41_baseline.py`](src/day_41_baseline.py) | baseline (Ollama / Cursor) |
| [`homeworks/src/day_41_finetune_client.py`](src/day_41_finetune_client.py) | OpenAI FT: upload→job→poll (только dry-run) |

```bash
source .venv/bin/activate

# Сборка (без LLM — knowledge-templates)
python homeworks/src/day_41_build_dataset.py --no-llm
# С LLM-синтетикой (Ollama если up, иначе CURSOR_API_KEY):
python homeworks/src/day_41_build_dataset.py

# Валидация
python homeworks/src/day_41_validate_jsonl.py \
  homeworks/artifacts/day_41/train.jsonl \
  homeworks/artifacts/day_41/eval.jsonl

# Baseline: auto = Ollama → Cursor
python homeworks/src/day_41_baseline.py --n 10
python homeworks/src/day_41_baseline.py --n 10 --provider cursor
python homeworks/src/day_41_baseline.py --n 10 --provider ollama

# Fine-tune client — только dry-run (job не запускать)
python homeworks/src/day_41_finetune_client.py
```

Валидация: **100/100 OK**. Fine-tune job **не запускался** (только dry-run).

## Baseline

Точка отсчёта без fine-tune: локальная модель (Ollama) или Cursor (`composer-2.5`). Зафиксировано в `baseline_responses.json`. Критерии — `evaluation_criteria.md` (точность FAR/AIM, формат, стиль инструктора, A/B 0–2).

## Unsloth (позже)

Те же `train.jsonl` / `eval.jsonl` → локальный SFT (LoRA / QLoRA) через Unsloth. OpenAI FT-клиент — усиление ДЗ (dry-run); основной тюнинг — локальный.

## Дальше

Локальный тюнинг (Unsloth / LoRA) на тех же JSONL — отдельно от day_41.

---

## Исходное ТЗ

Выберите задачу, под которую будете тюнить модель:
👉 классификация (тональность / категории тикетов / спам) 
👉 генерация (код в вашем стеке / ответы в стиле компании / саммари) 
👉 extraction (сущности из текста / парсинг документов)

Соберите датасет:
👉 минимум 50 примеров в формате JSONL 
👉 каждый пример — объект с массивом messages: system + user + assistant 
👉 assistant — это эталонный ответ, которому модель будет учиться 
👉 минимум 20% данных — реальные, остальное можно сгенерировать через ИИ

Подготовьте данные:
👉 уберите мусор: дубли, пустые, слишком короткие/длинные 
👉 разделите на train (80%) и eval (20%) 
👉 напишите скрипт валидации: проверьте что каждая строка — валидный JSON, что все три роли на месте, что нет пустых content

Замерьте baseline:
👉 возьмите 10 примеров из eval 
👉 прогоните их через базовую модель БЕЗ файнтюна 
👉 зафиксируйте ответы — это ваша точка отсчёта 
👉 запишите критерии: по чему будете определять "стало лучше" (точность, формат, стиль)

Усиление:
👉 напишите клиент на любом языке, который автоматизирует загрузку файла и запуск файнтюна через OpenAI API (upload file → create fine-tuning job → poll status) 
👉 пока не запускайте — только подготовьте код

Результат:
Датасет в JSONL (train + eval) + скрипт валидации + 10 baseline-ответов + критерии оценки + клиент для запуска файнтюна
