# Day 41 — Aviation Dataset (Fine-Tuning prep)

## Задача для демо

**Генерация:** чат-ассистент «FAA Flight Instructor» для подготовки к private / commercial oral.

Почему эта задача хороша для полноценного демо:
- домен узкий (FAR / AIM / PHAK) — base-модель часто путает номера параграфов;
- эталонный стиль короткий и «инструкторский» — после SFT видно сжатие формата;
- на baseline уже есть ловушки: например **14 CFR 93.93** (base → Grand Canyon, gold → Los Angeles SFRA) и **14 CFR 107.77** (base → alcohol, gold → смена имени на сертификате).

Источник данных: [`ziksy/faa-aviation-training`](https://huggingface.co/datasets/ziksy/faa-aviation-training).

| | |
|--|--|
| Всего | 100 (train 80 / eval 20) |
| Real | 30 (30%) |
| Synthetic | 70 |
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
| `README.md` | краткая копия рядом с артефактами |
| `Modelfile.aviation-faa` | шаблон Ollama после экспорта GGUF |
| `train_unsloth.py` | скрипт локального LoRA/QLoRA |

## Окружение (данные / baseline)

```bash
# из корня репозитория
source .venv/bin/activate
pip install -r homeworks/artifacts/day_41/requirements.txt
```

Провайдеры baseline: **Ollama** или **Cursor** (`CURSOR_API_KEY`). OpenRouter не используем.

## Скрипты day_41

| Скрипт | |
|--------|--|
| [`day_41_build_dataset.py`](src/day_41_build_dataset.py) | сборка JSONL |
| [`day_41_validate_jsonl.py`](src/day_41_validate_jsonl.py) | валидация |
| [`day_41_baseline.py`](src/day_41_baseline.py) | baseline |
| [`day_41_finetune_client.py`](src/day_41_finetune_client.py) | OpenAI FT dry-run |
| [`day_41_demo_compare.py`](src/day_41_demo_compare.py) | Base vs FT side-by-side |
| [`day_41_with_venv.sh`](src/day_41_with_venv.sh) | `source .venv` + команда |

```bash
source .venv/bin/activate

python homeworks/src/day_41_build_dataset.py --no-llm
python homeworks/src/day_41_validate_jsonl.py \
  homeworks/artifacts/day_41/train.jsonl \
  homeworks/artifacts/day_41/eval.jsonl
python homeworks/src/day_41_baseline.py --n 10 --provider ollama
python homeworks/src/day_41_finetune_client.py   # только dry-run
```

Валидация: **100/100 OK**. OpenAI FT job **не запускался**.

---

## Локальное обучение (Ollama → Unsloth → снова Ollama)

Цель демо: взять локальную base-модель, дообучить на `train.jsonl`, поднять как отдельный Ollama-тег и сравнить ответы.

### 1. Запуск Ollama и base-модели

```bash
# macOS: приложение Ollama или сервис
ollama serve
# в другом терминале:
ollama pull qwen2.5:7b
ollama list
ollama run qwen2.5:7b
```

Проверка API:

```bash
curl -s http://127.0.0.1:11434/api/tags | head
```

Снять baseline на Ollama (если ещё не снят):

```bash
source .venv/bin/activate
OLLAMA_MODEL=qwen2.5:7b python homeworks/src/day_41_baseline.py --n 10 --provider ollama
```

### 2. Обучение LoRA/QLoRA (Unsloth)

Нужны GPU (рекомендуется) и отдельное venv под training (зависимости тяжёлые).

```bash
# отдельное окружение (не путать с app .venv, если конфликт torch)
python3 -m venv .venv-unsloth
source .venv-unsloth/bin/activate
pip install -U pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets transformers trl peft accelerate bitsandbytes

# обучение на нашем JSONL
python homeworks/artifacts/day_41/train_unsloth.py \
  --train homeworks/artifacts/day_41/train.jsonl \
  --eval homeworks/artifacts/day_41/eval.jsonl \
  --out homeworks/artifacts/day_41/unsloth_out \
  --model unsloth/Qwen2.5-7B-Instruct \
  --max-steps 60
```

Скрипт пишет LoRA-адаптер в `homeworks/artifacts/day_41/unsloth_out/`.  
На машине без GPU можно уменьшить модель (`unsloth/Qwen2.5-3B-Instruct`) или гонять в Colab / облаке с тем же `train.jsonl`.

### 3. Экспорт в GGUF и регистрация в Ollama

```bash
source .venv-unsloth/bin/activate
python homeworks/artifacts/day_41/train_unsloth.py --export-gguf \
  --out homeworks/artifacts/day_41/unsloth_out \
  --gguf-out homeworks/artifacts/day_41/aviation-faa-q4_k_m.gguf

# Modelfile уже лежит рядом с артефактами — поправьте FROM на путь к GGUF
ollama create aviation-faa -f homeworks/artifacts/day_41/Modelfile.aviation-faa
ollama run aviation-faa
```

---

## Примеры запросов для демо

Один и тот же system prompt (как в датасете):

> You are an aviation expert and FAA-certified flight instructor. Answer questions accurately based on Federal Aviation Regulations, the Aeronautical Information Manual, and standard aviation knowledge.

### Ловушки из eval (хорошо показывают «до/после»)

1. `What should a pilot know about this topic from 14 CFR 93.93? Topic: 14 CFR 93.93: Description of area`  
   Ожидание: **Los Angeles** Special Flight Rules Area (не Grand Canyon).
2. `What specific requirements are stated in 14 CFR 107.77?`  
   Ожидание: смена **имени** на remote pilot certificate (не alcohol/drugs).

### Устные вопросы «как на oral» (обобщение)

3. `According to 14 CFR 91.103, what preflight action is required of the PIC?`
4. `Explain VFR weather minimums in Class G airspace below 1,200 feet AGL (day).`
5. `What is the right-of-way rule when two aircraft are converging at approximately the same altitude?`
6. `What fuel reserves are required for IFR flight under 14 CFR 91.167 (airplane)?`

Через Ollama CLI:

```bash
ollama run qwen2.5:7b "You are an FAA flight instructor. What specific requirements are stated in 14 CFR 107.77?"
ollama run aviation-faa "You are an FAA flight instructor. What specific requirements are stated in 14 CFR 107.77?"
```

Или пакетно:

```bash
source .venv/bin/activate
python homeworks/src/day_41_demo_compare.py \
  --base qwen2.5:7b \
  --tuned aviation-faa:latest
# до обучения:
python homeworks/src/day_41_demo_compare.py --base qwen2.5:7b --skip-tuned
```

---

## Как сравнивать: стало лучше или хуже

Критерии — [`evaluation_criteria.md`](artifacts/day_41/evaluation_criteria.md). Кратко:

| Балл | Смысл |
|------|--------|
| 0 | фактическая ошибка / выдуманный FAR / не по теме |
| 1 | по теме, но неточно или много воды |
| 2 | близко к gold / официальному тексту, стиль инструктора |

### Процедура демо (5–10 минут)

1. Взять 6 промптов выше (или 10 из `baseline_responses.json`).
2. Для каждого получить ответ **base** и **tuned** (скрипт `day_41_demo_compare.py` пишет `demo_compare.json`).
3. Не глядя на метку модели, выставить 0–2; потом раскрыть, кто base / tuned.
4. Посчитать:
   - средний балл base vs tuned;
   - win / tie / loss по парам;
   - отдельно: исчезли ли галлюцинации на 93.93 и 107.77.
5. «Стало лучше», если средний балл вырос **и** на ловушках 93.93 / 107.77 оценка не хуже, чем у base.

Таблица для записи на демо:

| # | Промпт (кратко) | Base 0–2 | Tuned 0–2 | Winner | Комментарий |
|---|-----------------|----------|-----------|--------|-------------|
| 1 | 93.93 area | | | | |
| 2 | 107.77 | | | | |
| 3 | 91.103 preflight | | | | |
| … | | | | | |

Признаки регрессии (хуже):
- верный факт base пропал у tuned;
- модель зациклилась на шаблоне из train и игнорирует вопрос;
- train loss ↓, но eval/human score ↓ → overfitting (early stop / больше данных).

---

## OpenAI FT-клиент (усиление ДЗ)

Код готов, **не запускать** job без явной команды:

```bash
source .venv/bin/activate
python homeworks/src/day_41_finetune_client.py          # dry-run
# python homeworks/src/day_41_finetune_client.py --execute   # только с OPENAI_API_KEY
```

Основной путь тюнинга для этого ДЗ — **локальный** (Unsloth + Ollama).

---

## Исходное ТЗ

Выберите задачу: классификация / **генерация** / extraction.

Датасет ≥50 JSONL (`system`+`user`+`assistant`), ≥20% real, train/eval, валидация, baseline, критерии, FT-клиент (без запуска).
