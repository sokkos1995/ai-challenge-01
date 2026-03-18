# Минимальный запрос к LLM через API (без OpenAI)

Этот проект показывает минимальный пример, который:
- отправляет запрос в LLM через API,
- получает ответ,
- выводит ответ в консоль (CLI).

## Файл проекта

- `llm_cli.py` — скрипт для отправки запроса и вывода ответа.
- `llm_request.sh` — bash/curl скрипт для отправки запроса.
- `constraints_comparison.md` — сравнение ответов с и без ограничений.

## Подготовка

Нужен Python 3.8+ и API-ключ от любого провайдера:
- `OPENROUTER_API_KEY` (OpenRouter),
- или `GROQ_API_KEY` (Groq, бесплатный tier).

Создайте файл `.env` в корне проекта:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=ваш_api_ключ
EOF
```

или:

```bash
cat > .env <<'EOF'
LLM_PROVIDER=groq
GROQ_API_KEY=ваш_api_ключ
EOF
```

Чтобы переменные из `.env` появились в текущей shell-сессии:

```bash
set -a
source .env
set +a
```

Проверка:

```bash
echo "$OPENROUTER_API_KEY"
echo "$LLM_PROVIDER"
```

Примечание: оба скрипта (`llm_cli.py` и `llm_request.sh`) умеют читать `.env` сами, но `source .env` полезен, когда переменные нужны в текущем терминале для других команд.

Где взять ключ:
- зарегистрироваться на [OpenRouter](https://openrouter.ai/),
- открыть [Keys](https://openrouter.ai/keys),
- создать API key.

## Запуск

```bash
python3 llm_cli.py "Объясни, что такое LLM в 2 предложениях"
```

Если аргумент не передан, скрипт использует промпт по умолчанию.

Параметры сэмплинга можно задавать флагами:

```bash
python3 llm_cli.py --temperature 0.2 --top-p 0.9 --top-k 40 "Привет"
```

Минимальная статистика запроса:

```bash
python3 llm_cli.py -v "Привет"
```

При `-v` скрипт выводит в `stderr`:
- время ответа (`latency_ms`);
- провайдера и модель;
- `finish_reason`;
- токены (`prompt_tokens`, `completion_tokens`, `total_tokens`);
- `cost` (если провайдер вернул это поле).

Ограничения на формат/длину/завершение:

```bash
python3 llm_cli.py \
  --response-format "Ответ строго в JSON: {\"summary\": string, \"items\": [string]}" \
  --max-output-tokens 120 \
  --stop-sequence "<END>" \
  --finish-instruction "Заверши ответ строкой <END>" \
  "Дай краткое описание LLM"
```

По умолчанию:
- если есть `GROQ_API_KEY`, используется Groq (`llama-3.1-8b-instant`);
- иначе используется OpenRouter (`openrouter/auto`).

Если модель недоступна, скрипт автоматически пробует фолбэки:
- `qwen/qwen-2.5-7b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`

Можно переопределить модель и URL:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=ваш_api_ключ
LLM_MODEL=mistralai/mistral-7b-instruct:free
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
EOF

python3 llm_cli.py "Напиши краткий план обучения ML"
```

Можно задать свой список фолбэков:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=ваш_api_ключ
LLM_FALLBACK_MODELS=google/gemma-2-9b-it:free,qwen/qwen-2.5-7b-instruct:free
EOF
```

## Запуск через bash + curl

```bash
chmod +x llm_request.sh
./llm_request.sh "Привет! Ответь в 1 предложении."
```

С флагами:

```bash
./llm_request.sh --temperature 0.2 --top-p 0.9 --top-k 40 "Привет"
```

С ограничениями:

```bash
./llm_request.sh \
  --response-format "Ответ в 3 пунктах, каждый с префиксом '- '" \
  --max-output-tokens 80 \
  --stop-sequence "<END>" \
  --finish-instruction "Заверши ответ строкой <END>" \
  "Объясни, что такое LLM"
```

Минимальная статистика для bash-скрипта:

```bash
./llm_request.sh -v "Привет"
```

При `-v`:
- в `stdout` остается только текст ответа;
- в `stderr` выводится статистика (`latency_ms`, токены, `finish_reason`, модель, `cost` если есть).

## Контроль формата, длины и завершения

Оба скрипта поддерживают одинаковые опции:

- `--response-format` / `LLM_RESPONSE_FORMAT`  
  Явно задает формат ответа (например, JSON, список, фиксированная структура).
- `--max-output-tokens` / `LLM_MAX_OUTPUT_TOKENS`  
  Ограничивает максимальную длину ответа по токенам (через `max_tokens`).
- `--stop-sequence` / `LLM_STOP_SEQUENCES`  
  Задает стоп-последовательности, при встрече которых генерация останавливается.
  Для CLI можно передавать флаг несколько раз.
  Для env-переменной используется список через запятую.
- `--finish-instruction` / `LLM_FINISH_INSTRUCTION`  
  Добавляет явную инструкцию о том, как модель должна завершить ответ (альтернатива stop sequence).

Пример через переменные окружения:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=ваш_api_ключ
LLM_RESPONSE_FORMAT='JSON: {"answer": string, "confidence": number}'
LLM_MAX_OUTPUT_TOKENS=100
LLM_STOP_SEQUENCES='<END>,###'
LLM_FINISH_INSTRUCTION='Остановись сразу после строки <END>.'
EOF

python3 llm_cli.py "Что такое LLM?"
./llm_request.sh "Что такое LLM?"
```

Если нужен другой endpoint/model:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=ваш_api_ключ
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=mistralai/mistral-7b-instruct:free
EOF

./llm_request.sh "Дай 3 идеи pet-проекта"
```

## Пример `.env` файла

```dotenv
# Обязательное: один из ключей
OPENROUTER_API_KEY=ваш_api_ключ
# GROQ_API_KEY=ваш_api_ключ

# Опционально: выбор провайдера (auto/openrouter/groq)
LLM_PROVIDER=auto

# Опционально: endpoint и модель
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=openrouter/auto
LLM_FALLBACK_MODELS=qwen/qwen-2.5-7b-instruct:free,google/gemma-2-9b-it:free,mistralai/mistral-7b-instruct:free

# Опционально: сэмплинг
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_TOP_K=40

# Опционально: ограничения ответа
LLM_RESPONSE_FORMAT='Ответ в 3 пунктах, каждый с префиксом "- "'
LLM_MAX_OUTPUT_TOKENS=120
LLM_STOP_SEQUENCES='<END>,###'
LLM_FINISH_INSTRUCTION='Заверши ответ строкой <END>'
```

Загрузка этого файла в текущую shell-сессию:

```bash
set -a
source .env
set +a
```

## Разбор ошибок

- `SSL: CERTIFICATE_VERIFY_FAILED`  
  На системе не хватает корректного набора CA-сертификатов.  
  Решение:
  1. `pip install certifi`
  2. добавить `SSL_CERT_FILE` в `.env`:
     `SSL_CERT_FILE=/полный/путь/к/cacert.pem`

- `HTTP 403: error code 1010`  
  Это блокировка со стороны Cloudflare/OpenRouter (часто регион/политика).  
  Решение:
  - попробовать другой IP/VPN,
  - или использовать другой совместимый endpoint через `LLM_API_URL`.

- `HTTP 404: No endpoints found for ...`  
  У модели сейчас нет доступных endpoint.  
  Решение:
  - скрипт уже делает авто-фолбэк,
  - или переключиться на Groq через `.env`:
    - `LLM_PROVIDER=groq`
    - `GROQ_API_KEY=...`

## Пример кода

```python
import json
import os
import urllib.request

api_key = os.getenv("OPENROUTER_API_KEY")
payload = {
    "model": "meta-llama/llama-3.1-8b-instruct:free",
    "messages": [{"role": "user", "content": "Привет! Что такое LLM?"}]
}

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print(data["choices"][0]["message"]["content"])
```

Пример запроса
```bash
./llm_request.sh "Что такое LLM?"
LLM — большая языковая модель, обученная на больших корпусах текста для генерации и понимания естественного языка.
```

Пример запроса на питоне
```bash
python3 llm_cli.py "Привет"                                                                
Привет! Рад помочь. Что хочешь сделать: ответить на вопрос, написать текст, перевести, разобрать код, или что-то ещё? Сформулируй запрос — и я постараюсь помочь.
```