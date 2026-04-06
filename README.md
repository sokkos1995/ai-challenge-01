# Минимальный запрос к LLM через API (без OpenAI)

Этот проект показывает минимальный пример, который:
- отправляет запрос в LLM через API,
- получает ответ,
- выводит ответ в консоль (CLI).
- использует отдельную сущность агента (`SimpleLLMAgent`) с инкапсулированной логикой запроса/ответа.

## Файл проекта

- `app/agent.py` — сущность агента (`SimpleLLMAgent`): инкапсулирует логику запроса/ответа к LLM API.
- `app/services/` — сервисный слой: работа с провайдером, памятью, историей чата, персонализацией и token accounting.
- `app/cli_utils.py` — CLI-утилиты (`parse_args`, `resolve_prompt`, `print_verbose_stats`).
- `app/cli.py` — основной CLI-поток (single prompt + chat), использует модули пакета `app`.
- `llm_cli.py` — тонкая точка входа, делегирует запуск в `app.cli.main`.
- `llm_request.sh` — bash/curl скрипт для отправки запроса.
- `prompt.txt` — пример длинного промпта (удобно подставлять через `-f`).
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

Если аргумент не передан и не задан файл промпта (см. ниже), скрипт использует короткий промпт по умолчанию.

### Простой чат-режим (CLI-агент)

```bash
python3 llm_cli.py --chat
```

В этом режиме агент принимает сообщения по одному, отправляет их в LLM API и выводит ответы.  
Выход: `exit`, `quit` или `q`.

Режим с компрессией истории включается отдельным флагом:

```bash
python3 llm_cli.py --chat --summary
```

В режиме `--summary`:
- последние `N` сообщений хранятся "как есть";
- более старые реплики сворачиваются в отдельный `summary`;
- в запрос к модели подставляется `summary` + свежий "хвост" диалога.

Начиная с `day_10` можно выбирать стратегию управления контекстом через `--context-strategy`:
- `full` — отправлять всю историю чата;
- `summary` — как `--summary` (summary + последние `N`);
- `sliding` — отправлять только последние `N` сообщений;
- `facts` — отправлять отдельно накопленные key-value `facts` + последние `N` сообщений;
- `branching` — поддерживать 2 ветки от checkpoint (см. команды ниже);
- `memory` — явные memory layers: `short-term`, `working`, `long-term`.

Персонализация пользователя включается через `--user-id`:

```bash
python3 llm_cli.py --chat --context-strategy memory --user-id 123
```

Что происходит:
- для нового `user_id` создается профиль в скрытой папке `.llm_users/`;
- при первом запуске проводится мини-интервью (`role`, `stack`, стиль/детальность ответа, формат, ограничения);
- профиль автоматически подмешивается в `system prompt` каждого запроса;
- chat history и memory layers становятся раздельными для каждого `user_id`.

Поддерживаемые поля профиля:
- `role` — кто пользователь и в каком контексте работает;
- `stack` — основной стек/инструменты;
- `answer_detail` — предпочитаемая детализация (`brief` / `detailed`);
- `answer_format` — предпочитаемый формат ответа (`bullets`, `step-by-step`, `text`, `json`, ...);
- `constraints` — дополнительные ограничения.

Важно:
- персонализация реализована в Python CLI (`python3 llm_cli.py`);
- `llm_request.sh` остается stateless-скриптом без `--user-id` и пользовательского профиля.

Команды внутри `branching`:
- `@checkpoint` — сохранить checkpoint;
- `@fork` — создать 2 ветки (`1` и `2`) и активировать ветку `1`;
- `@switch 1` / `@switch 2` — переключиться на ветку;
- `@branches` (или `@branch-info`) — показать состояние веток.

Команды для memory layers (режим `--context-strategy memory`):
- `@mem show` — показать снимок всех слоев памяти;
- `@mem clear short|work|long|all` — очистить выбранный слой;
- `@mem short note <text>` — явно сохранить заметку в краткосрочную память;
- `@mem work <field>=<value>` — обновить рабочую память (`task`, `state`, `plan_status`, `validation_status`, `paused`, `step`, `total`, `expected_action`, `plan+`, `done+`, `note+`);
- `@mem long profile <key>=<value>` — сохранить профиль пользователя в долговременную память;
- `@mem long knowledge <key>=<value>` — сохранить знание в долговременную память;
- `@mem long decision <text>` — добавить решение в долговременную память.

Task state machine в рабочей памяти:
- стадии: `PLANNING`, `EXECUTION`, `VALIDATION`, `DONE`, `REJECTED`;
- разрешенные переходы: `PLANNING -> EXECUTION|REJECTED`, `EXECUTION -> VALIDATION|PLANNING|REJECTED`, `VALIDATION -> DONE|EXECUTION|REJECTED`, `DONE -> REJECTED`, `REJECTED` — терминальное состояние;
- явные статусы контроля: `plan_status = DRAFT|APPROVED`, `validation_status = PENDING|PASSED|FAILED`;
- `PLANNING -> EXECUTION` разрешен только если план не пустой и имеет статус `APPROVED`;
- `VALIDATION -> DONE` разрешен только если `validation_status = PASSED`;
- задача может быть отменена на любом этапе через переход в `REJECTED`, в том числе если она стоит на паузе;
- при изменении плана approval сбрасывается обратно в `DRAFT`, а после новых изменений в реализации валидация снова становится `PENDING`;
- пауза доступна на любой стадии через отдельный флаг `paused`;
- состояние задачи, статусы контроля, план, ожидаемое действие и заметки автоматически подмешиваются в system prompt через prompt builder;
- в `memory`-режиме отдельный lifecycle guard не дает ассистенту перепрыгнуть этап, например начать реализацию до утверждения плана или выдать финал до завершенной валидации.

Команды для task state machine:
- `@task show` — показать текущее состояние задачи;
- `@task pause` — поставить задачу на паузу;
- `@task resume` — снять задачу с паузы;
- `@task reject` — отменить задачу и перевести ее в `REJECTED`;
- `@task approve-plan` — утвердить текущий план;
- `@task reject-plan` — вернуть план в `DRAFT`;
- `@task validate pass|fail` — явно зафиксировать результат валидации;
- `@task plan+ <text>` — добавить пункт в план;
- `@task done+ <text>` — добавить выполненный пункт;
- `@task expected <text>` — обновить ожидаемое следующее действие;
- `@task state <PLANNING|EXECUTION|VALIDATION|DONE|REJECTED>` — перевести задачу в следующий допустимый stage.

Команды для персонализации (доступны при запуске с `--user-id`):
- `@personalization show` — показать текущий профиль пользователя;
- `@personalization interview` — заново пройти мини-интервью;
- `@personalization <key>=<value>` — дополнить или обновить персонализацию вручную.

Пример:

```bash
python3 llm_cli.py --chat --context-strategy memory --user-id 123
```

```text
@personalization show
@personalization constraints=Не предлагай JavaScript
@personalization answer_format=step-by-step
```

Настройки через env:
- `LLM_CHAT_KEEP_LAST_N` — размер окна последних сообщений (по умолчанию `10`);
- `LLM_CHAT_SUMMARY_BATCH_SIZE` — размер батча старых сообщений для очередного обновления summary (по умолчанию `10`);
- `LLM_MEMORY_BASE_PATH` — базовый путь для memory layers (по умолчанию `.llm_memory`; создаются файлы `.short.db`, `.work.db`, `.long.db`).
- `LLM_USERS_BASE_PATH` — корень для пользовательских профилей и per-user сессий (по умолчанию `.llm_users`);
- `LLM_USER_ID` — user id по умолчанию, если не хочется передавать `--user-id` каждый раз.

Что хранится на диске при персонализации:
- `.llm_users/users.db` — профили пользователей и флаг завершенности интервью;
- `.llm_users/sessions/<user_id>/...` — отдельные chat/memory SQLite-файлы для конкретного пользователя.

Команды внутри чата:
- `@tokens` / `@tokens off` — включить/выключить вывод токенов;
- `@summary` — показать текущее накопленное summary (только в режиме `--summary`).

### Промпт из файла (`-f` / `--prompt-file`)

Текст запроса можно читать из файла в кодировке UTF-8:

```bash
python3 llm_cli.py -f prompt.txt
python3 llm_cli.py -v --temperature 0.2 -f prompt.txt
```

Если указан **`-f`**, позиционные аргументы с текстом промпта **игнорируются**.

Путь к файлу по умолчанию можно задать в **`.env`** или в окружении переменной **`LLM_PROMPT_FILE`** (например, `LLM_PROMPT_FILE=prompt.txt`) — тогда при необходимости можно вызывать `python3 llm_cli.py -v` без `-f`, если значение подхватилось из окружения после `source .env`.

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

# Опционально: файл с текстом промпта (как у флага -f; пустую строку не задавайте)
# LLM_PROMPT_FILE=prompt.txt

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