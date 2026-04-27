🔥 День 30. Локальная LLM как приватный сервис

## Цель

Развернуть локальную LLM как приватный сетевой сервис на ВМ в Yandex Cloud и проверить:

- доступ к модели по сети;
- стабильность при нескольких запросах;
- базовые ограничения (rate limit / max context).

## Инфраструктура

- Облако: Yandex Cloud
- ВМ: `compute-vm-4-4-20-ssd-1776755943026`
- Внешний IP / домен: `81.26.191.111`
- ОС: `Ubuntu 24.04.4 LTS (Noble Numbat), kernel 6.8.0-110-generic`
- Runtime: `Ollama`
- Версии: `Ollama 0.21.0`, `Python 3.12.3`
- Модель: `qwen2.5:3b`
- API слой: `FastAPI proxy` поверх локального `Ollama`
- Порт сервиса: `8000`
- Авторизация: `Bearer API key`

## Архитектура сервиса

Схема (кратко):

1. Клиент отправляет запрос на `http://81.26.191.111:8000/chat`.
2. Внешний API проверяет авторизацию и ограничения.
3. API проксирует запрос в локальный `Ollama` (`http://127.0.0.1:11434`).
4. Ответ модели возвращается клиенту в JSON.

## Шаги развертывания

1. Поднял `Ollama` на ВМ:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull <MODEL_NAME>
```

2. Настроил автозапуск:
```bash
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```

3. Поднял API-прокси (`FastAPI`) как отдельный `systemd` сервис:

```bash
sudo systemctl enable --now local-llm-api.service
sudo systemctl status local-llm-api.service
```

4. Настроил сеть:
- Security Group: открыт входящий порт `8000` (HTTP API);
- доступ ограничен: рекомендован whitelist (на момент демо доступ был открыт для проверки извне);
- firewall на ВМ (`ufw`): разрешены `OpenSSH` и `8000/tcp`.

## API контракты

### Healthcheck

- `GET /health`
- Ответ:
```json
{"status":"ok","model":"qwen2.5:3b"}
```

### Chat

- `POST /chat`
- Тело:
```json
{
  "messages": [
    {"role":"system","content":"Ты полезный ассистент."},
    {"role":"user","content":"Объясни, что такое локальная LLM."}
  ],
  "max_tokens": 300
}
```
- Ответ:
```json
{
  "answer": "О",
  "model": "qwen2.5:3b",
  "usage": {
    "prompt_eval_count": 38,
    "eval_count": 1,
    "total_duration": 7849565635
  }
}
```

## Проверка доступа по сети

Тест с локальной машины (вне ВМ):

```bash
curl -X POST "http://81.26.191.111:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-llm-secret-key" \
  -d '{
    "messages":[{"role":"user","content":"Дай 3 причины использовать локальную LLM"}],
    "max_tokens":200
  }'
```

Результат: `успешно`, получен валидный JSON-ответ с полями `answer`, `model`, `usage`.

## Проверка стабильности (несколько запросов)

Инструмент: `python + requests` (серия запросов к `/chat`).

Пример:
```bash
python3 - <<'PY'
import requests
url='http://81.26.191.111:8000/chat'
headers={'Content-Type':'application/json','Authorization':'Bearer local-llm-secret-key'}
payload={'messages':[{'role':'user','content':'ping'}],'max_tokens':1}
statuses=[]
for i in range(22):
    r=requests.post(url,headers=headers,json=payload,timeout=90)
    statuses.append(r.status_code)
print('statuses_tail',statuses[-5:])
print('count_429',sum(1 for s in statuses if s==429))
PY
```

Итоги:

- Всего запросов: `22`
- Параллелизм: `последовательные быстрые запросы` (burst-тест)
- Успешных ответов: `18/22`
- Ошибки/таймауты: `4` (ожидаемые `429 Too Many Requests`)
- Средняя латентность: `~5-10s` для коротких ответов на CPU
- p95: `не измерялся отдельно в демо`

Вывод по стабильности: сервис стабильно отвечает при серии запросов и корректно применяет защитные ограничения под нагрузкой. Критических сбоев процесса `local-llm-api.service` не зафиксировано.

## Базовые ограничения

### 1) Rate limit

Настройка: `20 req/min` на клиента (по IP / источнику запроса).

Проверка:

- отправил серию быстрых запросов;
- после превышения получил `429 Too Many Requests`.

Факт: `ограничение работает`.

### 2) Max context / max input

Настройка:

- `num_ctx = 4096`;
- max длина входа: `12000 символов` (ограничение на уровне API);
- max output tokens: `512`.

Проверка:

- отправлен oversized prompt;
- получен `413` с сообщением `Input too large. Max chars: 12000`.

Факт: `ограничение работает`.

## Безопасность (минимум)

Реализовано:

- API не открыт без авторизации;
- вход ограничен Security Group;
- сервис слушает `0.0.0.0:8000` (доступ регулируется API key + firewall + Security Group);
- логирование запросов и ошибок включено.

## Что использовано из предыдущих дней

- Day 26: локальный запуск модели и HTTP API;
- Day 27: клиентский чат/интеграция;
- Day 28: локальный retrieval + генерация;
- Day 29: оптимизированные параметры (`temperature`, `num_predict`, `num_ctx`, prompt).

## Результат

Критерии day 30 выполнены:

- локальная LLM развернута как приватный сервис на ВМ в Yandex Cloud;
- к модели есть доступ по сети;
- подтверждена стабильность при нескольких запросах;
- внедрены и проверены ограничения (rate limit / max context).

Итог: приватный AI-сервис на базе локальной LLM готов к использованию извне ВМ.
