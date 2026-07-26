# День 39. Local Boost — Ollama + Continue.dev

## Цель

Подключить локальную LLM как код-ассистент в IDE: чат + tab autocomplete, перенести правила из `AGENTS.md` в системный промпт Continue, прогнать задачи «фича» и «агент», сравнить с облачным Cursor Agent.

## Железо и связка

| Параметр | Значение |
|----------|----------|
| Машина | MacBook Air (Mac14,2), Apple M2, 16 GB |
| Runtime | Ollama (`http://localhost:11434`) |
| IDE | **VS Code** + расширение **Continue** (в Cursor Marketplace Continue часто нет) |
| Облачный baseline | Cursor Agent (сравнение в таблице ниже) |
| Правила | [`.continue/rules/hw01-agents.md`](../.continue/rules/hw01-agents.md) ← сжатый [`AGENTS.md`](../AGENTS.md) |
| Конфиг проекта | [`.continue/config.yaml`](../.continue/config.yaml) |

## Установка (чеклист)

### 1. Ollama

```bash
# сервер
ollama serve

# модели (уже проверены на этом Mac)
ollama pull qwen2.5-coder:1.5b   # autocomplete
ollama pull qwen2.5-coder:3b     # сравнение
ollama pull qwen2.5-coder:7b     # основной chat
ollama pull deepseek-coder:6.7b  # сравнение
```

Smoke:

```bash
ollama run qwen2.5-coder:1.5b "Reply with one word: pong"
# → pong (~2.5s cold-ish)
```

### 2. Настройка Continue в VS Code

Почему «не видно настроек»: у Continue **нет** привычной страницы в VS Code Settings (`Cmd+,`) с полями temperature/моделей. Всё живёт в YAML:

| Что | Где |
|-----|-----|
| Глобальный конфиг (то, что открывает шестерёнка в UI) | `~/.continue/config.yaml` |
| Конфиг этого репо | [`.continue/config.yaml`](../.continue/config.yaml) |
| Правила hw01 | [`.continue/rules/hw01-agents.md`](../.continue/rules/hw01-agents.md) |

#### 2.1. Поставить расширение

1. Открыть **Visual Studio Code** (не Cursor).
2. Extensions (`Cmd+Shift+X`) → поиск **`Continue`** → publisher **Continue** → **Install**.  
   Marketplace: [Continue - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue).
3. **File → Open Folder…** → корень репозитория `hw01` (чтобы подтянулись `.continue/rules`).

#### 2.2. Подтянуть модели Ollama в Local Config

Если в панели Continue пустой список моделей / нет наших имён — UI смотрит на **глобальный** `~/.continue/config.yaml`, а не обязательно на файл в репо. Скопируй project-config:

```bash
mkdir -p ~/.continue
cp /path/to/hw01/.continue/config.yaml ~/.continue/config.yaml
# или из корня репо:
# cp .continue/config.yaml ~/.continue/config.yaml
```

Правила из `.continue/rules/` подхватываются из открытой папки проекта отдельно — их копировать не нужно.

#### 2.3. Открыть панель и проверить конфиг

1. Continue Chat: **`Cmd+L`** (или иконка Continue в Activity Bar слева).
2. Над полем ввода — селектор конфига / агента → наведи на **Local Config** → иконка **шестерёнки (gear)** → откроется `~/.continue/config.yaml`.
3. Убедись, что в файле есть блоки с `provider: ollama` и моделями `qwen2.5-coder:7b`, `qwen2.5-coder:1.5b` и т.д. Сохрани файл — Continue перечитывает конфиг сам.
4. В селекторе модели чата выбери **Qwen2.5-Coder 7B (chat)**.
5. Autocomplete: модель с `roles: [autocomplete]` → **Qwen2.5-Coder 1.5B**; в любом `.py` набери начало функции и прими Tab.
6. Rules (иконка «ручка» / список rules над чатом) → **hw01 AGENTS rules** (`alwaysApply: true`).

#### 2.4. Если всё ещё пусто — быстрый чеклист

| Симптом | Что сделать |
|---------|-------------|
| Нет расширения / не находится в Cursor | Ставить в **VS Code** Marketplace, не в Cursor |
| Модели не в списке | `cp .continue/config.yaml ~/.continue/config.yaml`, снова `Cmd+L` |
| Ошибки сети / timeout | `ollama serve` + `curl http://127.0.0.1:11434/api/tags` |
| Autocomplete молчит | В Continue Settings (gear в UI, не VS Code Settings) включён Tab autocomplete; модель 1.5b скачана (`ollama list`) |
| Rules нет | Открыта именно папка `hw01`, не родительский каталог |
| **Message exceeds context limit** на коротком вопросе | См. §2.5 ниже |

#### 2.5. Ошибка «Message exceeds context limit»

Даже короткое сообщение («какая модель?») может не пройти: Continue кладёт в промпт **system + rules (`> 2 rules`) + открытые/прикреплённые файлы + (в Agent) схемы tools**. Бюджет входа ≈ `contextLength − maxTokens`. При `8192 / 2048` на M2 этого часто не хватает.

Что сделано в конфиге:

- `contextLength: 16384`
- `maxTokens: 1024`

Обнови глобальный файл (если ещё не):

```bash
cp .continue/config.yaml ~/.continue/config.yaml
```

В Continue:

1. Сохрани YAML → подожди авто-reload (или Reload Window).
2. **Новый чат** (не продолжай старый с ошибкой).
3. Переключись с **Agent** на обычный **Chat**, если вопрос простой.
4. Убери лишний контекст: не жми `@codebase` / не таскай огромные файлы; закрой лишние вкладки или сними чипы `@file` над инпутом.

На скрине уже выбрано **Qwen2.5-Coder 7B (chat)** → это **локальная** модель через Ollama (`qwen2.5-coder:7b` @ `localhost:11434`), не облако.

Документация Continue: [Configure Continue](https://docs.continue.dev/customize/deep-dives/configuration), [Autocomplete](https://docs.continue.dev/customize/deep-dives/autocomplete).

### 3. Параметры генерации (в конфиге)

| Режим | temperature | top_p | context / prompt |
|-------|-------------|-------|------------------|
| Chat / edit | 0.2 | 0.9 | `contextLength: 16384`, `maxTokens: 1024` |
| Autocomplete | 0.1 | 0.9 | `maxPromptTokens: 1024`, debounce 250ms, `onlyMyCode: true` |

Контекст файлов: Continue видит открытые/упомянутые файлы (`@file`, codebase); autocomplete — prefix/suffix текущего буфера в пределах `maxPromptTokens`. Менять temperature/top_p/context — только правкой YAML (см. §2), не через Settings VS Code.

---

## Модели

| Роль | Модель | Размер на диске |
|------|--------|-----------------|
| Chat (primary) | `qwen2.5-coder:7b` | 4.7 GB |
| Autocomplete | `qwen2.5-coder:1.5b` | 986 MB |
| Сравнение | `deepseek-coder:6.7b` | 3.8 GB |
| Сравнение | `qwen2.5-coder:3b` | 1.9 GB |

---

## Задачи (те же смыслы, что у облачного ассистента)

Промпты: [`homeworks/artifacts/day_39/prompts.md`](artifacts/day_39/prompts.md).

Сравнительный прогон feature/agent — через Ollama HTTP API (`/api/chat`) с тем же system = `.continue/rules/hw01-agents.md` (эквивалент Continue Chat + rules). UI Continue (чат / Tab autocomplete) — ручной чеклист выше; бенч:

```bash
python3 homeworks/src/day_39_run_local_bench.py
```

### A. Генерация фичи (как День 1)

Сгенерировать `SlugNormalizeService` по шаблону AGENTS (dataclass + Service, без I/O).

| Модель | Время | С первого раза? | Качество кода |
|--------|-------|-----------------|---------------|
| qwen2.5-coder:7b | **23.5 s** | Частично | Структура отличная; slug-логика неполная (слабо non-alnum) |
| deepseek-coder:6.7b | 51.8 s | Нет | Лучше regex-нормализация, но **синтаксическая ошибка** `-) :` и лишний текст |
| qwen2.5-coder:3b | **14.6 s** | Частично | Шаблон ок; хаотичные `.replace`, плохое схлопывание `--` |

Сырые ответы: `homeworks/artifacts/day_39/feature_*.md`.

**Облачный Cursor (baseline дней 36–38):** типично с первого раза даёт полный сервис + тесты + вписывание в дерево `app/`, соблюдает антипаттерны из `AGENTS.md` без урезания.

### B. Агентный режим / Bug Fix (как День 2)

Симптом: `needs_interview()` True после заполнения профиля; формат ответа Причина → Что починить → Что проверить → Риски.

| Модель | Время | Формат | Понимание причины |
|--------|-------|--------|-------------------|
| qwen2.5-coder:7b | **11.9 s** | Строго соблюдён | Поверхностно |
| deepseek-coder:6.7b | 29.5 s | Соблюдён | Видит роль `_interview_completed`, фикс спорный |
| qwen2.5-coder:3b | **9.0 s** | Частично | Неверная трактовка `_is_interview_completed` |

Сырые ответы: `homeworks/artifacts/day_39/agent_*.md`.

**Облачный Cursor + bug-fix:** читает несколько файлов (`storage`, тесты), предлагает точечный фикс и pytest — локальные 3–7B без tool-loop этого не повторяют.

Детальный скоринг: [`artifacts/day_39/evaluation.md`](artifacts/day_39/evaluation.md). Тайминги: [`artifacts/day_39/timings.json`](artifacts/day_39/timings.json).

---

## Сравнение: облачный Cursor vs локальная модель

| Критерий | Cursor Agent (облако) | Локально (лучший: Qwen2.5-Coder 7B + 1.5B) |
|----------|------------------------|--------------------------------------------|
| Качество кода | Высокое: типы, тесты, соответствие AGENTS | Среднее: шаблон сервиса ок, edge-cases и синтаксис плавают |
| Скорость | Сеть + очередь; интерактивно быстро на коротких правках | Фича ~15–25 s, агент ~10–12 s на M2; autocomplete почти мгновенный (1.5B) |
| Понимание контекста проекта | Сильное: multi-file, rules, skills, субагенты | Слабое–среднее: нужен ручной `@file` / context 16k |
| Работа без интернета | Нет (облако) | **Да** (после `ollama pull`) |
| Агентность (tools, pytest, git) | Полный цикл | Continue Agent ограниченнее; 7B без надёжного tool-use как у Cursor |
| Приватность / стоимость токенов | Данные у провайдера; платный план | Локально; $0 за inference |

---

## Для чего хватает локальной / где облако незаменимо

**Локальной достаточно:**

- Tab autocomplete и мелкие дописывания в текущем файле.
- Черновик `*_service.py` / dataclass по шаблону.
- Офлайн-работа, эксперименты без утечки кода в облако.
- Быстрые Q&A по уже открытому фрагменту.

**Облако (Cursor) незаменимо:**

- Многофайловый рефакторинг и агентный Bug Fix с pytest.
- Соблюдение длинного `AGENTS.md` + skills/субагенты (`result-verifier`, `pre-push-secrets`).
- Интеграции API (сначала docs) и сложная диагностика по всему репо.
- Качество «с первого раза» на нетривиальных фичах (как дни 35–38).

---

## Лучшая связка на M2 16GB

| Слот | Выбор |
|------|--------|
| Chat / edit | **`qwen2.5-coder:7b`** |
| Autocomplete | **`qwen2.5-coder:1.5b`** |
| temperature / top_p | **0.2 / 0.9** (чат), **0.1** (autocomplete) |
| context | **16384** chat (`maxTokens: 1024`), **1024** autocomplete prompt tokens |
| Rules | `.continue/rules/hw01-agents.md` всегда apply |

DeepSeek 6.7b на этом железе медленнее и чаще «размазывает» ответ; 3b — только если нужна скорость ценой качества рассуждений.

---

## Результат (deliverables)

- Локальные модели в Ollama + конфиг Continue (чат + autocomplete).
- Правила `AGENTS.md` в system/rules Continue.
- Артефакты прогона 3 моделей + таблица сравнения + рекомендация выше.

Повторить бенч:

```bash
ollama serve   # если не запущен
python3 homeworks/src/day_39_run_local_bench.py
```
