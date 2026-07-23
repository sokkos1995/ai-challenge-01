---
name: add-app-service
description: Добавляет новый app/services/*_service.py по паттернам hw01 (dataclass DTO, типизированный API, при необходимости подключение в agent/cli). Используйте при создании сервиса, фичи в app/services или выносе логики из SimpleLLMAgent.
---

# Добавление app-сервиса

## Шаги

1. Прочитайте `AGENTS.md` и небольшой существующий сервис (например `task_lifecycle_guard_service.py`) как шаблон.
2. Создайте `app/services/<name>_service.py`:
   - `from __future__ import annotations`
   - импорты: stdlib → third-party → `app.*`
   - опциональный `@dataclass` DTO
   - класс `<Name>Service` с типизированным `__init__` и публичными методами
3. Данные сохраняйте только через `app/storage.py` или отдельный storage-хелпер — без inline SQL.
4. Подключайте сервис в `SimpleLLMAgent` и/или CLI/command handlers только если фиче это нужно.
5. Добавьте `tests/test_<name>_service.py` с точечными unit-тестами (`tmp_path`, без сети).
6. Запустите `python3 -m pytest tests/test_<name>_service.py -q`.

## Нельзя

- Класть CLI-циклы с `print` внутрь сервиса.
- Хардкодить API keys или имена моделей (используйте config/env).
- Размазывать `Any` по публичному API.
