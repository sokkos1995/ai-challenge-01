# Day 39 — evaluation notes (local models)

## Feature task scoring (0–5)

| Model | Structure/AGENTS | Spec (slug rules) | First-try usable | Speed | Notes |
|-------|------------------|-------------------|------------------|-------|-------|
| qwen2.5-coder:7b | 5 | 2 | 3 | 4 (23.5s) | Шаблон сервиса ок; slug почти только по пробелам, non-alnum слабо |
| deepseek-coder:6.7b | 3 | 4 | 2 | 2 (51.8s) | Логика slug лучше (re), но синтаксис `-) :` и лишний текст |
| qwen2.5-coder:3b | 4 | 2 | 3 | 5 (14.6s) | Шаблон ок; replace-хак, плохое схлопывание `--` |

## Agent task scoring (0–5)

| Model | Format | Root cause quality | Speed | Notes |
|-------|--------|--------------------|-------|-------|
| qwen2.5-coder:7b | 5 | 2 | 5 (11.9s) | Формат соблюдён; причина поверхностная |
| deepseek-coder:6.7b | 4 | 3 | 3 (29.5s) | Видит флаг `_interview_completed`, фикс спорный |
| qwen2.5-coder:3b | 3 | 1 | 5 (9.0s) | Неверная трактовка `_is_interview_completed` |

## Winner on this hardware (M2 16GB)

- **Chat/agent:** `qwen2.5-coder:7b`, temperature `0.2`, top_p `0.9`, context `8192`
- **Autocomplete:** `qwen2.5-coder:1.5b`, maxPromptTokens `1024`, temperature `0.1`
- DeepSeek 6.7b — запасной вариант, медленнее и менее стабилен по синтаксису на M2
- 3b — только для быстрых черновиков / autocomplete-adjacent задач
