# Day 39 — prompts (feature + agent)

## Feature prompt (День 1 style)

```text
Сгенерируй полный файл app/services/slug_normalize_service.py для hw01.

Требования:
- from __future__ import annotations
- frozen dataclass SlugNormalizeResult с полями: ok: bool, slug: str, detail: str
- класс SlugNormalizeService:
  - __init__(self, max_length: int = 64)
  - normalize(self, text: str) -> SlugNormalizeResult
  - поведение: trim, lower, заменить пробелы/не-алфанумерик на "-", схлопнуть "--", обрезать по max_length, strip "-" по краям
  - пустой вход -> ValueError("text must not be empty")
- без print, без SQL, без I/O
- только код файла, без пояснений вокруг
```

## Agent prompt (День 2 / Bug Fix style)

```text
Режим Bug Fix (только анализ, код не пиши целиком).

Симптом: PersonalizationService.needs_interview() иногда возвращает True даже после того,
как пользователь заполнил все REQUIRED_PROFILE_KEYS через save_interview_answers.

Ниже фрагмент app/services/personalization_service.py. Найди вероятную причину и план фикса.

Ответ строго в формате:
Причина: ...
Что починить: ...
Что проверить: ...
Риски: ...
```
