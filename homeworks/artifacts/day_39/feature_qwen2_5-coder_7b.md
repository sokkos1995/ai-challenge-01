# Feature — qwen2.5-coder:7b

- elapsed_sec: 23.51
- prompt_eval_count: 1020
- eval_count: 201

```
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlugNormalizeResult:
    ok: bool
    slug: str
    detail: str


class SlugNormalizeService:
    """Одна ответственность. Публичные методы — глаголы предметной области."""

    def __init__(self, max_length: int = 64) -> None:
        self._max_length = max(1, max_length)

    def normalize(self, text: str) -> SlugNormalizeResult:
        if not text:
            raise ValueError("text must not be empty")

        slug = text.strip().lower()
        slug = '-'.join(filter(None, slug.split()))
        slug = '--'.join(slug.split('--'))
        slug = slug[:self._max_length].strip('-')

        return SlugNormalizeResult(ok=True, slug=slug, detail=f"Normalized to: {slug}")
```
```
