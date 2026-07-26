# Feature — qwen2.5-coder:3b

- elapsed_sec: 14.64
- prompt_eval_count: 1020
- eval_count: 321

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
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text must not be empty")

        slug = (
            cleaned.lower()
            .replace(" ", "-")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(".", "-")
            .replace(",", "-")
            .replace("?", "-")
            .replace("!", "-")
            .replace("(", "-")
            .replace(")", "-")
            .replace("{", "-")
            .replace("}", "-")
            .replace("[", "-")
            .replace("]", "-")
            .replace(";", "-")
            .replace(":", "-")
            .replace("'", "")
            .replace('"', "")
        )

        slug = "--".join(slug.split("--"))
        if len(slug) > self._max_length:
            slug = slug[:self._max_length]

        return SlugNormalizeResult(ok=True, slug=slug, detail=f"Normalized '{text}' to '{slug}'")
```
```
