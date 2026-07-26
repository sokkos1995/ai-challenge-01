# Feature — deepseek-coder:6.7b

- elapsed_sec: 51.79
- prompt_eval_count: 1419
- eval_count: 634

```
```python
from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class SlugNormalizeResult:
    ok: bool
    slug: str
    detail: str

class SlugNormalizeService:
    def __init__(self, max_length: int = 64):
        self._max_length = max(1, max_length)

    def normalize(self, text: str) -: SlugNormalizeResult:
        if not text.strip():
            raise ValueError("text must not be empty")
        
        normalized = re.sub(r'[^a-zA-Z0-9 ]', '', text).lower()
        normalized = re.sub(r'\s+', '-', normalized)
        normalized = re.sub(r'-+', '-', normalized)
        normalized = normalized.strip('-')
        
        if len(normalized) > self._max_length:
            detail = f"Truncated to {self._max_length} characters."
            slug = normalized[:self._max_length]
        else:
            detail = ""
            slug = normalized
            
        return SlugNormalizeResult(ok=True, slug=slug, detail=detail)
```

В этом файле мы определяем класс `SlugNormalizeService` с методом `normalize()`. Метод принимает строку и возвращает объект типа `SlugNormalizeResult`, который содержит результат нормализации слаг-строки.

Нормализация выполняется следующим образом:
1. Удаляем все символы кроме букв и цифр, заменяем пробелы на дефисы, схлопываем несколько дефисов в один.
2. Преобразуем строку к нижнему регистру.
3. Удаляем начальные и конечные дефисы.
4. Если результирующая слаг-строка превышает заданную длину, обрезаем ее до максимальной длины и указываем в `detail` сообщении что строка была обрезанна.
5. Если слаг-строка не превышает заданную длину, поле `detail` останется пустым.
```
```

```
