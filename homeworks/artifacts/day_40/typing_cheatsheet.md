# Python typing cheatsheet (day40)

```python
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Optional, TypedDict, Union

# 1) list
names: list[str] = ["ada", "grace"]

# 2) dict
ages: dict[str, int] = {"ada": 36}

# 3) Optional / |
maybe: Optional[int] = None
also: int | None = 1

# 4) Union
Id = Union[int, str]
ticket: Id = "abc"

# 5) Callable
def apply(fn: Callable[[int], str], value: int) -> str:
    return fn(value)

# 6) TypedDict
class User(TypedDict):
    id: int
    name: str

user: User = {"id": 1, "name": "ada"}

# 7) Iterable / Mapping
def sum_ints(values: Iterable[int]) -> int:
    return sum(values)

def get_name(row: Mapping[str, str]) -> str:
    return row["name"]

# 8) tuple fixed length
Point = tuple[float, float]
origin: Point = (0.0, 0.0)
```
