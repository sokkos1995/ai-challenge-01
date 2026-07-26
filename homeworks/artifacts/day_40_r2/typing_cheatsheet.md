# Python typing cheatsheet (day40-r2)

```python
from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping
from typing import Optional, TypedDict, Union

names: list[str] = ["ada"]
ages: dict[str, int] = {"ada": 36}
maybe: Optional[int] = None
also: int | None = 1
Id = Union[int, str]
ticket: Id = "abc"

def apply(fn: Callable[[int], str], value: int) -> str:
    return fn(value)

class User(TypedDict):
    id: int
    name: str

user: User = {"id": 1, "name": "ada"}

def sum_ints(values: Iterable[int]) -> int:
    return sum(values)

def get_name(row: Mapping[str, str]) -> str:
    return row["name"]

Point = tuple[float, float]
origin: Point = (0.0, 0.0)
```
