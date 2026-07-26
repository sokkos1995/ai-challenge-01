"""Fixed last_n."""
from __future__ import annotations
from typing import TypeVar
T = TypeVar("T")

def last_n(items: list[T], n: int) -> list[T]:
    if n <= 0:
        return []
    return items[-n:]
