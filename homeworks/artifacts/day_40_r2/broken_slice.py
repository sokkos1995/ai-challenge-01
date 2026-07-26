"""last_n helper (day40-r2)."""
from __future__ import annotations
from typing import TypeVar
T = TypeVar("T")

def last_n(items: list[T], n: int) -> list[T]:
    if n <= 0:
        return []
    return items[-n:]

if __name__ == "__main__":
    print(last_n([1, 2, 3, 4, 5], 2))
