"""Intentionally broken: off-by-one in last_n (day40 bug task)."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def last_n(items: list[T], n: int) -> list[T]:
    if n <= 0:
        return []
    return items[-n:]


if __name__ == "__main__":
    print(last_n([1, 2, 3, 4, 5], 2))
