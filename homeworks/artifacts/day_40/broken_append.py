"""Intentionally broken: mutable default argument (day40 bug task)."""

from __future__ import annotations


def append_item(value: str, bucket: list[str] | None = None) -> list[str]:
    if bucket is None:
        bucket = []
    bucket.append(value)
    return bucket


if __name__ == "__main__":
    print(append_item("a"))
    print(append_item("b"))
