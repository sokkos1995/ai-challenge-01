"""FizzBuzz CLI (day40 feature)."""

from __future__ import annotations

import sys


def fizzbuzz(n: int) -> list[str]:
    lines: list[str] = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            lines.append("FizzBuzz")
        elif i % 3 == 0:
            lines.append("Fizz")
        elif i % 5 == 0:
            lines.append("Buzz")
        else:
            lines.append(str(i))
    return lines


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: fizzbuzz.py N", file=sys.stderr)
        return 2
    try:
        n = int(args[0])
    except ValueError:
        print("usage: fizzbuzz.py N", file=sys.stderr)
        return 2
    if n < 1:
        print("usage: fizzbuzz.py N", file=sys.stderr)
        return 2
    for line in fizzbuzz(n):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
