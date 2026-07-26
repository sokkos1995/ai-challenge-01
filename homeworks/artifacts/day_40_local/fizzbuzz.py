from __future__ import annotations

import sys


def fizzbuzz(n: int) -> list[str]:
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    if len(argv) != 2:
        print("Usage: python fizzbuzz.py <n>")
        return 2

    try:
        n = int(argv[1])
    except ValueError:
        print("Invalid argument. Please provide a valid integer.")
        return 2

    result = fizzbuzz(n)
    for item in result:
        print(item)

    return 0


if __name__ == "__main__":
    sys.exit(main())
