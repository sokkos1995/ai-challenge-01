"""Temp converter (day40-r2)."""
from __future__ import annotations
import sys

def c2f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)

def f2c(f: float) -> float:
    return round((f - 32) * 5 / 9, 1)

def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in {"c2f", "f2c"}:
        print("usage: temp_convert.py c2f|f2c <number>", file=sys.stderr)
        return 2
    try:
        value = float(args[1])
    except ValueError:
        print("usage: temp_convert.py c2f|f2c <number>", file=sys.stderr)
        return 2
    print(c2f(value) if args[0] == "c2f" else f2c(value))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
