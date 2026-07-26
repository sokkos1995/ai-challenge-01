"""Password generator (day40 feature)."""

from __future__ import annotations

import argparse
import secrets
import string
import sys


def generate(length: int) -> str:
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(alphabet) for _ in range(length))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a random password.")
    parser.add_argument("--length", type=int, default=16)
    args = parser.parse_args(argv)
    if args.length < 8:
        print("length must be >= 8", file=sys.stderr)
        return 2
    print(generate(args.length))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
