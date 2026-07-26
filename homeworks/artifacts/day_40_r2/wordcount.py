"""Word/line/char counter (day40-r2)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

def count_text(text: str) -> tuple[int, int, int]:
    words = len(text.split())
    if not text:
        lines = 0
    else:
        lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    return words, lines, len(text)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count words, lines, and characters.")
    parser.add_argument("path", nargs="?", help="file path (default: stdin)")
    args = parser.parse_args(argv)
    text = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
    words, lines, chars = count_text(text)
    print(f"words={words} lines={lines} chars={chars}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
