"""Word/line/char counter (day40 feature)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def count_text(text: str) -> tuple[int, int, int]:
    lines = text.splitlines() if text else []
    # Keep trailing empty line semantics consistent with wc-ish: empty → 0 lines
    if text and text.endswith("\n") and lines == []:
        lines = [""]
    words = text.split()
    chars = len(text)
    line_count = text.count("\n") + (0 if not text or text.endswith("\n") else 1)
    if not text:
        line_count = 0
    return len(words), line_count, chars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count words, lines, and characters.")
    parser.add_argument("path", nargs="?", help="file path (default: stdin)")
    args = parser.parse_args(argv)
    if args.path:
        text = Path(args.path).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    words, lines, chars = count_text(text)
    print(f"words={words} lines={lines} chars={chars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
