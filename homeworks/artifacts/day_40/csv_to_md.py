"""CSV to GitHub-flavored markdown table (day40 feature)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def csv_to_md(text: str) -> str:
    reader = csv.reader(text.splitlines())
    rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        text = Path(args[0]).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    print(csv_to_md(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
