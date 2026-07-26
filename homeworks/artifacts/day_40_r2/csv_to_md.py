"""CSV to markdown table (day40-r2)."""
from __future__ import annotations
import csv, sys
from pathlib import Path

def csv_to_md(text: str) -> str:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    text = Path(args[0]).read_text(encoding="utf-8") if args else sys.stdin.read()
    print(csv_to_md(text))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
