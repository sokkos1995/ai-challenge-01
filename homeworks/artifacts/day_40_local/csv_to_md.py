import csv
from typing import List, Optional


def csv_to_md(text: str) -> str:
    reader = csv.reader(text.splitlines())
    headers = next(reader)
    rows = list(reader)

    md_table = f"| {' | '.join(headers)} |\n"
    md_table += f"| {' | '.join(['---'] * len(headers))} |\n"

    for row in rows:
        md_table += f"| {' | '.join(row)} |\n"

    return md_table


def main(argv: Optional[List[str]] = None) -> int:
    import sys
    if argv is None:
        argv = sys.argv

    if len(argv) != 2:
        print("Usage: python csv_to_md.py <input.csv>")
        return 1

    with open(argv[1], 'r') as file:
        text = file.read()

    md_table = csv_to_md(text)
    print(md_table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
