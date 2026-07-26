"""Anagram checker (day40-r2)."""
from __future__ import annotations
import sys
from collections import Counter

def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if not ch.isspace())

def is_anagram(a: str, b: str) -> bool:
    return Counter(normalize(a)) == Counter(normalize(b))

def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: anagrams.py <str1> <str2>", file=sys.stderr)
        return 2
    print("YES" if is_anagram(args[0], args[1]) else "NO")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
