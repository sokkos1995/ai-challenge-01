from __future__ import annotations

import sys
from typing import List


@dataclass(frozen=True)
class AnagramResult:
    is_anagram: bool


class AnagramsService:
    """Одна ответственность. Публичные методы — глаголы предметной области."""

    def __init__(self) -> None:
        pass

    def is_anagram(self, str1: str, str2: str) -> AnagramResult:
        cleaned_str1 = ''.join(sorted(str1.lower().replace(' ', '')))
        cleaned_str2 = ''.join(sorted(str2.lower().replace(' ', '')))
        return AnagramResult(is_anagram=cleaned_str1 == cleaned_str2)


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    if len(argv) != 3:
        print("Usage: python anagrams.py <string1> <string2>")
        return 1

    str1, str2 = argv[1], argv[2]
    service = AnagramsService()
    result = service.is_anagram(str1, str2)
    print("YES" if result.is_anagram else "NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
