import argparse
from typing import Tuple


def count_text(text: str) -> Tuple[int, int, int]:
    words = len(text.split())
    lines = text.count('\n') + 1 if '\n' in text else 1
    chars = len(text)
    return words, lines, chars


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Count words, lines, and characters in a text.")
    parser.add_argument('path', type=str, nargs='?', help='Path to the file containing the text')
    args = parser.parse_args(argv)

    if args.path:
        with open(args.path, 'r') as file:
            text = file.read()
    else:
        import sys
        text = sys.stdin.read()

    words, lines, chars = count_text(text)
    print(f"Words: {words}")
    print(f"Lines: {lines}")
    print(f"Chars: {chars}")

    return 0


if __name__ == "__main__":
    exit(main())
