"""Email validation helper (day40-r2)."""
from __future__ import annotations
import re, sys
_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def is_valid_email(s: str) -> bool:
    return bool(_PATTERN.match(s))

def main() -> None:
    if len(sys.argv) != 2:
        print("usage: validate_email.py <email>", file=sys.stderr)
        raise SystemExit(2)
    if is_valid_email(sys.argv[1]):
        print("OK")
    else:
        print("INVALID")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
