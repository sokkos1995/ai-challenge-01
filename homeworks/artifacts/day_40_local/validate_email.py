"""Email CLI with helper."""
from __future__ import annotations

import re
import sys


def is_valid_email(email: str) -> bool:
    """Validate an email address using a regex pattern."""
    _PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(_PATTERN, email) is not None


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: validate_email.py <email>", file=sys.stderr)
        raise SystemExit(2)

    email = sys.argv[1]
    if is_valid_email(email):
        print("OK")
    else:
        print("INVALID")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
