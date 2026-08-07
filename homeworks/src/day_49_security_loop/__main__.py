"""python -m homeworks.src.day_49_security_loop → offline loop by default for safety."""

from __future__ import annotations

from .run_loop import main

if __name__ == "__main__":
    raise SystemExit(main(["--offline"]))
