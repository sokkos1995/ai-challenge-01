#!/usr/bin/env python3
"""Validate day_41 chat JSONL: JSON, roles system/user/assistant, non-empty content."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_ROLES = ("system", "user", "assistant")


def validate_line(line: str, line_no: int) -> list[str]:
    errors: list[str] = []
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        return [f"L{line_no}: invalid JSON ({exc})"]

    if not isinstance(obj, dict):
        return [f"L{line_no}: root must be object"]

    messages = obj.get("messages")
    if not isinstance(messages, list):
        return [f"L{line_no}: missing messages list"]

    if len(messages) != 3:
        errors.append(f"L{line_no}: expected exactly 3 messages, got {len(messages)}")

    roles = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"L{line_no}: messages[{i}] must be object")
            continue
        role = msg.get("role")
        content = msg.get("content")
        roles.append(role)
        if role not in REQUIRED_ROLES:
            errors.append(f"L{line_no}: messages[{i}] bad role={role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"L{line_no}: messages[{i}] empty content")

    if len(messages) >= 3:
        expected = list(REQUIRED_ROLES)
        if roles[:3] != expected:
            errors.append(
                f"L{line_no}: roles must be {expected}, got {roles[:3]}"
            )
    else:
        for role in REQUIRED_ROLES:
            if role not in roles:
                errors.append(f"L{line_no}: missing role {role}")

    return errors


def validate_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    ok = 0
    if not path.is_file():
        return 0, [f"file not found: {path}"]
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                errors.append(f"L{line_no}: empty line")
                continue
            line_errors = validate_line(line, line_no)
            if line_errors:
                errors.extend(line_errors)
            else:
                ok += 1
    return ok, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate chat JSONL for fine-tuning")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSONL files to validate",
    )
    args = parser.parse_args()

    total_ok = 0
    all_errors: list[str] = []
    for path in args.paths:
        ok, errors = validate_file(path)
        total_ok += ok
        all_errors.extend(errors)
        print(f"{path}: {ok} ok, {len(errors)} errors")

    if all_errors:
        for err in all_errors[:50]:
            print(err, file=sys.stderr)
        if len(all_errors) > 50:
            print(f"... and {len(all_errors) - 50} more", file=sys.stderr)
        print(f"FAILED: {total_ok} valid lines, {len(all_errors)} errors")
        return 1

    print(f"OK: {total_ok} valid lines across {len(args.paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
