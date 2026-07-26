"""JSON pretty-printer (day40-r2)."""
from __future__ import annotations
import json, sys
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        raw = Path(args[0]).read_text(encoding="utf-8") if args else sys.stdin.read()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
