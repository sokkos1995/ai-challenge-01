"""CLI entry: python -m homeworks.src.day_49_security_loop.run_loop"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "homeworks" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "homeworks" / "src"))

from day_49_security_loop.gateway_client import GatewayClient  # noqa: E402
from day_49_security_loop.loop import DEFAULT_ART, SecurityLoop  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day 49 security execution loop")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Deterministic fixtures + in-process gateway guards (no live LLM)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=int(os.getenv("DAY49_MAX_ITERS", "3")),
        help="Max generate→test→security iterations per task",
    )
    parser.add_argument(
        "--art-dir",
        type=Path,
        default=None,
        help="Artifacts directory (default homeworks/artifacts/day_49)",
    )
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8848"),
        help="day_48 gateway base URL",
    )
    args = parser.parse_args(argv)

    art = args.art_dir or DEFAULT_ART
    offline = args.offline or os.getenv("DAY49_OFFLINE", "").strip() in {"1", "true", "yes"}

    if offline:
        loop = SecurityLoop(art_dir=art, offline=True, max_iters=args.max_iters)
    else:
        gateway = GatewayClient(base_url=args.gateway_url, mode="redact")
        loop = SecurityLoop(
            art_dir=art,
            gateway=gateway,
            offline=False,
            max_iters=args.max_iters,
        )

    results = loop.run()
    committed = sum(1 for r in results if r.commit_status == "committed")
    print(f"done: {committed}/{len(results)} committed → {art}", flush=True)
    for r in results:
        print(
            f"  {r.task_id}: {r.commit_status} "
            f"sec={r.security_caught or '-'} gw={r.gateway_caught or 'clean'}",
            flush=True,
        )
    return 0 if committed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
