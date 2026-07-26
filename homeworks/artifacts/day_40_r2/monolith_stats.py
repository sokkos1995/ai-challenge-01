"""Stats helpers mean/median/mode (day40-r2)."""
from __future__ import annotations
import sys
from collections import Counter

def mean(nums: list[float]) -> float | None:
    return None if not nums else sum(nums) / len(nums)

def median(nums: list[float]) -> float | None:
    if not nums:
        return None
    ordered = sorted(nums)
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

def mode(nums: list[float]) -> float | None:
    return None if not nums else float(Counter(nums).most_common(1)[0][0])

def stats(nums: list[float]) -> dict[str, float | None]:
    return {"mean": mean(nums), "median": median(nums), "mode": mode(nums)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: monolith_stats.py <num> [num...]", file=sys.stderr)
        raise SystemExit(2)
    result = stats([float(x) for x in sys.argv[1:]])
    print(f"mean={result['mean']} median={result['median']} mode={result['mode']}")
