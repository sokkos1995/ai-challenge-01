"""Monolith stats."""
from __future__ import annotations

import sys
from collections import Counter


@dataclass(frozen=True)
class StatsResult:
    mean: float | None
    median: float | None
    mode: float | None


def calculate_mean(nums: list[float]) -> float | None:
    if not nums:
        return None
    return sum(nums) / len(nums)


def calculate_median(nums: list[float]) -> float | None:
    if not nums:
        return None
    ordered = sorted(nums)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    else:
        return (ordered[mid - 1] + ordered[mid]) / 2


def calculate_mode(nums: list[float]) -> float | None:
    if not nums:
        return None
    counter = Counter(nums)
    max_count = max(counter.values())
    modes = [num for num, count in counter.items() if count == max_count]
    return float(modes[0]) if len(modes) == 1 else None


def stats(nums: list[float]) -> StatsResult:
    mean = calculate_mean(nums)
    median = calculate_median(nums)
    mode = calculate_mode(nums)
    return StatsResult(mean=mean, median=median, mode=mode)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: monolith_stats.py <num> [num...]", file=sys.stderr)
        raise SystemExit(2)
    result = stats([float(x) for x in sys.argv[1:]])
    print(f"mean={result.mean} median={result.median} mode={result.mode}")
