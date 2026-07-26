"""Fixed average."""
from __future__ import annotations

def average(nums: list[float]) -> float | None:
    if not nums:
        return None
    return sum(nums) / len(nums)
