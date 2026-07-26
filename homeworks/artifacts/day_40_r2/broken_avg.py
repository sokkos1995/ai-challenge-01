"""Average helper (day40-r2)."""
from __future__ import annotations

def average(nums: list[float]) -> float | None:
    if not nums:
        return None
    return sum(nums) / len(nums)

if __name__ == "__main__":
    print(average([1.0, 2.0, 3.0]))
    print(average([]))
