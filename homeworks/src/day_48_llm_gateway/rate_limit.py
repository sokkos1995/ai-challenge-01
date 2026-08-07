"""Per-IP sliding-window rate limiter."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allow at most ``limit`` events per ``window_sec`` for each key (IP)."""

    def __init__(self, *, limit: int = 30, window_sec: float = 60.0) -> None:
        self.limit = max(1, limit)
        self.window_sec = max(1.0, window_sec)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        ts = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            cutoff = ts - self.window_sec
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(ts)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
