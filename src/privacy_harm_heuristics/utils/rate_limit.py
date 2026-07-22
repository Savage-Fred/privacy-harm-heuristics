"""Lightweight pacing limiter to throttle request rate.

Designed for simplicity and robustness in single-process jobs. Use per-client
instances to avoid coupling unrelated API consumers.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class PaceLimiter:
    """Enforce a minimum interval between operations.

    Args:
        per_minute: Target operations per minute. ``<= 0`` disables limiting.
    """

    def __init__(self, per_minute: Optional[float] = None) -> None:
        self._lock = threading.Lock()
        self._last_ts: Optional[float] = None
        if per_minute and per_minute > 0:
            self._min_interval = 60.0 / float(per_minute)
        else:
            self._min_interval = 0.0

    def wait(self) -> None:
        """Sleep as needed to maintain the configured pace."""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            last = self._last_ts
            if last is None:
                self._last_ts = now
                return
            elapsed = now - last
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
            self._last_ts = now
