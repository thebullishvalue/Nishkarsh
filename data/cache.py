"""
Nishkarsh v1.2.0 — TTL-based memory and disk cache for data fetching.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

DATA — Provides a simple key-value cache with time-to-live expiry.

Provides a simple key-value cache with time-to-live expiry.
Cache keys are derived from function arguments via MD5 hashing.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "nishkarsh"
DEFAULT_TTL_SECONDS = 3600  # 1 hour


class Cache:
    """Time-to-live cache with optional disk persistence.

    Parameters
    ----------
    ttl : int
        Cache entry lifetime in seconds.
    disk_dir : Path | None
        Directory for disk persistence. ``None`` disables disk caching.
    """

    def __init__(
        self,
        ttl: int = DEFAULT_TTL_SECONDS,
        disk_dir: Path | None = None,
    ) -> None:
        self.ttl = ttl
        self._memory: dict[str, tuple[Any, float]] = {}
        self._disk_dir = disk_dir or DEFAULT_CACHE_DIR
        if disk_dir is not None:
            self._disk_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(*args: Any) -> str:
        """Derive an MD5 cache key from positional arguments."""
        raw = "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *args: Any) -> Any | None:
        """Retrieve a cached value, or ``None`` if expired/missing."""
        key = self._key(*args)
        if key in self._memory:
            val, ts = self._memory[key]
            if time.time() - ts < self.ttl:
                return val
            del self._memory[key]

        # Try disk
        disk_path = self._disk_dir / f"{key}.pkl"
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as f:
                    val, ts = pickle.load(f)
                if time.time() - ts < self.ttl:
                    self._memory[key] = (val, ts)
                    return val
                disk_path.unlink()
            except Exception:
                pass
        return None

    def put(self, *args: Any, value: Any) -> None:
        """Store a value in the cache with the current timestamp."""
        key = self._key(*args)
        ts = time.time()
        self._memory[key] = (value, ts)

        # Persist to disk
        disk_path = self._disk_dir / f"{key}.pkl"
        try:
            with open(disk_path, "wb") as f:
                pickle.dump((value, ts), f)
        except Exception:
            pass

    def invalidate(self, *args: Any) -> None:
        """Remove a specific entry from both memory and disk."""
        key = self._key(*args)
        self._memory.pop(key, None)
        disk_path = self._disk_dir / f"{key}.pkl"
        if disk_path.exists():
            disk_path.unlink()

    def clear(self) -> None:
        """Remove all entries from memory."""
        self._memory.clear()
