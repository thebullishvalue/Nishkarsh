"""
Nishkarsh v2.0.0 — TTL memory + disk cache with last-good snapshots. NO PICKLE.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Storage (audit A5-16 — the v1 cache unpickled any ``*.pkl`` in its folder, so a planted
file executed code):
  * ``pd.DataFrame``            → Parquet (one file; the entry's metadata in the schema).
  * ``pd.Series``               → Parquet (single column; name kept in the metadata).
  * ``dict[str, DataFrame]``    → ONE long Parquet (rows keyed by ``__key__``); per-key
                                  columns, dtypes and index name are kept in the metadata,
                                  so the round trip is exact.
  * JSON-able values            → JSON envelope (list / dict / str / int / float / bool / None;
                                  tuples come back as lists).
  * anything else               → ``TypeError`` (never pickled).
Every write is atomic: a unique temp file in the same folder, then ``os.replace``.

Layout: ``<root>/<namespace>/<version>/<sha256(key)>.{parquet,json}``. The version folder IS
the namespace version: bumping it orphans old entries; legacy v1 ``*.pkl`` files sit one level
up and are never opened (``purge_legacy_pickles`` deletes them without reading).

Stale lookup (audit A5-06): every entry carries a ``tag`` (e.g. a hash of the symbol set);
``latest(tag)`` returns the newest snapshot for that tag whatever its date window, and the
caller trims it to the window it wants — a day rollover no longer orphans the snapshot.
At most ``max_per_tag`` snapshots are kept per tag.

Force refresh is an explicit ``force=`` argument on the fetchers (audit A5-05).
``begin_force_refresh()`` survives only as a deprecated, process-wide shim for the v1 app.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_FORMAT = 2
DEFAULT_TTL_SECONDS = 3600
_META_KEY = b"nishkarsh_cache"
_ENVELOPE = "__nishkarsh_cache__"


def default_cache_root() -> Path:
    """``$NISHKARSH_CACHE_DIR`` if set, else ``~/.cache/nishkarsh``."""
    env = os.environ.get("NISHKARSH_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "nishkarsh"


DEFAULT_CACHE_DIR = default_cache_root()


@dataclass
class CacheEntry:
    value: Any
    ts: float
    meta: dict = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.ts)


# ── serialisation ────────────────────────────────────────────────────────────

def _json_default(o: Any) -> Any:
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"value of type {type(o).__name__} is not cacheable (JSON)")


def _clone(value: Any) -> Any:
    """Defensive copy so callers can never mutate a cached object in place."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return value


def _is_frames(value: Any) -> bool:
    return (isinstance(value, dict) and len(value) > 0
            and all(isinstance(k, str) for k in value)
            and all(isinstance(v, pd.DataFrame) for v in value.values()))


def _check_columns(df: pd.DataFrame) -> None:
    if isinstance(df.columns, pd.MultiIndex) or not all(isinstance(c, str) for c in df.columns):
        raise TypeError("only DataFrames with flat string column names are cacheable")


def _write_parquet(path: Path, df: pd.DataFrame, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=True)
    md = dict(table.schema.metadata or {})
    md[_META_KEY] = json.dumps(meta, default=_json_default).encode()
    table = table.replace_schema_metadata(md)
    _atomic(path, lambda tmp: pq.write_table(table, tmp))


def _read_parquet(path: Path) -> tuple[pd.DataFrame, dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {}).get(_META_KEY, b"{}"))
    return table.to_pandas(), meta


def _read_meta_only(path: Path) -> dict | None:
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            md = pq.read_schema(path).metadata or {}
            return json.loads(md.get(_META_KEY, b"{}"))
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("meta", {})
    except Exception as e:  # noqa: BLE001 — a corrupt entry is skipped, never fatal
        log.warning("Cache: unreadable entry %s (%s)", path.name, e)
    return None


def _atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def encode_value(value: Any, meta: dict) -> tuple[str, Any]:
    """Return (suffix, payload-writer) for ``value``; raises TypeError if not cacheable."""
    meta = dict(meta)
    if isinstance(value, pd.DataFrame):
        _check_columns(value)
        meta["kind"] = "frame"
        return ".parquet", lambda p: _write_parquet(p, value, meta)
    if isinstance(value, pd.Series):
        meta["kind"] = "series"
        meta["series_name"] = value.name if isinstance(value.name, (str, type(None))) else str(value.name)
        return ".parquet", lambda p: _write_parquet(p, value.to_frame("__value__"), meta)
    if _is_frames(value):
        keys = list(value.keys())
        info = {}
        for k, df in value.items():
            _check_columns(df)
            info[k] = {"columns": list(df.columns),
                       "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                       "index_name": df.index.name if isinstance(df.index.name, (str, type(None))) else str(df.index.name),
                       "index_dtype": str(df.index.dtype)}
        parts = []
        for k, df in value.items():
            d = df.copy()
            d.index = d.index.rename("__idx__")
            d.insert(0, "__key__", k)
            parts.append(d.reset_index())
        long = pd.concat(parts, ignore_index=True, sort=False)
        meta.update({"kind": "frames", "keys": keys, "frames": info})
        return ".parquet", lambda p: _write_parquet(p, long, meta)
    meta["kind"] = "json"
    text = json.dumps({_ENVELOPE: CACHE_FORMAT, "meta": meta, "value": value},
                      default=_json_default, allow_nan=True)

    def _w(p):
        def writer(tmp):
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        _atomic(Path(p), writer)
    return ".json", _w


def decode_path(path: Path) -> CacheEntry:
    """Read an entry written by :func:`encode_value` (never executes code)."""
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            env = json.load(f)
        if not isinstance(env, dict) or env.get(_ENVELOPE) != CACHE_FORMAT:
            raise ValueError("not a Nishkarsh cache envelope")
        meta = env.get("meta", {})
        return CacheEntry(env.get("value"), float(meta.get("ts", 0.0)), meta)
    if path.suffix != ".parquet":
        raise ValueError(f"refusing to read {path.suffix} cache file")
    df, meta = _read_parquet(path)
    kind = meta.get("kind")
    if kind == "frame":
        value: Any = df
    elif kind == "series":
        value = df["__value__"].rename(meta.get("series_name"))
    elif kind == "frames":
        value = {}
        groups = {k: g for k, g in df.groupby("__key__", sort=False)} if len(df) else {}
        for k in meta["keys"]:
            inf = meta["frames"][k]
            g = groups.get(k)
            if g is None:
                g = pd.DataFrame(columns=["__idx__"] + inf["columns"])
            sub = g.set_index("__idx__")[inf["columns"]]
            try:
                sub = sub.astype(inf["dtypes"])
            except (TypeError, ValueError):
                pass
            try:
                sub.index = sub.index.astype(inf["index_dtype"])
            except (TypeError, ValueError):
                pass
            sub.index.name = inf["index_name"]
            value[k] = sub
    else:
        raise ValueError(f"unknown cache entry kind {kind!r}")
    return CacheEntry(value, float(meta.get("ts", 0.0)), meta)


# ── the cache ────────────────────────────────────────────────────────────────

class Cache:
    """TTL cache with a memory tier, an atomic pickle-free disk tier and tagged snapshots.

    Args:
        ttl: entry lifetime in seconds for the "fresh" path (``get``).
        disk_dir: cache root (``None`` → ``default_cache_root()``).
        version: namespace version — entries live under ``<root>/<namespace>/<version>/``.
        namespace: sub-folder, e.g. "ohlcv".
        max_per_tag: snapshots kept per tag (older ones are deleted on ``put``).
    """

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS, disk_dir: Path | None = None,
                 version: str = "v2", namespace: str = "", max_per_tag: int = 3) -> None:
        self.ttl = ttl
        self.version = version
        self.namespace = namespace
        self.max_per_tag = max_per_tag
        self._memory: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.hits = self.misses = self.stale_hits = self.writes = 0
        self.last_fetch_time: float | None = None
        self._disk_ok = True
        self.relocate(disk_dir)

    # -- location ---------------------------------------------------------
    def relocate(self, disk_dir: Path | None) -> None:
        """Point this cache at a new root (clears the memory tier)."""
        base = Path(disk_dir) if disk_dir is not None else default_cache_root()
        with self._lock:
            self._root = (base / self.namespace) if self.namespace else base
            self._disk_dir = self._root / self.version
            self._memory.clear()
            try:
                self._disk_dir.mkdir(parents=True, exist_ok=True)
                self._disk_ok = True
            except OSError as e:
                self._disk_ok = False
                log.warning("Cache '%s': disk tier unavailable (%s) — memory only", self.namespace, e)

    @property
    def disk_dir(self) -> Path:
        return self._disk_dir

    def _key(self, *args: Any) -> str:
        raw = json.dumps([self.version, *[str(a) for a in args]], ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:40]

    def _paths(self, key: str) -> list[Path]:
        return [self._disk_dir / f"{key}.parquet", self._disk_dir / f"{key}.json"]

    def _load(self, key: str) -> CacheEntry | None:
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        if not self._disk_ok:
            return None
        for p in self._paths(key):
            if p.exists():
                try:
                    entry = decode_path(p)
                except Exception as e:  # noqa: BLE001
                    log.warning("Cache '%s': disk read failed for %s: %s", self.namespace, key[:8], e)
                    return None
                with self._lock:
                    self._memory[key] = entry
                return entry
        return None

    # -- public API -------------------------------------------------------
    def get_entry(self, *args: Any) -> CacheEntry | None:
        """The stored entry for this exact key regardless of age (copy of the value)."""
        e = self._load(self._key(*args))
        return None if e is None else CacheEntry(_clone(e.value), e.ts, dict(e.meta))

    def get(self, *args: Any) -> Any | None:
        """Value if fresh (age < ttl), else None."""
        e = self._load(self._key(*args))
        with self._lock:
            if e is not None and time.time() - e.ts < self.ttl:
                self.hits += 1
                return _clone(e.value)
            self.misses += 1
        return None

    def get_stale(self, *args: Any) -> Any | None:
        """Last-good value for this exact key even if expired (fetch-failure fallback)."""
        e = self._load(self._key(*args))
        if e is None:
            return None
        with self._lock:
            self.stale_hits += 1
        return _clone(e.value)

    def put(self, *args: Any, value: Any, tag: str | None = None, meta: dict | None = None) -> bool:
        """Store ``value`` (both tiers). Returns False if it could not be persisted to disk
        (unsupported type / IO error) — it is then still held in memory for this process."""
        key = self._key(*args)
        ts = time.time()
        m = dict(meta or {})
        m.update({"ts": ts, "tag": tag, "version": self.version, "namespace": self.namespace,
                  "format": CACHE_FORMAT})
        entry = CacheEntry(_clone(value), ts, m)
        with self._lock:
            self._memory[key] = entry
            self.writes += 1
            self.last_fetch_time = ts
        if not self._disk_ok:
            return False
        try:
            suffix, write = encode_value(value, m)
            write(self._disk_dir / f"{key}{suffix}")
            for other in self._paths(key):          # a key changed kind → drop the old file
                if other.suffix != suffix and other.exists():
                    other.unlink()
        except Exception as e:  # noqa: BLE001
            log.warning("Cache '%s': disk write failed for %s: %s", self.namespace, key[:8], e)
            return False
        if tag is not None:
            self._prune(tag)
        return True

    def _disk_metas(self) -> list[tuple[Path, dict]]:
        if not self._disk_ok:
            return []
        out = []
        for p in list(self._disk_dir.glob("*.parquet")) + list(self._disk_dir.glob("*.json")):
            if p.name.startswith(".tmp-"):
                continue
            m = _read_meta_only(p)
            if m is not None:
                out.append((p, m))
        return out

    def latest(self, tag: str, *, max_age_seconds: float | None = None) -> CacheEntry | None:
        """Newest entry carrying ``tag`` (any key / window), optionally age-capped."""
        best_key, best_ts = None, -1.0
        with self._lock:
            for k, e in self._memory.items():
                if e.meta.get("tag") == tag and e.ts > best_ts:
                    best_key, best_ts = k, e.ts
        for p, m in self._disk_metas():
            if m.get("tag") == tag and float(m.get("ts", 0)) > best_ts:
                best_key, best_ts = p.stem, float(m.get("ts", 0))
        if best_key is None:
            return None
        if max_age_seconds is not None and time.time() - best_ts > max_age_seconds:
            return None
        e = self._load(best_key)
        if e is None:
            return None
        with self._lock:
            self.stale_hits += 1
        return CacheEntry(_clone(e.value), e.ts, dict(e.meta))

    def _prune(self, tag: str) -> None:
        entries = sorted(((float(m.get("ts", 0)), p) for p, m in self._disk_metas() if m.get("tag") == tag),
                         reverse=True)
        for _, p in entries[self.max_per_tag:]:
            try:
                p.unlink()
            except OSError:
                pass
            with self._lock:
                self._memory.pop(p.stem, None)

    def invalidate(self, *args: Any) -> None:
        key = self._key(*args)
        with self._lock:
            self._memory.pop(key, None)
        for p in self._paths(key):
            try:
                p.unlink()
            except OSError:
                pass

    def clear(self) -> None:
        """Clear the memory tier (disk snapshots are kept as the stale fallback)."""
        with self._lock:
            self._memory.clear()

    def stats(self) -> dict[str, Any]:
        """Snapshot of cache stats for diagnostics."""
        n_disk = 0
        if self._disk_ok:
            try:
                n_disk = sum(1 for p in self._disk_dir.iterdir()
                             if p.suffix in (".parquet", ".json") and not p.name.startswith(".tmp-"))
            except OSError:
                pass
        with self._lock:
            total = self.hits + self.misses
            return {
                "namespace": self.namespace or self._root.name,
                "version": self.version,
                "ttl_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "stale_hits": self.stale_hits,
                "writes": self.writes,
                "hit_rate": (self.hits / total) if total else 0.0,
                "memory_entries": len(self._memory),
                "disk_entries": n_disk,
                "last_fetch_time": self.last_fetch_time,
            }


# ── Module-level cache instances (one per source; independent versions) ─────
# v2 = pickle-free format. Bump a version to invalidate that namespace.
sheets_cache = Cache(ttl=3600, version="v2", namespace="sheets")
ohlcv_cache = Cache(ttl=3600, version="v2", namespace="ohlcv")
macro_cache = Cache(ttl=3600, version="v2", namespace="macro")
constituents_cache = Cache(ttl=3600, version="v2", namespace="constituents")


def all_caches() -> list[Cache]:
    """All module-level cache instances, for diagnostics."""
    return [sheets_cache, ohlcv_cache, macro_cache, constituents_cache]


def set_cache_root(path: Path | str) -> None:
    """Re-root every module-level cache (tests, isolated runs)."""
    for c in all_caches():
        c.relocate(Path(path))


def purge_legacy_pickles(root: Path | str | None = None) -> int:
    """Delete v1 ``*.pkl`` cache files WITHOUT opening them. Returns the count removed."""
    base = Path(root) if root is not None else default_cache_root()
    n = 0
    if base.exists():
        for p in base.rglob("*.pkl"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n


# ── Deprecated v1 force-refresh shim ─────────────────────────────────────────
# v1 keyed this by Streamlit session id, which worker threads do not carry, so the
# "Refresh Data" button was a no-op for macro/OHLCV (audit A5-05). Callers should pass
# ``force=True`` to the fetchers. Until app.py is rewritten, this opens a PROCESS-WIDE
# window during which fetchers treat ``force=False`` as ``force=True`` (cache is bypassed,
# snapshots are kept as the failure fallback). No Streamlit import, no session magic.
_FORCE_UNTIL = 0.0


def begin_force_refresh(window: float = 300.0) -> None:
    """DEPRECATED — pass ``force=True`` to the fetchers instead."""
    global _FORCE_UNTIL
    _FORCE_UNTIL = time.time() + float(window)


def force_window_active() -> bool:
    return time.time() < _FORCE_UNTIL
