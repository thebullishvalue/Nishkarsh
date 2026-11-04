"""
Nishkarsh v2.0.0 — the data-source boundary the pipeline codes against (contract C-DATA).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

    FetchResult(value, provenance, as_of, missing, note)
        provenance ∈ {"live", "cache", "stale", "fallback", "empty"}
    DataSources (Protocol):
        constituents() -> FetchResult[list[str]]
        macro(start, end, force=False) -> FetchResult[DataFrame]          (ticker columns, NOT lagged)
        ohlcv(symbols, start, end, force=False) -> FetchResult[dict[str, DataFrame]]
        index_levels(symbol, start, end, force=False) -> FetchResult[pd.Series]   (Close levels)

Implementations:
    LiveSources()          — the real fetchers (yfinance / niftyindices / Google Sheets).
    SnapshotSources(dir)   — offline replay of a directory written by ``save_snapshot`` /
                             ``snapshot_from_sources`` (Parquet + JSON; no pickle).
    SyntheticSources(seed) — deterministic, offline world from ``data.synthetic``.
All three also offer ``sheet(...)`` (not part of the Protocol).

Windows are inclusive [start, end]; every implementation trims to the requested window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import pandas as pd

Provenance = Literal["live", "cache", "stale", "fallback", "empty"]
PROVENANCES: tuple[str, ...] = ("live", "cache", "stale", "fallback", "empty")
SNAPSHOT_SCHEMA = "nishkarsh.snapshot/1"

__all__ = ["FetchResult", "DataSources", "LiveSources", "SnapshotSources", "SyntheticSources",
           "save_snapshot", "snapshot_from_sources", "PROVENANCES", "Provenance"]


@dataclass(frozen=True)
class FetchResult:
    value: Any
    provenance: Provenance
    as_of: str | None = None
    missing: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {PROVENANCES}, got {self.provenance!r}")

    @property
    def ok(self) -> bool:
        return self.provenance != "empty"

    def to_dict(self) -> dict:
        """JSON-safe description WITHOUT the value (for run reports)."""
        return {"provenance": self.provenance, "as_of": self.as_of,
                "missing": list(self.missing), "note": self.note}


@runtime_checkable
class DataSources(Protocol):
    def constituents(self) -> FetchResult: ...
    def macro(self, start, end, force: bool = False) -> FetchResult: ...
    def ohlcv(self, symbols, start, end, force: bool = False) -> FetchResult: ...
    def index_levels(self, symbol, start, end, force: bool = False) -> FetchResult: ...


def _norm(symbols) -> list[str]:
    out, seen = [], set()
    for s in symbols or []:
        u = str(s).strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _bounds(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def _trim(obj, s: pd.Timestamp, e: pd.Timestamp):
    return obj.loc[(obj.index >= s) & (obj.index < e + pd.Timedelta(days=1))]


def _as_of(objs) -> str | None:
    last = [o.index.max() for o in objs if o is not None and len(o)]
    return str(pd.Timestamp(max(last)).date()) if last else None


# ── Live ─────────────────────────────────────────────────────────────────────

class LiveSources:
    """The production sources (network). Thin delegation to data.fetcher / data.constituents."""

    name = "live"

    def constituents(self, force: bool = False) -> FetchResult:
        from data.constituents import fetch_constituents
        return fetch_constituents(force=force)

    def macro(self, start, end, force: bool = False) -> FetchResult:
        from data.fetcher import fetch_macro
        return fetch_macro(start, end, force=force)

    def ohlcv(self, symbols, start, end, force: bool = False) -> FetchResult:
        from data.fetcher import fetch_ohlcv
        return fetch_ohlcv(symbols, start, end, force=force)

    def index_levels(self, symbol, start, end, force: bool = False) -> FetchResult:
        from data.fetcher import fetch_index_levels
        return fetch_index_levels(symbol, start, end, force=force)

    def sheet(self, url: str, force: bool = False) -> FetchResult:
        from data.fetcher import fetch_sheet
        return fetch_sheet(url, force=force)


# ── Snapshot (offline replay) ────────────────────────────────────────────────

_FILES = {"constituents": "constituents", "macro": "macro", "ohlcv": "ohlcv",
          "index": "index_levels", "sheet": "sheet"}


def _as_result(x: Any) -> FetchResult | None:
    if x is None:
        return None
    if isinstance(x, FetchResult):
        return x
    return FetchResult(x, "live", None, [], "")


def save_snapshot(directory, *, constituents=None, macro=None, ohlcv=None,
                  index_levels=None, sheet=None) -> Path:
    """Write a replayable snapshot (Parquet/JSON, atomic per file). Each argument may be a
    ``FetchResult`` or a bare value. ``index_levels`` is ``{symbol: Series | FetchResult}``."""
    from data.cache import encode_value
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SNAPSHOT_SCHEMA, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                "components": {}}

    def _write(name: str, r: FetchResult | None, value: Any) -> None:
        if r is None:
            return
        meta = {"component": name, **r.to_dict()}
        for p in d.glob(f"{_FILES[name]}.*"):
            p.unlink()
        suffix, writer = encode_value(value, meta)
        writer(d / f"{_FILES[name]}{suffix}")
        manifest["components"][name] = {**r.to_dict(), "file": f"{_FILES[name]}{suffix}"}

    rc = _as_result(constituents)
    _write("constituents", rc, list(rc.value) if rc else None)
    rm = _as_result(macro)
    _write("macro", rm, rm.value if rm else None)
    ro = _as_result(ohlcv)
    _write("ohlcv", ro, dict(ro.value) if ro else None)
    if index_levels:
        frames, notes, provs = {}, [], []
        for sym, v in index_levels.items():
            r = _as_result(v)
            s = r.value if isinstance(r.value, pd.Series) else pd.Series(dtype="float64")
            frames[str(sym).upper()] = s.astype("float64").to_frame("Close")
            provs.append(r.provenance)
            notes.append(f"{sym}: {r.provenance}")
        worst = next((p for p in ("empty", "fallback", "stale", "cache") if p in provs), "live")
        _write("index", FetchResult(None, worst, None, [], "; ".join(notes)), frames)
    rs = _as_result(sheet)
    _write("sheet", rs, rs.value if rs else None)
    from data.cache import _atomic
    text = json.dumps(manifest, indent=2)
    _atomic(d / "manifest.json", lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    return d


def snapshot_from_sources(sources: DataSources, directory, start, end, *,
                          index_symbols=("^NSEI",), sheet: pd.DataFrame | None = None) -> Path:
    """Pull everything from ``sources`` for [start, end] and save it as a snapshot."""
    c = sources.constituents()
    return save_snapshot(
        directory, constituents=c, macro=sources.macro(start, end),
        ohlcv=sources.ohlcv(list(c.value or []), start, end),
        index_levels={s: sources.index_levels(s, start, end) for s in index_symbols},
        sheet=sheet)


class SnapshotSources:
    """Replay a snapshot directory offline. Provenance is ``cache`` (stored data, as intended)
    unless the captured component was itself ``stale`` / ``fallback`` / ``empty``."""

    name = "snapshot"

    def __init__(self, directory) -> None:
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"snapshot directory not found: {self.dir}")
        self._loaded: dict[str, Any] = {}

    def _load(self, name: str):
        if name not in self._loaded:
            from data.cache import decode_path
            hits = [p for p in self.dir.glob(f"{_FILES[name]}.*") if p.suffix in (".parquet", ".json")]
            self._loaded[name] = decode_path(hits[0]) if hits else None
        return self._loaded[name]

    def _prov(self, meta: dict) -> tuple[str, str]:
        orig = meta.get("provenance", "live")
        prov = orig if orig in ("stale", "fallback", "empty") else "cache"
        return prov, f"snapshot replay of {self.dir.name} (captured as {orig}; {meta.get('note', '')})".strip()

    def constituents(self, force: bool = False) -> FetchResult:
        e = self._load("constituents")
        if e is None:
            return FetchResult([], "empty", None, [], f"no constituents in snapshot {self.dir.name}")
        prov, note = self._prov(e.meta)
        return FetchResult(list(e.value or []), prov, e.meta.get("as_of"), [], note)

    def macro(self, start, end, force: bool = False) -> FetchResult:
        e = self._load("macro")
        if e is None or not isinstance(e.value, pd.DataFrame):
            return FetchResult(pd.DataFrame(), "empty", None, [], f"no macro in snapshot {self.dir.name}")
        s, t = _bounds(start, end)
        panel = _trim(e.value, s, t)
        prov, note = self._prov(e.meta)
        if panel.empty:
            return FetchResult(panel, "empty", None, [], note + "; window outside snapshot")
        return FetchResult(panel, prov, _as_of([panel]), list(e.meta.get("missing", [])), note)

    def ohlcv(self, symbols, start, end, force: bool = False) -> FetchResult:
        syms = _norm(symbols)
        e = self._load("ohlcv")
        stored = e.value if e is not None and isinstance(e.value, dict) else {}
        s, t = _bounds(start, end)
        out = {k: _trim(stored[k], s, t) for k in syms if k in stored}
        out = {k: v for k, v in out.items() if len(v)}
        if not out:                         # index symbols are saved separately (Close only)
            ix = self._load("index")
            if ix is not None:
                out = {k: _trim(ix.value[k], s, t) for k in syms if k in ix.value}
                out = {k: v for k, v in out.items() if len(v)}
                e = ix if out else e
        missing = [k for k in syms if k not in out]
        if not out:
            return FetchResult({}, "empty", None, missing, f"symbols not in snapshot {self.dir.name}")
        prov, note = self._prov(e.meta)
        return FetchResult(out, prov, _as_of(out.values()), missing, note)

    def index_levels(self, symbol, start, end, force: bool = False) -> FetchResult:
        sym = _norm([symbol])[0]
        e = self._load("index")
        if e is None or sym not in (e.value or {}):
            return FetchResult(pd.Series(dtype="float64", name=sym), "empty", None, [sym],
                               f"{sym} not in snapshot {self.dir.name}")
        s, t = _bounds(start, end)
        ser = _trim(e.value[sym]["Close"], s, t).rename(sym)
        prov, note = self._prov(e.meta)
        if ser.empty:
            return FetchResult(ser, "empty", None, [sym], note + "; window outside snapshot")
        return FetchResult(ser, prov, _as_of([ser]), [], note)

    def sheet(self, force: bool = False) -> FetchResult:
        e = self._load("sheet")
        if e is None:
            return FetchResult(None, "empty", None, [], f"no sheet in snapshot {self.dir.name}")
        prov, note = self._prov(e.meta)
        return FetchResult(e.value, prov, None, [], note)


# ── Synthetic (deterministic, offline) ───────────────────────────────────────

class SyntheticSources:
    """Deterministic offline sources over a ``data.synthetic.SyntheticWorld``.

    ``SyntheticSources(seed, **world_kwargs)`` builds the same world as
    ``synthetic_world(seed, **world_kwargs)``. Failure injection for tests:
    ``empty=("macro", "ohlcv", "index", "constituents")`` makes those components return
    provenance "empty"; ``drop_symbols=(...)`` makes those symbols ``missing``.
    Provenance of served data is "live" (freshly produced by this source); every note starts
    with "SYNTHETIC" so a run report can never mistake it for market data.
    """

    name = "synthetic"

    def __init__(self, seed: int = 0, *, world=None, empty: tuple[str, ...] = (),
                 drop_symbols: tuple[str, ...] = (), **world_kwargs) -> None:
        if world is None:
            from data.synthetic import build_world
            world = build_world(seed, **world_kwargs)
        self.world = world
        self.seed = world.seed
        self.empty = frozenset(empty)
        self.drop = frozenset(_norm(drop_symbols))
        self.calls: list[tuple] = []

    def _note(self, s=None, e=None, got=None) -> str:
        n = f"SYNTHETIC seed={self.seed}"
        if s is not None and got:
            n += f"; requested {s.date()}..{e.date()}, served {got}"
        return n

    def constituents(self, force: bool = False) -> FetchResult:
        self.calls.append(("constituents",))
        if "constituents" in self.empty:
            return FetchResult([], "empty", None, [], self._note() + "; injected empty")
        return FetchResult(list(self.world.constituents), "live",
                           str(self.world.dates[-1].date()), [], self._note())

    def macro(self, start, end, force: bool = False) -> FetchResult:
        self.calls.append(("macro", str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())))
        if "macro" in self.empty:
            return FetchResult(pd.DataFrame(), "empty", None, [], self._note() + "; injected empty")
        s, t = _bounds(start, end)
        panel = _trim(self.world.macro, s, t).copy()
        if panel.empty:
            return FetchResult(panel, "empty", None, list(self.world.macro.columns), self._note() + "; window outside world")
        a = _as_of([panel])
        return FetchResult(panel, "live", a, [], self._note(s, t, f"{panel.index[0].date()}..{a}"))

    def _index_ohlcv(self, sym: str) -> pd.DataFrame | None:
        return self.world.index_ohlcv.get(sym)

    def ohlcv(self, symbols, start, end, force: bool = False) -> FetchResult:
        syms = _norm(symbols)
        self.calls.append(("ohlcv", tuple(syms)))
        if "ohlcv" in self.empty:
            return FetchResult({}, "empty", None, syms, self._note() + "; injected empty")
        s, t = _bounds(start, end)
        out = {}
        for k in syms:
            if k in self.drop:
                continue
            src = self.world.ohlcv.get(k)
            if src is None:
                src = self._index_ohlcv(k)
            if src is None:
                continue
            sub = _trim(src, s, t).copy()
            if sub["Close"].notna().any():
                out[k] = sub
        missing = [k for k in syms if k not in out]
        if not out:
            return FetchResult({}, "empty", None, missing, self._note() + "; no requested symbol in world/window")
        a = _as_of(out.values())
        return FetchResult(out, "live", a, missing, self._note(s, t, f"..{a}"))

    def index_levels(self, symbol, start, end, force: bool = False) -> FetchResult:
        sym = _norm([symbol])[0]
        self.calls.append(("index_levels", sym))
        src = self._index_ohlcv(sym)
        if "index" in self.empty or src is None or sym in self.drop:
            return FetchResult(pd.Series(dtype="float64", name=sym), "empty", None, [sym],
                               self._note() + ("; injected empty" if "index" in self.empty else f"; {sym} unknown"))
        s, t = _bounds(start, end)
        ser = _trim(src["Close"], s, t).astype("float64").rename(sym)
        if ser.empty:
            return FetchResult(ser, "empty", None, [sym], self._note() + "; window outside world")
        a = _as_of([ser])
        return FetchResult(ser, "live", a, [], self._note(s, t, f"{ser.index[0].date()}..{a}"))

    def sheet(self, force: bool = False) -> FetchResult:
        return FetchResult(self.world.sheet.copy(), "live", str(self.world.dates[-1].date()), [], self._note())


def describe(results: Mapping[str, FetchResult]) -> dict:
    """{component: FetchResult.to_dict()} — the run report's provenance block."""
    return {k: (v.to_dict() if isinstance(v, FetchResult) else None) for k, v in results.items()}
