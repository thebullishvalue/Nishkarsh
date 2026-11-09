"""
Display views — small, pure re-shapings of frames the run already published, for the engine
pages (Aarambh · Nirnay · Convergence) to draw.

Nothing here estimates, fits, scores or grades anything. Each function only re-orients a
published column by the one configured Nirnay polarity (``core.config.NIRNAY_SIGN``), reads a
stock's published zone on a date, or takes the cross-section median of the per-stock adaptive
cuts the Nirnay engine already published (so the average oscillator is drawn against the cuts
a typical stock was judged by on that date — causal per date, never today's cut painted back
over history). The UI formats, sorts and windows what these return; it computes nothing.

Polarity: under "trend" (the shipped setting) a stock reading OVERBOUGHT is bullish evidence and
OVERSOLD bearish; under "contrarian" the reverse. ``side`` is always bullish-positive:
+1 = the bullish side under the configured polarity, −1 = the bearish side, 0 = neither.

Pure: no Streamlit, no IO; every function returns an empty frame for missing input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import NIRNAY_POLARITY, NIRNAY_SIGN
from convergence.verdict import _Z_LEAN

__all__ = ["PILLAR_NO_LEAN", "ZONE_WORDS", "breadth_view", "stock_snapshot", "cut_bands", "avg_oscillator_view",
           "lean_phrase", "zone_side"]

#: |pillar z| inside which a pillar reads "no lean" — the same display band the read verdict's
#: gate rows use (``convergence.verdict``), so a chart's shaded band and the Overview's words agree.
PILLAR_NO_LEAN = float(_Z_LEAN)

_TREND = NIRNAY_POLARITY == "trend"
#: (bullish-side zone, bearish-side zone) under the configured polarity.
ZONE_WORDS = ("Overbought", "Oversold") if _TREND else ("Oversold", "Overbought")


def lean_phrase(z) -> str:
    """A pillar's lean in words — the read verdict's gate wording without the number:
    "leans bullish" / "leans bearish" outside ±PILLAR_NO_LEAN, "no lean" inside, "" missing."""
    try:
        v = float(z)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return "leans bullish" if v >= PILLAR_NO_LEAN else "leans bearish" if v <= -PILLAR_NO_LEAN else "no lean"


def zone_side(condition) -> int:
    """+1 for the bullish-side zone, −1 for the bearish-side zone, 0 otherwise."""
    c = str(condition).strip().capitalize() if condition is not None else ""
    return 1 if c == ZONE_WORDS[0] else -1 if c == ZONE_WORDS[1] else 0


def _dated(frame) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    f = frame.copy()
    if isinstance(f.columns, pd.MultiIndex):
        f.columns = [c[0] for c in f.columns]
    idx = pd.DatetimeIndex(pd.to_datetime(f.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    f.index = idx.normalize()
    f = f[~f.index.isna()]
    return f[~f.index.duplicated(keep="last")].sort_index()


def _col(f: pd.DataFrame, name: str) -> pd.Series:
    if name not in f.columns:
        return pd.Series(np.nan, index=f.index, dtype="float64")
    return pd.to_numeric(f[name], errors="coerce").astype("float64")


def breadth_view(nirnay_daily) -> pd.DataFrame:
    """Nirnay's breadth, oriented: ``bull_pct`` / ``bear_pct`` (% of live stocks in the bullish- /
    bearish-side zone), ``net`` = bull_pct − bear_pct (points, bullish-positive — the Overview's
    "Nirnay Net Breadth"), ``n_live``. Indexed by date, oldest first."""
    f = _dated(nirnay_daily)
    if f.empty:
        return pd.DataFrame(columns=["bull_pct", "bear_pct", "net", "n_live"], dtype="float64")
    ob, os_ = _col(f, "Overbought_Pct"), _col(f, "Oversold_Pct")
    bull, bear = (ob, os_) if _TREND else (os_, ob)
    out = pd.DataFrame({"bull_pct": bull, "bear_pct": bear, "net": NIRNAY_SIGN * (ob - os_),
                        "n_live": _col(f, "n_live")}, index=f.index)
    return out


def _live(df: pd.DataFrame) -> pd.DataFrame:
    """A stock's LIVE rows — the engine's own definition (genuine bar, past warm-up, finite
    oscillator), so a stock counted here is a stock the daily breadth counted."""
    from engines.nirnay import _live_rows
    try:
        return _live_rows(df)
    except Exception:  # noqa: BLE001 — a malformed frame is simply not live
        return pd.DataFrame()


def stock_snapshot(constituent_dfs, as_of=None) -> pd.DataFrame:
    """Every stock live on ``as_of`` (default: the latest date any stock is live): its oscillator
    (``osc`` = Unified_Osc, native ±10, + = strong), published ``zone`` (Condition), the adaptive
    cuts it was judged against (``os_cut``/``ob_cut``) and ``side`` (bullish-positive, by the
    configured polarity). Indexed by the symbol without its exchange suffix; unsorted."""
    cols = ["symbol", "osc", "zone", "os_cut", "ob_cut", "side"]
    if not isinstance(constituent_dfs, dict) or not constituent_dfs:
        return pd.DataFrame(columns=cols).set_index("symbol")
    live = {str(k): _live(v) for k, v in constituent_dfs.items() if isinstance(v, pd.DataFrame) and len(v)}
    live = {k: v for k, v in live.items() if len(v)}
    if not live:
        return pd.DataFrame(columns=cols).set_index("symbol")
    day = pd.Timestamp(as_of).normalize() if as_of is not None else max(v.index.max() for v in live.values())
    rows = []
    for sym, f in live.items():
        if day not in f.index:
            continue
        r = f.loc[day]
        zone = str(r.get("Condition", "Neutral") or "Neutral")
        rows.append({"symbol": sym.split(".")[0], "osc": float(pd.to_numeric(r.get("Unified_Osc"), errors="coerce")),
                     "zone": zone, "os_cut": pd.to_numeric(r.get("OS_Cut"), errors="coerce"),
                     "ob_cut": pd.to_numeric(r.get("OB_Cut"), errors="coerce"), "side": zone_side(zone)})
    if not rows:
        return pd.DataFrame(columns=cols).set_index("symbol")
    return pd.DataFrame(rows, columns=cols).set_index("symbol")


def cut_bands(constituent_dfs, index=None) -> pd.DataFrame:
    """Per date, the median across LIVE stocks of the adaptive oversold / overbought cuts each
    stock was judged against that day (``os_cut``, ``ob_cut``). Causal by construction: each
    stock's cut on a date uses only its own past. Reindexed on ``index`` when given."""
    out_cols = ["os_cut", "ob_cut"]
    if not isinstance(constituent_dfs, dict) or not constituent_dfs:
        return pd.DataFrame(columns=out_cols, index=index, dtype="float64")
    os_parts, ob_parts = [], []
    for v in constituent_dfs.values():
        if not isinstance(v, pd.DataFrame) or not len(v):
            continue
        f = _live(v)
        if f.empty or not {"OS_Cut", "OB_Cut"} <= set(f.columns):
            continue
        os_parts.append(pd.to_numeric(f["OS_Cut"], errors="coerce").rename(len(os_parts)))
        ob_parts.append(pd.to_numeric(f["OB_Cut"], errors="coerce").rename(len(ob_parts)))
    if not os_parts:
        return pd.DataFrame(columns=out_cols, index=index, dtype="float64")
    out = pd.DataFrame({"os_cut": pd.concat(os_parts, axis=1).median(axis=1, skipna=True),
                        "ob_cut": pd.concat(ob_parts, axis=1).median(axis=1, skipna=True)}).sort_index()
    return out.reindex(index) if index is not None else out


def avg_oscillator_view(nirnay_daily, constituent_dfs) -> pd.DataFrame:
    """The stocks' average oscillator (``avg`` = Avg_Signal, native ±10) beside that date's median
    adaptive cuts, with ``zone`` (+1 above the overbought cut, −1 below the oversold cut, 0
    between — native orientation) and ``side`` (bullish-positive, by the configured polarity)."""
    f = _dated(nirnay_daily)
    if f.empty:
        return pd.DataFrame(columns=["avg", "os_cut", "ob_cut", "zone", "side"], dtype="float64")
    avg = _col(f, "Avg_Signal")
    cuts = cut_bands(constituent_dfs, f.index)
    zone = pd.Series(np.where(avg > cuts["ob_cut"], 1.0, np.where(avg < cuts["os_cut"], -1.0, 0.0)), index=f.index)
    zone[avg.isna() | cuts["os_cut"].isna() | cuts["ob_cut"].isna()] = np.nan
    return pd.DataFrame({"avg": avg, "os_cut": cuts["os_cut"], "ob_cut": cuts["ob_cut"], "zone": zone,
                         "side": NIRNAY_SIGN * zone}, index=f.index)
