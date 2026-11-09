"""
Nishkarsh — the small pieces the three engine pages (Aarambh · Nirnay · Convergence) share.

Reading, windowing and drawing helpers only: a frame re-indexed on dates, the page's chart
window cut, the one IC tone scale, the one "Walk-Forward IC" KPI card, and the sign-fill
traces every oscillator panel uses. Nothing here computes a statistic; colours resolve per
render (``chart_rgba`` reads the active appearance), never at import.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import TIMEFRAME_TRADING_DAYS
from ui.components import _d, num, signed, window_cutoff
from ui.theme import chart_rgba

#: Below this many sessions in the window a chart is not drawn (a calm "building … history"
#: empty state instead) — a collapsed scale flat-lines and reads as a confident neutral.
MIN_SESSIONS = 10


def by_date(frame) -> pd.DataFrame:
    """``frame`` re-indexed on normalized dates, de-duplicated, oldest first (empty when absent)."""
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


def series_by_date(s) -> pd.Series:
    """A Series re-indexed on normalized dates (empty float Series when absent)."""
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return pd.Series(dtype="float64")
    out = pd.to_numeric(s, errors="coerce").copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).normalize()
    out = out[~out.index.isna()]
    return out[~out.index.duplicated(keep="last")].sort_index()


def active_window() -> str:
    """The page's chart window (``tf_selected``), always one of the offered values."""
    tf = st.session_state.get("tf_selected")
    return tf if tf in (*TIMEFRAME_TRADING_DAYS, "ALL") else "6M"


def window_slice(obj, last=None):
    """``obj`` (frame or series, date-indexed) cut to the active chart window ending at ``last``
    (default: its own last date). ALL returns it whole."""
    if obj is None or len(obj) == 0:
        return obj
    last = obj.index.max() if last is None else last
    cut = window_cutoff(last, active_window())
    return obj if cut is None else obj[obj.index >= cut]


def ic_tone(ic) -> str:
    """The one IC scale (Tattva rule 59): ≥ 0.05 success · > 0 warning · ≤ 0 danger · — neutral."""
    v = num(ic)
    if v is None:
        return "neutral"
    return "success" if v >= 0.05 else "warning" if v > 0 else "danger"


def ic_card(edge, what: str, h: int, tooltip: str = "") -> dict:
    """The "Walk-Forward IC" KPI card for one graded series (its edge contract)."""
    ic = num(_d(edge).get("ic"))
    return {"label": "Walk-Forward IC", "value": signed(ic, "+.3f") if ic is not None else "—",
            "subtext": f"{what}, non-overlapping {h}d" if ic is not None else "not yet graded",
            "color_class": ic_tone(ic), "tooltip": tooltip}


def add_sign_fill(fig: go.Figure, x, y, *, above: str = "emerald", below: str = "rose", alpha: float = 0.12,
                  row: int | None = None, col: int | None = None) -> None:
    """Two hover-less fill traces: ``above`` tint between the series and zero where it is positive,
    ``below`` where it is negative. Colour carries the sign's meaning; the line is drawn by the caller."""
    s = pd.Series(pd.to_numeric(pd.Series(list(y)), errors="coerce").to_numpy(dtype=float))
    kw = {"row": row, "col": col} if row is not None else {}
    for vals, name in ((s.clip(lower=0), above), (s.clip(upper=0), below)):
        fig.add_trace(go.Scatter(x=list(x), y=vals, mode="lines", fill="tozeroy",
                                 fillcolor=chart_rgba(name, alpha), line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), **kw)
