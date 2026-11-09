"""
Nishkarsh — Data: the record behind every page, one row per session, exportable.

Tattva's "The Record" grammar: a section header whose description states the sign
conventions → a filter strip (Search) → the joined record in the one table primitive, windowed
by the chart window in its own header (the newest first, capped on screen) → the export (every
row, full precision) → one note. Every disclosure about the data lives in the notice rail under
the command bar, not here. The record is a JOIN of frames the run already published, by date —
nothing is computed.

Record grammar (the contract the other pages' tables follow): index = Date (date-only),
newest first; one column per published series in the units the pages print it in; signed
bullish-positive columns tinted by sign; missing = "—"; on-screen rows capped, export complete.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.config import TIMEFRAME_TRADING_DAYS
from ui.components import (
    _d, index_name, render_note, render_section_header, render_table_panel, window_cutoff,
)
from ui.copy import data_cols

#: The conclusion frame's published series → their record names.
_CONCLUSION = (("NKO", "NKO"), ("P_top_down", "Top-down"), ("P_bottom_up", "Bottom-up"),
               ("P_agreement", "Agreement"), ("two_system", "Two systems"), ("C", "C"), ("zA", "zA"),
               ("zN", "zN"), ("state", "State"), ("kappa", "Kappa"), ("agreement_ratio", "Agreement ratio"),
               ("SAO", "Signed agreement"))
#: Nirnay's daily aggregate → record names (in % of live stocks, as the Nirnay page prints them).
_NIRNAY = (("Overbought_Pct", "Overbought %"), ("Oversold_Pct", "Oversold %"),
           ("Avg_Signal", "Avg oscillator"), ("n_live", "Live stocks"))
#: Bullish-positive signed columns, tinted by sign on screen.
SIGN_COLS = {"NKO", "Top-down", "Bottom-up", "Agreement", "C", "zA", "zN"}
#: Header overrides the table's prettifier would otherwise re-case.
COL_LABELS = {"zA": "zA", "zN": "zN"}
CAP = 300


def _by_date(frame) -> pd.DataFrame:
    """A frame re-indexed on normalized dates, de-duplicated (empty when absent)."""
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    f = frame[~frame.index.duplicated(keep="last")].copy()
    f.index = pd.to_datetime(f.index, errors="coerce").normalize()
    return f[~f.index.isna()]


def build_record(ss) -> pd.DataFrame:
    """The joined, full-precision record: the Nifty, the read and its pillars, the conclusion,
    Nirnay's breadth and Aarambh's published columns — one row per session, oldest first."""
    ts = ss.get("aarambh_ts")
    if ts is None or getattr(ts, "empty", True):
        return pd.DataFrame()
    a = ts.copy()
    if "Date" in a.columns:
        a.index = pd.to_datetime(a["Date"], errors="coerce").dt.normalize()
    a = a[~a.index.isna()]
    a = a[~a.index.duplicated(keep="last")]
    out = pd.DataFrame(index=a.index)
    con = _d(ss.get("conclusion"))
    lv = ss.get("label_levels")
    if isinstance(lv, pd.Series) and len(lv):
        s = lv.copy()
        s.index = pd.to_datetime(s.index, errors="coerce").normalize()
        out[index_name(con)] = s[~s.index.duplicated(keep="last")].reindex(out.index)
    fr = _by_date(ss.get("conclusion_frame"))
    for src, name in _CONCLUSION:
        if src in fr.columns:
            col = fr[src].reindex(out.index)
            if src == "two_system":
                col = col.map(lambda x: "yes" if x is True or x == 1 else "no" if x is False or x == 0 else None)
            out[name] = col
    nd = _by_date(ss.get("nirnay_daily"))
    for src, name in _NIRNAY:
        if src in nd.columns:
            out[name] = pd.to_numeric(nd[src], errors="coerce").reindex(out.index)
    forward = bool(_d(ss.get("run_config")).get("forward_mode", True))
    for c in data_cols(forward):
        if c != "Date" and c in a.columns:
            out[c] = a[c]
    for c in ("BullishDiv", "BearishDiv", "Valid"):
        if c in out.columns:
            out[c] = out[c].map(lambda x: "yes" if bool(x) else "no")
    out.index.name = "Date"
    return out


def render_data_page() -> None:
    ss = st.session_state
    record = build_record(ss)
    name = index_name(_d(ss.get("conclusion")))
    render_section_header(
        "The Record",
        f"Every published series behind the read, one row per session, in the units the pages show "
        f"them in: the {name}, the Nishkarsh Oscillator and its pillars, the conclusion, Nirnay's "
        "breadth and Aarambh's columns. NKO, its three pillars, C, zA and zN are + = bullish; "
        "Aarambh's ConvictionScore and AvgZ are − = bullish (stretched up reads positive).",
        icon="database", accent="cyan",
    )
    if record.empty:
        render_note("No published record for this run.")
        return

    tf = ss.get("tf_selected")
    if tf not in (*TIMEFRAME_TRADING_DAYS, "ALL"):
        tf = "6M"
    cut = window_cutoff(record.index.max(), tf)
    shown = record if cut is None else record[record.index >= cut]

    with st.container(key="filterbar"):
        term = st.text_input("Search", placeholder="Filter by any value…", key="data_search")
    view = shown.iloc[::-1].copy()                      # newest first
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].round(2)
    if term:
        mask = view.astype(str).apply(lambda c: c.str.contains(term, case=False, regex=False, na=False)).any(axis=1)
        view = view[mask]
    n = len(view)
    # HTML tables render every row into the DOM, so the screen is capped — on the NEWEST rows
    # (the view is newest-first, so that is its head; Tattva 2.8.0's tail kept the oldest).
    render_table_panel(
        view.head(CAP), "data-record", index_label="Date",
        units=f"{n:,} sessions × {len(view.columns)} series",
        max_height=560, window=True,
        footer=(f"Showing the newest {CAP:,} of {n:,} — the export has all of them." if n > CAP else ""),
        sign_color_cols=SIGN_COLS, col_labels=COL_LABELS,
    )
    target = _d(ss.get("run_config")).get("target") or "nifty50"
    safe = "".join(ch for ch in str(target) if ch.isalnum() or ch in "_-")
    st.download_button(
        "Download the record (CSV)", record.reset_index().to_csv(index=False).encode("utf-8"),
        f"nishkarsh_{safe}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
    )
    render_note(f"Full history · {len(record):,} sessions · Aarambh, Nirnay and Convergence joined on date "
                "at full precision.")
