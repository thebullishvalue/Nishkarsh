"""
Nishkarsh v2 — Data tab: the engine frame + the conclusion columns, unrounded (A4-22).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

No destructive rounding (forward-mode values are ~1e-2); literal (non-regex) search; the
page-level timeframe is the only date filter. Columns are mode-aware (``ui.copy``).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from ui.components import render_section_header
from ui.copy import SIGN_ENGINE, SIGN_Z, mode_copy

_CONCLUSION_COLS = ("C", "C_aarambh", "zA", "zN", "state", "kappa")


def _table(ts: pd.DataFrame, frame: pd.DataFrame | None, forward: bool) -> pd.DataFrame:
    cols = [c for c in mode_copy(forward)["data_cols"] if c in ts.columns]
    out = ts[cols].copy()
    if frame is not None and len(frame) and "Date" in out.columns:
        f = frame[[c for c in _CONCLUSION_COLS if c in frame.columns]]
        f = f[~f.index.duplicated(keep="last")]
        out = out.join(f, on=pd.DatetimeIndex(pd.to_datetime(out["Date"])).normalize())
    for c in ("BullishDiv", "BearishDiv"):
        if c in out.columns:
            out[c] = out[c].map(lambda v: "●" if bool(v) else "○")
    return out


def render_data_tab(ts_f: pd.DataFrame, ts: pd.DataFrame, target: str, forward_mode: bool) -> None:
    """Data table for the selected timeframe + full-history CSV export of the same columns."""
    forward = bool(forward_mode)
    frame = st.session_state.get("conclusion_frame")
    render_section_header(f"Time Series Data ({len(ts_f)} observations)",
                          f"C, zA, zN, ForecastZ: {SIGN_Z} · ConvictionScore, AvgZ: {SIGN_ENGINE} · values unrounded",
                          icon="database", accent="cyan")
    shown = _table(ts_f, frame, forward)
    term = st.text_input("Search", placeholder="Literal text match in any cell…", key="data_search")
    if term:
        mask = shown.astype(str).apply(lambda col: col.str.contains(term, case=False, regex=False, na=False)).any(axis=1)
        shown = shown[mask]
    st.dataframe(shown, width="stretch", height=520,
                 column_config={c: st.column_config.NumberColumn(format="%.6g")
                                for c in shown.columns if pd.api.types.is_float_dtype(shown[c])})
    csv = _table(ts, frame, forward).to_csv(index=False).encode("utf-8")
    st.download_button("Download full CSV", csv,
                       f"nishkarsh_{''.join(ch for ch in str(target) if ch.isalnum() or ch in '_-')}"
                       f"_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
