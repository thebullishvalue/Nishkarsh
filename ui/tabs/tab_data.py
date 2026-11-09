"""
Nishkarsh — Data: is the data sound? Each source's freshness (where the header pill points), the
run's data notes as a plain list, and the merged series — searchable, unrounded, exportable.
"""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st

from ui.components import _d, _str, chip_html, data_notes, esc, pretty_date, section_head
from ui.copy import data_cols

_SOURCES = (("sheet", "Your sheet", "valuation, macro and breadth series"),
            ("constituents", "Constituents", "the Nifty 50 list"),
            ("ohlcv", "Stock prices", "daily bars for every constituent"),
            ("index", "Index", "the Nifty 50 level"),
            ("macro", "Market macro", "rates, currencies and commodities"))
_PROV = {"live": ("Live", "good"), "cache": ("Cached", "muted"), "stale": ("Stale", "caution"),
         "fallback": ("Fallback", "caution"), "empty": ("Empty", "bad"), "synthetic": ("Synthetic", "caution")}
_URL = re.compile(r"(https?://|docs\.google\.com/)\S+", re.I)


def redact(text: str) -> str:
    """Hide any link in a note (a sheet URL is a secret and must never reach the page)."""
    return _URL.sub("[link hidden]", str(text))


_CONCLUSION_COLS = ("NKO", "P_top_down", "P_bottom_up", "P_agreement", "two_system", "C", "zA", "zN", "state", "kappa")


def _sources_html(rc: dict) -> str:
    prov = _d(rc.get("provenance"))
    rows = []
    for key, name, what in _SOURCES:
        p = prov.get(key)
        if not isinstance(p, dict):
            rows.append(f'<tr><td><b>{esc(name)}</b><div class="muted">{esc(what)}</div></td>'
                        f'<td>{chip_html("Not used", "muted")}</td><td>—</td><td class="wrap muted">'
                        'Not part of this run.</td></tr>')
            continue
        note = _str(p.get("note"))
        word = "synthetic" if "SYNTHETIC" in note.upper() else _str(p.get("provenance")).lower()
        label, cls = _PROV.get(word, (word.capitalize() or "—", "muted"))
        miss = [str(m) for m in (p.get("missing") or [])]
        extra = (f" · {len(miss)} missing: " + ", ".join(miss[:8]) + (" …" if len(miss) > 8 else "")
                 if miss and not all(m in note for m in miss[:8]) else "")
        rows.append(f'<tr><td><b>{esc(name)}</b><div class="muted">{esc(what)}</div></td>'
                    f'<td>{chip_html(label, cls)}</td><td class="num">{esc(pretty_date(p.get("as_of")))}</td>'
                    f'<td class="wrap muted">{esc(redact((note or "—") + extra))}</td></tr>')
    return ('<div class="nk-table-wrap"><table class="nk-table"><thead><tr><th>Source</th><th>Status</th>'
            f'<th class="num">As of</th><th>Note</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _table(ts: pd.DataFrame, frame: pd.DataFrame | None, forward: bool) -> pd.DataFrame:
    cols = [c for c in data_cols(forward) if c in ts.columns]
    out = ts[cols].copy()
    if frame is not None and len(frame) and "Date" in out.columns:
        f = frame[[c for c in _CONCLUSION_COLS if c in frame.columns]]
        f = f[~f.index.duplicated(keep="last")]
        f.index = pd.DatetimeIndex(f.index).normalize()
        keys = pd.DatetimeIndex(pd.to_datetime(out["Date"])).normalize()
        out = pd.concat([out.reset_index(drop=True), f.reindex(keys).reset_index(drop=True)], axis=1)
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"]).dt.date
        lead = ["Date"] + [c for c in _CONCLUSION_COLS if c in out.columns]
        out = out[lead + [c for c in out.columns if c not in lead]]
    for c in ("BullishDiv", "BearishDiv"):
        if c in out.columns:
            out[c] = out[c].map(lambda v: "●" if bool(v) else "○")
    return out


def render_data_tab(ts_f: pd.DataFrame, ts: pd.DataFrame, target: str, forward_mode: bool) -> None:
    ss = st.session_state
    rc = _d(ss.get("run_config"))
    forward = bool(forward_mode)
    st.markdown(section_head("Sources and freshness", "Where each input came from on this run, and how current it is.",
                             level="top")
                + _sources_html(rc), unsafe_allow_html=True)

    notes = data_notes(rc)
    items = "".join(f"<li>{esc(redact(n))}</li>" for n in notes) or "<li>No data notes for this run.</li>"
    h0 = _str(rc.get("history_start"))
    st.markdown(section_head("Data notes", "Anything about the inputs worth knowing before reading the results.")
                + f'<ul class="nk-notes">{items}</ul>'
                + f'<p class="nk-method">History used from {esc(pretty_date(h0) if h0 else "—")} · '
                  f'{len(rc.get("features") or [])} model inputs · target {esc(_str(rc.get("target")) or "—")} · '
                  f'accuracy measured against {esc(_str(rc.get("label_src")) or "—")}.</p>', unsafe_allow_html=True)

    frame = ss.get("conclusion_frame")
    st.markdown(section_head("The series", f"{len(ts_f):,} rows in the selected range, unrounded. NKO and the pillars "
                             "are + = bullish; ConvictionScore and AvgZ are − = bullish."), unsafe_allow_html=True)
    shown = _table(ts_f, frame, forward)
    term = st.text_input("Search", placeholder="Search any cell…", key="data_search", label_visibility="collapsed")
    if term:
        mask = shown.astype(str).apply(lambda col: col.str.contains(term, case=False, regex=False, na=False)).any(axis=1)
        shown = shown[mask]
    st.dataframe(shown, width="stretch", height=480, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.6g")
                                for c in shown.columns if pd.api.types.is_float_dtype(shown[c])})
    csv = _table(ts, frame, forward).to_csv(index=False).encode("utf-8")
    st.download_button("Download all rows (CSV)", csv,
                       f"nishkarsh_{''.join(ch for ch in str(target) if ch.isalnum() or ch in '_-')}"
                       f"_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
