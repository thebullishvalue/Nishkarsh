"""
Nishkarsh — Precedent: what happened after reads like today's.

Renders ``session_state["precedent"]`` (computed ONCE per run by ``pipeline.run.compute_precedent``
from the Nishkarsh Oscillator state); this page recomputes nothing. A base rate is coloured only
when its exact binomial p (vs the usual rise rate over the same history) is below EDGE_ALPHA;
otherwise it is description in neutral colours. Realized outcomes are coloured by their own sign —
they are facts, not calls.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import EDGE_ALPHA
from ui.charts import CONFIG, nko_figure, precedent_figure
from ui.components import (
    PILLARS, _d, _list, _str, esc, fmt, index_name, nko_side, num, precedent_line, pretty_date, section_head, signed,
    z_side,
)
from ui.copy import NKO_PILLAR_PLAIN, PRECEDENT_METHOD


def _fwd(a: dict, h: int):
    f = _d(a.get("fwd"))
    return f.get(int(h), f.get(str(h)))


def _base_rate_table(summary: dict[int, dict]) -> str:
    rows = []
    for h in sorted(summary):
        s = summary[h]
        n, pos, u, p = num(s.get("n")), num(s.get("positive_pct")), num(s.get("unconditional_positive_pct")), \
            num(s.get("binomial_p"))
        k = int(round((pos or 0) / 100 * n)) if n else None
        sig = p is not None and p < EDGE_ALPHA and u is not None and pos is not None
        cls = ("bull" if pos > u else "bear") if sig else ""
        rows.append(f'<tr><td>{int(h)} sessions</td>'
                    f'<td class="num {cls}">{esc(f"{k} of {int(n)}" if k is not None else "—")}</td>'
                    f'<td class="num {cls}">{esc(fmt(pos, ".0f", suffix="%"))}</td>'
                    f'<td class="num">{esc(fmt(u, ".0f", suffix="%"))}</td>'
                    f'<td class="num">{esc(signed(s.get("median"), "+.1f"))}{"%" if num(s.get("median")) is not None else ""}</td>'
                    f'<td class="num">{esc(fmt(p, ".2f"))}</td>'
                    f'<td class="muted">{"significant" if sig else "descriptive"}</td></tr>')
    return ('<div class="nk-table-wrap"><table class="nk-table"><thead><tr><th>Horizon</th><th class="num">Rose</th>'
            '<th class="num">Share up</th><th class="num">Usually</th><th class="num">Median move</th>'
            '<th class="num">p</th><th></th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _matches_table(analogs: list[dict], holds: tuple[int, ...]) -> str:
    head = ("<tr><th>#</th><th>Date</th><th class=\"num\">Read then</th>"
            + "".join(f'<th class="num">{esc(NKO_PILLAR_PLAIN[p][0])}</th>' for p in PILLARS)
            + "".join(f'<th class="num">Nifty +{int(h)}d</th>' for h in holds) + "</tr>")
    rows = []
    for i, a in enumerate(analogs, start=1):
        pil = _d(a.get("pillars"))
        cells = [f'<td class="rank">{i}</td><td>{esc(pretty_date(a.get("date")))}</td>',
                 f'<td class="num {nko_side(a.get("nko"))}">{esc(signed(a.get("nko"), "+.0f"))}</td>']
        cells += [f'<td class="num {z_side(pil.get(p), 1)}">{esc(signed(pil.get(p), "+.1f"))}</td>' for p in PILLARS]
        for h in holds:
            v = num(_fwd(a, h))
            cells.append(f'<td class="num {z_side(v, 1)}">{esc(signed(v, "+.1f"))}{"%" if v is not None else ""}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="nk-table-wrap"><table class="nk-table"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def render_precedent_tab(horizon: int = 10) -> None:
    ss = st.session_state
    prec = _d(ss.get("precedent"))
    con = _d(ss.get("conclusion"))
    k = _d(con.get("nko"))
    analogs = [a for a in _list(prec.get("analogs")) if isinstance(a, dict)]
    summary = {int(h): _d(v) for h, v in _d(prec.get("summary")).items() if num(h) is not None}
    line = precedent_line(prec.get("hero"), {"BULLISH": 1, "BEARISH": -1}.get(_str(k.get("direction")), 0))
    st.markdown(section_head("After readings like today", level="top") + f'<p class="nk-lead">{esc(line["text"])}</p>',
                unsafe_allow_html=True)
    if not analogs:
        st.markdown('<div class="nk-empty">There is not enough two-system history to find past moments like today '
                    'in this run.</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="nk-method">{esc(PRECEDENT_METHOD)}</p>', unsafe_allow_html=True)
        return
    holds = tuple(sorted(summary)) or tuple(sorted({int(h) for a in analogs for h in _d(a.get("fwd"))}))
    if summary:
        st.markdown(section_head("Base rates by horizon", "How often the Nifty rose after these moments, against how "
                                 "often it rose over any period of the same length.", level="sm")
                    + _base_rate_table(summary), unsafe_allow_html=True)

    st.markdown(section_head("The matches", f"{len(analogs)} past dates, closest first: the read and its pillars as "
                             "they stood then, and what the Nifty did next.", level="sm")
                + _matches_table(analogs, holds), unsafe_allow_html=True)

    frame = ss.get("conclusion_frame")
    if frame is not None and len(frame) and "NKO" in frame.columns:
        nko = pd.to_numeric(frame["NKO"], errors="coerce")
        first = nko.first_valid_index()
        fr = frame[frame.index >= first] if first is not None else frame
        lv = ss.get("label_levels")
        price = lv[lv.index >= first] if isinstance(lv, pd.Series) and first is not None else None
        if price is not None and len(price.dropna()):
            matches = [(pd.Timestamp(str(a.get("date"))), str(i), num(_fwd(a, horizon)))
                       for i, a in enumerate(analogs, start=1)]
            st.markdown(section_head("Where they happened", f"The {index_name(con)} with each match numbered by "
                                     f"closeness; the shading is the next {int(horizon)} sessions — green where it "
                                     "rose, red where it fell. The white diamond is today.", level="sm"),
                        unsafe_allow_html=True)
            st.plotly_chart(precedent_figure(price, index_name(con), matches, horizon, today=k.get("as_of")
                                             and pd.Timestamp(str(k.get("as_of")))),
                            width="stretch", key="prec_chart", config=CONFIG)
            st.markdown(f'<p class="nk-method"><b>Method.</b> {esc(PRECEDENT_METHOD)}</p>', unsafe_allow_html=True)
            return
        marks = []
        for i, a in enumerate(analogs, start=1):
            d, v = pd.Timestamp(str(a.get("date"))), num(a.get("nko"))
            if v is None and d in nko.index:
                v = num(nko.loc[d])
            if v is not None:
                marks.append((d, v, str(i)))
        v0, d0 = num(k.get("value")), k.get("as_of")
        today = (pd.Timestamp(str(d0)), v0) if v0 is not None and d0 else None
        st.markdown(section_head("Where they sit in the read's history", "Amber dots are the matches, numbered by "
                                 "closeness; the white diamond is today.", level="sm"), unsafe_allow_html=True)
        st.plotly_chart(nko_figure(fr, price, index_name(con), height=300, marks=marks, today=today,
                                   line=True), width="stretch",
                        key="prec_chart", config=CONFIG)
    st.markdown(f'<p class="nk-method"><b>Method.</b> {esc(PRECEDENT_METHOD)}</p>', unsafe_allow_html=True)
