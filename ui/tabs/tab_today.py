"""
Nishkarsh — Today: what does the read say today, why, what followed similar reads, and how far
to trust it — at a glance, in about two screens.

A. the read (headline, one explanatory sentence, three stat chips, the gauge, one track-record
line) · B. the read over time against the Nifty · C. what drives today's read (pillars and their
inputs on one diverging scale) · D. one line on what followed similar reads. Renders
``conclusion["nko"]``, ``conclusion_frame``, ``label_levels`` and ``precedent["hero"]`` only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from convergence.nishkarsh import COMPONENTS
from core.config import CONV2_TIER_WEAK
from ui.charts import CONFIG, nko_figure, since, single_system_runs
from ui.components import (
    PILLARS, _d, _list, _str, build_hero_verdict, component_label, esc, index_name, num, render_hero, section_head, signed, z_side,
    zbar_html,
)
from ui.copy import NKO_COMPONENT_PLAIN, NKO_PILLAR_PLAIN

SPAN = {"3M": "the last 3 months", "6M": "the last 6 months", "1Y": "the last year", "2Y": "the last 2 years"}


def drivers_rows(k: dict) -> list[dict]:
    """Pillars (bold) and, under each, its inputs — today's z on one scale; top-3 drivers marked."""
    comps = {_str(_d(c).get("id")): _d(c) for c in _list(k.get("components"))}
    top = {_str(_d(c).get("id")) for c in _list(k.get("drivers"))}
    pil = _d(k.get("pillars"))
    rows = []
    for p in PILLARS:
        rows.append({"kind": "pillar", "label": NKO_PILLAR_PLAIN[p][0], "z": num(pil.get(p)), "driver": False,
                     "hint": NKO_PILLAR_PLAIN[p][1]})
        for pil_, cid, lab, _ in COMPONENTS:
            if pil_ == p:
                rows.append({"kind": "input", "label": component_label(cid, lab), "z": num(comps.get(cid, {}).get("z")),
                             "driver": cid in top, "hint": NKO_COMPONENT_PLAIN.get(cid, ("", ""))[1]})
    return rows


def _drivers_html(k: dict) -> str:
    rows = "".join(
        f'<div class="nk-drow {r["kind"]}{" driver" if r["driver"] else ""}" title="{esc(r["hint"])}">'
        f'<span class="lab">{esc(r["label"])}'
        + ('<i class="nk-dmark">top driver</i>' if r["driver"] else "")
        + f'</span><div class="bar">{zbar_html(r["z"], 3.0)}</div>'
        f'<span class="val {z_side(r["z"], 1)}">{esc(signed(r["z"], "+.1f"))}</span></div>'
        for r in drivers_rows(k))
    axis = ('<div class="nk-drow axis"><span class="lab"></span><div class="bar nk-daxis"><span>−3 bearish</span>'
            '<span>0</span><span>+3 bullish</span></div><span class="val"></span></div>')
    return f'<div class="nk-dchart">{axis}{rows}</div>'


def render_today_tab(start, tf_label: str) -> None:
    ss = st.session_state
    con, rc = _d(ss.get("conclusion")), _d(ss.get("run_config"))
    k = _d(con.get("nko"))
    v = build_hero_verdict(con, _d(ss.get("precedent")).get("hero"), rc)
    render_hero(v)

    # B. the read over time
    frame = ss.get("conclusion_frame")
    if frame is not None and len(frame) and "NKO" in frame.columns:
        fr = since(frame, start)
        lv = ss.get("label_levels")
        price = since(lv, start) if isinstance(lv, pd.Series) else None
        name = index_name(con)
        span = SPAN.get(str(tf_label), "all history")
        single = bool(single_system_runs(fr))
        st.markdown(section_head("The read over time", f"Each day's read over {span}, with the {name} on the right "
                                 "axis.") +
                    '<div class="nk-legend"><span><i class="sw bull"></i>Bullish</span><span><i class="sw bear"></i>'
                    f'Bearish</span><span><i class="sw flat"></i>No lean (±{CONV2_TIER_WEAK:g})</span><span><i class="sw line"></i>'
                    f'{esc(name)}</span>' + ('<span><i class="sw single"></i>Top-down only</span>' if single else "")
                    + '</div>', unsafe_allow_html=True)
        if len(fr):
            st.plotly_chart(nko_figure(fr, price, name, height=300), width="stretch", key="today_history",
                            config=CONFIG)

    # C. what drives today's read
    if k:
        st.markdown(section_head("What drives today's read",
                                 "The read is the average of three pillars, each built from the inputs under it. "
                                 "Every bar is today's value against its own past (+ = bullish).")
                    + _drivers_html(k), unsafe_allow_html=True)

    # D. after readings like today
    pr = _d(v.get("precedent"))
    st.markdown(section_head("After readings like today") +
                f'<div class="nk-oneline"><p>{esc(pr.get("text", "—"))}</p><span>More on the Precedent page</span></div>',
                unsafe_allow_html=True)
