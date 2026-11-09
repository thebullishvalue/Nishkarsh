"""
Nishkarsh v2 — Precedent tab: past moments like today, in the Nishkarsh Oscillator's own state.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Renders ``session_state["precedent"]`` (computed ONCE per run by ``pipeline.run.compute_precedent``
from the NKO state: the read, its three pillars and its 20-day change — A4-16); this tab recomputes
nothing. A base rate counts as evidence only when its exact binomial p (vs the unconditional rise
rate over the same history) is below EDGE_ALPHA; otherwise it is shown in neutral colours as
description (A4-14). Realized outcomes are coloured by their own sign — they are facts, not calls.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from convergence.nishkarsh import PILLARS
from core.config import CONV2_TIER_WEAK, EDGE_ALPHA
from ui.components import _d, chip_html, esc, fmt, nko_side, num, pretty_date, signed, z_side
from ui.copy import NKO_PILLAR_PLAIN, PRECEDENT
from ui.theme import chart_layout

_AMBER = "#D4A853"


def _fwd(a: dict, h: int):
    f = _d(a.get("fwd"))
    return f.get(int(h), f.get(str(h)))


def _rate_card(h: int, s: dict) -> str:
    n, pos = num(s.get("n")), num(s.get("positive_pct"))
    u, p = num(s.get("unconditional_positive_pct")), num(s.get("binomial_p"))
    k = int(round((pos or 0) / 100 * n)) if n else 0
    sig = p is not None and p < EDGE_ALPHA and u is not None and pos is not None
    chip = chip_html(("Significant · leans to a rise" if pos > u else "Significant · leans to a fall"),
                     "good" if pos > u else "bad") if sig else chip_html("Descriptive", "muted")
    return (f'<div class="nk-rate"><div class="nk-rate-h">After {int(h)} trading days</div>'
            f'<div class="nk-rate-big">{k}/{esc(fmt(n, ".0f"))} <span>rose</span></div>'
            f'<div class="nk-rate-sub">vs {esc(fmt(u, ".0f", suffix="%"))} of all {int(h)}-day periods · '
            f'binomial p {esc(fmt(p, ".2f"))} · median {esc(signed(s.get("median"), "+.2f"))}%</div>{chip}</div>')


def _today_line(k: dict) -> str:
    pil = _d(k.get("pillars"))
    parts = [f'<span class="nk-state-item">Read <b class="{nko_side(k.get("value"))}">'
             f'{esc(signed(k.get("value"), "+.0f"))}</b></span>']
    parts += [f'<span class="nk-state-item">{esc(NKO_PILLAR_PLAIN[p][0])} <b class="{z_side(pil.get(p))}">'
              f'{esc(signed(pil.get(p)))}</b></span>' for p in PILLARS]
    return (f'<div class="nk-state"><span class="nk-state-lab">Today · {esc(pretty_date(k.get("as_of")))}</span>'
            + "".join(parts) + '</div>')


def _chart(frame: pd.DataFrame, price, analogs: list[dict], k: dict) -> go.Figure:
    nko = pd.to_numeric(frame["NKO"], errors="coerce")
    first = nko.first_valid_index()
    if first is not None:
        nko = nko[nko.index >= first]
        price = price[price.index >= first] if price is not None and len(price) else price
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_hrect(y0=-CONV2_TIER_WEAK, y1=CONV2_TIER_WEAK, fillcolor="rgba(148,163,184,0.06)", line_width=0,
                  layer="below")
    if price is not None and len(price):
        fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name="Nifty", showlegend=False,
                                 line=dict(color="rgba(226,232,240,0.35)", width=1),
                                 hovertemplate="Nifty %{y:,.0f}<extra></extra>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=nko.index, y=nko.values, mode="lines", name="Nishkarsh read", showlegend=False,
                             line=dict(color="rgba(148,163,184,0.85)", width=1.3), connectgaps=False,
                             hovertemplate="Read %{y:+.0f}<extra></extra>"), secondary_y=False)
    xs, ys, txt = [], [], []
    for i, a in enumerate(analogs, start=1):
        d = pd.Timestamp(str(a.get("date")))
        v = num(a.get("nko"))
        if v is None and d in nko.index:
            v = num(nko.loc[d])
        if v is None:
            continue
        xs.append(d)
        ys.append(v)
        txt.append(f"#{i} · {pretty_date(a.get('date'))}")
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[t.split(" ")[0] for t in txt],
                                 textposition="top center", textfont=dict(size=10, color=_AMBER),
                                 marker=dict(size=10, color=_AMBER, line=dict(color="#0B1020", width=2)),
                                 customdata=txt, name="Past moments", showlegend=False,
                                 hovertemplate="%{customdata}<br>read then %{y:+.0f}<extra></extra>"),
                      secondary_y=False)
    v0, d0 = num(k.get("value")), k.get("as_of")
    if v0 is not None and d0:
        fig.add_trace(go.Scatter(x=[pd.Timestamp(str(d0))], y=[v0], mode="markers", name="Today", showlegend=False,
                                 marker=dict(size=13, color="#E2E8F0", symbol="diamond",
                                             line=dict(color="#0B1020", width=2)),
                                 hovertemplate="Today %{y:+.0f}<extra></extra>"), secondary_y=False)
    fig.update_layout(**chart_layout(height=330, show_legend=False, margin=dict(t=12, l=8, r=8, b=8)))
    tick = dict(family="Inter, sans-serif", size=11, color="#7C879A")
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=tick)
    fig.update_yaxes(range=[-105, 105], tickvals=[-100, -60, -30, 0, 30, 60, 100],
                     ticktext=["−100", "−60", "−30", "0", "+30", "+60", "+100"], showgrid=True,
                     gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(148,163,184,0.25)",
                     tickfont=tick, secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickformat=",.0f", tickfont=tick, secondary_y=True)
    return fig


def _table(analogs: list[dict], holds: tuple[int, ...]) -> str:
    head = ("<tr><th>#</th><th>Date</th><th>Similarity</th><th>Read then</th>"
            + "".join(f"<th>{esc(NKO_PILLAR_PLAIN[p][0])}</th>" for p in PILLARS)
            + "".join(f'<th class="num">Nifty +{int(h)}d</th>' for h in holds) + "</tr>")
    rows = []
    for i, a in enumerate(analogs, start=1):
        sim = num(a.get("similarity"))
        w = 0.0 if sim is None else max(0.0, min(100.0, sim * 100))
        pil = _d(a.get("pillars"))
        cells = [f'<td class="rank">{i}</td><td>{esc(pretty_date(a.get("date")))}</td>',
                 f'<td><div class="nk-simbar" title="{esc(fmt(sim, ".2f"))}"><div style="width:{w:.0f}%"></div></div>'
                 f'<span class="nk-sim-v">{esc(fmt(sim, ".2f"))}</span></td>',
                 f'<td class="num {nko_side(a.get("nko"))}">{esc(signed(a.get("nko"), "+.0f"))}</td>']
        cells += [f'<td class="num {z_side(pil.get(p))}">{esc(signed(pil.get(p)))}</td>' for p in PILLARS]
        for h in holds:
            v = num(_fwd(a, h))
            cells.append(f'<td class="num {z_side(v)}">{esc(signed(v, "+.1f"))}{"%" if v is not None else ""}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="nk-table-wrap"><table class="nk-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_precedent_tab(horizon: int = 10) -> None:
    """Precedent tab — reads ``precedent``, ``conclusion["nko"]``, ``conclusion_frame`` and ``label_levels``."""
    ss = st.session_state
    prec = _d(ss.get("precedent"))
    k = _d(_d(ss.get("conclusion")).get("nko"))
    analogs = [a for a in (prec.get("analogs") or []) if isinstance(a, dict)]
    summary = {int(h): _d(v) for h, v in _d(prec.get("summary")).items() if num(h) is not None}
    st.markdown(f'<div class="nk-section-head"><h3>Past moments like today</h3><p>{esc(PRECEDENT["desc"])}</p></div>',
                unsafe_allow_html=True)
    if k:
        st.markdown(_today_line(k), unsafe_allow_html=True)
    if not analogs:
        st.info("No past moments like today for this run (not enough two-system history yet, or the precedent "
                "step failed — see Data notes on the Overview).")
        return
    holds = tuple(sorted(summary)) or tuple(sorted({int(h) for a in analogs for h in _d(a.get("fwd"))}))
    if summary:
        cards = "".join(_rate_card(h, summary[h]) for h in holds if h in summary)
        st.markdown(f'<div class="nk-rates">{cards}</div>', unsafe_allow_html=True)
        hp = int(_d(prec.get("hero")).get("horizon") or horizon)
        st.markdown(f'<div class="nk-card-foot">{esc(PRECEDENT["base_rate"])} The Overview uses the {hp}-day row.</div>',
                    unsafe_allow_html=True)

    frame = ss.get("conclusion_frame")
    if frame is not None and len(frame) and "NKO" in frame.columns:
        lv = ss.get("label_levels")
        with st.container(border=True, key="nk_prec_chart"):
            st.markdown('<div class="nk-card-head"><div><h3>Where those moments sit in the read\'s history</h3>'
                        '<p>The Nishkarsh Oscillator over all history; amber dots are the matches (numbered by '
                        'closeness), the white diamond is today. The Nifty is drawn faintly behind (right axis).'
                        '</p></div></div>', unsafe_allow_html=True)
            st.plotly_chart(_chart(frame, lv if isinstance(lv, pd.Series) else None, analogs, k), width="stretch",
                            key="prec_chart", config={"displayModeBar": False})

    st.markdown(f'<div class="nk-section-head sm"><h3>The closest matches</h3><p>{len(analogs)} past dates, '
                f'closest first. Similarity is {esc(PRECEDENT["similarity_note"])}. The read and pillars are as '
                'they stood on that date; the Nifty columns are what happened next.</p></div>',
                unsafe_allow_html=True)
    st.markdown(_table(analogs, holds), unsafe_allow_html=True)
