"""
Nishkarsh — Precedent: what followed reads like today's (≈ Tattva's Precedent page).

PLACEHOLDER (phase A): renders the page's first section header — the base-rate sentence from
the run's verdict — and an empty state. Phase B fills it per DESIGN_SPEC_v4 §3 "Precedent":
metric cards per horizon (``precedent["summary"]``; neutral unless p < EDGE_ALPHA) → the
interpretation card → "Where They Happened" (``precedent_figure`` below, inside
``render_chart_panel(..., windowed=False)``) → "Analog Skill" (Not yet graded) → "Top Analog
Periods" (``.analog-card`` grid). Unwindowed page.

``precedent_figure`` is the retired page's Nifty-with-matches chart, restyled onto the design
system (``chart_layout``/``style_axes``/``chart_color``; colours resolve per render).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import _d, _str, render_empty_state, render_section_header
from ui.theme import chart_color, chart_layout, chart_rgba, panel_bg, style_axes


def precedent_figure(price: pd.Series, index_name: str, matches: list[tuple], horizon: int,
                     today=None, height: int = 340) -> go.Figure:
    """The index with each past match marked where it happened, and the ``horizon`` sessions after
    it shaded by what the index did (emerald rose / rose fell — the analog's own recorded outcome,
    a fact, not a lean). ``matches`` = [(date, label, fwd_pct or None)]; ``today`` = a date."""
    price = price[~price.index.duplicated(keep="last")].sort_index().dropna()
    idx = pd.DatetimeIndex(price.index)
    fig = go.Figure()
    for d, _, fwd in matches:
        if fwd is None:
            continue
        i = idx.searchsorted(d)
        if i >= len(idx):
            continue
        j = min(i + int(horizon), len(idx) - 1)
        fig.add_vrect(x0=idx[i], x1=idx[j], line_width=0, layer="below",
                      fillcolor=chart_rgba("emerald" if fwd > 0 else "rose", 0.12))
    fig.add_trace(go.Scatter(x=idx, y=price.values, mode="lines", name=index_name, showlegend=False,
                             line=dict(color=chart_color("accent"), width=1.5)))
    pts = [(idx[i], float(price.iloc[i]), lab) for d, lab, _ in matches
           if (i := idx.searchsorted(d)) < len(idx)]
    if pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers+text",
                                 text=[p[2] for p in pts], textposition="top center", name="Match",
                                 textfont=dict(size=9, family="JetBrains Mono, monospace",
                                               color=chart_color("amber")),
                                 marker=dict(size=8, color=chart_color("amber"),
                                             line=dict(color=panel_bg(), width=1)),
                                 showlegend=False))
    if today is not None and len(idx):
        k = min(idx.searchsorted(today), len(idx) - 1)
        fig.add_trace(go.Scatter(x=[idx[k]], y=[float(price.iloc[k])], mode="markers", name="Today",
                                 showlegend=False,
                                 marker=dict(size=10, color=chart_color("accent"), symbol="diamond",
                                             line=dict(color=panel_bg(), width=1))))
    fig.update_layout(**chart_layout(height=height, show_legend=False))
    style_axes(fig, y_title=index_name)
    fig.update_yaxes(tickformat=",.0f")
    return fig


def render_precedent_page() -> None:
    v = _d(_d(st.session_state.get("conclusion")).get("verdict"))
    desc = _str(_d(v.get("precedent")).get("text"),
                "Past dates whose read, pillars and 5- and 20-day change most resembled today's, and what the "
                "Nifty did over the next 3, 5, 10 and 20 sessions — one match per episode.")
    render_section_header("Precedent — What Followed Reads Like Today's", desc, icon="compass", accent="accent")
    render_empty_state("Being rebuilt", "This page is moving onto the new design system. The base rate is on "
                       "Overview's Inference Chain (the Precedent row) meanwhile.", eyebrow="Precedent")
