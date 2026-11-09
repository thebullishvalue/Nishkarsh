"""
Nishkarsh — Nirnay: what the fifty stocks are doing (≈ Tattva's Swayam page; hue: cyan; unit
word = stocks).

PLACEHOLDER (phase A): renders the page's first section header — with the answer from the run's
verdict — and an empty state. Phase B fills it per DESIGN_SPEC_v4 §3 "Nirnay": KPI strip ×6 →
"Breadth Over Time" (window=True) → "Walk-Forward Edge" (``pillar_edge["bottom_up"]``) → "Which
Stocks Carry the Breadth" → "Regime" → "Average Oscillator" → "Individual Stocks" (filterbar
selectbox + table panel). No Nirnay data → ``render_empty_state("No Nirnay breadth for this run",
eyebrow="Nirnay")``. Windowed page.
"""

from __future__ import annotations

import streamlit as st

from ui.components import _d, _str, num, render_empty_state, render_section_header, signed
from ui.copy import NIRNAY_READ_SENTENCE


def render_nirnay_page() -> None:
    v = _d(_d(st.session_state.get("conclusion")).get("verdict"))
    b = _d(_d(v.get("kpi")).get("breadth"))
    net, bull, bear = num(b.get("net_pts")), num(b.get("bull_pct")), num(b.get("bear_pct"))
    n = num(b.get("n_live"))
    desc = (f"Net breadth {signed(net, '+.0f')} pts: {bull:.0f}% of {int(n or 0)} stocks read "
            f"{_str(b.get('bull_label'))} and {bear:.0f}% {_str(b.get('bear_label'))}. {NIRNAY_READ_SENTENCE}"
            if net is not None and bull is not None and bear is not None else
            "Nirnay reads the Nifty's fifty stocks from the bottom up; there is no breadth for this run.")
    render_section_header("Nirnay — What the Fifty Stocks Are Doing", desc, icon="activity", accent="cyan")
    render_empty_state("Being rebuilt", "This page is moving onto the new design system. Nirnay's net breadth "
                       "and the bottom-up pillar are on Overview meanwhile.", eyebrow="Nirnay")
