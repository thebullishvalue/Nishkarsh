"""
Nishkarsh — Aarambh: the top-down forecast (≈ Tattva's Mūla page; hue: violet).

PLACEHOLDER (phase A): renders the page's first section header — with the answer from the run's
verdict — and an empty state. Phase B fills it per DESIGN_SPEC_v4 §3 "Aarambh": KPI strip ×5 →
"Forecast & The Nifty" (``render_chart_panel(..., window=True)``, the page's one window control)
→ "Its Inputs" → "Model Quality" → "Walk-Forward Edge" (``render_edge_panel(
conclusion["nko"]["pillar_edge"]["top_down"], "aarambh", what="the top-down pillar")``) + the
stage-by-stage and cost-aware tables. Windowed page (the shell passes ``windowed=True``).
"""

from __future__ import annotations

import streamlit as st

from ui.components import _d, num, render_empty_state, render_section_header, signed


def render_aarambh_page() -> None:
    v = _d(_d(st.session_state.get("conclusion")).get("verdict"))
    td = num(_d(v.get("kpi")).get("top_down_z"))
    lean = ("leans bullish" if td is not None and td >= 0.25 else "leans bearish" if td is not None and td <= -0.25
            else "has no lean")
    desc = (f"Aarambh forecasts the Nifty's next {int(num(v.get('horizon')) or 10)} sessions from the top down; "
            f"standardized against its own past, with its extension and lookback breadth, the top-down pillar "
            f"reads {signed(td, '+.2f')}σ: it {lean}." if td is not None else
            "Aarambh forecasts the Nifty from the top down; there is no top-down reading for this run.")
    render_section_header("Aarambh — The Top-Down Forecast", desc, icon="trending", accent="violet")
    render_empty_state("Being rebuilt", "This page is moving onto the new design system. The top-down pillar "
                       "and its inputs are on Overview's Inference Chain meanwhile.", eyebrow="Aarambh")
