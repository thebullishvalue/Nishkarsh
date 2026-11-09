"""
Nishkarsh v2 — Nirnay tab: constituent breadth and regime read-outs (C-NIR daily frame).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Every percentage is of the names with LIVE (warm, real-bar) data that day (``n_live``); zone
membership uses each stock's own adaptive cuts (OS_Cut / OB_Cut). Descriptive only: Nirnay's
role in the conclusion is the measured, lift-gated one shown in the Convergence tab.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import (
    COLOR_AMBER, COLOR_CYAN, COLOR_GREEN, COLOR_MUTED, COLOR_PURPLE, COLOR_RED,
    UI_CHART_HEIGHT_LARGE, UI_CHART_HEIGHT_MEDIUM,
)
from ui.components import (
    esc, fmt, num, render_interpretation_card, render_metric_card, render_pillar_banner, render_section_header,
    section_gap,
)
from ui.copy import NIRNAY
from ui.theme import chart_layout, style_axes


def _s(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column or an all-NaN series — never a fabricated default (A4-21)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def _last(df: pd.DataFrame, col: str):
    return num(_s(df, col).iloc[-1]) if len(df) else None


def _lines(df, specs, key, y_title, y_range=None, height=UI_CHART_HEIGHT_MEDIUM, fill=False):
    fig = go.Figure()
    for col, name, color in specs:
        if col in df.columns:
            kw = dict(fill="tozeroy") if fill else {}
            fig.add_trace(go.Scatter(x=df.index, y=_s(df, col), mode="lines", name=name,
                                     line=dict(color=color, width=1.5), connectgaps=False, **kw))
    fig.update_layout(**chart_layout(height=height))
    style_axes(fig, y_title=y_title, y_range=y_range)
    st.plotly_chart(fig, width="stretch", key=key)


def render_nirnay_tab(start_date=None) -> None:
    """Nirnay tab — reads ``nirnay_daily``, ``nirnay_constituent_dfs`` and the run report's cuts."""
    ss = st.session_state
    daily = ss.get("nirnay_daily")
    dfs = ss.get("nirnay_constituent_dfs") or {}
    rep_n = (ss.get("run_report") or {}).get("nirnay") or {}
    st.markdown('<div class="tab-bg nirnay"></div>', unsafe_allow_html=True)
    render_pillar_banner(ss.get("conclusion"), "bottom_up")
    if daily is None or len(daily) == 0:
        st.info("No Nirnay constituent data for this run — the Nishkarsh read runs in single-system mode "
                "(top-down pillar only).")
        return
    full = daily[~daily.index.duplicated(keep="last")].sort_index()
    df = full if start_date is None else full[pd.DatetimeIndex(full.index) >= pd.Timestamp(start_date)]
    if df.empty:
        st.info("No Nirnay data in the selected time range.")
        return

    n_live, n_uni = _last(df, "n_live"), _last(df, "n_universe")
    live_txt = f"of {fmt(n_live, '.0f')} stocks with live data"
    c = st.columns(6, gap="small")
    with c[0]:
        render_metric_card("Oversold", fmt(_last(df, "Oversold_Pct"), ".0f", suffix="%"),
                           f"{live_txt}, past their own oversold cut", tooltip=NIRNAY["zone_desc"])
    with c[1]:
        render_metric_card("Overbought", fmt(_last(df, "Overbought_Pct"), ".0f", suffix="%"),
                           f"{live_txt}, past their own overbought cut", tooltip=NIRNAY["zone_desc"])
    with c[2]:
        render_metric_card("Average oscillator", fmt(_last(df, "Avg_Signal"), "+.2f"),
                           "±10 scale · below 0 = oversold", tooltip=NIRNAY["avg_tip"])
    with c[3]:
        render_metric_card("Coverage", f"{fmt(n_live, '.0f')} / {fmt(n_uni, '.0f')}",
                           f"stocks with live data ({fmt(_last(df, 'coverage'), '.0%')})", tooltip=NIRNAY["coverage_desc"])
    with c[4]:
        render_metric_card("Cut crossings today", f"{fmt(_last(df, 'Buy_Signals'), '.0f')} · {fmt(_last(df, 'Sell_Signals'), '.0f')}",
                           "into oversold · into overbought (descriptive)", tooltip=NIRNAY["signals_desc"])
    with c[5]:
        render_metric_card("History", f"{len(full):,}", "trading days with live data",
                           tooltip="Days with at least one live name; days with none are absent (NO_DATA).")
    render_interpretation_card(
        "Adaptive cuts",
        esc(f"Zone membership uses each stock's own adaptive cut: median live OS_Cut "
            f"{fmt(rep_n.get('os_cut_median_live'), '+.2f')} / OB_Cut {fmt(rep_n.get('ob_cut_median_live'), '+.2f')} "
            f"on the latest bar. Dispersion (cross-sectional std of the oscillator) today: "
            f"{fmt(_last(df, 'Dispersion'), '.2f')}."), "neutral")

    section_gap()
    render_section_header("HMM State Probabilities (cross-sectional mean)", NIRNAY["hmm_desc"], icon="eye", accent="violet")
    _lines(df, (("avg_hmm_bull", "mean P(bull)", COLOR_GREEN), ("avg_hmm_bear", "mean P(bear)", COLOR_RED)),
           "nirnay_hmm_regime", "probability", y_range=[0, 1], height=300)

    section_gap()
    render_section_header("Zone Breadth Over Time", NIRNAY["zone_desc"], icon="layers", accent="emerald")
    _lines(df, (("Oversold_Pct", "oversold % of live", COLOR_GREEN), ("Overbought_Pct", "overbought % of live", COLOR_RED)),
           "nirnay_os_ob", "% of live names", y_range=[0, 100], height=UI_CHART_HEIGHT_LARGE, fill=True)

    render_section_header("Regime Composition", NIRNAY["regime_desc"], icon="bar-chart", accent="cyan")
    _lines(df, (("Regime_Bull_Pct", "bull %", COLOR_GREEN), ("Regime_Bear_Pct", "bear %", COLOR_RED),
                ("Regime_Neutral_Pct", "neutral %", COLOR_MUTED), ("Regime_Transition_Pct", "transition %", COLOR_AMBER)),
           "nirnay_regime_mix", "% of live names", y_range=[0, 100])

    render_section_header("Coverage", NIRNAY["coverage_desc"], icon="database")
    _lines(df, (("n_live", "live names", COLOR_CYAN), ("n_universe", "universe", COLOR_MUTED)),
           "nirnay_coverage", "names", height=240)

    section_gap()
    render_section_header("Cut Crossings Over Time", NIRNAY["signals_desc"], icon="zap", accent="rose")
    _lines(df, (("Buy_Signals", "oversold-side crossings", COLOR_GREEN), ("Sell_Signals", "overbought-side crossings", COLOR_RED),
                ("Change_Points", "CUSUM change points", COLOR_PURPLE)), "nirnay_signal_counts", "count")

    render_section_header("Average Unified Oscillator", NIRNAY["avg_tip"], icon="activity")
    _lines(df, (("Avg_Signal", "Avg_Signal (− = oversold)", COLOR_MUTED),), "nirnay_avg_signal",
           "mean oscillator (±10; − = oversold)")

    section_gap()
    render_section_header("Individual Constituents",
                          "Per-stock oscillators, adaptive cuts, warm-up flag and regime read-outs (last 100 bars).",
                          icon="database")
    if dfs:
        sym = st.selectbox("Select Symbol", sorted(dfs.keys()), key="nirnay_sym_select")
        cdf = dfs.get(sym)
        if cdf is not None and len(cdf):
            cdf = cdf.iloc[-100:].copy()
            if isinstance(cdf.columns, pd.MultiIndex):
                cdf.columns = [c[0] for c in cdf.columns]
            cols = [c for c in ("Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "OS_Cut", "OB_Cut", "Condition", "Warmup",
                                "Regime", "HMM_Bull", "HMM_Bear", "Vol_Regime", "Change_Point", "Confidence")
                    if c in cdf.columns]
            st.dataframe(cdf[cols] if cols else cdf, width="stretch")
