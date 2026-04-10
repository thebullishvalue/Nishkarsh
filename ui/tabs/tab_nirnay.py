"""
Nirnay tab — Constituent regime intelligence, zone distribution, signals, HMM.
Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import chart_layout, style_axes
from ui.components import render_metric_card, render_section_header, section_gap

# ── Theme palette ───────────────────────────────────────────────────────────
EMERALD = "#34D399"
ROSE = "#FB7185"
AMBER = "#D4A853"
SLATE = "#94A3B8"


def render_nirnay_tab() -> None:
    """Nirnay tab — all plots rebuilt with consistent Obsidian Quant styling."""
    nirnay_daily = st.session_state.get("nirnay_daily")
    nirnay_constituent_dfs = st.session_state.get("nirnay_constituent_dfs", {})

    if nirnay_daily is None or nirnay_daily.empty:
        st.info("No Nirnay constituent data available.")
        return

    # ── Normalize columns ───────────────────────────────────────────────
    df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")].copy()
    col_map = {}
    for c in df_n.columns:
        cl = c.lower().replace("-", "_")
        if cl in ("oversold_pct",):          col_map[c] = "Oversold_Pct"
        elif cl in ("overbought_pct",):      col_map[c] = "Overbought_Pct"
        elif cl in ("neutral_pct",):         col_map[c] = "Neutral_Pct"
        elif cl in ("buy_signals", "buy_signal_count"): col_map[c] = "Buy_Signals"
        elif cl in ("sell_signals", "sell_signal_count"): col_map[c] = "Sell_Signals"
        elif cl in ("avg_signal", "avg_unified_osc"):   col_map[c] = "Avg_Signal"
        elif cl in ("oversold",):            col_map[c] = "Oversold"
        elif cl in ("overbought",):          col_map[c] = "Overbought"
        elif cl in ("neutral",):             col_map[c] = "Neutral"
        elif cl in ("total_analyzed", "num_constituents"): col_map[c] = "Total_Analyzed"
        elif cl in ("avg_hmm_bull",):        col_map[c] = "avg_hmm_bull"
        elif cl in ("avg_hmm_bear",):        col_map[c] = "avg_hmm_bear"
    df_n = df_n.rename(columns=col_map)

    for col, default in [
        ("Oversold_Pct", 0), ("Overbought_Pct", 0), ("Neutral_Pct", 0),
        ("Buy_Signals", 0), ("Sell_Signals", 0), ("Avg_Signal", 0),
        ("Oversold", 0), ("Overbought", 0), ("Neutral", 0),
        ("Total_Analyzed", 0), ("avg_hmm_bull", 0.33), ("avg_hmm_bear", 0.33),
    ]:
        if col not in df_n.columns:
            df_n[col] = default

    dates = list(df_n.index)

    # ═══════════════════════════════════════════════════════════════════════
    # METRIC CARDS — summary row
    # ═══════════════════════════════════════════════════════════════════════
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        v = df_n["Oversold_Pct"].iloc[-1]
        render_metric_card("OVERSOLD", f"{v:.0f}%", "Constituents", "success" if v > 60 else "neutral")
    with c2:
        v = df_n["Overbought_Pct"].iloc[-1]
        render_metric_card("OVERBOUGHT", f"{v:.0f}%", "Constituents", "danger" if v > 60 else "neutral")
    with c3:
        v = df_n["Avg_Signal"].iloc[-1]
        render_metric_card("AVG SIGNAL", f"{v:.2f}", "Unified Osc", "success" if v < -1 else "danger" if v > 1 else "neutral")
    with c4:
        v = int(df_n["Buy_Signals"].iloc[-1])
        render_metric_card("BUY SIGNALS", str(v), "Today", "success" if v > 0 else "neutral")
    with c5:
        v = int(df_n["Sell_Signals"].iloc[-1])
        render_metric_card("SELL SIGNALS", str(v), "Today", "danger" if v > 0 else "neutral")
    with c6:
        render_metric_card("TRADING DAYS", str(len(df_n)), "Analyzed", "info")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 1. ZONE DISTRIBUTION OVER TIME
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Zone Distribution Over Time",
        "Percentage of constituents in each zone daily",
        icon="layers",
        accent="emerald",
    )

    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=dates, y=df_n["Oversold_Pct"].values,
        mode="lines", name="Oversold %",
        fill="tozeroy", fillcolor="rgba(52, 211, 153, 0.12)",
        line=dict(color=EMERALD, width=1.5),
    ))
    fig_zones.add_trace(go.Scatter(
        x=dates, y=df_n["Overbought_Pct"].values,
        mode="lines", name="Overbought %",
        fill="tozeroy", fillcolor="rgba(251, 113, 133, 0.12)",
        line=dict(color=ROSE, width=1.5),
    ))
    ymax = max(df_n["Oversold_Pct"].max(), df_n["Overbought_Pct"].max()) * 1.15

    fig_zones.update_layout(**chart_layout(height=380))
    style_axes(fig_zones, y_title="% of Constituents", y_range=[0, ymax])
    st.plotly_chart(fig_zones, width='stretch', key="nirnay_os_ob")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. RAW ZONE COUNTS
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header("Raw Zone Counts", icon="bar-chart", accent="cyan")

    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        x=dates, y=df_n["Oversold"].values, name="Oversold",
        marker=dict(color="rgba(52,211,153,0.85)"),
    ))
    fig_counts.add_trace(go.Bar(
        x=dates, y=df_n["Overbought"].values, name="Overbought",
        marker=dict(color="rgba(251,113,133,0.85)"),
    ))

    fig_counts.update_layout(**chart_layout(height=320), barmode="group")
    style_axes(fig_counts, y_title="Count")
    st.plotly_chart(fig_counts, width='stretch', key="nirnay_counts")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SIGNAL COUNTS OVER TIME
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header("Signal Counts Over Time", icon="zap", accent="rose")

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Scatter(
        x=dates, y=df_n["Buy_Signals"].values,
        mode="lines+markers", name="Buy Signals",
        line=dict(color=EMERALD, width=1.5),
        marker=dict(size=3, color=EMERALD),
    ))
    fig_signals.add_trace(go.Scatter(
        x=dates, y=df_n["Sell_Signals"].values,
        mode="lines+markers", name="Sell Signals",
        line=dict(color=ROSE, width=1.5),
        marker=dict(size=3, color=ROSE),
    ))

    fig_signals.update_layout(**chart_layout(height=350))
    style_axes(fig_signals, y_title="Signal Count")
    st.plotly_chart(fig_signals, width='stretch', key="nirnay_signal_counts")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. AVERAGE UNIFIED SIGNAL
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Average Unified Signal",
        "Negative = bullish bias \u00b7 Positive = bearish bias",
        icon="activity",
    )

    avg_vals = df_n["Avg_Signal"].values
    colors = [EMERALD if v < -2 else ROSE if v > 2 else "rgba(148,163,184,0.75)" for v in avg_vals]

    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(
        x=dates, y=np.clip(avg_vals, 0, None),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.05)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_n.add_trace(go.Scatter(
        x=dates, y=np.clip(avg_vals, None, 0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.05)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_n.add_trace(go.Scatter(
        x=dates, y=avg_vals,
        mode="lines+markers", name="Avg Signal",
        line=dict(color=SLATE, width=1.5),
        marker=dict(size=3, color=colors),
    ))
    fig_n.add_hline(y=2, line_color="rgba(251,113,133,0.2)", line_width=0.5, line_dash="dot")
    fig_n.add_hline(y=-2, line_color="rgba(52,211,153,0.2)", line_width=0.5, line_dash="dot")
    fig_n.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)

    fig_n.update_layout(**chart_layout(height=340))
    style_axes(fig_n, y_title="Avg Signal", y_range=[-6, 6])
    st.plotly_chart(fig_n, width='stretch', key="nirnay_avg_signal")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. HMM STATE PROBABILITIES
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header("HMM State Probabilities", icon="eye", accent="violet")

    fig_hmm = go.Figure()
    if "avg_hmm_bull" in df_n.columns:
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=df_n["avg_hmm_bull"].values,
            mode="lines", name="P(Bull)",
            line=dict(color=EMERALD, width=1.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        ))
    if "avg_hmm_bear" in df_n.columns:
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=df_n["avg_hmm_bear"].values,
            mode="lines", name="P(Bear)",
            line=dict(color=ROSE, width=1.5),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.08)",
        ))
    if "avg_hmm_bull" in df_n.columns and "avg_hmm_bear" in df_n.columns:
        neutral_vals = 1.0 - df_n["avg_hmm_bull"].values - df_n["avg_hmm_bear"].values
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=neutral_vals,
            mode="lines", name="P(Neutral)",
            line=dict(color=SLATE, width=1, dash="dot"),
        ))
    fig_hmm.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.08)", line_width=0.5)

    fig_hmm.update_layout(**chart_layout(height=300))
    style_axes(fig_hmm, y_title="Probability", y_range=[0, 1])
    st.plotly_chart(fig_hmm, width='stretch', key="nirnay_hmm_regime")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 6. INDIVIDUAL CONSTITUENTS
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header("Individual Constituents", icon="database")

    if nirnay_constituent_dfs:
        sym = st.selectbox("Select Symbol", sorted(nirnay_constituent_dfs.keys()), key="nirnay_sym_select")
        if sym and sym in nirnay_constituent_dfs:
            cdf = nirnay_constituent_dfs[sym].iloc[-100:].copy()
            if isinstance(cdf.columns, pd.MultiIndex):
                cdf.columns = [c[0] for c in cdf.columns]
            cols_show = [c for c in ["Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition", "Regime"] if c in cdf.columns]
            st.dataframe(cdf[cols_show] if cols_show else cdf, width='stretch')
