"""
Nirnay tab — ALL original Nirnay plots organized with dividers.
Midnight Bloomberg Terminal design language.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_chart_theme
from ui.components import render_metric_card
from core.config import COLOR_GREEN, COLOR_RED, COLOR_AMBER, COLOR_CYAN, COLOR_MUTED


def render_nirnay_tab() -> None:
    """ALL Nirnay plots from original system, organized with dividers."""
    nirnay_daily = st.session_state.get("nirnay_daily")
    nirnay_constituent_dfs = st.session_state.get("nirnay_constituent_dfs", {})

    if nirnay_daily is None or nirnay_daily.empty:
        st.info("No Nirnay constituent data available.")
        return

    # Normalize column names
    df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")].copy()
    col_map = {}
    for c in df_n.columns:
        cl = c.lower().replace("-", "_")
        if cl in ("oversold_pct",):
            col_map[c] = "Oversold_Pct"
        elif cl in ("overbought_pct",):
            col_map[c] = "Overbought_Pct"
        elif cl in ("neutral_pct",):
            col_map[c] = "Neutral_Pct"
        elif cl in ("buy_signals", "buy_signal_count"):
            col_map[c] = "Buy_Signals"
        elif cl in ("sell_signals", "sell_signal_count"):
            col_map[c] = "Sell_Signals"
        elif cl in ("avg_signal", "avg_unified_osc"):
            col_map[c] = "Avg_Signal"
        elif cl in ("oversold",):
            col_map[c] = "Oversold"
        elif cl in ("overbought",):
            col_map[c] = "Overbought"
        elif cl in ("neutral",):
            col_map[c] = "Neutral"
        elif cl in ("total_analyzed", "num_constituents"):
            col_map[c] = "Total_Analyzed"
        elif cl in ("avg_hmm_bull",):
            col_map[c] = "avg_hmm_bull"
        elif cl in ("avg_hmm_bear",):
            col_map[c] = "avg_hmm_bear"
    df_n = df_n.rename(columns=col_map)

    for col, default in [
        ("Oversold_Pct", 0), ("Overbought_Pct", 0), ("Neutral_Pct", 0),
        ("Buy_Signals", 0), ("Sell_Signals", 0), ("Avg_Signal", 0),
        ("Oversold", 0), ("Overbought", 0), ("Neutral", 0),
        ("Total_Analyzed", 0), ("avg_hmm_bull", 0.33), ("avg_hmm_bear", 0.33),
    ]:
        if col not in df_n.columns:
            df_n[col] = default

    # Metric cards row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        overs_pct = df_n["Oversold_Pct"].iloc[-1]
        render_metric_card("OVERSOLD", f"{overs_pct:.0f}%", "Constituents",
                           "success" if overs_pct > 60 else "neutral")
    with c2:
        overb = df_n["Overbought_Pct"].iloc[-1]
        render_metric_card("OVERBOUGHT", f"{overb:.0f}%", "Constituents",
                           "danger" if overb > 60 else "neutral")
    with c3:
        avg_sig = df_n["Avg_Signal"].iloc[-1]
        render_metric_card("AVG SIGNAL", f"{avg_sig:.2f}", "Unified Osc",
                           "success" if avg_sig < -1 else "danger" if avg_sig > 1 else "neutral")
    with c4:
        buys = int(df_n["Buy_Signals"].iloc[-1])
        render_metric_card("BUY SIGNALS", str(buys), "Today",
                           "success" if buys > 0 else "neutral")
    with c5:
        sells = int(df_n["Sell_Signals"].iloc[-1])
        render_metric_card("SELL SIGNALS", str(sells), "Today",
                           "danger" if sells > 0 else "neutral")
    with c6:
        n_days = len(df_n)
        render_metric_card("TRADING DAYS", str(n_days), "Analyzed", "info")

    st.markdown("---")

    # ─── Overbought / Oversold Distribution ────────────────────────────
    st.markdown("##### Zone Distribution Over Time")
    st.caption("Percentage of constituents in each zone daily")

    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=list(df_n.index), y=df_n["Oversold_Pct"].values,
        mode="lines", name="Oversold %",
        fill="tozeroy", fillcolor="rgba(52, 211, 153, 0.15)",
        line=dict(color=COLOR_GREEN, width=1.5),
    ))
    fig_zones.add_trace(go.Scatter(
        x=list(df_n.index), y=df_n["Overbought_Pct"].values,
        mode="lines", name="Overbought %",
        fill="tozeroy", fillcolor="rgba(251, 113, 133, 0.15)",
        line=dict(color=COLOR_RED, width=1.5),
    ))
    ymax = max(df_n["Oversold_Pct"].max(), df_n["Overbought_Pct"].max()) * 1.1
    fig_zones.update_layout(
        height=380,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=0.5, title="% of Constituents", range=[0, ymax], zeroline=True, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)", gridwidth=0.5),
        margin=dict(l=10, r=10, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"), bordercolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_zones, width='stretch', key="nirnay_os_ob")

    st.markdown("---")

    # ─── Raw Counts ────────────────────────────────────────────────────
    st.markdown("##### Raw Zone Counts")

    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        x=list(df_n.index), y=df_n["Oversold"].values,
        name="Oversold",
        marker=dict(color=f"rgba({int(COLOR_GREEN[1:3],16)},{int(COLOR_GREEN[3:5],16)},{int(COLOR_GREEN[5:7],16)},0.7)"),
    ))
    fig_counts.add_trace(go.Bar(
        x=list(df_n.index), y=df_n["Overbought"].values,
        name="Overbought",
        marker=dict(color=f"rgba({int(COLOR_RED[1:3],16)},{int(COLOR_RED[3:5],16)},{int(COLOR_RED[5:7],16)},0.7)"),
    ))
    fig_counts.update_layout(
        height=320, barmode="group",
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=0.5, title="Count", zeroline=True, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)", gridwidth=0.5),
        margin=dict(l=10, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"), bordercolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_counts, width='stretch', key="nirnay_counts")

    st.markdown("---")

    # ─── Buy / Sell Signals ────────────────────────────────────────────
    st.markdown("##### Signal Counts Over Time")

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Scatter(
        x=list(df_n.index), y=df_n["Buy_Signals"].values,
        mode="lines+markers", name="Buy Signals",
        line=dict(color=COLOR_GREEN, width=1.5),
        marker=dict(size=4, color=COLOR_GREEN),
    ))
    fig_signals.add_trace(go.Scatter(
        x=list(df_n.index), y=df_n["Sell_Signals"].values,
        mode="lines+markers", name="Sell Signals",
        line=dict(color=COLOR_RED, width=1.5),
        marker=dict(size=4, color=COLOR_RED),
    ))
    fig_signals.update_layout(
        height=350,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=0.5, title="Signal Count", zeroline=True, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)", gridwidth=0.5),
        margin=dict(l=10, r=10, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"), bordercolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_signals, width='stretch', key="nirnay_signal_counts")

    st.markdown("---")

    # ─── Average Signal ────────────────────────────────────────────────
    st.markdown("##### Average Unified Signal")
    st.caption("Negative = bullish bias · Positive = bearish bias")

    avg_vals = df_n["Avg_Signal"].values
    colors = [COLOR_GREEN if v < -2 else COLOR_RED if v > 2 else COLOR_MUTED for v in avg_vals]

    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(
        x=list(df_n.index), y=np.clip(avg_vals, 0, None),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
        line=dict(width=0), showlegend=False,
    ))
    fig_n.add_trace(go.Scatter(
        x=list(df_n.index), y=np.clip(avg_vals, None, 0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
        line=dict(width=0), showlegend=False,
    ))
    fig_n.add_trace(go.Scatter(
        x=list(df_n.index), y=avg_vals,
        mode="lines+markers", name="Avg Signal",
        line=dict(color="#E2E8F0", width=1.5),
        marker=dict(size=4, color=colors),
    ))
    fig_n.add_hline(y=2, line_color=f"rgba({int(COLOR_RED[1:3],16)},{int(COLOR_RED[3:5],16)},{int(COLOR_RED[5:7],16)},0.3)", line_width=0.5, dash="dash")
    fig_n.add_hline(y=-2, line_color=f"rgba({int(COLOR_GREEN[1:3],16)},{int(COLOR_GREEN[3:5],16)},{int(COLOR_GREEN[5:7],16)},0.3)", line_width=0.5, dash="dash")
    fig_n.add_hline(y=0, line_color="rgba(255,255,255,0.08)", line_width=0.5)
    fig_n.update_layout(
        height=340,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=0.5, title="Avg Signal", range=[-6, 6], zeroline=True, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)", gridwidth=0.5),
        margin=dict(l=10, r=10, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"), bordercolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_n, width='stretch', key="nirnay_avg_signal")

    st.markdown("---")

    # ─── HMM Regime ────────────────────────────────────────────────────
    st.markdown("##### HMM State Probabilities")

    fig_hmm = go.Figure()
    if "avg_hmm_bull" in df_n.columns:
        fig_hmm.add_trace(go.Scatter(
            x=list(df_n.index), y=df_n["avg_hmm_bull"].values,
            mode="lines", name="P(Bull)",
            line=dict(color=COLOR_GREEN, width=1.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.1)",
        ))
    if "avg_hmm_bear" in df_n.columns:
        fig_hmm.add_trace(go.Scatter(
            x=list(df_n.index), y=df_n["avg_hmm_bear"].values,
            mode="lines", name="P(Bear)",
            line=dict(color=COLOR_RED, width=1.5),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.1)",
        ))
    if "avg_hmm_bull" in df_n.columns and "avg_hmm_bear" in df_n.columns:
        neutral_vals = 1.0 - df_n["avg_hmm_bull"].values - df_n["avg_hmm_bear"].values
        fig_hmm.add_trace(go.Scatter(
            x=list(df_n.index), y=neutral_vals,
            mode="lines", name="P(Neutral)",
            line=dict(color=COLOR_MUTED, width=1, dash="dot"),
        ))
    fig_hmm.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.1)", line_width=0.5)
    fig_hmm.update_layout(
        height=280,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=0.5, range=[0, 1], title="Probability", zeroline=True, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)", gridwidth=0.5),
        margin=dict(l=10, r=10, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"), bordercolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hmm, width='stretch', key="nirnay_hmm_regime")

    st.markdown("---")

    # ─── Individual Constituents ───────────────────────────────────────
    st.markdown("##### Individual Constituents")

    if nirnay_constituent_dfs:
        sym = st.selectbox("Select Symbol", sorted(nirnay_constituent_dfs.keys()), key="nirnay_sym_select")
        if sym and sym in nirnay_constituent_dfs:
            cdf = nirnay_constituent_dfs[sym].iloc[-100:].copy()
            if isinstance(cdf.columns, pd.MultiIndex):
                cdf.columns = [c[0] for c in cdf.columns]
            cols_show = [c for c in ["Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition", "Regime"] if c in cdf.columns]
            st.dataframe(cdf[cols_show] if cols_show else cdf, width='stretch')
