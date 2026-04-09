"""
Aarambh tab — ALL original correl.py plots organized with dividers.
Midnight Bloomberg Terminal design language.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui.theme import apply_chart_theme
from ui.components import render_metric_card, render_info_box
from core.config import OU_PROJECTION_DAYS, COLOR_AMBER, COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_MUTED

# Color constants for Obsidian Quant theme
EMERALD = "#34D399"
ROSE = "#FB7185"
AMBER = "#D4A853"
CYAN = "#22D3EE"
VIOLET = "#A78BFA"
SLATE = "#94A3B8"
SLATE_DIM = "#64748B"


def render_aarambh_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target):
    """ALL Aarambh plots from correl.py, organized with dividers."""

    # ─── Base Conviction ───────────────────────────────────────────────
    st.markdown("##### Base Conviction Score")
    st.caption("Raw breadth differential: Oversold% − Overbought% across all lookbacks. Negative = oversold bias.")

    if "ConvictionRaw" in ts_filtered.columns:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(lower=0),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)", line=dict(width=0), showlegend=False,
        ))
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(upper=0),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.08)", line=dict(width=0), showlegend=False,
        ))
        conv_colors = []
        marker_sizes = []
        for c in ts_filtered["ConvictionRaw"]:
            if c > 40: conv_colors.append(ROSE); marker_sizes.append(7)
            elif c >= 20: conv_colors.append("rgba(251,113,133,0.6)"); marker_sizes.append(5)
            elif c < -40: conv_colors.append(EMERALD); marker_sizes.append(7)
            elif c <= -20: conv_colors.append("rgba(52,211,153,0.6)"); marker_sizes.append(5)
            else: conv_colors.append("rgba(148,163,184,0.4)"); marker_sizes.append(4)
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"], mode="lines+markers", name="Raw Conviction",
            line=dict(color=AMBER, width=1.5),
            marker=dict(size=marker_sizes, color=conv_colors),
        ))
        fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
        fig_raw.add_hline(y=40, line_dash="dash", line_color="rgba(251,113,133,0.2)", line_width=0.5)
        fig_raw.add_hline(y=-40, line_dash="dash", line_color="rgba(52,211,153,0.2)", line_width=0.5)
        fig_raw.update_layout(
            height=340,
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", range=[-100, 100], title="Conviction"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)"),
            margin=dict(t=10, l=10, r=10, b=35),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(10, 14, 23, 0.9)",
                font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
                bordercolor="rgba(255,255,255,0.1)",
            ),
        )
        st.plotly_chart(fig_raw, width='stretch', key="aarambh_raw_conviction")

    st.markdown("---")

    # ─── DDM Conviction ────────────────────────────────────────────────
    st.markdown("##### SPRT DDM Confidence Boundaries")
    st.caption("Drift-Diffusion accumulation with mean-reverting variance.")

    fig_conv = go.Figure()
    if "ConvictionUpper" in ts_filtered.columns:
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionUpper"], mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionLower"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(245,158,11,0.06)",
        ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(lower=0),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)", line=dict(width=0), showlegend=False,
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(upper=0),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)", line=dict(width=0), showlegend=False,
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"], mode="lines", name="DDM Conviction",
        line=dict(color=AMBER, width=2),
    ))
    fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
    fig_conv.update_layout(
        height=340,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", range=[-100, 100], title="Conviction"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)"),
        margin=dict(t=10, l=10, r=10, b=35),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10, 14, 23, 0.9)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
            bordercolor="rgba(255,255,255,0.1)",
        ),
    )
    st.plotly_chart(fig_conv, width='stretch', key="aarambh_ddm")

    # Interpretation card
    conviction_val = signal["conviction_score"]
    conviction_upper = signal["conviction_upper"]
    conviction_lower = signal["conviction_lower"]
    if conviction_val > 40:
        regime_title = "STRONG OVERBOUGHT"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates extreme overbought conditions."
    elif conviction_val > 20:
        regime_title = "MODERATE OVERBOUGHT"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} suggests overbought bias."
    elif conviction_val > -20:
        regime_title = "NEUTRAL ZONE"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} — no strong directional bias."
    elif conviction_val > -40:
        regime_title = "MODERATE OVERSOLD"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} suggests oversold bias."
    else:
        regime_title = "STRONG OVERSOLD"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates extreme oversold conditions."
    band_width = conviction_upper - conviction_lower
    confidence_note = f"Narrow band ({band_width:.0f} pts) = high conviction." if band_width < 30 else f"Wide band ({band_width:.0f} pts) = elevated uncertainty." if band_width > 60 else f"Moderate uncertainty ({band_width:.0f} pts)."
    render_info_box(title=regime_title, content=f"{regime_desc} {confidence_note}")

    st.markdown("---")

    # ─── Market State ──────────────────────────────────────────────────
    st.markdown("##### Market State")
    reg_col1, reg_col2, reg_col3 = st.columns(3)
    with reg_col1:
        render_metric_card("OVERSOLD BREADTH", f'{signal["oversold_breadth"]:.0f}%', "Lookbacks in undervalued zones",
                           "success" if signal["oversold_breadth"] > 60 else "neutral")
    with reg_col2:
        render_metric_card("OVERBOUGHT BREADTH", f'{signal["overbought_breadth"]:.0f}%', "Lookbacks in overvalued zones",
                           "danger" if signal["overbought_breadth"] > 60 else "neutral")
    with reg_col3:
        curr_regime = signal["regime"]
        regime_short = curr_regime.replace("STRONGLY ", "").replace("OVERSOLD", "OS").replace("OVERBOUGHT", "OB")
        regime_color = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else "neutral"
        render_metric_card("CURRENT REGIME", regime_short, f"OU t½ = {signal['ou_half_life']:.0f}d", regime_color)

    total = len(ts)
    os_total = regime_stats["strongly_oversold"] + regime_stats["oversold"]
    ob_total = regime_stats["strongly_overbought"] + regime_stats["overbought"]
    neutral_count = regime_stats["neutral"]
    os_pct = os_total / total * 100 if total > 0 else 0
    ob_pct = ob_total / total * 100 if total > 0 else 0
    neutral_pct = neutral_count / total * 100 if total > 0 else 0

    if os_pct > 50:
        interp_title = "OVERSOLD DOMINANT"
        interp_text = f"Market spent {os_pct:.0f}% in oversold ({os_total}/{total}). Neutral: {neutral_pct:.0f}%. Overbought: {ob_pct:.0f}%."
    elif ob_pct > 50:
        interp_title = "OVERBOUGHT DOMINANT"
        interp_text = f"Market spent {ob_pct:.0f}% in overbought ({ob_total}/{total}). Neutral: {neutral_pct:.0f}%. Oversold: {os_pct:.0f}%."
    else:
        interp_title = "BALANCED REGIME"
        interp_text = f"Oscillating regimes. Neutral: {neutral_pct:.0f}% ({neutral_count}). Oversold: {os_pct:.0f}%. Overbought: {ob_pct:.0f}%."
    render_info_box(title=interp_title, content=interp_text)

    st.markdown("---")

    # ─── Model Quality ─────────────────────────────────────────────────
    st.markdown("##### Model Quality")
    qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
    with qual_col1:
        r2 = model_stats["r2_oos"]
        render_metric_card("OOS R²", f"{r2:.3f}", "Walk-forward fit", "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger")
    with qual_col2:
        r2_rw = model_stats.get("r2_vs_rw", 0.0)
        render_metric_card("R² vs RW", f"{r2_rw:+.3f}", "Vs random walk", "success" if r2_rw > 0.05 else "warning" if r2_rw > -0.05 else "danger")
    with qual_col3:
        h = signal["hurst"]
        h_label = "Mean-Revert" if h < 0.40 else "Trending" if h > 0.60 else "Random Walk"
        render_metric_card("DFA Hurst", f"{h:.2f}", h_label, "success" if h < 0.40 else "danger" if h > 0.60 else "neutral")
    with qual_col4:
        sp = signal["model_spread"]
        render_metric_card("Model Spread", f"{sp:.2f}", "Ensemble std dev", "success" if sp < 0.5 else "warning" if sp < 1.5 else "danger")

    st.markdown("---")

    # ─── Fair Value ────────────────────────────────────────────────────
    st.markdown("##### Actual vs Walk-Forward Fair Value")
    st.caption("Out-of-sample ensemble prediction with model uncertainty bands")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["Actual"], mode="lines", name="Actual",
        line=dict(color=AMBER, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value (OOS)",
        line=dict(color=CYAN, width=1.5, dash="dot"),
    ), row=1, col=1)
    if "ModelSpread" in ts_filtered.columns:
        upper = ts_filtered["FairValue"] + ts_filtered["ModelSpread"]
        lower = ts_filtered["FairValue"] - ts_filtered["ModelSpread"]
        fig.add_trace(go.Scatter(x=x_axis, y=upper, mode="lines", line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(34,182,238,0.08)"), row=1, col=1)
    bar_colors = [EMERALD if r < 0 else ROSE for r in ts_filtered["Residual"]]
    fig.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["Residual"], name="Residual (OOS)",
        marker_color=bar_colors, showlegend=False, opacity=0.7,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=2, col=1)

    if hasattr(engine, "ou_projection") and len(engine.ou_projection) > 0 and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
        from pandas import bdate_range
        last_date = ts["Date"].iloc[-1]
        proj_dates = bdate_range(start=last_date, periods=OU_PROJECTION_DAYS + 1)[1:]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=engine.ou_projection, mode="lines", name="OU Projection",
            line=dict(color=AMBER, width=1, dash="dot"), opacity=0.5,
        ), row=2, col=1)
        if len(engine.ou_projection_upper) > 0:
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_upper, mode="lines", line=dict(width=0), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(212,168,83,0.1)"), row=2, col=1)

    fig.update_layout(
        height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, family="JetBrains Mono, monospace")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10),
        hovermode="x unified",
        margin=dict(t=10, l=10, r=10, b=35),
        hoverlabel=dict(
            bgcolor="rgba(10, 14, 23, 0.9)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
            bordercolor="rgba(255,255,255,0.1)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.02)", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.02)", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", title_text=active_target, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", title_text="Residual", row=2, col=1)
    st.plotly_chart(fig, width='stretch', key="aarambh_fairvalue")

    st.markdown("---")

    # ─── Market Breadth ────────────────────────────────────────────────
    st.markdown("##### Market Breadth")
    st.caption("Oversold/overbought breadth across lookback windows")

    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OversoldBreadth"],
        fill="tozeroy", fillcolor="rgba(52,211,153,0.1)",
        line=dict(color=EMERALD, width=1.5), name="Oversold",
    ))
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OverboughtBreadth"],
        fill="tozeroy", fillcolor="rgba(251,113,133,0.1)",
        line=dict(color=ROSE, width=1.5), name="Overbought",
    ))
    fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(212,168,83,0.2)", line_width=0.5)
    fig_zones.update_layout(
        height=340,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", range=[0, 100], title="Breadth %"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)"),
        margin=dict(t=10, l=10, r=10, b=35),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, family="JetBrains Mono, monospace")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10, 14, 23, 0.9)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
            bordercolor="rgba(255,255,255,0.1)",
        ),
    )
    st.plotly_chart(fig_zones, width='stretch', key="aarambh_breadth")

    st.markdown("---")

    # ─── Signal Frequency ──────────────────────────────────────────────
    st.markdown("##### Signal Frequency")
    st.caption("Z-score threshold crossovers")

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy",
        marker=dict(color=EMERALD, opacity=0.7),
    ))
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell",
        marker=dict(color=ROSE, opacity=0.7),
    ))
    fig_signals.update_layout(
        height=280,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", title="Count"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)"),
        margin=dict(t=10, l=10, r=10, b=35),
        barmode="relative",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, family="JetBrains Mono, monospace")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10, 14, 23, 0.9)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
            bordercolor="rgba(255,255,255,0.1)",
        ),
    )
    st.plotly_chart(fig_signals, width='stretch', key="aarambh_signal_freq")

    st.markdown("---")

    # ─── Average Z-Score ───────────────────────────────────────────────
    st.markdown("##### Average Z-Score")
    st.caption("Multi-lookback z-score composite")

    fig_z = go.Figure()
    bar_colors = [EMERALD if z < -1 else ROSE if z > 1 else "rgba(148,163,184,0.4)" for z in ts_filtered["AvgZ"]]
    fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered["AvgZ"], marker_color=bar_colors, opacity=0.7))
    fig_z.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
    fig_z.add_hline(y=2, line_dash="dash", line_color="rgba(251,113,133,0.2)", line_width=0.5)
    fig_z.add_hline(y=-2, line_dash="dash", line_color="rgba(52,211,153,0.2)", line_width=0.5)
    fig_z.update_layout(
        height=280,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", title="Z-Score"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)"),
        margin=dict(t=10, l=10, r=10, b=35),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10, 14, 23, 0.9)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
            bordercolor="rgba(255,255,255,0.1)",
        ),
    )
    st.plotly_chart(fig_z, width='stretch', key="aarambh_avg_z")

    st.markdown("---")

    # ─── Current Lookback States ───────────────────────────────────────
    st.markdown("##### Current Lookback States")

    from core.config import LOOKBACK_WINDOWS
    for lb in LOOKBACK_WINDOWS:
        if f"Z_{lb}" not in ts_filtered.columns:
            continue
        z = ts_filtered[f"Z_{lb}"].iloc[-1]
        zone = ts_filtered[f"Zone_{lb}"].iloc[-1]
        zone_color = COLOR_GREEN if "Under" in zone else COLOR_RED if "Over" in zone else COLOR_MUTED
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0.2rem;border-bottom:1px solid var(--border);">
            <span style="font-family:var(--data);color:var(--ink-secondary);font-size:0.78rem;">{lb}-Day Lookback</span>
            <span style="color:{zone_color};font-weight:600;font-family:var(--data);font-size:0.78rem;">{zone} ({z:+.2f})</span>
        </div>
        """, unsafe_allow_html=True)
