"""
Aarambh tab — Walk-forward valuation plots, conviction, breadth, model quality.
Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui.theme import chart_layout, style_axes
from ui.components import render_metric_card, render_info_box, render_section_header, section_gap
from core.config import OU_PROJECTION_DAYS, COLOR_GREEN, COLOR_RED

# ── Theme palette ───────────────────────────────────────────────────────────
EMERALD = "#34D399"
ROSE = "#FB7185"
AMBER = "#D4A853"
CYAN = "#22D3EE"
VIOLET = "#A78BFA"
SLATE = "#94A3B8"


def _conviction_colors(values):
    """Map conviction values to semantic colors and marker sizes."""
    colors, sizes = [], []
    for c in values:
        if c > 40:
            colors.append(ROSE); sizes.append(7)
        elif c >= 20:
            colors.append("rgba(251,113,133,0.85)"); sizes.append(5)
        elif c < -40:
            colors.append(EMERALD); sizes.append(7)
        elif c <= -20:
            colors.append("rgba(52,211,153,0.85)"); sizes.append(5)
        else:
            colors.append("rgba(148,163,184,0.75)"); sizes.append(4)
    return colors, sizes


def render_aarambh_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target):
    """Aarambh tab — all plots rebuilt with consistent Obsidian Quant styling."""

    # ═══════════════════════════════════════════════════════════════════════
    # 1. BASE CONVICTION SCORE
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Base Conviction Score",
        "Raw breadth differential: Oversold% - Overbought% across all lookbacks",
        icon="activity",
    )

    if "ConvictionRaw" in ts_filtered.columns:
        raw = ts_filtered["ConvictionRaw"]
        conv_colors, marker_sizes = _conviction_colors(raw)

        fig_raw = go.Figure()
        # Positive fill (bearish zone)
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=raw.clip(lower=0),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # Negative fill (bullish zone)
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=raw.clip(upper=0),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # Main line
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=raw, mode="lines+markers", name="Raw Conviction",
            line=dict(color=SLATE, width=1.5),
            marker=dict(size=marker_sizes, color=conv_colors),
        ))
        # Threshold lines
        fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
        fig_raw.add_hline(y=40, line_dash="dot", line_color="rgba(251,113,133,0.18)", line_width=0.5, annotation_text="OB", annotation_position="right")
        fig_raw.add_hline(y=-40, line_dash="dot", line_color="rgba(52,211,153,0.18)", line_width=0.5, annotation_text="OS", annotation_position="right")

        fig_raw.update_layout(**chart_layout(height=340, show_legend=False))
        style_axes(fig_raw, y_title="Conviction", y_range=[-100, 100])
        st.plotly_chart(fig_raw, width='stretch', key="aarambh_raw_conviction")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. DDM CONVICTION (SPRT BOUNDARIES)
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "SPRT DDM Confidence Boundaries",
        "Drift-Diffusion accumulation with mean-reverting variance",
        icon="shield",
        accent="cyan",
    )

    fig_conv = go.Figure()
    # Confidence band
    if "ConvictionUpper" in ts_filtered.columns:
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionUpper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionLower"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(148,163,184,0.06)",
            showlegend=False, hoverinfo="skip",
        ))
    # Positive/negative fills
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(lower=0),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(upper=0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # DDM line
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"], mode="lines", name="DDM Conviction",
        line=dict(color=SLATE, width=2),
    ))
    fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)

    fig_conv.update_layout(**chart_layout(height=340, show_legend=False))
    style_axes(fig_conv, y_title="Conviction", y_range=[-100, 100])
    st.plotly_chart(fig_conv, width='stretch', key="aarambh_ddm")

    # Interpretation card
    cv = signal["conviction_score"]
    cu, cl = signal["conviction_upper"], signal["conviction_lower"]
    if cv > 40:
        regime_title, regime_desc = "STRONG OVERBOUGHT", f"DDM conviction at {cv:+.0f} indicates extreme overbought conditions."
    elif cv > 20:
        regime_title, regime_desc = "MODERATE OVERBOUGHT", f"DDM conviction at {cv:+.0f} suggests overbought bias."
    elif cv > -20:
        regime_title, regime_desc = "NEUTRAL ZONE", f"DDM conviction at {cv:+.0f} — no strong directional bias."
    elif cv > -40:
        regime_title, regime_desc = "MODERATE OVERSOLD", f"DDM conviction at {cv:+.0f} suggests oversold bias."
    else:
        regime_title, regime_desc = "STRONG OVERSOLD", f"DDM conviction at {cv:+.0f} indicates extreme oversold conditions."
    bw = cu - cl
    conf_note = f"Narrow band ({bw:.0f} pts) = high conviction." if bw < 30 else f"Wide band ({bw:.0f} pts) = elevated uncertainty." if bw > 60 else f"Moderate uncertainty ({bw:.0f} pts)."
    render_info_box(title=regime_title, content=f"{regime_desc} {conf_note}")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. MARKET STATE — metric cards
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header("Market State", icon="crosshair", accent="emerald")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "OVERSOLD BREADTH", f'{signal["oversold_breadth"]:.0f}%',
            "Lookbacks in undervalued zones",
            "success" if signal["oversold_breadth"] > 60 else "neutral",
        )
    with c2:
        render_metric_card(
            "OVERBOUGHT BREADTH", f'{signal["overbought_breadth"]:.0f}%',
            "Lookbacks in overvalued zones",
            "danger" if signal["overbought_breadth"] > 60 else "neutral",
        )
    with c3:
        curr_regime = signal["regime"]
        regime_short = curr_regime.replace("STRONGLY ", "").replace("OVERSOLD", "OS").replace("OVERBOUGHT", "OB")
        regime_color = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else "neutral"
        render_metric_card("CURRENT REGIME", regime_short, f"OU t\u00bd = {signal['ou_half_life']:.0f}d", regime_color)

    # Regime distribution interpretation
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

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. MODEL QUALITY
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header("Model Quality", icon="cpu", accent="violet")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        r2 = model_stats["r2_oos"]
        render_metric_card("OOS R\u00b2", f"{r2:.3f}", "Walk-forward fit",
                           "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger")
    with q2:
        r2_rw = model_stats.get("r2_vs_rw", 0.0)
        render_metric_card("R\u00b2 vs RW", f"{r2_rw:+.3f}", "Vs random walk",
                           "success" if r2_rw > 0.05 else "warning" if r2_rw > -0.05 else "danger")
    with q3:
        h = signal["hurst"]
        h_label = "Mean-Revert" if h < 0.40 else "Trending" if h > 0.60 else "Random Walk"
        render_metric_card("DFA Hurst", f"{h:.2f}", h_label,
                           "success" if h < 0.40 else "danger" if h > 0.60 else "neutral")
    with q4:
        sp = signal["model_spread"]
        render_metric_card("Model Spread", f"{sp:.2f}", "Ensemble std dev",
                           "success" if sp < 0.5 else "warning" if sp < 1.5 else "danger")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. ACTUAL vs FAIR VALUE — dual-panel chart
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Actual vs Walk-Forward Fair Value",
        "Out-of-sample ensemble prediction with model uncertainty bands",
        icon="trending",
        accent="cyan",
    )

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.62, 0.38],
        shared_xaxes=True, vertical_spacing=0.06,
    )
    # Actual
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["Actual"], mode="lines", name="Actual",
        line=dict(color=SLATE, width=1.5),
    ), row=1, col=1)
    # Fair Value
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value (OOS)",
        line=dict(color=CYAN, width=1.5, dash="dot"),
    ), row=1, col=1)
    # Model spread band
    if "ModelSpread" in ts_filtered.columns:
        upper = ts_filtered["FairValue"] + ts_filtered["ModelSpread"]
        lower = ts_filtered["FairValue"] - ts_filtered["ModelSpread"]
        fig.add_trace(go.Scatter(
            x=x_axis, y=upper, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(34,211,238,0.07)",
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
    # Residual bars
    bar_colors = [EMERALD if r < 0 else ROSE for r in ts_filtered["Residual"]]
    fig.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["Residual"], name="Residual (OOS)",
        marker_color=bar_colors, showlegend=False, opacity=0.65,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=2, col=1)

    # OU projection overlay
    if hasattr(engine, "ou_projection") and len(engine.ou_projection) > 0 and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
        from pandas import bdate_range
        last_date = ts["Date"].iloc[-1]
        proj_dates = bdate_range(start=last_date, periods=OU_PROJECTION_DAYS + 1)[1:]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=engine.ou_projection, mode="lines", name="OU Projection",
            line=dict(color=SLATE, width=1, dash="dot"), opacity=0.45,
        ), row=2, col=1)
        if len(engine.ou_projection_upper) > 0:
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(148,163,184,0.08)", showlegend=False, hoverinfo="skip"), row=2, col=1)

    fig.update_layout(**chart_layout(height=540))
    style_axes(fig, y_title=active_target, row=1, col=1)
    style_axes(fig, y_title="Residual", row=2, col=1)

    st.plotly_chart(fig, width='stretch', key="aarambh_fairvalue")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 6. MARKET BREADTH
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Market Breadth",
        "Oversold / overbought breadth across lookback windows",
        icon="bar-chart",
        accent="emerald",
    )

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
    fig_zones.add_hline(y=60, line_dash="dot", line_color="rgba(212,168,83,0.18)", line_width=0.5)

    fig_zones.update_layout(**chart_layout(height=340))
    style_axes(fig_zones, y_title="Breadth %", y_range=[0, 100])
    st.plotly_chart(fig_zones, width='stretch', key="aarambh_breadth")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 7. SIGNAL FREQUENCY
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Signal Frequency",
        "Z-score threshold crossovers",
        icon="zap",
        accent="rose",
    )

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy",
        marker=dict(color=EMERALD, opacity=0.85),
    ))
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell",
        marker=dict(color=ROSE, opacity=0.85),
    ))

    fig_signals.update_layout(**chart_layout(height=280), barmode="relative")
    style_axes(fig_signals, y_title="Count")
    st.plotly_chart(fig_signals, width='stretch', key="aarambh_signal_freq")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 8. AVERAGE Z-SCORE
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Average Z-Score",
        "Multi-lookback z-score composite",
        icon="target",
    )

    fig_z = go.Figure()
    bar_colors = [EMERALD if z < -1 else ROSE if z > 1 else "rgba(148,163,184,0.75)" for z in ts_filtered["AvgZ"]]
    fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered["AvgZ"], marker_color=bar_colors, opacity=0.85, showlegend=False))
    fig_z.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
    fig_z.add_hline(y=2, line_dash="dot", line_color="rgba(251,113,133,0.18)", line_width=0.5)
    fig_z.add_hline(y=-2, line_dash="dot", line_color="rgba(52,211,153,0.18)", line_width=0.5)

    fig_z.update_layout(**chart_layout(height=280, show_legend=False))
    style_axes(fig_z, y_title="Z-Score")
    st.plotly_chart(fig_z, width='stretch', key="aarambh_avg_z")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 9. CURRENT LOOKBACK STATES
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header("Current Lookback States", icon="grid")

    from core.config import LOOKBACK_WINDOWS
    rows_html = []
    for lb in LOOKBACK_WINDOWS:
        if f"Z_{lb}" not in ts_filtered.columns:
            continue
        z = ts_filtered[f"Z_{lb}"].iloc[-1]
        zone = ts_filtered[f"Zone_{lb}"].iloc[-1]
        zone_color = COLOR_GREEN if "Under" in zone else COLOR_RED if "Over" in zone else SLATE
        rows_html.append(
            f'<div class="lookback-row">'
            f'<span class="label">{lb}-Day Lookback</span>'
            f'<span class="value" style="color:{zone_color};">{zone} ({z:+.2f})</span>'
            f'</div>'
        )
    if rows_html:
        st.markdown(
            f'<div style="background:var(--glass);border:1px solid var(--border);border-radius:var(--r-md);overflow:hidden;">{"".join(rows_html)}</div>',
            unsafe_allow_html=True,
        )
