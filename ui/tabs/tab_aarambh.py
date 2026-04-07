"""
Aarambh tab — ALL original correl.py plots organized with dividers.
Dashboard + Breadth Topology + ML Diagnostics combined.
"""

from __future__ import annotations

import html as html_mod
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui.theme import apply_chart_theme, COLOR_GOLD, COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_MUTED
from ui.components import render_metric_card
from core.config import OU_PROJECTION_DAYS


def render_aarambh_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target):
    """ALL Aarambh plots from correl.py, organized with dividers."""

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 1: BASE CONVICTION — Full width
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("##### Base Conviction Score")
    st.markdown(
        '<p style="color: #888; font-size: 0.85rem;">Raw breadth differential: Oversold% − Overbought% across all lookbacks. '
        'Negative = oversold bias, Positive = overbought bias.</p>',
        unsafe_allow_html=True,
    )

    if "ConvictionRaw" in ts_filtered.columns:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(lower=0),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)", line=dict(width=0), showlegend=False,
        ))
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(upper=0),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.12)", line=dict(width=0), showlegend=False,
        ))
        conv_colors = []
        marker_sizes = []
        for c in ts_filtered["ConvictionRaw"]:
            if c > 40: conv_colors.append(COLOR_RED); marker_sizes.append(10)
            elif c >= 20: conv_colors.append("rgba(239, 68, 68, 0.75)"); marker_sizes.append(8)
            elif c < -40: conv_colors.append(COLOR_GREEN); marker_sizes.append(10)
            elif c <= -20: conv_colors.append("rgba(16, 185, 129, 0.75)"); marker_sizes.append(8)
            else: conv_colors.append(COLOR_MUTED); marker_sizes.append(8)
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"], mode="lines+markers", name="Raw Conviction",
            line=dict(color=COLOR_GOLD, width=2.0),
            marker=dict(size=marker_sizes, color=conv_colors),
            hovertemplate="Date: %{x}<br>Conviction: %{y:.1f}<extra></extra>"
        ))
        fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        fig_raw.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.25)", line_width=1)
        fig_raw.add_hline(y=20, line_dash="dot", line_color="rgba(239,68,68,0.15)", line_width=1)
        fig_raw.add_hline(y=-20, line_dash="dot", line_color="rgba(16,185,129,0.15)", line_width=1)
        fig_raw.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.25)", line_width=1)
        fig_raw.update_layout(
            height=350, xaxis_title=None, yaxis_title="Conviction",
            yaxis=dict(range=[-100, 100], tickfont=dict(size=10)),
            margin=dict(t=10, l=60, r=20, b=10), showlegend=False,
        )
        apply_chart_theme(fig_raw)
        st.plotly_chart(fig_raw, width="stretch", key="aarambh_raw_conviction")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 2: DDM CONVICTION
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("##### SPRT DDM Confidence Boundaries")
    st.markdown(
        '<p style="color: #888; font-size: 0.85rem;">Drift-Diffusion accumulation with mean-reverting variance.</p>',
        unsafe_allow_html=True,
    )
    fig_conv = go.Figure()
    if "ConvictionUpper" in ts_filtered.columns:
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionUpper"], mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionLower"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(255,195,0,0.10)", name="95% Band",
        ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(lower=0),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)", line=dict(width=0), showlegend=False,
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(upper=0),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.12)", line=dict(width=0), showlegend=False,
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"], mode="lines", name="DDM Conviction",
        line=dict(color=COLOR_GOLD, width=2.5),
    ))
    fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig_conv.update_layout(
        height=350, xaxis_title=None, yaxis_title="Conviction",
        yaxis=dict(range=[-100, 100], tickfont=dict(size=10)),
        margin=dict(t=10, l=60, r=20, b=10), showlegend=False,
    )
    apply_chart_theme(fig_conv)
    st.plotly_chart(fig_conv, width="stretch", key="aarambh_ddm")

    # Interpretation Card
    conviction_val = signal["conviction_score"]
    conviction_upper = signal["conviction_upper"]
    conviction_lower = signal["conviction_lower"]
    if conviction_val > 40:
        regime_title = "STRONG OVERBOUGHT"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates extreme overbought conditions. The accumulated evidence strongly favors mean-reversion lower."
    elif conviction_val > 20:
        regime_title = "MODERATE OVERBOUGHT"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} suggests overbought bias. Evidence accumulation is positive but not at extreme levels."
    elif conviction_val > -20:
        regime_title = "NEUTRAL ZONE"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates balanced evidence. The Drift-Diffusion model shows no strong directional bias."
    elif conviction_val > -40:
        regime_title = "MODERATE OVERSOLD"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} suggests oversold bias. Evidence accumulation is negative but not at extreme levels."
    else:
        regime_title = "STRONG OVERSOLD"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates extreme oversold conditions. The accumulated evidence strongly favors mean-reversion higher."
    band_width = conviction_upper - conviction_lower
    confidence_note = f"Narrow band ({band_width:.0f} points) suggests high conviction in current reading." if band_width < 30 else f"Wide band ({band_width:.0f} points) suggests elevated uncertainty." if band_width > 60 else f"Moderate uncertainty range ({band_width:.0f} points)."
    st.markdown(f"""
    <div class="metric-card" style="padding: 1.25rem;">
        <h4 style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{html_mod.escape(regime_title)}</h4>
        <p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.7; margin: 0;">{html_mod.escape(regime_desc)}</p>
        <p style="color: var(--text-muted); font-size: 0.75rem; line-height: 1.6; margin: 0.75rem 0 0 0; font-style: italic;">{html_mod.escape(confidence_note)}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 4: MARKET STATE
    # ══════════════════════════════════════════════════════════════════════════════
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
    st.markdown("")

    total = len(ts)
    os_total = regime_stats["strongly_oversold"] + regime_stats["oversold"]
    ob_total = regime_stats["strongly_overbought"] + regime_stats["overbought"]
    neutral_count = regime_stats["neutral"]
    os_pct = os_total / total * 100 if total > 0 else 0
    ob_pct = ob_total / total * 100 if total > 0 else 0
    neutral_pct = neutral_count / total * 100 if total > 0 else 0

    if os_pct > 50:
        interp_title = "OVERSOLD DOMINANT"
        intensity = f"Strong oversold signals ({regime_stats['strongly_oversold']} periods) outweigh moderate oversold ({regime_stats['oversold']} periods)." if regime_stats["strongly_oversold"] > regime_stats["oversold"] else f"Moderate oversold signals ({regime_stats['oversold']} periods) dominate with {regime_stats['strongly_oversold']} strong oversold periods."
        interp_text = f"Market has spent {os_pct:.0f}% of the analyzed period in oversold conditions ({os_total} out of {total} observations). {intensity} Neutral regimes account for only {neutral_pct:.0f}% ({neutral_count} periods), while overbought conditions are rare at {ob_pct:.0f}% ({ob_total} periods)."
    elif ob_pct > 50:
        interp_title = "OVERBOUGHT DOMINANT"
        intensity = f"Strong overbought signals ({regime_stats['strongly_overbought']} periods) outweigh moderate overbought ({regime_stats['overbought']} periods)." if regime_stats["strongly_overbought"] > regime_stats["overbought"] else f"Moderate overbought signals ({regime_stats['overbought']} periods) dominate with {regime_stats['strongly_overbought']} strong overbought periods."
        interp_text = f"Market has spent {ob_pct:.0f}% of the analyzed period in overbought conditions ({ob_total} out of {total} observations). {intensity} Neutral regimes account for only {neutral_pct:.0f}% ({neutral_count} periods), while oversold conditions are rare at {os_pct:.0f}% ({os_total} periods)."
    else:
        interp_title = "BALANCED REGIME"
        interp_text = f"Market has oscillated between regimes without establishing clear directional dominance. Neutral conditions prevail at {neutral_pct:.0f}% ({neutral_count} periods), while oversold ({os_pct:.0f}%, {os_total} periods) and overbought ({ob_pct:.0f}%, {ob_total} periods) conditions are roughly balanced."
    st.markdown(f"""
    <div class="metric-card" style="padding: 1.25rem;">
        <h4 style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{interp_title}</h4>
        <p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.7; margin: 0;">{html_mod.escape(interp_text)}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 5: MODEL QUALITY
    # ══════════════════════════════════════════════════════════════════════════════
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
        h_label = "MR" if h < 0.40 else "Trend" if h > 0.60 else "RW"
        render_metric_card("DFA Hurst", f"{h:.2f}", h_label, "success" if h < 0.40 else "danger" if h > 0.60 else "neutral")
    with qual_col4:
        sp = signal["model_spread"]
        render_metric_card("Model Spread", f"{sp:.2f}", "Ensemble std dev", "success" if sp < 0.5 else "warning" if sp < 1.5 else "danger")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 6: FAIR VALUE
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("##### Actual vs Walk-Forward Fair Value")
    st.markdown('<p style="color: #888; font-size: 0.85rem;">Out-of-sample ensemble prediction with model uncertainty bands</p>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["Actual"], mode="lines", name="Actual",
        line=dict(color=COLOR_GOLD, width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value (OOS)",
        line=dict(color=COLOR_CYAN, width=2, dash="dot"),
    ), row=1, col=1)
    if "ModelSpread" in ts_filtered.columns:
        upper = ts_filtered["FairValue"] + ts_filtered["ModelSpread"]
        lower = ts_filtered["FairValue"] - ts_filtered["ModelSpread"]
        fig.add_trace(go.Scatter(x=x_axis, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(6,182,212,0.08)", name="Model Uncertainty", hoverinfo="skip"), row=1, col=1)
    bar_colors = [COLOR_GREEN if r < 0 else COLOR_RED for r in ts_filtered["Residual"]]
    fig.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["Residual"], name="Residual (OOS)",
        marker_color=bar_colors, showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=COLOR_GOLD, line_width=1, row=2, col=1)
    if hasattr(engine, "ou_projection") and len(engine.ou_projection) > 0 and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
        from pandas import bdate_range
        last_date = ts["Date"].iloc[-1]
        proj_dates = bdate_range(start=last_date, periods=OU_PROJECTION_DAYS + 1)[1:]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=engine.ou_projection, mode="lines", name="OU Projection",
            line=dict(color=COLOR_GOLD, width=1.5, dash="dot"), opacity=0.5,
        ), row=2, col=1)
        if len(engine.ou_projection_upper) > 0:
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(255,195,0,0.15)", name="θ Uncertainty", hoverinfo="skip"), row=2, col=1)
    fig.update_layout(height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text=active_target, row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch", key="aarambh_fairvalue")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # BREADTH TOPOLOGY
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("##### Market Breadth")
    st.markdown(
        '<p style="color:#888;font-size:0.85rem;">Oversold/overbought breadth across lookback windows</p>',
        unsafe_allow_html=True,
    )

    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OversoldBreadth"],
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
        line=dict(color=COLOR_GREEN, width=2), name="Oversold",
    ))
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OverboughtBreadth"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=COLOR_RED, width=2), name="Overbought",
    ))
    fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.2)", line_width=1)
    fig_zones.update_layout(
        height=350, xaxis_title=None, yaxis_title="Breadth %",
        yaxis=dict(range=[0, 100], tickfont=dict(size=10)),
        margin=dict(t=10, l=60, r=20, b=10),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    apply_chart_theme(fig_zones)
    st.plotly_chart(fig_zones, width="stretch", key="aarambh_breadth")

    st.markdown("---")
    st.markdown("##### Signal Frequency")
    st.markdown(
        '<p style="color:#888;font-size:0.85rem;">Z-score threshold crossovers (±1σ)</p>',
        unsafe_allow_html=True,
    )

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy",
        marker=dict(color=COLOR_GREEN, opacity=0.8),
    ))
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell",
        marker=dict(color=COLOR_RED, opacity=0.8),
    ))
    fig_signals.update_layout(
        height=300, xaxis_title=None, yaxis_title="Count",
        barmode="relative", margin=dict(t=10, l=60, r=20, b=10),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    apply_chart_theme(fig_signals)
    st.plotly_chart(fig_signals, width="stretch", key="aarambh_signal_freq")

    st.markdown("---")
    st.markdown("##### Average Z-Score")
    st.markdown(
        '<p style="color:#888;font-size:0.85rem;">Multi-lookback z-score composite</p>',
        unsafe_allow_html=True,
    )

    fig_z = go.Figure()
    bar_colors = [COLOR_GREEN if z < -1 else COLOR_RED if z > 1 else COLOR_MUTED for z in ts_filtered["AvgZ"]]
    fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered["AvgZ"], marker_color=bar_colors, name="Avg Z", opacity=0.8))
    fig_z.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig_z.add_hline(y=2, line_dash="dash", line_color="rgba(239,68,68,0.25)", line_width=1)
    fig_z.add_hline(y=-2, line_dash="dash", line_color="rgba(16,185,129,0.25)", line_width=1)
    fig_z.update_layout(
        height=300, xaxis_title=None, yaxis_title="Z-Score",
        margin=dict(t=10, l=60, r=20, b=10), showlegend=False,
    )
    apply_chart_theme(fig_z)
    st.plotly_chart(fig_z, width="stretch", key="aarambh_avg_z")

    st.markdown("---")
    st.markdown("##### Current Lookback States")

    from core.config import LOOKBACK_WINDOWS
    for lb in LOOKBACK_WINDOWS:
        if f"Z_{lb}" not in ts_filtered.columns:
            continue
        z = ts_filtered[f"Z_{lb}"].iloc[-1]
        zone = ts_filtered[f"Zone_{lb}"].iloc[-1]
        zone_color = COLOR_GREEN if "Under" in zone else COLOR_RED if "Over" in zone else COLOR_MUTED
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #2A2A2A;">
            <span style="color: #888; font-size: 0.85rem;">{lb}-Day</span>
            <span style="color: {zone_color}; font-weight: 600; font-size: 0.85rem;">{zone} ({z:+.2f})</span>
        </div>
        """, unsafe_allow_html=True)
