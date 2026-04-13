"""
Nishkarsh v1.2.0 — Aarambh tab: Walk-forward valuation plots, conviction, breadth, model quality.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

UI — AARAMBH engine visualization: ensemble regression outputs with conformal bounds.

Section order (logical analytical flow):
  Phase 1 — Trust:     Model Quality
  Phase 2 — Anchor:    Actual vs Fair Value
  Phase 3 — Signal:    Base Conviction → DDM-Filtered → Market Breadth
  Phase 4 — State:     Market State → Current Lookback States
  Phase 5 — Extremes:  Signal Frequency → Average Z-Score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui.theme import chart_layout, style_axes
from ui.components import (
    render_metric_card,
    render_info_box,
    render_interpretation_card,
    render_section_header,
    render_collapsible_section,
    render_collapsible_section_close,
    section_gap,
)
from core.config import (
    OU_PROJECTION_DAYS,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_PURPLE,
    COLOR_MUTED,
    UI_CONVICTION_STRONG,
    UI_CONVICTION_MODERATE,
    UI_BREADTH_HIGH,
    UI_R2_STRONG,
    UI_R2_ACCEPTABLE,
    UI_MODEL_SPREAD_LOW,
    UI_MODEL_SPREAD_HIGH,
    UI_BAND_NARROW,
    UI_BAND_WIDE,
    UI_CHART_HEIGHT_MEDIUM,
    UI_CHART_HEIGHT_XLARGE,
    UI_CHART_HEIGHT_SMALL,
)

# ── Alias colors for tab-local use ────────────────────────────────────────
EMERALD = COLOR_GREEN
ROSE = COLOR_RED
AMBER = COLOR_AMBER
CYAN = COLOR_CYAN
VIOLET = COLOR_PURPLE
SLATE = COLOR_MUTED

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "conviction_raw": (
        "Difference between the percentage of lookback windows calling the market oversold "
        "vs. overbought. Positive = more windows see overbought (bearish); negative = more "
        "see oversold (bullish). Values beyond +/-40 are extremes. This is the unsmoothed "
        "signal before any temporal filtering."
    ),
    "ddm_conviction": (
        "Conviction score smoothed through a Drift-Diffusion Model that accumulates evidence "
        "over time and reverts toward zero when signals are inconsistent. The shaded band is "
        "the confidence interval — narrow bands mean high conviction, wide bands mean the "
        "model is uncertain. Use this (not raw conviction) for trade decisions."
    ),
    "oversold_breadth": (
        "Share of lookback windows that classify the current PE ratio as below fair value. "
        "Above 60% means most models agree the market is cheap — a bullish signal when it "
        "starts to turn down."
    ),
    "overbought_breadth": (
        "Share of lookback windows that classify the current PE ratio as above fair value. "
        "Above 60% means most models agree the market is expensive — a caution signal when "
        "it starts to turn up."
    ),
    "current_regime": (
        "Classification derived from OU half-life (speed of mean reversion) and DDM "
        "conviction score. 'Oversold' = market prices in excessive pessimism; 'Overbought' "
        "= excessive optimism. Half-life tells you how fast reversion is expected."
    ),
    "oos_r2": (
        "Fraction of variance explained by the model on walk-forward out-of-sample data. "
        "Above 0.7 = strong predictive power; below 0.4 = model struggles to forecast. "
        "Primary metric for whether to trust the fair-value estimate."
    ),
    "r2_vs_rw": (
        "R-squared minus the R-squared of a naive random-walk forecast. Positive = the "
        "model adds value beyond 'tomorrow = today'; negative = a random walk would have "
        "done better. This is your reality check on model edge."
    ),
    "hurst": (
        "Hurst exponent via Detrended Fluctuation Analysis. Below 0.40 = mean-reverting "
        "(signals reliable). Above 0.60 = trending (signals lag, produce false reversals). "
        "Near 0.50 = random walk — no valuation edge."
    ),
    "model_spread": (
        "Standard deviation of predictions across ensemble models. Below 0.5 = models agree "
        "(signal is stable). Above 1.5 = models disagree (treat with caution even if "
        "conviction is high)."
    ),
}


def _conviction_colors(values):
    """Convert conviction scores to color-coded markers for chart rendering.

    Rose/red tones = overbought (bearish signal), emerald/green = oversold (bullish),
    slate/gray = neutral. Marker size scales with conviction magnitude (4 → 7px).
    Thresholds: |40| = strong, |20| = moderate, <20 = noise.
    """
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


# ═══════════════════════════════════════════════════════════════════════
#  CHART BUILDERS (extracted so they can be reused in any section order)
# ═══════════════════════════════════════════════════════════════════════

def _render_raw_conviction_chart(ts_filtered, x_axis):
    """Section: Base Conviction Score — unsmoothed signal."""
    if "ConvictionRaw" not in ts_filtered.columns:
        return
    raw = ts_filtered["ConvictionRaw"]
    conv_colors, marker_sizes = _conviction_colors(raw)

    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=x_axis, y=raw.clip(lower=0),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_raw.add_trace(go.Scatter(
        x=x_axis, y=raw.clip(upper=0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_raw.add_trace(go.Scatter(
        x=x_axis, y=raw, mode="lines+markers", name="Raw Conviction",
        line=dict(color=SLATE, width=1.5),
        marker=dict(size=marker_sizes, color=conv_colors),
    ))
    fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
    fig_raw.add_hline(y=40, line_dash="dot", line_color="rgba(251,113,133,0.18)", line_width=0.5, annotation_text="OB", annotation_position="right")
    fig_raw.add_hline(y=-40, line_dash="dot", line_color="rgba(52,211,153,0.18)", line_width=0.5, annotation_text="OS", annotation_position="right")

    fig_raw.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False))
    style_axes(fig_raw, y_title="Conviction", y_range=[-100, 100])
    st.plotly_chart(fig_raw, width='stretch', key="aarambh_raw_conviction")


def _render_ddm_conviction_chart(ts_filtered, x_axis, signal):
    """Section: DDM-Filtered Conviction with Confidence Bands + interpretation card."""
    fig_conv = go.Figure()
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
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"], mode="lines", name="DDM Conviction",
        line=dict(color=SLATE, width=2),
    ))
    fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)

    fig_conv.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False))
    style_axes(fig_conv, y_title="Conviction", y_range=[-100, 100])
    st.plotly_chart(fig_conv, width='stretch', key="aarambh_ddm")

    # Interpretation card
    cv = signal["conviction_score"]
    cu, cl = signal["conviction_upper"], signal["conviction_lower"]
    bw = cu - cl

    if cv > UI_CONVICTION_STRONG:
        regime_title = "STRONG OVERBOUGHT"
        regime_color = "danger"
        regime_body = (
            f"Conviction {cv:+.0f} — top decile. Most windows price the market above fair value. "
            f"Elevated drawdown risk from this zone. "
        )
    elif cv > UI_CONVICTION_MODERATE:
        regime_title = "MODERATE OVERBOUGHT"
        regime_color = "warning"
        regime_body = (
            f"Conviction {cv:+.0f} — tilts overbought. Not at extremes, but evidence suggests "
            f"fair value sits below current price. "
        )
    elif cv > -UI_CONVICTION_MODERATE:
        regime_title = "NEUTRAL"
        regime_color = "neutral"
        regime_body = (
            f"Conviction {cv:+.0f} — noise band. Windows are split — no reliable directional signal. "
            f"Stand aside or maintain current allocation. "
        )
    elif cv > -UI_CONVICTION_STRONG:
        regime_title = "MODERATE OVERSOLD"
        regime_color = "success"
        regime_body = (
            f"Conviction {cv:+.0f} — tilts oversold. Market prices in more pessimism than the "
            f"ensemble justifies. Watch for conviction to roll over before entering. "
        )
    else:
        regime_title = "STRONG OVERSOLD"
        regime_color = "success"
        regime_body = (
            f"Conviction {cv:+.0f} — bottom decile. Most windows agree the market is below fair value. "
            f"Historically the most favorable return regime. "
        )

    if bw < UI_BAND_NARROW:
        regime_body += f"Band narrow ({bw:.0f} pts) — model is confident."
    elif bw > UI_BAND_WIDE:
        regime_body += f"Band wide ({bw:.0f} pts) — models disagree, treat with caution."
    else:
        regime_body += f"Band moderate ({bw:.0f} pts) — some uncertainty."

    render_interpretation_card(title=regime_title, body=regime_body, color=regime_color)


def _render_market_breadth_chart(ts_filtered, x_axis):
    """Section: Market Breadth — oversold/overbought zone convergence."""
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
    fig_zones.add_hline(y=UI_BREADTH_HIGH, line_dash="dot", line_color="rgba(212,168,83,0.18)", line_width=0.5)

    fig_zones.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig_zones, y_title="Breadth %", y_range=[0, 100])
    st.plotly_chart(fig_zones, width='stretch', key="aarambh_breadth")


def _render_market_state_cards(signal, regime_stats, ts):
    """Section: Market State — metric cards + regime distribution interpretation."""
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "OVERSOLD BREADTH", f'{signal["oversold_breadth"]:.0f}%',
            "Fraction of models that see cheap valuation. Rising = bullish pressure building.",
            "success" if signal["oversold_breadth"] > UI_BREADTH_HIGH else "neutral",
            tooltip=TOOLTIPS["oversold_breadth"],
        )
    with c2:
        render_metric_card(
            "OVERBOUGHT BREADTH", f'{signal["overbought_breadth"]:.0f}%',
            "Fraction of models that see expensive valuation. Rising = caution strengthening.",
            "danger" if signal["overbought_breadth"] > UI_BREADTH_HIGH else "neutral",
            tooltip=TOOLTIPS["overbought_breadth"],
        )
    with c3:
        curr_regime = signal["regime"]
        regime_short = curr_regime.replace("STRONGLY ", "").replace("OVERSOLD", "OS").replace("OVERBOUGHT", "OB")
        regime_color = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else "neutral"
        render_metric_card(
            "CURRENT REGIME", regime_short,
            f"Mean reversion half-life: {signal['ou_half_life']:.0f}d. Shorter = faster snap-back to fair value.",
            regime_color,
            tooltip=TOOLTIPS["current_regime"],
        )

    # Regime distribution interpretation
    total = len(ts)
    os_total = regime_stats["strongly_oversold"] + regime_stats["oversold"]
    ob_total = regime_stats["strongly_overbought"] + regime_stats["overbought"]
    neutral_count = regime_stats["neutral"]
    os_pct = os_total / total * 100 if total > 0 else 0
    ob_pct = ob_total / total * 100 if total > 0 else 0
    neutral_pct = neutral_count / total * 100 if total > 0 else 0

    if os_pct > 50:
        interp_title = "MARKET LEANS CHEAP"
        interp_color = "success"
        interp_text = (
            f"{os_pct:.0f}% of history ({os_total}/{total} periods) classified oversold. "
            f"Valuation sits near the lower end of its range. "
            f"Neutral {neutral_pct:.0f}%, overbought {ob_pct:.0f}%."
        )
    elif ob_pct > 50:
        interp_title = "MARKET LEANS EXPENSIVE"
        interp_color = "danger"
        interp_text = (
            f"{ob_pct:.0f}% of history ({ob_total}/{total} periods) classified overbought. "
            f"Valuation sits near the upper end of its range. "
            f"Neutral {neutral_pct:.0f}%, oversold {os_pct:.0f}%."
        )
    else:
        interp_title = "MARKET OSCILLATES EVENLY"
        interp_color = "neutral"
        interp_text = (
            f"No dominant regime. Neutral {neutral_pct:.0f}% ({neutral_count} periods), "
            f"oversold {os_pct:.0f}%, overbought {ob_pct:.0f}%. "
            f"Mean-reversion signals equally likely in both directions."
        )
    render_interpretation_card(title=interp_title, body=interp_text, color=interp_color)


def _render_model_quality_cards(model_stats, signal):
    """Section: Model Quality — four metric cards (OOS R², R² vs RW, Hurst, Model Spread)."""
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        r2 = model_stats["r2_oos"]
        render_metric_card(
            "OOS R²", f"{r2:.3f}",
            "Variance explained on unseen data. Below 0.4 = model cannot reliably forecast.",
            "success" if r2 > UI_R2_STRONG else "warning" if r2 > UI_R2_ACCEPTABLE else "danger",
            tooltip=TOOLTIPS["oos_r2"],
        )
    with q2:
        r2_rw = model_stats.get("r2_vs_rw", 0.0)
        render_metric_card(
            "R² vs RW", f"{r2_rw:+.3f}",
            "Edge over naive forecast. Negative = 'tomorrow = today' would have beaten the model.",
            "success" if r2_rw > 0.05 else "warning" if r2_rw > -0.05 else "danger",
            tooltip=TOOLTIPS["r2_vs_rw"],
        )
    with q3:
        h = signal["hurst"]
        h_label = "Mean-reverting (signals reliable)" if h < 0.40 else "Trending (signals lag)" if h > 0.60 else "Random walk (no edge)"
        render_metric_card(
            "DFA Hurst", f"{h:.2f}", h_label,
            "success" if h < 0.40 else "danger" if h > 0.60 else "neutral",
            tooltip=TOOLTIPS["hurst"],
        )
    with q4:
        sp = signal["model_spread"]
        render_metric_card(
            "Model Spread", f"{sp:.2f}",
            "Disagreement among ensemble models. Above 1.5 = models conflict — distrust signal.",
            "success" if sp < UI_MODEL_SPREAD_LOW else "warning" if sp < UI_MODEL_SPREAD_HIGH else "danger",
            tooltip=TOOLTIPS["model_spread"],
        )


def _render_fair_value_chart(engine, ts_filtered, x_axis, ts, active_target):
    """Section: Actual vs Walk-Forward Fair Value — dual-panel chart."""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.62, 0.38],
        shared_xaxes=True, vertical_spacing=0.06,
    )
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["Actual"], mode="lines", name="Actual",
        line=dict(color=SLATE, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value (OOS)",
        line=dict(color=CYAN, width=1.5, dash="dot"),
    ), row=1, col=1)
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
    bar_colors = [EMERALD if r < 0 else ROSE for r in ts_filtered["Residual"]]
    fig.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["Residual"], name="Residual (OOS)",
        marker_color=bar_colors, showlegend=False, opacity=0.65,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=2, col=1)

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

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_XLARGE))
    style_axes(fig, y_title=active_target, row=1, col=1)
    style_axes(fig, y_title="Residual", row=2, col=1)

    st.plotly_chart(fig, width='stretch', key="aarambh_fairvalue")


def _render_signal_frequency_chart(ts_filtered, x_axis):
    """Section: Signal Frequency — buy/sell threshold crossings."""
    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy",
        marker=dict(color=EMERALD, opacity=0.85),
    ))
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell",
        marker=dict(color=ROSE, opacity=0.85),
    ))

    fig_signals.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False), barmode="relative")
    style_axes(fig_signals, y_title="Count")
    st.plotly_chart(fig_signals, width='stretch', key="aarambh_signal_freq")


def _render_avg_zscore_chart(ts_filtered, x_axis):
    """Section: Average Z-Score — statistical extremes across windows."""
    fig_z = go.Figure()
    bar_colors = [EMERALD if z < -1 else ROSE if z > 1 else "rgba(148,163,184,0.75)" for z in ts_filtered["AvgZ"]]
    fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered["AvgZ"], marker_color=bar_colors, opacity=0.85, showlegend=False))
    fig_z.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
    fig_z.add_hline(y=2, line_dash="dot", line_color="rgba(251,113,133,0.18)", line_width=0.5)
    fig_z.add_hline(y=-2, line_dash="dot", line_color="rgba(52,211,153,0.18)", line_width=0.5)

    fig_z.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False))
    style_axes(fig_z, y_title="Z-Score")
    st.plotly_chart(fig_z, width='stretch', key="aarambh_avg_z")


def _render_lookback_states(ts_filtered):
    """Section: Current Lookback States — per-window z-score and zone."""
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


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION — sections arranged in logical analytical flow
# ═══════════════════════════════════════════════════════════════════════

def render_aarambh_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target):
    """Aarambh tab — walk-forward valuation with violet system identity.

    Analytical flow:
      1. Model Quality        — "Can I trust this?"
      2. Actual vs Fair Value  — "What are we valuing?"
      3. Base Conviction       — "What's the raw signal?"
      4. DDM-Filtered          — "What's the processed signal?"
      5. Market Breadth        — "How much agreement?"
      6. Market State          — "What's the regime?"
      7. Lookback States       — "What do individual windows say?"
      8. Signal Frequency      — "Where are the crossings?"
      9. Average Z-Score       — "What's statistically extreme?"
    """

    st.markdown(
        '<div class="tab-bg aarambh"></div>',
        unsafe_allow_html=True,
    )

    # ── Phase 1: TRUST ─────────────────────────────────────────────────
    render_section_header(
        "Model Quality",
        "Is the ensemble reliable? Weak metrics here mean treat all fair-value estimates with caution.",
        icon="cpu",
        accent="violet",
    )
    _render_model_quality_cards(model_stats, signal)

    section_gap()

    # ── Phase 2: ANCHOR ────────────────────────────────────────────────
    render_section_header(
        "Actual vs Walk-Forward Fair Value",
        "Actual PE vs model's fair-value estimate. Tight band + large divergence = strongest signals.",
        icon="trending",
        accent="cyan",
    )
    _render_fair_value_chart(engine, ts_filtered, x_axis, ts, active_target)

    section_gap()

    # ── Phase 3: SIGNAL ────────────────────────────────────────────────
    render_section_header(
        "Base Conviction Score",
        "Unsmoothed oversold/overbought window differential. Feeds the DDM evidence accumulator.",
        icon="activity",
    )
    _render_raw_conviction_chart(ts_filtered, x_axis)

    section_gap()

    render_section_header(
        "DDM-Filtered Conviction with Confidence Bands",
        "Evidence-accumulated signal with confidence bands. Narrow = high conviction. This is your primary trade signal.",
        icon="shield",
        accent="cyan",
    )
    _render_ddm_conviction_chart(ts_filtered, x_axis, signal)

    section_gap()

    render_section_header(
        "Market Breadth",
        "Windows that agree the market is cheap (green) vs expensive (red). Convergence near zero = fairly valued.",
        icon="bar-chart",
        accent="emerald",
    )
    _render_market_breadth_chart(ts_filtered, x_axis)

    section_gap()

    # ── Phase 4: STATE ─────────────────────────────────────────────────
    render_section_header(
        "Market State",
        "How many windows see the market as cheap vs expensive, plus the mean-reversion regime.",
        icon="crosshair",
        accent="emerald",
    )
    _render_market_state_cards(signal, regime_stats, ts)

    section_gap()

    render_section_header(
        "Current Lookback States",
        "Per-window z-score and zone. Uniform zones = high conviction. Mixed = low conviction.",
        icon="grid",
    )
    _render_lookback_states(ts_filtered)

    section_gap()

    # ── Phase 5: EXTREMES ──────────────────────────────────────────────
    render_section_header(
        "Signal Frequency",
        "Buy/sell threshold crossings per window. Clusters = conviction building.",
        icon="zap",
        accent="rose",
    )
    _render_signal_frequency_chart(ts_filtered, x_axis)

    section_gap()

    render_section_header(
        "Average Z-Score",
        "Mean z-score across all windows. Beyond ±2 (dotted) is statistically extreme. Green = cheap; red = expensive.",
        icon="target",
    )
    _render_avg_zscore_chart(ts_filtered, x_axis)
