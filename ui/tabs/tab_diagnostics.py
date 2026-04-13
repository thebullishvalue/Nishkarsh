"""
Nishkarsh v1.2.0 — Diagnostics tab: ML diagnostics from both engines.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

UI — Model quality assessment: feature importance, residuals, walk-forward performance.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import chart_layout, style_axes
from ui.components import render_metric_card, render_section_header, section_gap
from core.config import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_MUTED,
)

# ── Alias colors for tab-local use ────────────────────────────────────────
EMERALD = COLOR_GREEN
ROSE = COLOR_RED
AMBER = COLOR_AMBER
CYAN = COLOR_CYAN
SLATE = COLOR_MUTED

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "ou_half_life": (
        "Expected time (in days) for the pricing residual to close halfway back to fair value "
        "after a shock. Shorter half-lives = faster mean reversion = more frequent opportunities."
    ),
    "adf_pvalue": (
        "Tests whether the pricing residual has a unit root (drifts away from fair value). "
        "p < 0.05 rejects the unit root, confirming mean-reversion."
    ),
    "kpss_pvalue": (
        "Corroborating test: checks whether the residual is stationary around a trend. "
        "p > 0.05 fails to reject stationarity — second confirmation of mean-reversion."
    ),
    "hmm_cov_shrinkage": (
        "Covariance regularization for the regime detection model. "
        "Prevents overfitting when estimating regime volatility from limited data."
    ),
    "viterbi_persist": (
        "Probability the current regime (bull/bear) persists into the next period. "
        "Near 1.0 = stable regimes; below 0.9 = frequent switching, lower signal confidence."
    ),
}


def render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats):
    """ML Diagnostics with rose system identity."""

    # System identity background
    st.markdown(
        '<div class="tab-bg diagnostics"></div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 1. OU MEAN-REVERSION DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "OU Mean-Reversion Diagnostics",
        "Tests whether the pricing residual is stationary — the foundation all mean-reversion signals depend on.",
        icon="crosshair",
        accent="cyan",
    )

    theta_status = "Stable" if signal.get("theta_stable", True) else "Unstable"
    stationarity = "Stationary" if signal["adf_pvalue"] < 0.05 and signal["kpss_pvalue"] > 0.05 else "Non-Stationary"

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("OU HALF-LIFE", f"{signal['ou_half_life']:.0f}d", "Days to close half the pricing gap", "info",
                           tooltip=TOOLTIPS["ou_half_life"])
    with c2:
        adf_class = "success" if signal["adf_pvalue"] < 0.05 else "danger"
        render_metric_card("ADF P-VALUE", f"{signal['adf_pvalue']:.3f}", "Rejects drift if p < 0.05", adf_class,
                           tooltip=TOOLTIPS["adf_pvalue"])
    with c3:
        kpss_class = "success" if signal["kpss_pvalue"] > 0.05 else "danger"
        render_metric_card("KPSS P-VALUE", f"{signal['kpss_pvalue']:.3f}", "Confirms mean-reversion if p > 0.05", kpss_class,
                           tooltip=TOOLTIPS["kpss_pvalue"])

    # Status indicators
    ok_svg = f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{COLOR_GREEN}" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>'
    warn_svg = f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{COLOR_AMBER}" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    stat_icon = ok_svg if "Stationary" in stationarity else warn_svg
    theta_icon = ok_svg if "Stable" in theta_status else warn_svg

    st.markdown(
        f'<div style="display:flex;gap:var(--sp-6);margin-top:var(--sp-3);">'
        f'<span style="font-family:var(--data);font-size:0.78rem;color:var(--ink-secondary);display:inline-flex;align-items:center;gap:0.4rem;">Stationarity: {stat_icon} {stationarity}</span>'
        f'<span style="font-family:var(--data);font-size:0.78rem;color:var(--ink-secondary);display:inline-flex;align-items:center;gap:0.4rem;">\u03b8 Stability: {theta_icon} {theta_status}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. FEATURE IMPACT
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Feature Impact on Fair Value",
        "How much each predictor shifts the fair-value estimate now. Top features drive the signal — if they go stale, the signal degrades.",
        icon="bar-chart",
        accent="violet",
    )

    feature_history = engine.get_feature_impact_history()
    if not feature_history.empty:
        if hasattr(engine, "latest_feature_impacts") and engine.latest_feature_impacts:
            impacts = engine.latest_feature_impacts
            labels = list(impacts.keys())[::-1]
            vals = list(impacts.values())[::-1]

            # Gradient color scale from light slate to bright slate based on relative contribution
            colors = []
            max_val = max(vals) if vals else 1
            for v in vals:
                intensity = v / max_val
                # Light slate (148,163,184) to brighter slate (180,195,215)
                r = int(148 + (180 - 148) * intensity)
                g = int(163 + (195 - 163) * intensity)
                b = int(184 + (215 - 184) * intensity)
                alpha = 0.75 + 0.25 * intensity
                colors.append(f"rgba({r},{g},{b},{alpha:.2f})")

            fig_imp = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors),
            ))
            fig_imp.update_layout(**chart_layout(height=max(260, len(labels) * 32), show_legend=False))
            fig_imp.update_xaxes(
                showgrid=True, gridcolor="rgba(255,255,255,0.035)", gridwidth=0.5,
                title_text="Contribution %", zeroline=True,
                zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5,
            )
            fig_imp.update_yaxes(showgrid=False)
            st.plotly_chart(fig_imp, width='stretch', key="diagnostics_feature_impact")

        if not feature_history.empty and len(feature_history) > 0:
            st.markdown(
                '<div style="font-family:var(--display);font-size:0.72rem;font-weight:600;color:var(--ink-tertiary);'
                'text-transform:uppercase;letter-spacing:0.08em;margin:var(--sp-4) 0 var(--sp-2) 0;">Impact History (last 10)</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(feature_history.tail(10), width='stretch', height=200)
    else:
        st.info("Feature impact data not available.")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SIGNAL PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Signal Performance",
        "Walk-forward hit rates across 5D, 10D, 20D forward return horizons.",
        icon="trending",
        accent="emerald",
    )

    perf = engine.get_signal_performance()
    perf_rows = []
    for period in (5, 10, 20):
        p = perf[period]
        buy_sig = "\u2713" if p["buy_p_value"] < 0.05 else "~" if p["buy_p_value"] < 0.10 else "\u2014"
        sell_sig = "\u2713" if p["sell_p_value"] < 0.05 else "~" if p["sell_p_value"] < 0.10 else "\u2014"
        perf_rows.append({
            "Period": f"{period}D",
            "Buy HR": f"{p['buy_hit'] * 100:.1f}%" if p["buy_count"] > 0 else "\u2014",
            "Buy Avg \u0394": f"{p['buy_avg']:.2f}%" if p["buy_count"] > 0 else "\u2014",
            "Buy t": f"{p['buy_t_stat']:.2f} {buy_sig}" if p["buy_count"] > 0 else "\u2014",
            "Buy N": p["buy_count"],
            "Sell HR": f"{p['sell_hit'] * 100:.1f}%" if p["sell_count"] > 0 else "\u2014",
            "Sell Avg \u0394": f"{p['sell_avg']:.2f}%" if p["sell_count"] > 0 else "\u2014",
            "Sell t": f"{p['sell_t_stat']:.2f} {sell_sig}" if p["sell_count"] > 0 else "\u2014",
            "Sell N": p["sell_count"],
        })
    st.dataframe(pd.DataFrame(perf_rows), width='stretch', height=160)

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. HMM TELEMETRY
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Regime Detection (HMM)",
        "How the Hidden Markov Model classifies the market over time. Sustained P > 0.5 = confident. Frequent crossings = uncertainty.",
        icon="eye",
        accent="rose",
    )

    c1, c2 = st.columns(2)
    with c1:
        render_metric_card("COVARIANCE SHRINKAGE", "1e-4", "Regularization strength", "warning",
                           tooltip=TOOLTIPS["hmm_cov_shrinkage"])
    with c2:
        render_metric_card("REGIME PERSISTENCE", "0.98", "Probability regime holds next period", "info",
                           tooltip=TOOLTIPS["viterbi_persist"])

    nirnay_df = st.session_state.get("nirnay_results", pd.DataFrame())
    if not nirnay_df.empty and "avg_hmm_bull" in nirnay_df.columns:
        fig_hmm = go.Figure()
        fig_hmm.add_trace(go.Scatter(
            x=nirnay_df.index, y=nirnay_df["avg_hmm_bull"],
            name="P(Bull)", line=dict(color=EMERALD, width=1.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        ))
        fig_hmm.add_trace(go.Scatter(
            x=nirnay_df.index, y=nirnay_df["avg_hmm_bear"],
            name="P(Bear)", line=dict(color=ROSE, width=1.5),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.08)",
        ))
        fig_hmm.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.08)", line_width=0.5)

        fig_hmm.update_layout(**chart_layout(height=300))
        style_axes(fig_hmm, y_title="State Probability", x_title=x_title, y_range=[0, 1])
        st.plotly_chart(fig_hmm, width='stretch', key="diagnostics_hmm_plot")
