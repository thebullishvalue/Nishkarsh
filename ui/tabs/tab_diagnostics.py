"""
Diagnostics tab — ML diagnostics from both engines.
Midnight Bloomberg Terminal design language.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_chart_theme
from ui.components import render_metric_card
from core.config import COLOR_AMBER, COLOR_GREEN, COLOR_RED, COLOR_CYAN


def render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats):
    """ML Diagnostics: OU diagnostics, feature impacts, signal performance."""

    st.markdown("##### OU Mean-Reversion Diagnostics")
    theta_status = "Stable" if signal.get("theta_stable", True) else "Unstable"
    stationarity = "Stationary" if signal['adf_pvalue'] < 0.05 and signal['kpss_pvalue'] > 0.05 else "Non-Stationary"
    ou_col1, ou_col2, ou_col3 = st.columns(3)
    with ou_col1:
        render_metric_card("OU Half-Life", f"{signal['ou_half_life']:.0f}d", "Andrews MU estimator", "info")
    with ou_col2:
        adf_class = "success" if signal['adf_pvalue'] < 0.05 else "danger"
        render_metric_card("ADF p-value", f"{signal['adf_pvalue']:.3f}", "Unit root test", adf_class)
    with ou_col3:
        kpss_class = "success" if signal['kpss_pvalue'] > 0.05 else "danger"
        render_metric_card("KPSS p-value", f"{signal['kpss_pvalue']:.3f}", "Stationarity test", kpss_class)
    st.markdown("")
    stat_icon_ok = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>'
    stat_icon_warn = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D4A853" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    stat_icon = stat_icon_ok if "Stationary" in stationarity else stat_icon_warn
    theta_icon = stat_icon_ok if "Stable" in theta_status else stat_icon_warn
    st.markdown(f'<span style="font-family:var(--data);font-size:0.82rem;color:var(--ink-secondary);display:inline-flex;align-items:center;gap:0.5rem;">Stationarity: {stat_icon} {stationarity} &nbsp;|&nbsp; θ Stability: {theta_icon} {theta_status}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ─── Feature Impact ────────────────────────────────────────────────
    st.markdown("##### Feature Impact")
    st.caption("Current predictor contributions to fair value estimation")
    feature_history = engine.get_feature_impact_history()
    if not feature_history.empty:
        if hasattr(engine, "latest_feature_impacts") and engine.latest_feature_impacts:
            impacts = engine.latest_feature_impacts
            labels = list(impacts.keys())[::-1]
            vals = list(impacts.values())[::-1]
            colors = []
            max_val = max(vals) if vals else 1
            for v in vals:
                intensity = v / max_val
                r = int(6 + (245 - 6) * intensity)
                g = int(182 + (158 - 182) * intensity)
                b = int(212 + (11 - 212) * intensity)
                colors.append(f"rgba({r},{g},{b},{0.6 + 0.4 * intensity})")
            fig_imp = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors),
            ))
            fig_imp.update_layout(
                height=max(260, len(labels) * 30),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=0.5, title="Contribution %", zeroline=True, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5),
                yaxis=dict(showgrid=False),
                margin=dict(t=10, l=10, r=10, b=35),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"), bordercolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig_imp, width='stretch', key="diagnostics_feature_impact")
        if not feature_history.empty and len(feature_history) > 0:
            st.markdown("###### Impact History (last 10)")
            st.dataframe(feature_history.tail(10), width='stretch', height=180)
    else:
        st.info("Feature impact data not available.")

    st.markdown("---")

    # ─── Signal Performance ────────────────────────────────────────────
    st.markdown("##### Signal Performance")
    st.caption("Hit rates and t-statistics for conviction-based signals")
    perf = engine.get_signal_performance()
    perf_rows = []
    for period in (5, 10, 20):
        p = perf[period]
        buy_sig = "✓" if p["buy_p_value"] < 0.05 else "~" if p["buy_p_value"] < 0.10 else "—"
        sell_sig = "✓" if p["sell_p_value"] < 0.05 else "~" if p["sell_p_value"] < 0.10 else "—"
        perf_rows.append({
            "Period": f"{period}D",
            "Buy HR": f"{p['buy_hit'] * 100:.1f}%" if p["buy_count"] > 0 else "—",
            "Buy Avg Δ": f"{p['buy_avg']:.2f}%" if p["buy_count"] > 0 else "—",
            "Buy t": f"{p['buy_t_stat']:.2f} {buy_sig}" if p["buy_count"] > 0 else "—",
            "Buy N": p["buy_count"],
            "Sell HR": f"{p['sell_hit'] * 100:.1f}%" if p["sell_count"] > 0 else "—",
            "Sell Avg Δ": f"{p['sell_avg']:.2f}%" if p["sell_count"] > 0 else "—",
            "Sell t": f"{p['sell_t_stat']:.2f} {sell_sig}" if p["sell_count"] > 0 else "—",
            "Sell N": p["sell_count"],
        })
    st.dataframe(pd.DataFrame(perf_rows), width='stretch', height=140)

    st.markdown("---")

    # ─── HMM Telemetry ─────────────────────────────────────────────────
    st.markdown("##### Regime Intelligence (HMM Telemetry)")
    st.caption("Hidden Markov Model state probabilities and shrinkage penalty")

    hmm_col1, hmm_col2 = st.columns(2)
    with hmm_col1:
        render_metric_card("HMM Covariance Shrinkage", "1e-4", "Ledoit-Wolf Diagonal", "warning")
    with hmm_col2:
        render_metric_card("Viterbi Persist", "0.98", "Transition Trace", "info")

    nirnay_df = st.session_state.get("nirnay_results", pd.DataFrame())
    if not nirnay_df.empty and "avg_hmm_bull" in nirnay_df.columns:
        fig_hmm = go.Figure()
        fig_hmm.add_trace(go.Scatter(
            x=nirnay_df.index, y=nirnay_df["avg_hmm_bull"],
            name="P(Bull)", line=dict(color=COLOR_GREEN, width=1.5),
        ))
        fig_hmm.add_trace(go.Scatter(
            x=nirnay_df.index, y=nirnay_df["avg_hmm_bear"],
            name="P(Bear)", line=dict(color=COLOR_RED, width=1.5),
        ))
        fig_hmm.update_layout(
            height=280,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)", gridwidth=0.5, title=x_title),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=0.5, title="State Probability", zeroline=True, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5),
            margin=dict(t=10, l=10, r=10, b=35),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#94A3B8"),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"), bordercolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_hmm, width='stretch', key="diagnostics_hmm_plot")
