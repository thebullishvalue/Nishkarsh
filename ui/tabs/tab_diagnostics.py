"""
Diagnostics tab — ML diagnostics from both engines (exact match with correl.py).
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_chart_theme, COLOR_GOLD
from ui.components import render_metric_card


def render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats):
    """ML Diagnostics: OU diagnostics, feature impacts, signal performance."""

    st.markdown("##### OU Mean-Reversion Diagnostics")
    theta_status = "✅ Stable" if signal.get("theta_stable", True) else "⚠️ Unstable"
    stationarity = "Stationary ✅" if signal['adf_pvalue'] < 0.05 and signal['kpss_pvalue'] > 0.05 else "Non-Stationary ⚠️"
    ou_col1, ou_col2, ou_col3 = st.columns(3)
    with ou_col1:
        render_metric_card("OU Half-Life", f"{signal['ou_half_life']:.0f}d", "Andrews MU estimator", "primary")
    with ou_col2:
        adf_class = "success" if signal['adf_pvalue'] < 0.05 else "danger"
        render_metric_card("ADF p-value", f"{signal['adf_pvalue']:.3f}", "Unit root test", adf_class)
    with ou_col3:
        kpss_class = "success" if signal['kpss_pvalue'] > 0.05 else "danger"
        render_metric_card("KPSS p-value", f"{signal['kpss_pvalue']:.3f}", "Stationarity test", kpss_class)
    st.markdown("")
    st.markdown(f"**Stationarity:** {stationarity} | **θ Stability:** {theta_status}")
    st.markdown("---")

    st.markdown("##### Feature Impact")
    st.markdown('<p style="color: #888; font-size: 0.85rem;">Current predictor contributions to fair value estimation</p>', unsafe_allow_html=True)
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
                r = int(6 + (255 - 6) * intensity)
                g = int(182 + (212 - 182) * intensity)
                b = int(212 + (182 - 212) * intensity)
                colors.append(f"rgba({r},{g},{b},0.8)")
            fig_imp = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
            ))
            fig_imp.update_layout(
                height=max(280, len(labels) * 32),
                xaxis_title="Contribution %",
                xaxis=dict(tickfont=dict(size=9), zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(tickfont=dict(size=9)),
                margin=dict(t=10, l=10, r=20, b=10),
                showlegend=False,
            )
            apply_chart_theme(fig_imp)
            st.plotly_chart(fig_imp, width="stretch", key="diagnostics_feature_impact")
        if not feature_history.empty and len(feature_history) > 0:
            st.markdown("###### Impact History")
            st.dataframe(feature_history.tail(10), hide_index=True, height=200)
    else:
        st.info("Feature impact data not available for current configuration.")

    st.markdown("---")
    st.markdown("##### Signal Performance (with Significance)")
    st.markdown('<p style="color: #888;">Hit rates and t-statistics for conviction-based signals</p>', unsafe_allow_html=True)
    perf = engine.get_signal_performance()
    perf_rows = []
    for period in (5, 10, 20):
        p = perf[period]
        buy_sig = "✅" if p["buy_p_value"] < 0.05 else "⚠️" if p["buy_p_value"] < 0.10 else ""
        sell_sig = "✅" if p["sell_p_value"] < 0.05 else "⚠️" if p["sell_p_value"] < 0.10 else ""
        perf_rows.append({
            "Holding Period": f"{period} Days",
            "Buy Hit Rate": f"{p['buy_hit'] * 100:.1f}%" if p["buy_count"] > 0 else "N/A",
            "Buy Avg Fwd Chg": f"{p['buy_avg']:.2f}%" if p["buy_count"] > 0 else "N/A",
            "Buy t-stat": f"{p['buy_t_stat']:.2f} {buy_sig}" if p["buy_count"] > 0 else "N/A",
            "Buy Count": p["buy_count"],
            "Sell Hit Rate": f"{p['sell_hit'] * 100:.1f}%" if p["sell_count"] > 0 else "N/A",
            "Sell Avg Fwd Chg": f"{p['sell_avg']:.2f}%" if p["sell_count"] > 0 else "N/A",
            "Sell t-stat": f"{p['sell_t_stat']:.2f} {sell_sig}" if p["sell_count"] > 0 else "N/A",
            "Sell Count": p["sell_count"],
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(perf_rows), hide_index=True)
