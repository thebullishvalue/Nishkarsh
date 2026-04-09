"""
Convergence tab — Unified signal with timeframe filtering.
Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import apply_chart_theme
from ui.components import render_metric_card
from core.config import COLOR_GREEN, COLOR_RED, COLOR_AMBER, COLOR_CYAN, COLOR_MUTED


def render_convergence_tab(ts_filtered=None):
    """Render the convergence dashboard tab."""
    convergence_df = st.session_state.get("convergence_df")
    divergence_events = st.session_state.get("divergence_events")
    nishkarsh_result = st.session_state.get("nishkarsh_result")
    aarambh_ts = st.session_state.get("aarambh_ts")
    nirnay_daily = st.session_state.get("nirnay_daily")

    if convergence_df is None or convergence_df.empty:
        st.info("No convergence data available. Run the analysis first.")
        return

    # ─── Header Section ──────────────────────────────────────────────────
    st.markdown("##### Convergence Analysis")
    st.caption("Cross-system agreement between Aarambh (top-down) and Nirnay (bottom-up) analytical frameworks")

    # ─── Metric Cards ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if nishkarsh_result:
            score = nishkarsh_result.nishkarsh_conviction
            signal = nishkarsh_result.nishkarsh_signal
            color = "success" if "BUY" in signal else "danger" if "SELL" in signal else "neutral"
            render_metric_card("NISHKARSH CONVICTION", f"{score:+.0f}", signal, color)
        else:
            render_metric_card("NISHKARSH CONVICTION", "N/A", "Not computed", "neutral")

    with col2:
        if aarambh_ts is not None and "ConvictionBounded" in aarambh_ts.columns:
            a_conv = aarambh_ts["ConvictionBounded"].iloc[-1]
            render_metric_card("AARAMBH CONVICTION", f"{a_conv:+.0f}", "Fair value breadth",
                             "success" if a_conv < -20 else "danger" if a_conv > 20 else "neutral")
        else:
            render_metric_card("AARAMBH CONVICTION", "N/A", "", "neutral")

    with col3:
        if nirnay_daily is not None and not nirnay_daily.empty:
            df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")]
            n_avg = 0.0
            if "avg_unified_osc" in df_n.columns:
                n_avg = df_n["avg_unified_osc"].iloc[-1]
            elif "Avg_Signal" in df_n.columns:
                n_avg = df_n["Avg_Signal"].iloc[-1]
            elif "avg_signal" in df_n.columns:
                n_avg = df_n["avg_signal"].iloc[-1]
            render_metric_card("NIRNAY AVG SIGNAL", f"{n_avg:.1f}", "Component oscillator",
                             "success" if n_avg < -3 else "danger" if n_avg > 3 else "neutral")
        else:
            render_metric_card("NIRNAY AVG SIGNAL", "N/A", "No constituent data", "neutral")

    with col4:
        agreement = convergence_df["agreement_ratio"].iloc[-1]
        render_metric_card("AGREEMENT", f"{agreement:.0%}", "Cross-system alignment",
                          "success" if agreement > 0.7 else "warning" if agreement > 0.5 else "neutral")

    st.markdown("---")

    # ─── Unified Normalized Signal ─────────────────────────────────────
    st.markdown("##### Unified Signal — Normalized Convergence")
    st.caption(
        "Both signals normalized to [-1, 1] via z-score. "
        "Average shows where systems converge. Negative = bullish, Positive = bearish."
    )

    if ts_filtered is not None and not ts_filtered.empty:
        if "Date" in ts_filtered.columns:
            filtered_dates = set(pd.to_datetime(ts_filtered["Date"]).dt.date.astype(str))
        else:
            filtered_dates = set(ts_filtered.index.astype(str))
    else:
        filtered_dates = None

    aligned_dates = []
    aligned_aarambh_raw = []
    aligned_nirnay_raw = []

    if nirnay_daily is not None and not nirnay_daily.empty:
        df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")].copy()
        col_map = {}
        for c in df_n.columns:
            cl = c.lower().replace("-", "_")
            if cl in ("avg_signal", "avg_unified_osc"):
                col_map[c] = "Avg_Signal"
        df_n = df_n.rename(columns=col_map)
        if "Avg_Signal" not in df_n.columns:
            df_n["Avg_Signal"] = 0

        nirnay_lookup = {}
        for idx in df_n.index:
            key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
            nirnay_lookup[key] = float(df_n.loc[idx].get("Avg_Signal", 0))

        if aarambh_ts is not None and "ConvictionRaw" in aarambh_ts.columns:
            aarambh_ts_dedup = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")].copy()
            if "Date" in aarambh_ts_dedup.columns:
                date_series = aarambh_ts_dedup["Date"]
            else:
                date_series = aarambh_ts_dedup.index

            for d_val in date_series:
                ts_date = d_val
                ts_key = str(ts_date.date()) if hasattr(ts_date, "date") else str(pd.Timestamp(ts_date).date())

                if filtered_dates is not None and ts_key not in filtered_dates:
                    continue

                if ts_key in nirnay_lookup:
                    aligned_dates.append(ts_date if hasattr(ts_date, "date") else pd.Timestamp(ts_key))
                    aligned_aarambh_raw.append(float(aarambh_ts_dedup.loc[ts_date, "ConvictionRaw"]))
                    aligned_nirnay_raw.append(nirnay_lookup[ts_key])

    if aligned_dates:
        # Normalization using full dataset params
        if "conv_norm_params" not in st.session_state:
            all_a, all_n = [], []
            if nirnay_daily is not None and not nirnay_daily.empty:
                df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")].copy()
                col_map = {}
                for c in df_n.columns:
                    cl = c.lower().replace("-", "_")
                    if cl in ("avg_signal", "avg_unified_osc"):
                        col_map[c] = "Avg_Signal"
                df_n = df_n.rename(columns=col_map)
                if "Avg_Signal" not in df_n.columns:
                    df_n["Avg_Signal"] = 0
                nirnay_lookup_full = {}
                for idx in df_n.index:
                    key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
                    nirnay_lookup_full[key] = float(df_n.loc[idx].get("Avg_Signal", 0))

                if aarambh_ts is not None and "ConvictionRaw" in aarambh_ts.columns:
                    aarambh_ts_dedup_full = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
                    for ts_idx in aarambh_ts_dedup_full.index:
                        ts_key = str(ts_idx.date()) if hasattr(ts_idx, "date") else str(pd.Timestamp(ts_idx).date())
                        if ts_key in nirnay_lookup_full:
                            all_a.append(float(aarambh_ts_dedup_full.loc[ts_idx, "ConvictionRaw"]))
                            all_n.append(nirnay_lookup_full[ts_key])

            arr_a_full = np.array(all_a, dtype=np.float64) if all_a else np.array([])
            arr_n_full = np.array(all_n, dtype=np.float64) if all_n else np.array([])
            mu_a = np.mean(arr_a_full) if len(arr_a_full) > 0 else 0.0
            sigma_a = np.std(arr_a_full) if len(arr_a_full) > 0 else 1.0
            mu_n = np.mean(arr_n_full) if len(arr_n_full) > 0 else 0.0
            sigma_n = np.std(arr_n_full) if len(arr_n_full) > 0 else 1.0
            st.session_state["conv_norm_params"] = {
                "mu_a": mu_a, "sigma_a": max(sigma_a, 1e-10),
                "mu_n": mu_n, "sigma_n": max(sigma_n, 1e-10),
            }

        params = st.session_state["conv_norm_params"]

        def apply_norm(x_raw, mu, sigma):
            if sigma < 1e-10:
                return np.zeros_like(x_raw)
            z = (x_raw - mu) / sigma
            return np.clip(z / 3.0, -1.0, 1.0)

        arr_a = np.array(aligned_aarambh_raw, dtype=np.float64)
        arr_n = np.array(aligned_nirnay_raw, dtype=np.float64)
        norm_a = apply_norm(arr_a, params["mu_a"], params["sigma_a"])
        norm_n = apply_norm(arr_n, params["mu_n"], params["sigma_n"])
        norm_avg = (norm_a + norm_n) / 2.0

        def dynamic_range(vals, padding=0.15):
            valid = [v for v in vals if v is not None and not np.isnan(v)]
            if not valid:
                return (-1, 1)
            mn, mx = min(valid), max(valid)
            span = mx - mn if mx != mn else 1.0
            pad = span * padding
            return (round(mn - pad, 2), round(mx + pad, 2))

        # Pre-compute Row 2
        aligned_conv_raw = []
        for d in aligned_dates:
            d_str = str(d.date()) if hasattr(d, "date") else str(d)
            if aarambh_ts is not None:
                aarambh_ts_dedup2 = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
                if d in aarambh_ts_dedup2.index:
                    aligned_conv_raw.append(float(aarambh_ts_dedup2.loc[d]["ConvictionRaw"]))
                elif "Date" in aarambh_ts_dedup2.columns:
                    mask = aarambh_ts_dedup2["Date"].astype(str).str.contains(d_str)
                    if mask.any():
                        aligned_conv_raw.append(float(aarambh_ts_dedup2.loc[mask, "ConvictionRaw"].iloc[0]))
                    else:
                        aligned_conv_raw.append(None)
                else:
                    aligned_conv_raw.append(None)
            else:
                aligned_conv_raw.append(None)

        aligned_nirnay_vals = [aligned_nirnay_raw[i] for i in range(len(aligned_dates))]
        unified_y = dynamic_range(norm_avg)
        conv_y = dynamic_range(aligned_conv_raw)
        nirnay_y = dynamic_range(aligned_nirnay_vals)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.50, 0.25, 0.25],
            vertical_spacing=0.05,
        )

        avg_colors = []
        avg_marker_sizes = []
        for v in norm_avg:
            if v < -0.5:
                avg_colors.append("#34D399"); avg_marker_sizes.append(7)
            elif v <= -0.3:
                avg_colors.append("rgba(52,211,153,0.6)"); avg_marker_sizes.append(5)
            elif v > 0.5:
                avg_colors.append("#FB7185"); avg_marker_sizes.append(7)
            elif v >= 0.3:
                avg_colors.append("rgba(251,113,133,0.6)"); avg_marker_sizes.append(5)
            else:
                avg_colors.append("rgba(148,163,184,0.4)"); avg_marker_sizes.append(4)

        # Row 1: Unified
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=np.clip(norm_avg, 0, None),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.08)",
            line=dict(width=0), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=np.clip(norm_avg, None, 0),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
            line=dict(width=0), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=norm_a,
            mode="lines", name="Aarambh",
            line=dict(color="rgba(212,168,83,0.3)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=norm_n,
            mode="lines", name="Nirnay",
            line=dict(color="rgba(34,211,238,0.25)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=norm_avg,
            mode="lines+markers", name="Convergence",
            line=dict(color="#D4A853", width=2),
            marker=dict(size=avg_marker_sizes, color=avg_colors),
        ), row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(251,113,133,0.2)", line_width=0.5, row=1, col=1)
        fig.add_hline(y=-0.5, line_dash="dash", line_color="rgba(52,211,153,0.2)", line_width=0.5, row=1, col=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=1, col=1)

        # Row 2: Conviction
        conv_colors = []
        marker_sizes = []
        for c in [v for v in aligned_conv_raw if v is not None]:
            if c > 40: conv_colors.append("#FB7185"); marker_sizes.append(6)
            elif c >= 20: conv_colors.append("rgba(251,113,133,0.5)"); marker_sizes.append(4)
            elif c < -40: conv_colors.append("#34D399"); marker_sizes.append(6)
            elif c <= -20: conv_colors.append("rgba(52,211,153,0.5)"); marker_sizes.append(4)
            else: conv_colors.append("rgba(148,163,184,0.4)"); marker_sizes.append(4)

        valid_mask = [v is not None for v in aligned_conv_raw]
        conv_colors_padded, marker_sizes_padded = [], []
        ci = 0
        for v in valid_mask:
            if v: conv_colors_padded.append(conv_colors[ci]); marker_sizes_padded.append(marker_sizes[ci]); ci += 1
            else: conv_colors_padded.append("rgba(148,163,184,0.3)"); marker_sizes_padded.append(3)

        conv_vals = [v if v is not None else None for v in aligned_conv_raw]

        fig.add_trace(go.Scatter(
            x=aligned_dates, y=np.clip(conv_vals, 0, None),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.06)", line=dict(width=0), showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=np.clip(conv_vals, None, 0),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.06)", line=dict(width=0), showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=conv_vals, mode="lines+markers", name="Base Conviction",
            line=dict(color="#D4A853", width=1.5),
            marker=dict(size=marker_sizes_padded, color=conv_colors_padded),
        ), row=2, col=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=2, col=1)
        fig.add_hline(y=40, line_dash="dash", line_color="rgba(251,113,133,0.15)", line_width=0.5, row=2, col=1)
        fig.add_hline(y=-40, line_dash="dash", line_color="rgba(52,211,153,0.15)", line_width=0.5, row=2, col=1)

        # Row 3: Nirnay Avg Signal
        nirnay_colors = ["#34D399" if v < -2 else "#FB7185" if v > 2 else "rgba(148,163,184,0.4)" for v in aligned_nirnay_vals]

        fig.add_trace(go.Scatter(
            x=aligned_dates, y=np.clip(aligned_nirnay_vals, 0, None),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.06)", line=dict(width=0), showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=np.clip(aligned_nirnay_vals, None, 0),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.06)", line=dict(width=0), showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=aligned_nirnay_vals, mode="lines+markers", name="Avg Signal",
            line=dict(color="#94A3B8", width=1.2),
            marker=dict(size=3, color=nirnay_colors),
        ), row=3, col=1)
        fig.add_hline(y=2, line_dash="dash", line_color="rgba(251,113,133,0.2)", line_width=0.5, row=3, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="rgba(52,211,153,0.2)", line_width=0.5, row=3, col=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=3, col=1)

        fig.update_layout(
            height=820, showlegend=False,
            hovermode="x unified",
            margin=dict(t=10, l=10, r=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10),
            hoverlabel=dict(
                bgcolor="rgba(10, 14, 23, 0.9)",
                font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
                bordercolor="rgba(255,255,255,0.1)",
            ),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", title_text="Normalized", range=unified_y, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", title_text="Conviction", range=conv_y, row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", title_text="Avg Signal", range=nirnay_y, row=3, col=1)

        st.plotly_chart(fig, width='stretch', key="convergence_overlay")
        st.caption(f"{len(aligned_dates)} trading days with overlapping data. All three rows share one x-axis.")
    else:
        st.warning("No overlapping dates between Aarambh and Nirnay data sources.")
