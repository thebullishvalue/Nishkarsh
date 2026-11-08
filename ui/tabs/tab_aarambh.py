"""
Nishkarsh v2 — Aarambh tab: the walk-forward engine, mode-aware (A4-08, A6-08, R3/R8).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

FORWARD mode (product path): the out-of-sample forecast of the h-day forward change is the
directional head; its edge is graded by the conclusion layer (Diagnostics → Edge Ledger).
LEVELS mode: a fair-value regression, labelled "valuation context — falsified as a forecaster".
All wording comes from ``ui.copy.mode_copy``; every axis states its sign convention
(engine conviction / AvgZ / Z_*: − = bullish; ForecastZ: + = expected rise).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.config import (
    COLOR_CYAN, COLOR_GREEN, COLOR_MUTED, COLOR_RED, LOOKBACK_WINDOWS, OU_PROJECTION_DAYS,
    UI_CHART_HEIGHT_MEDIUM, UI_CHART_HEIGHT_SMALL, UI_CHART_HEIGHT_XLARGE,
)
from ui.components import (
    CALL_TIERS, esc, fmt, num, render_interpretation_card, render_metric_card, render_section_header, section_gap,
)
from ui.copy import SIGN_ENGINE, mode_copy, regime_label, zone_label
from ui.theme import chart_layout, style_axes

_MUTED_FILL = "rgba(148,163,184,0.75)"


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def _quality_cards(model_stats: dict, signal: dict, ts: pd.DataFrame, mc: dict, forward: bool) -> None:
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        render_metric_card(mc["r2_label"], fmt(model_stats.get("r2_oos"), ".3f"), mc["r2_sub"], "neutral",
                           tooltip=mc["r2_tip"])
    with q2:
        render_metric_card(mc["r2rw_label"], fmt(model_stats.get("r2_vs_rw"), "+.3f"), mc["r2rw_sub"], "neutral",
                           tooltip=mc["r2rw_tip"])
    with q3:
        render_metric_card("DFA HURST", fmt(signal.get("hurst"), ".2f"), mc["hurst_sub"], "neutral",
                           tooltip="< 0.5 anti-persistent, ≈ 0.5 random walk, > 0.5 persistent increments. "
                                   "A description of the series, not evidence of edge.")
    with q4:
        # Scale-aware: the spread is quoted in σ of the series it spreads around (A4-08).
        base = _col(ts, "Forecast" if forward else "Residual")
        sd = num(base[_col(ts, "Valid").fillna(0).astype(bool)].std()) if "Valid" in ts.columns else num(base.std())
        sp = num(signal.get("model_spread"))
        rel = (sp / sd) if (sp is not None and sd) else None
        render_metric_card(mc["spread_label"], fmt(rel, ".2f", suffix="σ"),
                           f"{mc['spread_sub']} · raw {fmt(sp, '.4g')}", "neutral")


def _edge_pointer(signal: dict, con: dict, mc: dict, forward: bool) -> None:
    trust = con.get("trust") or {}
    tier = str(trust.get("tier") or "INSUFFICIENT_DATA")
    body = "Conclusion layer: " + esc(trust.get("text") or "no edge graded in this run.")
    if not forward:
        body += "<br>Engine levels test: " + esc(signal.get("edge_assessment") or "—")
    body += "<br>Full evidence: Diagnostics → Edge Ledger."
    color = "success" if tier in CALL_TIERS else "danger" if tier == "INVERTED" else "neutral"
    render_interpretation_card(mc["edge_title"], body, color)


def _anchor_chart(engine, ts_f: pd.DataFrame, ts: pd.DataFrame, x, target: str, signal: dict, mc: dict,
                  forward: bool) -> None:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], shared_xaxes=True, vertical_spacing=0.06)
    if forward:
        a, b = _col(ts_f, "Label"), _col(ts_f, "Forecast")
        lower = _col(ts_f, "ForecastZ")
        lower_colors = [COLOR_GREEN if v > 0 else COLOR_RED if v < 0 else _MUTED_FILL for v in lower.fillna(0)]
    else:
        a, b = _col(ts_f, "Actual"), _col(ts_f, "FairValue")
        lower = _col(ts_f, "Residual")
        # residual < 0 = below fair value = the engine's bullish pole (− = bullish)
        lower_colors = [COLOR_GREEN if v < 0 else COLOR_RED if v > 0 else _MUTED_FILL for v in lower.fillna(0)]
    fig.add_trace(go.Scatter(x=x, y=a, mode="lines", name=mc["anchor_a"], line=dict(color=COLOR_MUTED, width=1.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=b, mode="lines", name=mc["anchor_b"],
                             line=dict(color=COLOR_CYAN, width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=lower, name=mc["anchor_lower"], marker_color=lower_colors, opacity=0.65,
                         showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=2, col=1)
    proj = getattr(engine, "ou_projection", None)
    if (not forward and signal.get("half_life_meaningful") is True and proj is not None and len(proj) > 0
            and pd.api.types.is_datetime64_any_dtype(ts["Date"])):
        proj_dates = pd.bdate_range(start=ts["Date"].iloc[-1], periods=OU_PROJECTION_DAYS + 1)[1:]
        fig.add_trace(go.Scatter(x=proj_dates[:len(proj)], y=list(proj)[:len(proj_dates)], mode="lines", name="OU projection",
                                 line=dict(color=COLOR_MUTED, width=1, dash="dot"), opacity=0.45), row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_XLARGE))
    style_axes(fig, y_title=(f"{mc['anchor_y']} {target}" if forward else target), row=1, col=1)
    style_axes(fig, y_title=mc["anchor_lower"], row=2, col=1)
    st.plotly_chart(fig, width="stretch", key="aarambh_anchor")


def _conviction_chart(ts_f: pd.DataFrame, x, signal: dict, forward: bool) -> None:
    fig = go.Figure()
    score = _col(ts_f, "ConvictionScore")
    if not forward and {"ConvictionUpper", "ConvictionLower"} <= set(ts_f.columns):
        fig.add_trace(go.Scatter(x=x, y=_col(ts_f, "ConvictionUpper"), mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=_col(ts_f, "ConvictionLower"), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(148,163,184,0.08)", name="uncertainty band"))
    fig.add_trace(go.Scatter(x=x, y=score.clip(lower=0), fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=score.clip(upper=0), fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=score, mode="lines", name="engine conviction", line=dict(color=COLOR_MUTED, width=2)))
    levels = signal.get("conviction_levels") or {}
    for name in ("moderate", "strong"):
        lv = num(levels.get(name))
        if lv:
            for s in (1, -1):
                fig.add_hline(y=s * lv, line_dash="dot", line_width=0.5, line_color="rgba(148,163,184,0.22)")
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=not forward))
    style_axes(fig, y_title=f"conviction ({SIGN_ENGINE})", y_range=[-100, 100])
    st.plotly_chart(fig, width="stretch", key="aarambh_conviction")
    if not forward and "ConvictionRaw" in ts_f.columns:
        raw = go.Figure(go.Scatter(x=x, y=_col(ts_f, "ConvictionRaw"), mode="lines", name="raw conviction",
                                   line=dict(color=COLOR_MUTED, width=1.2)))
        raw.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False))
        style_axes(raw, y_title=f"raw (pre-DDM) · {SIGN_ENGINE}", y_range=[-100, 100])
        st.plotly_chart(raw, width="stretch", key="aarambh_raw_conviction")


def _conviction_card(signal: dict, con: dict, forward: bool) -> None:
    cv = num(signal.get("conviction_score"))
    strength = str(signal.get("display_strength") or "—")
    tier = str((con.get("trust") or {}).get("tier") or "INSUFFICIENT_DATA")
    side = "—" if cv is None else ("bullish side" if cv < 0 else "bearish side" if cv > 0 else "flat")
    body = (f"Engine conviction {esc(fmt(cv, '+.1f'))} ({esc(SIGN_ENGINE)}) — {esc(side)}, engine strength "
            f"{esc(strength)}. Directional trust for this run: {esc(tier.replace('_', ' '))}.")
    if not forward:
        lo, hi = num(signal.get("conviction_lower")), num(signal.get("conviction_upper"))
        if lo is not None and hi is not None:
            body += f" Uncertainty band width {esc(fmt(hi - lo, '.0f'))} pts (not a calibrated interval)."
    render_interpretation_card("ENGINE CONVICTION", body, "neutral")


def _extension_section(ts_f: pd.DataFrame, x, mc: dict, forward: bool) -> None:
    render_section_header(mc["ext_title"], mc["ext_desc"], icon="bar-chart", accent="emerald")
    if {"OversoldBreadth", "OverboughtBreadth"} <= set(ts_f.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=_col(ts_f, "OversoldBreadth"), fill="tozeroy", fillcolor="rgba(52,211,153,0.1)",
                                 line=dict(color=COLOR_GREEN, width=1.5), name=mc["breadth_bull"]))
        fig.add_trace(go.Scatter(x=x, y=_col(ts_f, "OverboughtBreadth"), fill="tozeroy",
                                 fillcolor="rgba(251,113,133,0.1)", line=dict(color=COLOR_RED, width=1.5),
                                 name=mc["breadth_bear"]))
        fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
        style_axes(fig, y_title="% of lookback windows", y_range=[0, 100])
        st.plotly_chart(fig, width="stretch", key="aarambh_breadth")
    rows = []
    for lb in LOOKBACK_WINDOWS:
        if f"Z_{lb}" not in ts_f.columns or not len(ts_f):
            continue
        z = num(ts_f[f"Z_{lb}"].iloc[-1])
        zone = ts_f[f"Zone_{lb}"].iloc[-1] if f"Zone_{lb}" in ts_f.columns else ""
        color = COLOR_GREEN if "Under" in str(zone) else COLOR_RED if "Over" in str(zone) else COLOR_MUTED
        rows.append(f'<div class="lookback-row"><span class="label">{lb}-day lookback</span>'
                    f'<span class="value" style="color:{color};">{esc(zone_label(zone, forward))} '
                    f'({esc(fmt(z, "+.2f"))})</span></div>')
    if rows:
        st.caption(mc["lookback_desc"])
        st.markdown('<div style="background:var(--glass);border:1px solid var(--border);border-radius:var(--r-md);'
                    f'overflow:hidden;">{"".join(rows)}</div>', unsafe_allow_html=True)
    if "AvgZ" in ts_f.columns:
        z = _col(ts_f, "AvgZ")
        fig = go.Figure(go.Bar(x=x, y=z, opacity=0.85, showlegend=False,
                               marker_color=[COLOR_GREEN if v < -1 else COLOR_RED if v > 1 else _MUTED_FILL
                                             for v in z.fillna(0)]))
        for s in (2, -2):
            fig.add_hline(y=s, line_dash="dot", line_width=0.5, line_color="rgba(148,163,184,0.22)")
        fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False))
        style_axes(fig, y_title=f"AvgZ ({SIGN_ENGINE})")
        st.caption(mc["avgz_desc"])
        st.plotly_chart(fig, width="stretch", key="aarambh_avg_z")
    if {"BuySignalBreadth", "SellSignalBreadth"} <= set(ts_f.columns):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=_col(ts_f, "BuySignalBreadth"), name="bullish crossings",
                             marker=dict(color=COLOR_GREEN, opacity=0.85)))
        fig.add_trace(go.Bar(x=x, y=-_col(ts_f, "SellSignalBreadth"), name="bearish crossings",
                             marker=dict(color=COLOR_RED, opacity=0.85)))
        fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False), barmode="relative")
        style_axes(fig, y_title="crossings (bearish below 0)")
        st.caption(mc["freq_desc"])
        st.plotly_chart(fig, width="stretch", key="aarambh_signal_freq")


def _state_section(signal: dict, regime_stats: dict, mc: dict, forward: bool) -> None:
    render_section_header(mc["regime_title"], mc["regime_desc"], icon="crosshair", accent="emerald")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(mc["breadth_bull"].upper(), fmt(signal.get("oversold_breadth"), ".0f", suffix="%"),
                           "share of lookbacks in the bullish zone (descriptive)")
    with c2:
        render_metric_card(mc["breadth_bear"].upper(), fmt(signal.get("overbought_breadth"), ".0f", suffix="%"),
                           "share of lookbacks in the bearish zone (descriptive)")
    with c3:
        sub = "classification of today's extension"
        if not forward and signal.get("half_life_meaningful") is True:
            sub = f"OU half-life {fmt(signal.get('ou_half_life'), '.0f')}d (levels fit)"
        render_metric_card("CURRENT STATE", regime_label(signal.get("regime"), forward), sub)
    rs = regime_stats or {}
    os_t = int((num(rs.get("strongly_oversold")) or 0) + (num(rs.get("oversold")) or 0))
    ob_t = int((num(rs.get("strongly_overbought")) or 0) + (num(rs.get("overbought")) or 0))
    nt = int(num(rs.get("neutral")) or 0)
    total = os_t + ob_t + nt
    if total:
        render_interpretation_card(
            "HISTORY SHARE (DESCRIPTIVE)",
            esc(f"Of {total} classified out-of-sample rows: {os_t / total:.0%} {mc['regime_bull'].lower()}, "
                f"{ob_t / total:.0%} {mc['regime_bear'].lower()}, {nt / total:.0%} neutral. "
                "A description of where the series has sat — not a forecast."), "neutral")


def render_aarambh_tab(engine, ts_f: pd.DataFrame, ts: pd.DataFrame, signal: dict, model_stats: dict,
                       regime_stats: dict, target: str, forward_mode: bool, conclusion: dict | None = None) -> None:
    """Aarambh tab. ``ts_f`` = timeframe slice (RangeIndex + Date), ``ts`` = full engine frame."""
    forward = bool(forward_mode)
    mc = mode_copy(forward)
    con = conclusion or {}
    signal, model_stats = signal or {}, model_stats or {}
    st.markdown('<div class="tab-bg aarambh"></div>', unsafe_allow_html=True)
    render_section_header("Model Quality", mc["quality_desc"], icon="cpu", accent="violet")
    st.caption(mc["badge"])
    _quality_cards(model_stats, signal, ts, mc, forward)
    _edge_pointer(signal, con, mc, forward)
    if len(ts_f) == 0:
        st.info("No rows in the selected time range.")
        return
    x = ts_f["Date"]
    section_gap()
    render_section_header(mc["anchor_title"], mc["anchor_desc"], icon="trending", accent="cyan")
    _anchor_chart(engine, ts_f, ts, x, target, signal, mc, forward)
    section_gap()
    render_section_header(mc["conv_title"], mc["conv_desc"], icon="shield", accent="cyan")
    _conviction_chart(ts_f, x, signal, forward)
    _conviction_card(signal, con, forward)
    section_gap()
    _extension_section(ts_f, x, mc, forward)
    section_gap()
    _state_section(signal, regime_stats, mc, forward)
