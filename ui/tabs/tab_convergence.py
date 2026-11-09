"""
Nishkarsh v2 — Convergence tab: the conclusion C over time, its inputs and its evidence.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Renders ONLY the conclusion layer's outputs (C-V2 / C-PIPE keys): ``conclusion_frame``
(C, C_aarambh, zA, zN, state, κ), ``conclusion_events``, the ``conclusion`` dict's lift /
κ / MDE / OOS read-out, and the agreement anatomy (``anatomy_df`` — descriptive, NOT used in
the conclusion). Sign convention: C, zA, zN  + = bullish.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.config import (
    COLOR_AMBER, COLOR_CYAN, COLOR_GREEN, COLOR_MUTED, COLOR_PURPLE, COLOR_RED,
    CONV2_TAU, CONV2_TIER_MODERATE, CONV2_TIER_STRONG, CONV2_TIER_WEAK, EDGE_ALPHA,
    UI_CHART_HEIGHT_MEDIUM, UI_CHART_HEIGHT_STACKED,
)
from ui.components import (
    CALL_TIERS, esc, fmt, render_interpretation_card, render_metric_card, render_section_header, section_gap,
)
from ui.copy import CONVERGENCE, LEVELS_TAG, SIGN_C, SIGN_Z, TIERS_TEXT
from ui.theme import chart_layout, style_axes

_EVENT_STYLE = {"CONTRADICTION": (COLOR_RED, "x"), "AARAMBH_LEADS": (COLOR_PURPLE, "triangle-up"),
                "NIRNAY_LEADS": (COLOR_CYAN, "diamond")}


def _since(df: pd.DataFrame | None, start) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if start is None:
        return df
    return df[pd.DatetimeIndex(df.index) >= pd.Timestamp(start)]


def _conclusion_chart(fr: pd.DataFrame, ev: pd.DataFrame) -> None:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.32, 0.18], vertical_spacing=0.05)
    x = fr.index
    fig.add_trace(go.Scatter(x=x, y=fr["C_aarambh"], name="C_aarambh (κ ≡ 1)", mode="lines",
                             line=dict(color=COLOR_MUTED, width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=fr["C"], name="C (conclusion)", mode="lines",
                             line=dict(color=COLOR_AMBER, width=1.8)), row=1, col=1)
    for lvl in (CONV2_TIER_WEAK, CONV2_TIER_MODERATE, CONV2_TIER_STRONG):
        for s in (1, -1):
            fig.add_hline(y=s * lvl, line_dash="dot", line_width=0.5, line_color="rgba(148,163,184,0.25)",
                          row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=fr["zA"], name="zA (Aarambh)", mode="lines",
                             line=dict(color=COLOR_PURPLE, width=1.3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=fr["zN"], name="zN (Nirnay)", mode="lines",
                             line=dict(color=COLOR_CYAN, width=1.1), connectgaps=False), row=2, col=1)
    for s in (1, -1):
        fig.add_hline(y=s * CONV2_TAU, line_dash="dot", line_width=0.5, line_color="rgba(148,163,184,0.3)",
                      row=2, col=1)
    if len(ev):
        for typ, (color, symbol) in _EVENT_STYLE.items():
            e = ev[ev["type"] == typ]
            if len(e):
                fig.add_trace(go.Scatter(x=e.index, y=e["zA"], mode="markers", name=typ.replace("_", " ").title(),
                                         marker=dict(color=color, symbol=symbol, size=6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=fr["kappa"], name="κ", mode="lines", line_shape="hv",
                             line=dict(color=COLOR_GREEN, width=1.2)), row=3, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED))
    style_axes(fig, y_title=CONVERGENCE["c_axis"], y_range=[-100, 100], row=1, col=1)
    style_axes(fig, y_title=CONVERGENCE["z_axis"], row=2, col=1)
    style_axes(fig, y_title=CONVERGENCE["kappa_axis"], y_range=[0, 1.05], row=3, col=1)
    st.plotly_chart(fig, width="stretch", key="conv_conclusion_chart")


def _evidence_cards(con: dict) -> None:
    lift, oos = con.get("lift") or {}, con.get("oos") or {}
    lat = con.get("latest") or {}
    kappa_on = con.get("kappa_enabled") is True
    if not lift.get("available"):
        l_status, l_cls = "N/A — too few windows" if not con.get("single_system") else "N/A — single system", "neutral"
    else:
        l_status, l_cls = ("PASSED", "success") if lift.get("passed") is True else ("NOT PROVEN", "neutral")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_metric_card("CONVERGENCE LIFT", fmt(lift.get("lift"), "+.3f"),
                           f"{l_status} · IC|confirm {fmt(lift.get('ic_confirm'), '+.3f')} (n {fmt(lift.get('n_confirm'), '.0f')})"
                           f" vs IC|contradict {fmt(lift.get('ic_contradict'), '+.3f')} (n {fmt(lift.get('n_contradict'), '.0f')})"
                           f" · one-sided p {fmt(lift.get('p_one_sided'), '.3f')}", l_cls,
                           tooltip="IC(zA, forward return | CONFIRM) − IC(… | CONTRADICT) on non-overlapping windows; "
                                   "passes only if one-sided p < α and IC|confirm > 0 (walk-forward).")
    with c2:
        render_metric_card("MDE (80% POWER)", fmt(lift.get("mde"), "+.3f"),
                           "smallest lift this sample could detect — a lift below this is unmeasurable here",
                           "neutral")
    with c3:
        render_metric_card("κ TODAY", fmt(lat.get("kappa"), ".2f"),
                           f"Intelligence {'ON' if kappa_on else 'OFF (κ ≡ 1)'} · test κ {fmt(lift.get('kappa'), '.2f')}"
                           " · κ < 1 only on CONTRADICT days after a passed test", "neutral")
    with c4:
        adds = oos.get("adds_value") is True
        render_metric_card("OOS · C vs AARAMBH", f"{fmt(oos.get('ic_C'), '+.3f')} vs {fmt(oos.get('ic_A'), '+.3f')}",
                           f"n {fmt(oos.get('n'), '.0f')} · "
                           + ("Nirnay improved out-of-sample IC" if adds else "no out-of-sample improvement"),
                           "success" if adds else "neutral")


def render_convergence_tab(start_date=None) -> None:
    """Conclusion tab — reads ``conclusion``, ``conclusion_frame``, ``conclusion_events``, ``anatomy_df``."""
    ss = st.session_state
    con = ss.get("conclusion") or {}
    frame = ss.get("conclusion_frame")
    st.markdown('<div class="tab-bg convergence"></div>', unsafe_allow_html=True)
    render_section_header(CONVERGENCE["title"], CONVERGENCE["desc"], icon="target")
    if frame is None or len(frame) == 0:
        st.info("No conclusion for this run (Aarambh produced no valid out-of-sample forecast).")
        return

    if con.get("forward_mode") is False:
        st.caption(f"Levels mode: zA is the valuation head — {LEVELS_TAG}.")
    lat = con.get("latest") or {}
    trust = con.get("trust") or {}
    call = str(trust.get("tier")) in CALL_TIERS
    direction = str(lat.get("direction") or "NEUTRAL")
    c_cls = ("success" if direction == "BULLISH" else "danger" if direction == "BEARISH" else "neutral") if call else "neutral"
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_metric_card("Conclusion C", fmt(lat.get("C"), "+.1f"),
                           f"{lat.get('tier') or 'NO LEAN'} · {direction}" + ("" if call else " · no call (unproven)")
                           + f" · {SIGN_C}", c_cls, tooltip=TIERS_TEXT)
    with c2:
        render_metric_card("C_aarambh", fmt(lat.get("C_aarambh"), "+.1f"), f"pure Aarambh (κ ≡ 1) · {SIGN_Z}")
    with c3:
        render_metric_card("zA · zN", f"{fmt(lat.get('zA'), '+.2f')} · {fmt(lat.get('zN'), '+.2f')}",
                           f"causal z of each head · {SIGN_Z}")
    with c4:
        agr = con.get("agreement") or {}
        render_metric_card("State today", str(lat.get("state") or "—").replace("_", " ").capitalize(),
                           f"60-session same-sign rate {fmt(agr.get('same_sign_rate_60d'), '.0%')} (descriptive)",
                           tooltip=CONVERGENCE["state_desc"])
    st.caption(f"As of {con.get('as_of') or '—'} · horizon {fmt(con.get('horizon'), '.0f')} trading days · "
               f"label {str(con.get('label_src') or '—')[:60]}")

    nir = con.get("nirnay") or {}
    render_interpretation_card("Nirnay", esc(nir.get("text") or "Nirnay: — (no read-out in this run)."), "info")

    section_gap()
    render_section_header("Conclusion Over Time",
                          f"Top: C and C_aarambh ({SIGN_Z}; dotted = fixed tiers ±{CONV2_TIER_WEAK:g}/"
                          f"{CONV2_TIER_MODERATE:g}/{CONV2_TIER_STRONG:g}). Middle: zA, zN ({SIGN_Z}; dotted = ±τ "
                          f"{CONV2_TAU:g}) with divergence events. Bottom: κ.", icon="activity", accent="cyan")
    fr = _since(frame, start_date)
    ev = _since(ss.get("conclusion_events"), start_date)
    if len(fr):
        _conclusion_chart(fr.astype({c: "float64" for c in ("C", "C_aarambh", "zA", "zN", "kappa")}), ev)
    st.caption(CONVERGENCE["state_desc"])

    section_gap()
    render_section_header("Convergence-Lift Evidence",
                          f"Does Nirnay's agreement improve Aarambh's accuracy? Graded walk-forward at α = {EDGE_ALPHA:g} "
                          "(one threshold scale).", icon="shield", accent="emerald")
    _evidence_cards(con)
    if len(ev):
        with st.expander(f"Divergence events in view ({len(ev)})", expanded=False):
            st.dataframe(ev.sort_index(ascending=False), width="stretch", height=240)

    section_gap()
    render_section_header(CONVERGENCE["anatomy_title"], CONVERGENCE["anatomy_desc"], icon="layers")
    an = _since(ss.get("anatomy_df"), start_date)
    if len(an) == 0:
        st.info("Agreement anatomy unavailable (no warm Nirnay dates in view).")
        return
    fig = go.Figure()
    styles = {"agreement_ratio": (COLOR_AMBER, 1.6, None), "dim_direction": (COLOR_PURPLE, 1, "dot"),
              "dim_breadth": (COLOR_GREEN, 1, "dot"), "dim_magnitude": (COLOR_CYAN, 1, "dot"),
              "dim_regime": (COLOR_RED, 1, "dot")}
    for col, (color, w, dash) in styles.items():
        if col in an.columns:
            fig.add_trace(go.Scatter(x=an.index, y=pd.to_numeric(an[col], errors="coerce"), mode="lines",
                                     name=col, line=dict(color=color, width=w, dash=dash)))
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig, y_title="agreement (0 = opposed · 1 = aligned)")
    st.plotly_chart(fig, width="stretch", key="conv_anatomy")
    st.caption("Descriptive — not used in the conclusion.")
