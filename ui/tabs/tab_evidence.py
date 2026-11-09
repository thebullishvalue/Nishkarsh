"""
Nishkarsh v2 — Evidence tab: every figure behind the Overview, with a plain one-line gloss.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Absorbs the former Diagnostics and Convergence tabs: the verdict in detail (``build_hero_verdict``),
the Edge Ledger, the Agreement Oscillator's measurement, how the conclusion is built (C, κ,
zA/zN, Convergence Lift, agreement anatomy), model inputs and data health. Renders the
conclusion layer's outputs only — the UI grades nothing itself.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.config import COLOR_GREEN, COLOR_RED, EDGE_ALPHA
from ui.components import (
    _d, _str, build_hero_verdict, chip_html, esc, fmt, num, render_hero_card, render_metric_card,
    render_section_header, signed,
)
from ui.copy import AGREEMENT_BANDS_TEXT, OSC_SCALE_TEXT, OSC_STATUS, mode_copy
from ui.tabs import tab_convergence as conv
from ui.tabs import tab_diagnostics as diag
from ui.theme import chart_layout, style_axes

GLOSSARY = (
    ("Conviction (C)", "Aarambh's read on a −100 to +100 scale; + leans higher. Tiers: 10 weak, 30 moderate, 60 strong."),
    ("zA, zN", "Each system's reading as a z-score against its own past (+ = bullish)."),
    ("κ (kappa)", "Shrink factor: 1 = no change; below 1 only on disagreement days, and only after the lift test passes."),
    ("IC", "Information coefficient: rank correlation between a reading and the next 10 days' move; 0 = no link."),
    ("p-value", f"How surprising the result would be if there were no real link; below {EDGE_ALPHA:g} counts as evidence."),
    ("HAC t", "A t-statistic corrected for overlapping, autocorrelated days."),
    ("95% CI", "A range for the IC from resampling blocks of history (stationary bootstrap)."),
    ("MDE", "Minimum detectable effect: the smallest IC this much history can confirm with 80% power."),
    ("Convergence lift", "IC on days the systems agree minus IC on days they disagree."),
    ("Phase", "The test re-run from every possible start day; a wide range means the result is unstable."),
    ("Out-of-sample", "Measured only on history the models had not seen when they were fitted."),
    ("Sharpe, break-even", "Risk-adjusted return of trading the conclusion, and the cost per trade that would erase it."),
)


def _glossary() -> None:
    rows = "".join(f'<div class="nk-gloss-row"><b>{esc(k)}</b><span>{esc(v)}</span></div>' for k, v in GLOSSARY)
    with st.expander("Glossary — what the technical terms mean", expanded=False):
        st.markdown(f'<div class="nk-gloss">{rows}</div>', unsafe_allow_html=True)


def _render_oscillator_evidence(con: dict) -> None:
    render_section_header("Agreement Oscillator — measurement",
                          "Signed agreement = consensus (+1 both lean bullish, −1 both bearish, 0 split) × agreement "
                          "ratio, graded against the Nifty's forward move by the rule fixed before it was tested.",
                          icon="activity", accent="cyan")
    osc = _d(con.get("oscillator"))
    if not osc:
        st.info("The oscillator was not measured in this run.")
        return
    status = _str(osc.get("status")).upper()
    label, cls, _ = OSC_STATUS.get(status, OSC_STATUS["INSUFFICIENT_DATA"])
    st.markdown(f'<div class="nk-evi-status">{chip_html(label, cls)}<span>{esc(_str(osc.get("text")) or "—")}'
                f'</span></div>', unsafe_allow_html=True)
    ev = _d(osc.get("evidence"))
    ci = ev.get("boot_ci") if isinstance(ev.get("boot_ci"), (list, tuple)) and len(ev.get("boot_ci")) == 2 else (None, None)
    cols = st.columns(5, gap="small")
    cards = (("Today", signed(osc.get("value")), f"{_str(osc.get('side')).replace('-', ' ').lower() or '—'} · as of "
              f"{_str(osc.get('as_of')) or '—'}"),
             ("IC", signed(ev.get("ic"), "+.3f"),
              f"95% CI {signed(ci[0])}…{signed(ci[1])} · MDE {fmt(ev.get('mde'), '.2f')}"),
             ("Windows", fmt(ev.get("n"), ".0f"), "non-overlapping, h days apart"),
             ("One-sided p", fmt(ev.get("p"), ".3f"), f"needs < {EDGE_ALPHA:g}"),
             ("HAC t", signed(ev.get("hac_t")), "autocorrelation-robust t"))
    for col, (lab, val, sub) in zip(cols, cards):
        with col:
            render_metric_card(lab, val, sub)
    yearly = {str(k): num(v) for k, v in _d(ev.get("yearly")).items() if num(v) is not None}
    if yearly:
        fig = go.Figure(go.Bar(x=list(yearly), y=list(yearly.values()),
                               marker_color=[COLOR_GREEN if v > 0 else COLOR_RED for v in yearly.values()]))
        fig.update_layout(**chart_layout(height=220, show_legend=False))
        style_axes(fig, y_title="IC by calendar year")
        st.plotly_chart(fig, width="stretch", key="evi_osc_yearly", config={"displayModeBar": False})
    st.caption(f"Scale: {OSC_SCALE_TEXT}. Strength words on the Overview: {AGREEMENT_BANDS_TEXT}. Validated only if "
               f"p < {EDGE_ALPHA:g}, the bootstrap CI excludes 0 and at least 60% of years are positive "
               "(pre-registered); otherwise it carries no buy/sell wording.")


def render_evidence_tab(engine, signal: dict, start=None) -> None:
    """Evidence — verdict detail, Edge Ledger, oscillator measurement, conclusion build, inputs, data health."""
    ss = st.session_state
    con, rc = _d(ss.get("conclusion")), _d(ss.get("run_config"))
    forward = bool(rc.get("forward_mode", con.get("forward_mode", True)))
    mc = mode_copy(forward)
    st.markdown('<div class="nk-evi-intro">The full measurement behind the Overview. Every figure is computed on '
                'this run\'s data and graded out of sample; the glossary explains each term in one line.</div>',
                unsafe_allow_html=True)
    _glossary()

    render_section_header("The verdict in detail",
                          "The conclusion layer's full read-out for the latest date, with each piece of evidence.",
                          icon="target")
    prec = _d(ss.get("precedent"))
    render_hero_card(build_hero_verdict(con, signal, prec.get("hero"), rc))

    diag._render_edge_ledger(con)
    _render_oscillator_evidence(con)
    conv.render_convergence_tab(start)

    diag._render_feature_impact(engine, mc)
    if forward:
        st.caption(mc["levels_only_note"])
    else:
        diag._render_levels_only(engine, signal or {})
    diag._render_data_health(rc, _d(ss.get("run_report")))
    warns = [w for w in (rc.get("warnings") or []) if not str(w).upper().startswith("SYNTHETIC")]
    for w in warns:
        st.markdown(f'<div class="nk-note">{esc(w)}</div>', unsafe_allow_html=True)
