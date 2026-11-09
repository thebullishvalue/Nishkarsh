"""
Nishkarsh v2 — Evidence tab: every figure behind the Overview, with a plain one-line gloss.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Leads with the one read — the Nishkarsh Oscillator's pre-registered measurement, its pillar ICs and
its ten components — then the components and diagnostics behind it: the Aarambh forecast input
(C, κ; ``build_forecast_readout``), the Edge Ledger, the signed agreement (Agreement pillar), how
C is built (zA/zN, Convergence Lift, agreement anatomy), model inputs and data health. Renders
pipeline outputs only — the UI grades nothing itself.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.config import COLOR_GREEN, COLOR_RED, EDGE_ALPHA
from convergence.nishkarsh import COMPONENTS, PILLARS
from ui.components import (
    _d, _str, build_forecast_readout, chip_html, component_label, esc, fmt, num, render_forecast_card,
    render_metric_card, render_section_header, signed, z_side, zbar_html,
)
from ui.copy import NKO_DOCS, NKO_PILLAR_PLAIN, NKO_SCALE_TEXT, NKO_STATUS, OSC_SCALE_TEXT, OSC_STATUS, mode_copy
from ui.tabs import tab_convergence as conv
from ui.tabs import tab_diagnostics as diag
from ui.theme import chart_layout, style_axes

GLOSSARY = (
    ("Nishkarsh read (NKO)", "The one series every surface infers from: −100 to +100, + = bullish; ±10 no lean, ±30 moderate, ±60 strong."),
    ("Pillar", "Top-down (Aarambh), bottom-up (Nirnay) and agreement: each the mean of its inputs' causal z-scores, re-standardized."),
    ("Single-system", "A date without Nirnay data: only the top-down pillar counts, and it is left out of the measurement."),
    ("Conviction (C)", "The Aarambh forecast input on a −100 to +100 scale; + leans higher. Tiers: 10 weak, 30 moderate, 60 strong."),
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
    render_section_header("Component: signed agreement (the Agreement pillar's input)",
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
    _yearly_chart(ev, "evi_osc_yearly")
    st.caption(f"Scale: {OSC_SCALE_TEXT}. Measured on its own as the former Agreement Oscillator (pre-registered "
               f"rule: p < {EDGE_ALPHA:g}, bootstrap CI excludes 0, ≥ 60% of years positive). It now enters the "
               "Nishkarsh read as the Agreement pillar.")


def _yearly_chart(ev: dict, key: str) -> None:
    yearly = {str(k): num(v) for k, v in _d(ev.get("yearly")).items() if num(v) is not None}
    if not yearly:
        return
    fig = go.Figure(go.Bar(x=list(yearly), y=list(yearly.values()),
                           marker_color=[COLOR_GREEN if v > 0 else COLOR_RED for v in yearly.values()]))
    fig.update_layout(**chart_layout(height=220, show_legend=False))
    style_axes(fig, y_title="IC by calendar year")
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def _render_nko_evidence(con: dict) -> None:
    render_section_header("The Nishkarsh read — measurement",
                          "The one series the Overview, the hero and the Precedent tab infer from, graded against "
                          "the Nifty's forward move by the rule fixed before it was tested.", icon="target",
                          accent="cyan")
    k = _d(con.get("nko"))
    if not k:
        st.info("The Nishkarsh read was not built in this run.")
        return
    h = int(num(con.get("horizon")) or 10)
    status = _str(k.get("status")).upper()
    label, cls, _ = NKO_STATUS.get(status, NKO_STATUS["INSUFFICIENT_DATA"])
    st.markdown(f'<div class="nk-evi-status">{chip_html(label, cls)}<span>{esc(_str(k.get("text")) or "—")}'
                f'</span></div>', unsafe_allow_html=True)
    ev = _d(k.get("evidence"))
    ci = ev.get("boot_ci") if isinstance(ev.get("boot_ci"), (list, tuple)) and len(ev.get("boot_ci")) == 2 else (None, None)
    cols = st.columns(5, gap="small")
    cards = (("Today", signed(k.get("value"), "+.0f"),
              f"{_str(k.get('tier')).lower() or '—'} · as of {_str(k.get('as_of')) or '—'}"
              + ("" if k.get("two_system") is True else " · single-system")),
             ("IC", signed(ev.get("ic"), "+.3f"), f"95% CI {signed(ci[0])}…{signed(ci[1])} · MDE {fmt(ev.get('mde'), '.2f')}"),
             ("Windows", fmt(ev.get("n"), ".0f"), f"non-overlapping, {h} days apart, two-system dates"),
             ("One-sided p", fmt(ev.get("p"), ".3f"), f"needs < {EDGE_ALPHA:g}"),
             ("HAC t", signed(ev.get("hac_t")), "autocorrelation-robust t"))
    for col, (lab, val, sub) in zip(cols, cards):
        with col:
            render_metric_card(lab, val, sub)
    _yearly_chart(ev, "evi_nko_yearly")
    st.markdown(f'<div class="nk-rule"><b>Pre-registered rule.</b> VALIDATED only if, on two-system dates, the '
                f'one-sided p for IC(read, {h}-day forward Nifty return) &gt; 0 is below {EDGE_ALPHA:g}, the '
                f'block-bootstrap CI excludes 0 and at least 60% of calendar years are positive; otherwise UNPROVEN, '
                f'and the product never calls it a buy/sell signal. Construction and rule were fixed before the '
                f'first measurement — see <code>{esc(NKO_DOCS[0])}</code> and the results in '
                f'<code>{esc(NKO_DOCS[1])}</code>. Scale: {esc(NKO_SCALE_TEXT)}.</div>', unsafe_allow_html=True)

    pic = _d(k.get("pillar_ic"))
    pil = _d(k.get("pillars"))
    prow = "".join(
        f'<tr><td><b>{esc(NKO_PILLAR_PLAIN[p][0])}</b> <span class="muted">{esc(NKO_PILLAR_PLAIN[p][1])}</span></td>'
        f'<td class="num {z_side(pil.get(p))}">{esc(signed(pil.get(p)))}</td>'
        f'<td class="num">{esc(signed(_d(pic.get(p)).get("ic"), "+.3f"))}</td>'
        f'<td class="num">{esc(fmt(_d(pic.get(p)).get("p"), ".2f"))}</td>'
        f'<td class="num">{esc(fmt(_d(pic.get(p)).get("n"), ".0f"))}</td></tr>' for p in PILLARS)
    st.markdown('<div class="nk-section-head sm"><h3>Pillars</h3><p>Today\'s pillar z-scores and each pillar\'s '
                'IC on the same windows — reported for context, not used for any decision.</p></div>'
                '<div class="nk-table-wrap"><table class="nk-table"><thead><tr><th>Pillar</th><th class="num">Today'
                '</th><th class="num">IC</th><th class="num">p</th><th class="num">Windows</th></tr></thead>'
                f'<tbody>{prow}</tbody></table></div>', unsafe_allow_html=True)

    comps = {_str(_d(c).get("id")): _d(c) for c in (k.get("components") or [])}
    drivers = {_str(_d(c).get("id")) for c in (k.get("drivers") or [])}
    crow = "".join(
        f'<tr><td>{esc(NKO_PILLAR_PLAIN[pil_][0])}</td><td><b>{esc(component_label(cid, lab))}</b>'
        + (' <span class="nk-driver-tag">driver</span>' if cid in drivers else "")
        + f'</td><td class="muted wrap">{esc(_str(comps.get(cid, {}).get("source")) or src)}</td>'
        f'<td class="bar">{zbar_html(comps.get(cid, {}).get("z"))}</td>'
        f'<td class="num {z_side(comps.get(cid, {}).get("z"))}">{esc(signed(comps.get(cid, {}).get("z")))}</td></tr>'
        for pil_, cid, lab, src in COMPONENTS)
    st.markdown('<div class="nk-section-head sm"><h3>The ten components</h3><p>Each input oriented so + = bullish '
                '(Nirnay read contrarian) and causally z-scored against its own past; no weight or sign is fitted '
                'to returns.</p></div>'
                '<div class="nk-table-wrap"><table class="nk-table"><thead><tr><th>Pillar</th><th>Component</th>'
                '<th>Source</th><th>Today</th><th class="num">z</th></tr></thead>'
                f'<tbody>{crow}</tbody></table></div>', unsafe_allow_html=True)


def render_evidence_tab(engine, signal: dict, start=None) -> None:
    """Evidence — the Nishkarsh read's measurement first, then its components and diagnostics."""
    ss = st.session_state
    con, rc = _d(ss.get("conclusion")), _d(ss.get("run_config"))
    forward = bool(rc.get("forward_mode", con.get("forward_mode", True)))
    mc = mode_copy(forward)
    st.markdown('<div class="nk-evi-intro">The full measurement behind the Overview. The Nishkarsh read comes '
                'first; everything below it is one of its inputs or a diagnostic of one. Every figure is computed '
                'on this run\'s data; the glossary explains each term in one line.</div>', unsafe_allow_html=True)
    _glossary()
    _render_nko_evidence(con)

    st.markdown('<div class="nk-divider"><span>Components and diagnostics</span></div>', unsafe_allow_html=True)
    render_section_header("Component: the Aarambh forecast input",
                          "The C/κ conclusion — Aarambh's forecast, shrunk by κ only when the lift test passes. It "
                          "enters the read as the top-down pillar's forecast input.", icon="cpu")
    render_forecast_card(build_forecast_readout(con, signal, rc))

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
