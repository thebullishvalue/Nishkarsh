"""
Nishkarsh v2 — Diagnostics tab: the Edge Ledger, model inputs, and data-layer provenance.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

The EDGE LEDGER renders the conclusion layer's evidence fields verbatim (``conclusion`` dict,
C-V2): full-history Aarambh-head IC, out-of-sample C IC, Convergence Lift (with MDE / power),
the cost-aware backtest (incl. break-even), the signal chain and the trust tier. A row reads
PROVEN / PASSED / TRADEABLE only when the conclusion's own boolean says so; the threshold scale
is core/config ``EDGE_*`` — the UI grades nothing itself. Data Layer Health is the run's
provenance (``run_config["provenance"]``).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import (
    BACKTEST_REFERENCE_BPS, COLOR_GREEN, COLOR_RED, EDGE_ALPHA, EDGE_TRADEABLE_T,
)
from ui.components import (
    esc, fmt, num, render_interpretation_card, render_metric_card, render_section_header, section_gap,
)
from ui.copy import EDGE_SCALE_TEXT, LEVELS_TAG, SIGN_Z, mode_copy
from ui.theme import chart_layout, style_axes

_PROV_CLASS = {"live": "success", "cache": "info", "stale": "warning", "fallback": "warning", "empty": "danger"}


def _edge_status(e: dict) -> tuple[str, str]:
    """Status word for an IC evidence block — PROVEN only when the conclusion says ``significant``."""
    if not e or not e.get("available"):
        return "INSUFFICIENT DATA", "neutral"
    if e.get("significant") is True:
        return "PROVEN (p < α)", "success"
    ic, p = num(e.get("ic")), num(e.get("p"))
    if ic is not None and ic < 0 and p is not None and (1.0 - p) < EDGE_ALPHA:
        return "INVERTED", "danger"
    return "NOT PROVEN", "neutral"


def _edge_row(title: str, e: dict) -> None:
    status, cls = _edge_status(e)
    ci = e.get("boot_ci") if isinstance(e.get("boot_ci"), (list, tuple)) and len(e.get("boot_ci")) == 2 else (None, None)
    render_metric_card(
        title, fmt(e.get("ic"), "+.3f"),
        f"{status} · n {fmt(e.get('n'), '.0f')} non-overlapping · t {fmt(e.get('t'), '+.2f')} · p {fmt(e.get('p'), '.3f')}"
        f" · HAC t {fmt(e.get('hac_t'), '+.2f')} · 95% CI [{fmt(ci[0], '+.3f')}, {fmt(ci[1], '+.3f')}]"
        f" · phase {fmt(e.get('phase_min'), '+.3f')}…{fmt(e.get('phase_max'), '+.3f')}"
        f" · momentum baseline {fmt(e.get('baseline_momentum_ic'), '+.3f')}", cls,
        tooltip=("Spearman IC of the signal (+ = bullish) vs the forward return on h-spaced rows; one-sided "
                 "p for IC > 0; HAC = Newey–West t of the daily rank IC; CI = stationary block bootstrap."))


def _render_edge_ledger(con: dict) -> None:
    render_section_header("Edge Ledger",
                          "Every edge word in this app is backed by a row here, computed on the displayed series "
                          f"at the displayed horizon. {EDGE_SCALE_TEXT}", icon="shield", accent="emerald")
    if not con:
        st.info("No conclusion in this run — nothing to grade.")
        return
    trust = con.get("trust") or {}
    tier = str(trust.get("tier") or "INSUFFICIENT_DATA")
    render_interpretation_card(
        f"TRUST · {tier.replace('_', ' ')}", esc(trust.get("text") or "—"),
        "success" if tier in ("DIRECTIONAL_EDGE", "TRADEABLE_EDGE") else "danger" if tier == "INVERTED" else "neutral")
    st.caption(f"Horizon {fmt(con.get('horizon'), '.0f')}d · label: {con.get('label_src') or '—'} · "
               f"as of {con.get('as_of') or '—'} · signs {SIGN_Z}"
               + ("" if con.get("forward_mode") is not False else f" · levels mode: {LEVELS_TAG}"))

    c1, c2 = st.columns(2, gap="small")
    with c1:
        _edge_row("AARAMBH HEAD IC · FULL HISTORY", con.get("edge") or {})
    with c2:
        _edge_row("CONCLUSION C IC · OUT-OF-SAMPLE", con.get("edge_C") or {})

    lift, oos = con.get("lift") or {}, con.get("oos") or {}
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if not lift.get("available"):
            ls, lc = "NOT RUN / INSUFFICIENT", "neutral"
        else:
            ls, lc = ("PASSED", "success") if lift.get("passed") is True else ("NOT PROVEN", "neutral")
        render_metric_card(
            "CONVERGENCE LIFT", fmt(lift.get("lift"), "+.3f"),
            f"{ls} · IC|confirm {fmt(lift.get('ic_confirm'), '+.3f')} (n {fmt(lift.get('n_confirm'), '.0f')}) vs "
            f"IC|contradict {fmt(lift.get('ic_contradict'), '+.3f')} (n {fmt(lift.get('n_contradict'), '.0f')}) · "
            f"z {fmt(lift.get('z'), '+.2f')} · one-sided p {fmt(lift.get('p_one_sided'), '.3f')} · "
            f"MDE (80% power) {fmt(lift.get('mde'), '.3f')} · phase {fmt(lift.get('lift_phase_min'), '+.3f')}…"
            f"{fmt(lift.get('lift_phase_max'), '+.3f')} · κ {fmt(lift.get('kappa'), '.2f')}", lc)
    with c2:
        adds = oos.get("adds_value") is True
        render_metric_card(
            "OUT-OF-SAMPLE · C vs AARAMBH", f"{fmt(oos.get('ic_C'), '+.3f')} vs {fmt(oos.get('ic_A'), '+.3f')}",
            f"{'IMPROVED' if adds else 'NO IMPROVEMENT'} · n {fmt(oos.get('n'), '.0f')} · requires a passed latest "
            "refit, κ enabled and IC_C ≥ IC_A", "success" if adds else "neutral")

    bt = con.get("backtest") or {}
    by_cost = bt.get("by_cost") or {}
    tradeable = tier == "TRADEABLE_EDGE"
    rows = []
    for k in sorted(by_cost, key=lambda s: num(s) if num(s) is not None else 1e9):
        b = by_cost.get(k) or {}
        rows.append({"cost (bps, one-way)": fmt(k, "g"), "gross Sharpe": fmt(b.get("gross_sharpe"), "+.2f"),
                     "net Sharpe": fmt(b.get("net_sharpe"), "+.2f"), "HAC t": fmt(b.get("t"), "+.2f"),
                     "ann. return": fmt(b.get("ann_return"), "+.2%"),
                     "turnover / yr": fmt(b.get("turnover_per_year"), ".1f"),
                     "break-even bps": fmt(b.get("breakeven_bps"), ".1f"), "n": fmt(b.get("n"), ".0f")})
    no = bt.get("nonoverlap_reference") or {}
    if no:
        rows.append({"cost (bps, one-way)": f"{fmt(bt.get('reference_bps'), 'g')} · non-overlap",
                     "gross Sharpe": fmt(no.get("gross_sharpe"), "+.2f"), "net Sharpe": fmt(no.get("net_sharpe"), "+.2f"),
                     "HAC t": fmt(no.get("t"), "+.2f"), "ann. return": fmt(no.get("ann_return"), "+.2%"),
                     "turnover / yr": fmt(no.get("turnover_per_year"), ".1f"),
                     "break-even bps": fmt(no.get("breakeven_bps"), ".1f"), "n": fmt(no.get("n"), ".0f")})
    st.markdown(f"**Cost-aware backtest** · position = C/100 · mode {bt.get('mode') or '—'} · graded at "
                f"{BACKTEST_REFERENCE_BPS:g} bps: **{'TRADEABLE' if tradeable else 'NOT TRADEABLE'}** "
                f"(needs HAC t and non-overlap t ≥ {EDGE_TRADEABLE_T:g} and a proven IC).")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    chain = [c for c in (con.get("chain") or []) if isinstance(c, dict)]
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown("**Signal chain** (same rows, h-spaced IC, + = bullish)")
        if chain:
            st.dataframe(pd.DataFrame([{"stage": c.get("stage"), "IC": fmt(c.get("ic"), "+.3f"),
                                        "n": fmt(c.get("n"), ".0f")} for c in chain]),
                         width="stretch", hide_index=True)
    with c2:
        yearly = (con.get("edge") or {}).get("yearly") or {}
        ys = {str(k): num(v) for k, v in yearly.items() if num(v) is not None}
        if ys:
            fig = go.Figure(go.Bar(x=list(ys), y=list(ys.values()),
                                   marker_color=[COLOR_GREEN if v > 0 else COLOR_RED for v in ys.values()]))
            fig.update_layout(**chart_layout(height=240, show_legend=False))
            style_axes(fig, y_title="daily rank IC by year (+ = right way)")
            st.plotly_chart(fig, width="stretch", key="diag_yearly_ic")


def _render_feature_impact(engine, mc: dict) -> None:
    render_section_header(mc["feature_title"], mc["feature_desc"], icon="bar-chart", accent="violet")
    impacts = getattr(engine, "latest_feature_impacts", None) or {}
    if not impacts:
        st.info("Feature impact data not available.")
        return
    names = list(impacts.keys())
    is_pca = any(str(n).startswith("MACRO_PC") for n in names)
    roster = " · ".join(esc(n) for n in names[:18]) + (" …" if len(names) > 18 else "")
    lead = (f"<b>{len(names)} factors</b> (causal-PCA, orthogonal — not raw predictors)" if is_pca
            else f"<b>{len(names)} inputs</b> feeding this run")
    st.markdown(f'<div style="font-family:var(--data);font-size:0.72rem;color:var(--ink-tertiary);'
                f'margin:-0.2rem 0 0.6rem 0;line-height:1.5;">{lead}: {roster}</div>', unsafe_allow_html=True)
    vals = [num(v) or 0.0 for v in impacts.values()][::-1]
    fig = go.Figure(go.Bar(x=vals, y=[str(n) for n in names][::-1], orientation="h",
                           marker=dict(color="rgba(148,163,184,0.85)")))
    fig.update_layout(**chart_layout(height=max(260, len(names) * 28), show_legend=False))
    style_axes(fig, x_title="contribution %")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch", key="diagnostics_feature_impact")
    hist = engine.get_feature_impact_history() if hasattr(engine, "get_feature_impact_history") else None
    if isinstance(hist, pd.DataFrame) and len(hist):
        st.caption("Impact history (last 10 refits)")
        st.dataframe(hist.tail(10), width="stretch", height=200)


def _render_levels_only(engine, signal: dict) -> None:
    render_section_header("OU Mean-Reversion Diagnostics (levels mode)",
                          "Stationarity of the valuation residual — valuation context — falsified as a forecaster.",
                          icon="crosshair", accent="cyan")
    c1, c2, c3 = st.columns(3)
    meaningful = signal.get("half_life_meaningful") is True
    with c1:
        render_metric_card("OU HALF-LIFE", fmt(signal.get("ou_half_life"), ".0f", suffix="d") if meaningful else "—",
                           "days to close half the gap" if meaningful else "not meaningful on this fit (θ unstable / no reversion)")
    with c2:
        p = num(signal.get("adf_pvalue"))
        render_metric_card("ADF P-VALUE", fmt(p, ".3f"), f"rejects a unit root if p < {EDGE_ALPHA:g}",
                           "neutral" if p is None else "success" if p < EDGE_ALPHA else "danger")
    with c3:
        p = num(signal.get("kpss_pvalue"))
        render_metric_card("KPSS P-VALUE", fmt(p, ".3f"), f"fails to reject stationarity if p > {EDGE_ALPHA:g}",
                           "neutral" if p is None else "success" if p > EDGE_ALPHA else "danger")
    st.caption("Edge assessment (engine levels test): " + str(signal.get("edge_assessment") or "—"))
    if not hasattr(engine, "get_signal_performance"):
        return
    perf = engine.get_signal_performance() or {}
    rows = []
    for period, p in sorted(perf.items()):
        if not isinstance(p, dict):
            continue
        row = {"Period": f"{period}D"}
        for side in ("buy", "sell"):
            n = int(num(p.get(f"{side}_count")) or 0)
            pv = num(p.get(f"{side}_p_value"))
            mark = "✓" if pv is not None and pv < EDGE_ALPHA else "—"
            row[f"{side.title()} HR"] = fmt((num(p.get(f"{side}_hit")) or 0) * 100, ".1f", suffix="%") if n else "—"
            row[f"{side.title()} avg Δ"] = fmt(p.get(f"{side}_avg"), "+.2f", suffix="%") if n else "—"
            row[f"{side.title()} t"] = f"{fmt(p.get(f'{side}_t_stat'), '+.2f')} {mark}" if n else "—"
            row[f"{side.title()} N"] = n
        rows.append(row)
    if rows:
        st.caption(f"Engine signal hit rates by horizon (✓ = p < {EDGE_ALPHA:g}; levels test, not the Edge Ledger)")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_data_health(run_config: dict, run_report: dict) -> None:
    render_section_header("Data Layer Health",
                          "Provenance of every input this run used (live / cache / stale / fallback / empty).",
                          icon="database", accent="emerald")
    prov = run_config.get("provenance") or {}
    keys = list(prov) or ["sheet"]
    for col, k in zip(st.columns(len(keys), gap="small"), keys):
        p = prov.get(k) if isinstance(prov.get(k), dict) else {}
        word = str(p.get("provenance") or "n/a")
        missing = p.get("missing") or []
        with col:
            render_metric_card(f"{k.upper()}", word.upper(),
                               f"as of {p.get('as_of') or '—'} · {len(missing)} missing", _PROV_CLASS.get(word, "neutral"),
                               tooltip=str(p.get("note") or "") + (f" · missing: {', '.join(map(str, missing[:10]))}"
                                                                   if missing else ""))
    t = run_report.get("timings") or {}
    st.caption(f"Run {run_report.get('run_id') or '—'} · fingerprint {str(run_config.get('fingerprint') or '—')[:12]} · "
               f"history from {run_config.get('history_start') or '—'} · total {fmt(t.get('total', t.get('t_total')), '.1f', suffix='s')}")


def render_diagnostics_tab(engine, signal: dict) -> None:
    """Diagnostics — Edge Ledger (from the conclusion), model inputs, levels-only tests, provenance."""
    ss = st.session_state
    con, rc = ss.get("conclusion") or {}, ss.get("run_config") or {}
    forward = bool(rc.get("forward_mode", con.get("forward_mode", True)))
    mc = mode_copy(forward)
    st.markdown('<div class="tab-bg diagnostics"></div>', unsafe_allow_html=True)
    _render_edge_ledger(con)
    section_gap()
    _render_feature_impact(engine, mc)
    section_gap()
    if forward:
        st.caption(mc["levels_only_note"])
    else:
        _render_levels_only(engine, signal or {})
    section_gap()
    _render_data_health(rc, ss.get("run_report") or {})
