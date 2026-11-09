"""
Nishkarsh v2 — Precedent tab: historical analogs, rendered from ``session_state["precedent"]``.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

The analogs and their base rates are computed ONCE per run by the pipeline (A4-16) — this tab
recomputes nothing. A base rate counts as evidence only when its exact binomial p (vs the
unconditional rise rate over the same history) is below EDGE_ALPHA; otherwise it is shown in
neutral colours as description (A4-14). Colour convention everywhere on this tab: extension
Z < 0 (the engine's bullish pole, − = bullish) is green (A4-15).
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.config import COLOR_GREEN, COLOR_MUTED, COLOR_RED, EDGE_ALPHA
from ui.components import esc, fmt, num, render_interpretation_card, render_metric_card, render_section_header, section_gap
from ui.copy import PRECEDENT, SIGN_ENGINE, zone_label
from ui.theme import chart_layout, style_axes


def _badge(avgz) -> tuple[str, str]:
    z = num(avgz)
    if z is None:
        return "tier-hold", "badge-hold"
    if z <= -1.0:
        return ("tier-strong-buy", "badge-strong-buy") if z <= -2.0 else ("tier-buy", "badge-buy")
    if z >= 1.0:
        return "tier-caution", "badge-caution"
    return "tier-hold", "badge-hold"


def _zone_word(avgz, forward: bool) -> str:
    z = num(avgz)
    if z is None:
        return "—"
    return zone_label("Under" if z <= -1.0 else "Over" if z >= 1.0 else "Fair", forward)


def _tile(h: int, val) -> str:
    v = num(val)
    cls = "neutral" if v is None else ("pos" if v > 0 else "neg")
    return (f'<div class="analog-fwd-tile {cls}"><span class="analog-fwd-label">+{int(h)}D</span>'
            f'<span class="analog-fwd-value">{esc(fmt(v, "+.1f", suffix="%"))}</span></div>')


def _period_card(p: dict, target: str, holds: tuple[int, ...], forward: bool) -> None:
    sim = num(p.get("similarity"))
    sim_pct = 0.0 if sim is None else max(0.0, min(100.0, sim * 100))
    tier_cls, badge_cls = _badge(p.get("avgz"))
    z = num(p.get("avgz"))
    z_cls = "neutral" if z is None else ("pos" if z < 0 else "neg" if z > 0 else "neutral")
    fwd = p.get("fwd") if isinstance(p.get("fwd"), dict) else {}
    tiles = "".join(_tile(h, fwd.get(int(h), fwd.get(str(h)))) for h in holds)
    grid = f"grid-template-columns:repeat({max(1, len(holds))},1fr);"
    st.markdown(
        f"""\
<div class="position-card analog-card {tier_cls}">
  <div class="analog-card-head">
    <div class="analog-card-id">
      <div class="analog-eyebrow">Historical match</div>
      <div class="analog-symbol">{esc(p.get('date', '—'))}</div>
    </div>
    <span class="position-card-badge {badge_cls}">{esc(_zone_word(z, forward))}</span>
  </div>
  <div class="analog-stat-row">
    <div class="analog-stat">
      <span class="analog-stat-label">Similarity (relative)</span>
      <span class="analog-stat-value amber">{esc(fmt(sim_pct, '.1f', suffix='%'))}</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">Extension Z ({esc(SIGN_ENGINE)})</span>
      <span class="analog-stat-value {z_cls}">{esc(fmt(z, '+.2f'))}</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">{esc(target)} at T</span>
      <span class="analog-stat-value">{esc(fmt(p.get('price'), ',.2f'))}</span>
    </div>
  </div>
  <div class="analog-fwd-block">
    <div class="analog-fwd-block-label">Realized change of {esc(target)} after T</div>
    <div class="analog-fwd-grid" style="{grid}">{tiles}</div>
  </div>
  <div class="analog-card-foot">
    <span class="analog-foot-label">Similarity</span>
    <div class="conviction-bar"><div class="conviction-bar-fill fill-hold" style="width:{sim_pct:.0f}%;"></div></div>
    <span class="analog-foot-pct">{sim_pct:.0f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_precedent_tab(ts, target: str, horizon: int, forward_mode: bool) -> None:
    """Precedent tab. ``ts`` = full engine frame (RangeIndex + Date) — used only for the descriptive scatter."""
    prec = st.session_state.get("precedent") or {}
    analogs = [a for a in (prec.get("analogs") or []) if isinstance(a, dict)]
    summary = prec.get("summary") or {}
    forward = bool(forward_mode)
    render_section_header("Similar Historical Periods", PRECEDENT["desc"], icon="compass", accent="emerald")
    if not analogs:
        st.warning("No historical analogs for this run (not enough history, or the precedent step failed — see warnings).")
        return

    holds = tuple(sorted(int(h) for h in summary)) or tuple(sorted({int(h) for a in analogs for h in (a.get("fwd") or {})}))
    if summary:
        cols = st.columns(len(holds), gap="small")
        for col, h in zip(cols, holds):
            s = summary.get(h) or summary.get(str(h)) or {}
            n, pos = num(s.get("n")), num(s.get("positive_pct"))
            u, p = num(s.get("unconditional_positive_pct")), num(s.get("binomial_p"))
            k = int(round((pos or 0) / 100 * n)) if n else 0
            sig = p is not None and p < EDGE_ALPHA and u is not None and pos is not None
            cls = ("success" if pos > u else "danger") if sig else "neutral"
            with col:
                render_metric_card(f"After {h} days · analogs that rose", f"{k}/{fmt(n, '.0f')}",
                                   f"vs {fmt(u, '.0f', suffix='%')} unconditional · binomial p={fmt(p, '.2f')}"
                                   f" · median {fmt(s.get('median'), '+.2f', suffix='%')}"
                                   + (" · significant" if sig else " · not significant"), cls)
    render_interpretation_card("Empirical Base Rate — descriptive", esc(PRECEDENT["base_rate"]), "info")

    section_gap()
    render_section_header("Top Analog Periods",
                          f"{len(analogs)} historical matches · similarity is {PRECEDENT['similarity_note']}", icon="layers")
    grid = st.columns(2, gap="medium")
    for i, p in enumerate(analogs):
        with grid[i % 2]:
            _period_card(p, target, holds, forward)
            st.markdown('<div style="height: var(--sp-3);"></div>', unsafe_allow_html=True)

    # Descriptive scatter of the engine's own columns (no fit, no statistic computed here).
    col = f"FwdChg_{int(horizon)}"
    if ts is None or "AvgZ" not in getattr(ts, "columns", []) or col not in ts.columns:
        return
    section_gap()
    render_section_header(f"Extension Z vs Realized {int(horizon)}d Change of {target}",
                          f"One dot per non-overlapping {int(horizon)}d window · descriptive only — graded "
                          "evidence lives in Evidence → Edge ledger.", icon="chart", accent="rose")
    d = ts[["AvgZ", col]].iloc[:: int(horizon)].dropna()
    if len(d) < 5:
        st.caption("Too few windows to plot.")
        return
    fig = go.Figure(go.Scattergl(
        x=d["AvgZ"], y=d[col], mode="markers", name="window",
        marker=dict(size=5, opacity=0.6, color=[COLOR_GREEN if v < 0 else COLOR_RED if v > 0 else COLOR_MUTED
                                                for v in d["AvgZ"]]),
        hovertemplate="Z: %{x:.2f}<br>change: %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")
    layout = chart_layout(height=400, show_legend=False)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    style_axes(fig, y_title=f"{target} change T→T+{int(horizon)}d (%)", x_title=f"extension Z at T ({SIGN_ENGINE})")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "displaylogo": False})
