"""
Nishkarsh — Convergence: do the systems agree? (≈ Tattva's Convergence page; lean words only.)

Evidence about the read, not a second read: every number here is a reading of how far
Aarambh's top-down pillar and Nirnay's bottom-up pillar concur, in lean words (bullish /
bearish), never in action words. The read itself lives on Overview.

Reading order — the house convention every engine page follows:

  1 ANSWER    do they agree, and which way?      agreement · consensus · the two pillars · edge
  2 ANCHOR    how has that moved?                the read and the two pillars it averages
  3 STATE     what drives today's read?          the three pillars and their ten inputs, today
  4 DETAIL    what the agreement is made of      signed agreement · its four dimensions
  5 TRUST     has agreement helped, and been     agreement lift · walk-forward edge of the
              worth anything?                    agreement pillar
  6 RISK      where they disagreed               divergences (amber)

Sign: the read (±100), its pillars and every input are + = bullish. The read's markers are
classified at the FIXED bands (±10 no lean · ±30 moderate · ±60 strong), so a dot on any date
means what that date's reading meant. Windowed page: the window control sits in "The Read" and
frames every time chart here; "What Drives Today's Read" is today only.

``drivers_rows`` and ``divergence_rows`` are the pure row builders (tests/test_ui_pages.py).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from convergence.nishkarsh import COMPONENTS
from convergence.views import PILLAR_NO_LEAN, lean_phrase
from core.config import CONV2_TIER_MODERATE, CONV2_TIER_STRONG, CONV2_TIER_WEAK
from ui.components import (
    PILLARS, _d, _list, _str, esc, fmt, nko_side, nko_tier_of, num, pretty_date, render_chart_panel,
    render_edge_panel, render_empty_state, render_kpi_strip, render_note, render_section_header,
    render_table_panel, signed, tone_of,
)
from ui.copy import NIRNAY_READ_SENTENCE, NKO_COMPONENT_PLAIN, NKO_PILLAR_PLAIN
from ui.palette import UI_CHART_HEIGHT_MEDIUM, UI_CHART_HEIGHT_STACKED
from ui.tabs.engine_common import MIN_SESSIONS, add_sign_fill, by_date, ic_card, ic_tone, window_slice
from ui.tabs.page_overview import read_verdict
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, panel_bg, style_axes

#: Marker size per fixed tier (size carries strength; colour carries direction — never opacity).
_TIER_SIZE = {"STRONG": 8, "MODERATE": 6, "WEAK": 4, "NO LEAN": 3}
#: The agreement anatomy's four dimensions: (column, card label, base weight, what it compares).
DIMENSIONS = (("dim_direction", "Direction", 0.30, "same sign of the two leans"),
              ("dim_breadth", "Breadth", 0.25, "bullish breadth, top-down vs stocks"),
              ("dim_magnitude", "Magnitude", 0.25, "strength of the two readings"),
              ("dim_regime", "Regime", 0.20, "Aarambh's regime vs the stocks' majority"))
_DIV_WORD = {"AARAMBH_LEADS": "Aarambh leads", "NIRNAY_LEADS": "Nirnay leads", "CONTRADICTION": "Contradiction"}

TOOLTIPS = {
    "agreement": (
        "Today's agreement between Aarambh and Nirnay across four dimensions — direction, breadth, "
        "magnitude and regime — with fixed base weights. Descriptive: it is not fitted to returns. Read "
        "it against its own usual level, not against 50%."
    ),
    "consensus": (
        "Whether the two systems lean the same way today: Bullish or Bearish when Aarambh's forecast and "
        "Nirnay's stocks point the same way, Split when they point opposite ways or one is flat."
    ),
    "aarambh": "The top-down pillar: Aarambh's four inputs averaged, each against its own past. + = bullish.",
    "nirnay": "The bottom-up pillar: Nirnay's five stock inputs averaged, each against its own past. + = bullish.",
    "signed": (
        "The consensus (+1 both bullish, −1 both bearish, 0 split) times the agreement ratio: the "
        "agreement pillar's input. Its pillar is the same series standardized against its own past."
    ),
    "ic": (
        "Rank correlation of the agreement pillar with the Nifty's forward return over the horizon, scored "
        "on non-overlapping windows. 0 = no skill."
    ),
    "lift": (
        "Aarambh's IC on days Nirnay agrees minus its IC on days Nirnay disagrees, on non-overlapping "
        "windows, tested one-sided. Only if it passes may Nirnay shrink Aarambh (κ below 1) on "
        "disagreement days."
    ),
}


def _component_label(cid, fallback="") -> str:
    return NKO_COMPONENT_PLAIN.get(_str(cid), (_str(fallback) or "An input", ""))[0]


def drivers_rows(k: dict) -> list[dict]:
    """Pillars and, under each, its inputs — today's z on one scale; the top-3 drivers marked.
    ``[{kind: pillar|input, label, z, driver, hint}]``."""
    k = _d(k)
    comps = {_str(_d(c).get("id")): _d(c) for c in _list(k.get("components"))}
    top = {_str(_d(c).get("id")) for c in _list(k.get("drivers"))}
    pil = _d(k.get("pillars"))
    rows = []
    for p in PILLARS:
        rows.append({"kind": "pillar", "label": NKO_PILLAR_PLAIN[p][0], "z": num(pil.get(p)), "driver": False,
                     "hint": NKO_PILLAR_PLAIN[p][1]})
        for pil_, cid, lab, _ in COMPONENTS:
            if pil_ == p:
                rows.append({"kind": "input", "label": _component_label(cid, lab),
                             "z": num(comps.get(cid, {}).get("z")), "driver": cid in top,
                             "hint": NKO_COMPONENT_PLAIN.get(cid, ("", ""))[1]})
    return rows


def divergence_rows(events, frame_index, window: int = 20) -> tuple[pd.DataFrame, int]:
    """The divergence table (newest first, at most 15 rows) and how many fell in the last ``window``
    sessions of the conclusion frame. When none did, the most recent are shown."""
    cols = ["Type", "Aarambh", "Nirnay", "Persistent", "What happened"]
    e = by_date(events)
    if e.empty or "type" not in e.columns:
        return pd.DataFrame(columns=cols), 0
    e = e.sort_index(ascending=False)
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(frame_index), errors="coerce")).dropna().normalize().sort_values()
    start = idx[max(0, len(idx) - int(window))] if len(idx) else None
    recent = e[e.index >= start] if start is not None else e.iloc[:0]
    shown = (recent if len(recent) else e).head(15)

    def _what(r) -> str:
        za, zn = signed(r.get("zA"), "+.2f"), signed(r.get("zN"), "+.2f")
        t = _str(r.get("type"))
        if t == "AARAMBH_LEADS":
            return f"Aarambh reads {za}σ while the stocks are calm ({zn}σ)"
        if t == "NIRNAY_LEADS":
            return f"The stocks read {zn}σ while Aarambh is calm ({za}σ)"
        return f"Aarambh {za}σ and the stocks {zn}σ point opposite ways, repeatedly"

    tbl = pd.DataFrame({
        "Type": [_DIV_WORD.get(_str(t), _str(t, "—").replace("_", " ").title()) for t in shown["type"]],
        "Aarambh": pd.to_numeric(shown.get("zA"), errors="coerce").to_numpy(dtype=float),
        "Nirnay": pd.to_numeric(shown.get("zN"), errors="coerce").to_numpy(dtype=float),
        "Persistent": ["yes" if bool(x) else "" for x in shown.get("persistent", pd.Series(False, index=shown.index))],
        "What happened": [_what(r) for _, r in shown.iterrows()],
    }, index=shown.index)
    tbl.index.name = "Date"
    return tbl, int(len(recent))


# ═══════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════

def _marker_style(values) -> dict:
    """Colour carries direction, size carries the fixed tier; a 1px outline in the panel colour."""
    colors, sizes = [], []
    for v in values:
        side, tier = nko_side(v), nko_tier_of(v)
        colors.append(chart_color("emerald") if side == "bull" else chart_color("rose") if side == "bear"
                      else chart_rgba("slate", 0.5))
        sizes.append(_TIER_SIZE.get(tier, 3))
    return dict(size=sizes, color=colors, line=dict(width=1, color=panel_bg()))


def _read_chart(w: pd.DataFrame) -> None:
    """The read (top) over the two pillars it averages with their agreement."""
    x = list(w.index)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.46, 0.27, 0.27], vertical_spacing=0.05)
    v = pd.to_numeric(w["NKO"], errors="coerce").to_numpy(dtype=float)
    fig.add_hrect(y0=-CONV2_TIER_WEAK, y1=CONV2_TIER_WEAK, fillcolor=grid_rgba(0.05), line_width=0, row=1, col=1)
    add_sign_fill(fig, x, v, above="emerald", below="rose", alpha=0.06, row=1, col=1)
    words = [nko_tier_of(a).title() if nko_side(a) in ("flat", "none") else
             f"{nko_tier_of(a).title()} {'bullish' if nko_side(a) == 'bull' else 'bearish'}" for a in v]
    fig.add_trace(go.Scatter(x=x, y=v, mode="lines+markers", name="Nishkarsh read",
                             line=dict(color=chart_rgba("slate", 0.6), width=1.2), marker=_marker_style(v),
                             customdata=[[signed(a, "+.0f"), wd] for a, wd in zip(v, words)],
                             hovertemplate="%{customdata[0]} · %{customdata[1]}<extra>Read</extra>"), row=1, col=1)
    for lvl, nm in ((CONV2_TIER_MODERATE, "emerald"), (CONV2_TIER_STRONG, "emerald"),
                    (-CONV2_TIER_MODERATE, "rose"), (-CONV2_TIER_STRONG, "rose")):
        fig.add_hline(y=lvl, line_dash="dot", line_color=chart_rgba(nm, 0.25 if abs(lvl) < CONV2_TIER_STRONG else 0.35),
                      line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=1, col=1)

    for row, col, name, hue in ((2, "P_top_down", "Top-down pillar", "violet"),
                                (3, "P_bottom_up", "Bottom-up pillar", "cyan")):
        y = pd.to_numeric(w.get(col), errors="coerce") if col in w.columns else pd.Series(index=w.index, dtype=float)
        fig.add_hrect(y0=-PILLAR_NO_LEAN, y1=PILLAR_NO_LEAN, fillcolor=grid_rgba(0.05), line_width=0, row=row, col=1)
        fig.add_trace(go.Scatter(x=x, y=y.to_numpy(dtype=float), mode="lines", name=name,
                                 line=dict(color=chart_color(hue), width=1.2)), row=row, col=1)
        fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=row, col=1)
        if y.notna().sum() == 0:
            fig.add_annotation(text="no stock data in this window", showarrow=False, xref="x domain",
                               yref="y domain", x=0.5, y=0.5, row=row, col=1,
                               font=dict(size=10, family="JetBrains Mono, monospace", color=chart_color("slate")))

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED, show_legend=False))
    style_axes(fig, y_title="Read (±100)", y_range=[-100, 100], row=1, col=1)
    style_axes(fig, y_title="Top-down σ", row=2, col=1)
    style_axes(fig, y_title="Bottom-up σ", row=3, col=1)
    render_chart_panel(fig, "convergence_read", units="read ±100 · pillar σ", window=True)


def _drivers_chart(rows: list[dict], k: dict) -> None:
    """Today's pillars and their ten inputs on one z scale; the top-3 inputs marked."""
    rows = [r for r in rows if r["z"] is not None]
    labels = [(f"<b>{r['label'].upper()}</b>" if r["kind"] == "pillar" else r["label"]) for r in rows][::-1]
    vals = [r["z"] for r in rows][::-1]
    kinds = [r for r in rows][::-1]
    colors = [(chart_color("emerald") if z > 0 else chart_color("rose")) if (r["kind"] == "pillar" or r["driver"])
              else chart_rgba("emerald" if z > 0 else "rose", 0.45) for r, z in zip(kinds, vals)]
    text = ["top 3" if r["driver"] else "" for r in kinds]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=colors), text=text,
                           textposition="outside", cliponaxis=False,
                           textfont=dict(size=9, family="JetBrains Mono, monospace", color=chart_color("slate")),
                           customdata=[[signed(z, "+.2f")] for z in vals],
                           hovertemplate="%{y}: %{customdata[0]}σ<extra></extra>"))
    span = max([abs(z) for z in vals] + [1.0]) * 1.25
    fig.add_vline(x=0, line_color=grid_rgba(0.12), line_width=0.6)
    fig.update_layout(**chart_layout(height=max(240, len(labels) * 26), show_legend=False))
    style_axes(fig, x_title="z today · + = bullish", y_range=None)
    fig.update_xaxes(range=[-span, span])
    fig.update_yaxes(showgrid=False, zeroline=False)
    drivers = [_d(c) for c in _list(_d(k).get("drivers"))]
    foot = ("Largest inputs today: " + ", ".join(
        f"{_component_label(c.get('id'), c.get('label')).lower()} {signed(c.get('z'), '+.1f')}" for c in drivers)
        + ". Bold rows are the pillars; each is the average of the inputs under it, re-standardized."
        if drivers else "Bold rows are the pillars; each is the average of the inputs under it, re-standardized.")
    render_chart_panel(fig, "convergence_drivers", units="z today", windowed=False, footer=esc(foot))


def _anatomy_chart(w: pd.DataFrame) -> None:
    """Signed agreement (top) over the agreement ratio it is scaled by (bottom)."""
    x = list(w.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08)
    sao = pd.to_numeric(w.get("SAO"), errors="coerce").to_numpy(dtype=float)
    add_sign_fill(fig, x, sao, above="emerald", below="rose", alpha=0.08, row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=sao, mode="lines", name="Signed agreement",
                             line=dict(color=chart_color("accent"), width=1.4)), row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.08), line_width=0.5, row=1, col=1)
    ar = pd.to_numeric(w.get("agreement_ratio"), errors="coerce").to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=x, y=ar, mode="lines", name="Agreement ratio",
                             line=dict(color=chart_color("slate"), width=1.2)), row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM + 100))
    style_axes(fig, y_title="Signed (±1)", y_range=[-1.05, 1.05], row=1, col=1)
    style_axes(fig, y_title="Agreement", y_range=[0, 1], row=2, col=1)
    render_chart_panel(fig, "convergence_anatomy", units="signed agreement · agreement ratio")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def _consensus_word(c) -> tuple[str, str, str]:
    v = num(c)
    if v is None:
        return "—", "no stock reading today", "neutral"
    if v > 0:
        return "Bullish", "both systems lean bullish", "success"
    if v < 0:
        return "Bearish", "both systems lean bearish", "danger"
    return "Split", "they point opposite ways, or one is flat", "neutral"


def render_convergence_page() -> None:
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    k = _d(con.get("nko"))
    v = read_verdict()
    h = int(num(con.get("horizon")) or 10)
    fr = by_date(ss.get("conclusion_frame"))
    pil = _d(k.get("pillars"))
    td, bu, ag = (num(pil.get(p)) for p in PILLARS)
    as_of = pd.Timestamp(k["as_of"]).normalize() if _str(k.get("as_of")) else (fr.index.max() if len(fr) else None)
    row = fr.loc[as_of] if as_of is not None and as_of in fr.index else None
    ar = num(row.get("agreement_ratio")) if row is not None else None
    sao = num(row.get("SAO")) if row is not None else None
    cword, csub, ctone = _consensus_word(row.get("consensus") if row is not None else None)
    anat = _d(_d(ss.get("run_report")).get("anatomy"))
    usual = num(_d(anat.get("mean")).get("agreement_ratio"))
    edge = _d(_d(k.get("pillar_edge")).get("agreement")) or None
    single = con.get("single_system") is True or bu is None

    signal = _str(v.get("signal"), "NO READ")
    vt = _str(v.get("value_text"))
    if single:
        answer = (f"Top-down only today: Nirnay has no stock data for {pretty_date(as_of)}, so there is nothing to "
                  f"agree with — the read is Aarambh's alone ({signed(td, '+.2f')}σ)." if td is not None else
                  "There is no reading for the latest date of this run.")
    else:
        answer = (f"Aarambh's top-down pillar reads {signed(td, '+.2f')}σ and Nirnay's bottom-up "
                  f"{signed(bu, '+.2f')}σ; " + (f"they agree {ar:.0%} across direction, breadth, magnitude and "
                                               "regime, " if ar is not None else "")
                  + f"and with their agreement the read is {signal.lower()}" + (f" ({vt})." if vt and vt != "—"
                                                                               else "."))
    render_section_header("Convergence — Do the Systems Agree?", answer, icon="target", accent="accent")
    if not con or fr.empty:
        render_empty_state("No convergence read for this run",
                           "Convergence needs a conclusion over time, and this run published no conclusion frame.",
                           eyebrow="Convergence")
        return
    render_kpi_strip([
        {"label": "Agreement", "value": f"{ar:.0%}" if ar is not None else "—",
         "subtext": (f"usually {usual:.0%} · four dimensions" if usual is not None else "four dimensions"),
         "color_class": "neutral", "tooltip": TOOLTIPS["agreement"]},
        {"label": "Consensus", "value": cword, "subtext": csub, "color_class": ctone,
         "tooltip": TOOLTIPS["consensus"]},
        {"label": "Aarambh Reads", "value": f"{signed(td, '+.2f')}σ" if td is not None else "—",
         "subtext": f"top-down · {lean_phrase(td)}" if td is not None else "no top-down reading",
         "color_class": tone_of(td, 0.5), "tooltip": TOOLTIPS["aarambh"]},
        {"label": "Nirnay Reads", "value": f"{signed(bu, '+.2f')}σ" if bu is not None else "—",
         "subtext": f"bottom-up · {lean_phrase(bu)}" if bu is not None else "no stock data today",
         "color_class": tone_of(bu, 0.5), "tooltip": TOOLTIPS["nirnay"]},
        {"label": "Signed Agreement", "value": signed(sao, "+.2f"),
         "subtext": f"agreement pillar {signed(ag, '+.2f')}σ" if ag is not None else "consensus × agreement",
         "color_class": tone_of(ag, 0.5), "tooltip": TOOLTIPS["signed"]},
        ic_card(edge, "agreement pillar", h, TOOLTIPS["ic"]),
    ], max_cols=6, key="conv-read")

    # ── 2 · The read and its two pillars ─────────────────────────────────
    render_section_header(
        "The Read — Three Pillars, Averaged",
        (f"Top: the Nishkarsh read ({vt} today), each marker classified at the fixed bands — inside ±"
         f"{CONV2_TIER_WEAK:g} (shaded) no lean, ±{CONV2_TIER_MODERATE:g} and ±{CONV2_TIER_STRONG:g} (dotted) "
         "moderate and strong; size grows with the band. " if vt and vt != "—" else
         f"Top: the Nishkarsh read, each marker classified at the fixed bands (±{CONV2_TIER_WEAK:g} · "
         f"±{CONV2_TIER_MODERATE:g} · ±{CONV2_TIER_STRONG:g}). ")
        + f"Below: the top-down and bottom-up pillars (shaded ±{PILLAR_NO_LEAN:g}σ = no lean). + = bullish.",
        icon="layers", accent="accent",
    )
    w = window_slice(fr) if "NKO" in fr.columns else fr.iloc[:0]
    if "NKO" not in fr.columns or pd.to_numeric(w["NKO"], errors="coerce").notna().sum() < MIN_SESSIONS:
        render_empty_state(
            "Building read history",
            "The read is standardized against its own past, which needs a few weeks of readings in this window "
            "before it is drawn. Widen the chart window.", eyebrow="Convergence")
    else:
        w = w[pd.to_numeric(w["NKO"], errors="coerce").notna()]
        _read_chart(w)

    # ── 3 · What drives today's read ─────────────────────────────────────
    rows = drivers_rows(k)
    drivers = [_d(c) for c in _list(k.get("drivers"))]
    lead = drivers[0] if drivers else None
    render_section_header(
        "What Drives Today's Read",
        (_str(v.get("headline")) + (f" The largest single input is "
                                    f"{_component_label(lead.get('id'), lead.get('label')).lower()} "
                                    f"({signed(lead.get('z'), '+.1f')})." if lead else ""))
        if _str(v.get("headline")) else
        "The three pillars and their ten inputs today, on one scale — each standardized against its own past.",
        icon="bar-chart", accent="accent",
    )
    if any(r["z"] is not None for r in rows):
        _drivers_chart(rows, k)
    else:
        render_empty_state("No reading today", "No pillar or input has a reading for the latest date.",
                           eyebrow="Convergence")

    # ── 4 · Agreement anatomy ─────────────────────────────────────────────
    ad = by_date(ss.get("anatomy_df"))
    arow = ad.loc[as_of] if as_of is not None and as_of in ad.index else (ad.iloc[-1] if len(ad) else None)
    render_section_header(
        "Agreement Anatomy — What the Agreement Is Made Of",
        (f"Signed agreement is +1 when both systems lean bullish and −1 when both lean bearish, times how fully "
         f"they agree ({fmt(ar, '.0%')} today); it is 0 when they split. {NIRNAY_READ_SENTENCE}"),
        icon="scale", accent="accent",
    )
    if arow is None:
        render_empty_state("No agreement anatomy",
                           "The anatomy compares the two systems on dates both read; this run has no stock data.",
                           eyebrow="Convergence")
    else:
        mean = _d(anat.get("mean"))
        render_kpi_strip([
            {"label": lab, "value": fmt(num(arow.get(col)), ".0%"),
             "subtext": f"weight {wt:.0%} · usually {fmt(num(mean.get(col)), '.0%')}",
             "color_class": "neutral", "tooltip": f"{lab}: {what}. 1 = full agreement, 0 = no agreement."}
            for col, lab, wt, what in DIMENSIONS
        ], max_cols=4, key="conv-anatomy")
        aw = window_slice(fr[[c for c in ("SAO", "agreement_ratio") if c in fr.columns]].dropna(how="all"),
                          fr.index.max()) if {"SAO", "agreement_ratio"} & set(fr.columns) else pd.DataFrame()
        if len(aw) < MIN_SESSIONS:
            render_empty_state("Building agreement history", "Fewer than ten sessions both systems read fall in "
                               "this window.", eyebrow="Convergence")
        else:
            _anatomy_chart(aw)

    # ── 5 · Agreement lift ────────────────────────────────────────────────
    lift = _d(con.get("lift"))
    available = lift.get("available") is True
    lv, lp = num(lift.get("lift")), num(lift.get("p_one_sided"))
    passed = lift.get("passed") is True
    kappa = num(lift.get("kappa"))
    render_section_header(
        "Agreement Lift — Does Agreement Help Aarambh?",
        (f"Is Aarambh more accurate when Nirnay agrees than when it disagrees? Lift {signed(lv, '+.3f')} "
         f"(one-sided p {fmt(lp, '.2f')}) — "
         + ("passed, so Nirnay may shrink Aarambh on disagreement days" if passed else
            "not shown, so Nirnay does not shrink Aarambh")
         + f" (κ {fmt(kappa, '.2f')}).") if available else
        "Is Aarambh more accurate when Nirnay agrees than when it disagrees? Not measured this run — the test "
        "needs enough non-overlapping windows on each side, on dates both systems read.",
        icon="crosshair", accent="accent",
    )
    if available:
        render_kpi_strip([
            {"label": "When Nirnay Agrees", "value": signed(lift.get("ic_confirm"), "+.3f"),
             "subtext": f"Aarambh's IC · {fmt(lift.get('n_confirm'), ',.0f')} windows",
             "color_class": ic_tone(lift.get("ic_confirm"))},
            {"label": "When It Disagrees", "value": signed(lift.get("ic_contradict"), "+.3f"),
             "subtext": f"Aarambh's IC · {fmt(lift.get('n_contradict'), ',.0f')} windows",
             "color_class": ic_tone(lift.get("ic_contradict"))},
            {"label": "Lift", "value": signed(lv, "+.3f"),
             "subtext": f"one-sided p {fmt(lp, '.2f')} · {'passed' if passed else 'not passed'}",
             "color_class": "success" if passed else "neutral", "tooltip": TOOLTIPS["lift"]},
            {"label": "Smallest Detectable", "value": fmt(lift.get("mde"), ".2f"),
             "subtext": "lift this sample could confirm (80% power)", "color_class": "neutral"},
            {"label": "κ Applied", "value": fmt(kappa, ".2f"),
             "subtext": ("Intelligence on · below 1 only after a pass" if con.get("kappa_enabled") is True
                         else "Intelligence off · Aarambh unchanged"),
             "color_class": "info"},
        ], max_cols=5, key="conv-lift")
    else:
        render_empty_state("Not yet measured",
                           "The lift test compares Aarambh's accuracy on agreeing and disagreeing days, and needs "
                           "enough of each on dates both systems read.", eyebrow="Convergence")

    # ── 6 · Walk-forward edge of the agreement pillar ───────────────────
    ic = num(_d(edge).get("ic"))
    pos, nb = int(num(_d(edge).get("positive_blocks")) or 0), int(num(_d(edge).get("scored_blocks")) or 0)
    render_section_header(
        "Walk-Forward Edge",
        f"When the agreement pillar leaned one way, did the Nifty move that way over the next {h} sessions? "
        + (f"On {int(num(_d(edge).get('n')) or 0):,} non-overlapping windows its IC is {signed(ic, '+.3f')}"
           + (f", positive in {pos} of {nb} periods." if nb else ".") if ic is not None else
           "Not graded yet — it needs enough dates both systems read.")
        + " Graded exactly as the two other pillars are.",
        icon="trending", accent="accent",
    )
    render_edge_panel(edge, "agreement", what="the agreement pillar")

    # ── 7 · Divergences ───────────────────────────────────────────────────
    win = int(num(_d(con.get("divergence_recent")).get("window_days")) or 20)
    tbl, n_recent = divergence_rows(ss.get("conclusion_events"), fr.index, win)
    render_section_header(
        "Divergences",
        "Sessions where the systems pointed opposite ways, or one read strongly while the other was calm — "
        + (f"{n_recent} in the last {win} sessions." if n_recent else f"no events in the last {win} sessions.")
        + " The Overview counts the recent ones as a standing risk.",
        icon="alert-triangle", accent="amber",
    )
    if tbl.empty:
        render_note("No divergence events: wherever both systems read, they did not point opposite ways.")
    else:
        render_table_panel(
            tbl, "convergence-divergences", index_label="Date", windowed=False, max_height=420,
            units=(f"{n_recent} in the last {win} sessions" if n_recent else
                   f"no events in the last {win} sessions — most recent shown"),
            sign_color_cols={"Aarambh", "Nirnay"}, col_labels={"Aarambh": "Aarambh z", "Nirnay": "Nirnay z"})
