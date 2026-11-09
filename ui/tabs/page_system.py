"""
Nishkarsh — System: does the machinery behind the read hold up? (≈ Tattva's System page).

Every engine page grades its own lean; this page puts the grades side by side, keeps the
ledger of every claim the system makes, states what each model is as this run fitted it, where
each input came from, and the method and its limits.

Reading order — the house convention every analysis page follows:

  1 ANSWER    does any of it work?            The read's own grade + walk-forward edge, every engine
  2 TRUST     what does each claim say?       The Ledger (status words from the payload's own flags)
  3 STATE     what are the models?            Model facts: Aarambh · Nirnay · the read
  4 DETAIL    is the data sound?              Data layer health (provenance per source)
  5 DETAIL    how is it built, what can't it? Method & limits

Evidence lives on the page of the claim it supports (Tattva rule 9): Aarambh's stage-by-stage
accuracy and cost-aware backtest are on the Aarambh page, the agreement lift on Convergence —
so neither is repeated here. Long "how to read" text, including the glossary, is in the
metric-card TOOLTIPS beside the numbers it explains. Unwindowed page.

``build_ledger_rows`` is the Ledger (tested in tests/test_ui_pages.py): one row per claim,
status words from the payload's own flags. The six walk-forward edges come from
``convergence.system_views.system_edges`` (the payload's evidence in the one edge contract).
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from convergence.system_views import system_edges
from core.config import BACKTEST_REFERENCE_BPS, CONV2_ZMIN_PERIODS, EDGE_ALPHA, NIRNAY_POLARITY
from pipeline.shell import SOURCE_NAMES, redact
from ui.components import (
    PILLARS, _d, _list, _str, default_chart_context, edge_periods, edge_status_code, edge_tier, esc, fmt,
    is_synthetic, num, panel, plural, pretty_date, render_chart_panel, render_chip, render_empty_state,
    render_kpi_strip, render_note, render_section_header, render_sub_header, render_table_panel, signed,
)
from ui.copy import GLOSSARY, LEVELS_TAG, METHOD_LINES, NIRNAY_READ_SENTENCE, NKO_PILLAR_PLAIN
from ui.palette import UI_CHART_HEIGHT_MEDIUM
from ui.theme import chart_color, chart_layout, grid_rgba, style_axes

__all__ = ["build_ledger_rows", "edge_table_rows", "redact", "render_system_page"]

_GLOSS = dict(GLOSSARY)
_READ_STATUS = {"VALIDATED": ("Validated", "success"), "UNPROVEN": ("Unproven", "warning"),
                "INSUFFICIENT_DATA": ("Not enough data", "neutral")}

# ── tooltips: the glossary, where its numbers appear ──────────────────────────
TOOLTIPS = {
    "ic": f"Accuracy (IC): {_GLOSS['Accuracy (IC)']} Scored on non-overlapping windows; ± is one standard "
          "error — how much of the number is sampling noise.",
    "p": f"p-value: {_GLOSS['p-value']} One-sided here (the claim is that the read points the right way).",
    "mde": f"Smallest detectable: {_GLOSS['Smallest detectable']} A true IC below this would most likely go "
           "unnoticed on this much history.",
    "hac": f"HAC t: {_GLOSS['HAC t']} Uses every day's rank IC rather than one window per horizon.",
    "periods": "Calendar years scored separately. A durable edge is positive in most years, not large in one.",
    "windows": f"One scored window per horizon, no two overlapping, on dates both systems ran. Single-system: "
               f"{_GLOSS['Single-system']}",
    "train": f"Each refit trains only on rows before its forecasts and leaves a purge gap, so no forecast sees "
             f"its own outcome. Causal: {_GLOSS['Causal']}",
    "pillars": f"Pillar: {_GLOSS['Pillar']} The read is the plain average of the three, rescaled to ±100.",
    "standardize": f"Each input is a z-score against its own earlier values only. Causal: {_GLOSS['Causal']}",
    "two_system": f"Single-system: {_GLOSS['Single-system']}",
    "weights": "Fixed equal weights, never fitted to returns — nothing about the read is tuned to the outcome "
               "it is graded on. On a single-system date only the top-down pillar counts.",
    "cuts": "Each stock's oversold and overbought cuts adapt to its own oscillator history; the card shows the "
            "median across today's live stocks.",
    "polarity": f"{NIRNAY_READ_SENTENCE} An owner decision (version 2.3.0), recorded with its evidence in "
                "docs/research/NIRNAY_POLARITY_CHANGE.md.",
}


# ═══════════════════════════════════════════════════════════════════════════
#  THE LEDGER — pure (tested)
# ═══════════════════════════════════════════════════════════════════════════

def _ic_num(e: dict) -> str:
    e = _d(e)
    return f"IC {signed(e.get('ic'), '+.3f')} · p {fmt(e.get('p'), '.2f')} · {plural(e.get('n'), 'period')}"


def _ref_cost(bt: dict) -> dict:
    by = _d(_d(bt).get("by_cost"))
    for key, row in by.items():
        if num(key) is not None and abs(num(key) - BACKTEST_REFERENCE_BPS) < 1e-9:
            return _d(row)
    return {}


def build_ledger_rows(conclusion, precedent=None) -> list[dict]:
    """One row per claim the system makes: id · claim · status (label, class) · deciding number ·
    one-line meaning. Pure; status words follow the payload's booleans and status strings."""
    con = _d(conclusion)
    h = int(num(con.get("horizon")) or 10)
    k = _d(con.get("nko"))
    rows = []

    status = _str(k.get("status")).upper()
    status = status if status in _READ_STATUS else "INSUFFICIENT_DATA"
    rows.append({"id": "L-13", "claim": "The Nishkarsh read anticipates the Nifty", "status": _READ_STATUS[status],
                 "number": _ic_num(k.get("evidence")),
                 "meaning": {"VALIDATED": f"Its readings have lined up with the next {h} days, clearing its "
                                          "pre-registered bar.",
                             "UNPROVEN": "Not yet: its accuracy is too small or too inconsistent to tell from luck.",
                             "INSUFFICIENT_DATA": "Too little history with both systems running to measure."}[status]})

    pic = _d(k.get("pillar_ic"))
    rows.append({"id": "", "claim": "Its three pillars on their own", "status": ("Context", "neutral"),
                 "number": " · ".join(f"{NKO_PILLAR_PLAIN[p][0].lower()} {signed(_d(pic.get(p)).get('ic'), '+.3f')}"
                                      for p in PILLARS),
                 "meaning": "Measured on the same periods for context; they decide nothing."})

    osc = _d(con.get("oscillator"))
    o_status = _str(osc.get("status")).upper()
    o_status = o_status if o_status in _READ_STATUS else "INSUFFICIENT_DATA"
    rows.append({"id": "L-12", "claim": "Signed agreement anticipates the Nifty", "status": _READ_STATUS[o_status],
                 "number": _ic_num(osc.get("evidence")),
                 "meaning": "The agreement pillar's input, measured on its own as a read."})

    e = _d(con.get("edge"))
    code = edge_status_code(e)
    rows.append({"id": "L-03", "claim": "Aarambh's forecast points the right way",
                 "status": {"PROVEN": ("Proven", "success"), "NOT_PROVEN": ("Not proven", "warning"),
                            "INVERTED": ("Wrong way", "danger"), "INSUFFICIENT": ("Not enough data", "neutral")}[code],
                 "number": _ic_num(e),
                 "meaning": {"PROVEN": f"Its direction has matched the next {h} days more often than chance.",
                             "NOT_PROVEN": "Its direction has not matched the market often enough to rule out luck.",
                             "INVERTED": "It has pointed the wrong way more often than chance.",
                             "INSUFFICIENT": "Not enough history to judge it."}[code]})

    lift, oos = _d(con.get("lift")), _d(con.get("oos"))
    if lift.get("available") is not True:
        ls = ("Not enough data", "neutral")
    else:
        ls = ("Passed", "success") if lift.get("passed") is True else ("Not proven", "warning")
    rows.append({"id": "L-05", "claim": "Agreement makes Aarambh more accurate", "status": ls,
                 "number": f"lift {signed(lift.get('lift'), '+.3f')} · p {fmt(lift.get('p_one_sided'), '.2f')} · "
                           f"out-of-sample {signed(oos.get('ic_C'), '+.3f')} vs {signed(oos.get('ic_A'), '+.3f')}",
                 "meaning": ("Aarambh has been more accurate on days Nirnay agrees, by more than chance."
                             if ls[0] == "Passed" else
                             "Aarambh has not been reliably more accurate on days Nirnay agrees."
                             if ls[0] == "Not proven" else "Too few periods on each side to test it.")})

    bt = _d(con.get("backtest"))
    ref = _ref_cost(bt)
    tradeable = _str(_d(con.get("trust")).get("tier")) == "TRADEABLE_EDGE"
    rows.append({"id": "L-04", "claim": "Trading Aarambh's forecast survives costs",
                 "status": ("Tradeable", "success") if tradeable else
                 (("Not tradeable", "warning") if ref else ("Not enough data", "neutral")),
                 "number": f"net Sharpe {signed(ref.get('net_sharpe'))} · t {signed(ref.get('t'))} at "
                           f"{BACKTEST_REFERENCE_BPS:g} bps",
                 "meaning": (f"Net of {BACKTEST_REFERENCE_BPS:g} bps a trade, positioning on it has earned a "
                             "significant return." if tradeable else
                             f"Net of {BACKTEST_REFERENCE_BPS:g} bps a trade, positioning on it has not earned a "
                             "reliable return.")})

    ph = _d(_d(precedent).get("hero"))
    n, pos, u, p = (num(ph.get(x)) for x in ("n", "positive_pct", "unconditional_positive_pct", "binomial_p"))
    if not n or pos is None:
        ps, pnum = ("Not enough data", "neutral"), "—"
    else:
        sig = p is not None and p < EDGE_ALPHA
        ps = ("Significant", "success") if sig else ("Descriptive", "neutral")
        pnum = (f"{int(round(pos / 100 * n))} of {int(n)} rose vs {fmt(u, '.0f', suffix='%')} usually · "
                f"p {fmt(p, '.2f')}")
    rows.append({"id": "L-10", "claim": "Past moments like today point the way", "status": ps, "number": pnum,
                 "meaning": ("These moments' outcomes differ from the usual by more than chance." if ps[0] == "Significant"
                             else "A small sample; their outcomes are description, not evidence.")})
    return rows


def ledger_html(rows: list[dict]) -> str:
    """The Ledger as panel rows (claim + meaning · status chip · deciding number). Escaped."""
    head = ('<div class="ledger-row ledger-head"><span>ID</span><span>Claim</span><span>Status</span>'
            '<span>Deciding Number</span></div>')
    body = "".join(
        f'<div class="ledger-row"><span class="lg-id">{esc(r.get("id"))}</span>'
        f'<span class="lg-claim">{esc(r.get("claim"))}<span class="lg-mean">{esc(r.get("meaning"))}</span></span>'
        f'<span class="lg-status">{render_chip(str(_list(r.get("status"))[0]), str(_list(r.get("status"))[1]), as_html=True)}</span>'
        f'<span class="lg-num">{esc(r.get("number"))}</span></div>'
        for r in rows if len(_list(r.get("status"))) == 2)
    return f'<div class="ledger">{head}{body}</div>'


# ═══════════════════════════════════════════════════════════════════════════
#  1 · WALK-FORWARD EDGE — EVERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

#: (key in system_edges, engine, what is graded, chart hue or None = table only)
_IN_PROSE = {"read": "the read", "top_down": "the top-down pillar", "bottom_up": "the bottom-up pillar",
             "agreement": "the agreement pillar", "oscillator": "signed agreement", "aarambh": "Aarambh's forecast"}
_EDGE_ROWS = (
    ("read", "Nishkarsh read (L-13)", "the read — mean of the three pillars", "accent"),
    ("top_down", "Top-down pillar", "forecast · extension · lookback breadth · crossings", "violet"),
    ("bottom_up", "Bottom-up pillar", "the stocks' oscillator, zones, regimes, odds, crossings", "cyan"),
    ("agreement", "Agreement pillar", "signed agreement, standardized", "slate"),
    ("oscillator", "Signed agreement (L-12)", "the agreement oscillator on its own, every date", None),
    ("aarambh", "Aarambh forecast (L-03)", "the ensemble's forecast, every date", None),
)


def edge_table_rows(edges: dict) -> list[dict]:
    """The edge table, one row per graded series (formats only; strings carry their sign)."""
    out = []
    for key, name, what, _hue in _EDGE_ROWS:
        e = _d(_d(edges).get(key))
        ic, se = num(e.get("ic")), num(e.get("se"))
        pos, nb = edge_periods(e)
        out.append({"Engine": name, "Graded": what,
                    "IC": signed(ic, "+.3f") if ic is not None else "—",
                    "± SE": f"{se:.3f}" if se is not None else "—",
                    "Windows": int(num(e.get("n")) or 0) if ic is not None else 0,
                    "Periods +": f"{pos}/{nb}" if nb else "—",
                    "Read": edge_tier(ic)[1]})
    return out


def _edge_bars(edges: dict) -> go.Figure | None:
    """Grouped per-period IC bars: the read (accent) and its three pillars (violet / cyan / slate)."""
    series = []
    for key, name, _what, hue in _EDGE_ROWS:
        if hue is None:
            continue
        pts = [(str(b.get("start"))[:4], num(b.get("ic"))) for b in _list(_d(_d(edges).get(key)).get("blocks"))
               if isinstance(b, dict) and num(b.get("ic")) is not None and str(b.get("start"))[:4].isdigit()]
        if pts:
            series.append((name.split(" (")[0], hue, pts))
    if not series:
        return None
    fig = go.Figure()
    for name, hue, pts in series:
        fig.add_trace(go.Bar(x=[p for p, _ in pts], y=[v for _, v in pts], name=name,
                             marker=dict(color=chart_color(hue)),
                             customdata=[[signed(v, "+.3f")] for _, v in pts],
                             hovertemplate="%{fullData.name}: %{customdata[0]}<extra></extra>"))
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode="group")
    style_axes(fig, y_title="IC per period")
    fig.update_xaxes(type="category", categoryorder="category ascending")
    return fig


def _read_grade_items(con: dict, read_edge: dict, h: int) -> list[dict]:
    """The read's own grade (L-13) — six cards, the glossary in their tooltips."""
    ev = _d(_d(con.get("nko")).get("evidence"))
    ic, se, p = num(read_edge.get("ic")), num(read_edge.get("se")), num(ev.get("p"))
    tier, chip = edge_tier(ic)
    pos, nb = edge_periods(read_edge)
    n = int(num(read_edge.get("n")) or 0)
    return [
        {"label": "Read Edge (IC)", "value": signed(ic, "+.3f"),
         "subtext": f"{chip.title()} · ± {se:.3f} (1 SE)" if se is not None else chip.title(),
         "color_class": {"solid": "success", "modest": "success", "marginal": "warning",
                         "no_edge": "danger"}.get(tier, "neutral"), "tooltip": TOOLTIPS["ic"]},
        {"label": "p-Value", "value": fmt(p, ".2f"), "subtext": f"one-sided · evidence below {EDGE_ALPHA:g}",
         "color_class": "success" if p is not None and p < EDGE_ALPHA else "neutral", "tooltip": TOOLTIPS["p"]},
        {"label": "Min Detectable", "value": fmt(ev.get("mde"), ".2f"),
         "subtext": f"the least IC {plural(n, 'window')} can confirm", "color_class": "neutral",
         "tooltip": TOOLTIPS["mde"]},
        {"label": "HAC t", "value": signed(ev.get("hac_t"), "+.2f"), "subtext": "every day, corrected for overlap",
         "color_class": "neutral", "tooltip": TOOLTIPS["hac"]},
        {"label": "Positive Periods", "value": f"{pos}/{nb}" if nb else "—",
         "subtext": "calendar years with IC above zero",
         "color_class": ("success" if nb and pos > nb * 0.6 else "warning" if nb and pos >= nb * 0.5
                         else "danger" if nb else "neutral"), "tooltip": TOOLTIPS["periods"]},
        {"label": "Windows Scored", "value": f"{n:,}" if n else "—",
         "subtext": f"one per {h} sessions, both systems running", "color_class": "neutral",
         "tooltip": TOOLTIPS["windows"]},
    ]


def _render_edges(con: dict, edges: dict, h: int) -> None:
    rows = edge_table_rows(edges)
    graded = [(_IN_PROSE[key], num(_d(edges.get(key)).get("ic"))) for key, _n, _w, _h in _EDGE_ROWS
              if num(_d(edges.get(key)).get("ic")) is not None]
    read = _d(edges.get("read"))
    if graded:
        lo, hi = min(graded, key=lambda t: t[1]), max(graded, key=lambda t: t[1])
        ric = num(read.get("ic"))
        lead = (f"The read itself is {edge_tier(ric)[1].lower()} (IC {signed(ric, '+.3f')}"
                + (f" ± {num(read.get('se')):.3f}" if num(read.get("se")) is not None else "")
                + f" over {plural(read.get('n'), 'window')}); " if ric is not None else
                "The read itself has too little two-system history to grade; ")
        spread = (f"the {len(graded)} graded parts range from {signed(lo[1], '+.3f')} ({lo[0]}) to "
                  f"{signed(hi[1], '+.3f')} ({hi[0]})." if len(graded) > 1 else
                  f"only {lo[0]} has a graded edge ({signed(lo[1], '+.3f')}).")
        desc = (f"Each part's lean against the Nifty's next {h} sessions, on non-overlapping windows — the same "
                f"grading every engine page shows for itself. {lead}{spread}")
    else:
        desc = (f"Each part's lean against the Nifty's next {h} sessions, on non-overlapping windows — nothing has "
                "enough scored history to grade yet.")
    render_section_header("Walk-Forward Edge — Every Engine", desc, icon="trending", accent="accent")
    if not graded:
        render_empty_state("Nothing graded yet",
                           "The walk-forward edges need roughly a year of history past each engine's warm-up "
                           "before a non-overlapping read exists.", eyebrow="System")
        return
    if num(read.get("ic")) is not None:
        render_kpi_strip(_read_grade_items(con, read, h), max_cols=6, key="system-read")
    render_table_panel(pd.DataFrame(rows), "system-edges", units=f"non-overlapping {h}d IC", windowed=False,
                       label_col="Engine", max_height=320)
    fig = _edge_bars(edges)
    if fig is not None:
        render_chart_panel(fig, "system-edge-blocks", units="IC per calendar year, the read and its pillars",
                           windowed=False)
    se = num(read.get("se"))
    render_note("A durable edge is positive in most periods, not large in one."
                + (f" With {plural(read.get('n'), 'non-overlapping window')} the read's 1-SE band is "
                   f"±{se:.3f} — read every IC above against it." if se is not None else ""))


# ═══════════════════════════════════════════════════════════════════════════
#  3 · MODEL — what each system is, as this run fitted it
# ═══════════════════════════════════════════════════════════════════════════

def _aarambh_items(rr: dict, rc: dict, forward: bool, h: int) -> list[dict]:
    a = _d(rr.get("aarambh"))
    wf = _d(a.get("wf_meta"))
    feats = [str(f) for f in (_list(a.get("feature_names")) or _list(rc.get("features")))]
    lo, hi = num(wf.get("min_train")), num(wf.get("max_train"))
    return [
        {"label": "Mode", "value": "Forward" if forward else "Levels",
         "subtext": f"forecasts the next {h} sessions" if forward else LEVELS_TAG,
         "color_class": "neutral" if forward else "warning",
         "tooltip": ("Forward: Aarambh forecasts the Nifty's return over the next sessions, and that forecast is "
                     "what is graded. Levels: it fits the sheet's level as valuation context only.")},
        {"label": "Training Window",
         "value": f"{lo:,.0f}–{hi:,.0f}" if lo is not None and hi is not None else "—",
         "subtext": f"rows per fit · refit every {fmt(wf.get('refit'), '.0f')} sessions",
         "color_class": "neutral", "tooltip": TOOLTIPS["train"]},
        {"label": "Ensemble", "value": "4 models", "subtext": "Ridge · Huber · ElasticNet · PCA-OLS, equal weights",
         "color_class": "neutral",
         "tooltip": "The forecast is the equal-weight mean of the members that fit at each refit; nothing weights "
                    "them by past accuracy."},
        {"label": "Inputs", "value": f"{len(feats)}" if feats else "—", "subtext": "series from your sheet",
         "color_class": "neutral", "tooltip": ", ".join(feats) if feats else ""},
        {"label": "Forecasts Since", "value": pretty_date(a.get("first_valid")),
         "subtext": "the first out-of-sample forecast", "color_class": "neutral"},
    ]


def _nirnay_items(rr: dict) -> list[dict]:
    n = _d(rr.get("nirnay"))
    lat = _d(n.get("latest"))
    span = _list(n.get("span"))
    an, req = num(n.get("stocks_analyzed")), num(n.get("stocks_requested"))
    failed = len(_d(n.get("failed"))) if isinstance(n.get("failed"), dict) else len(_list(n.get("failed")))
    live, uni = num(lat.get("n_live")), num(lat.get("n_universe"))
    os_c, ob_c = num(n.get("os_cut_median_live")), num(n.get("ob_cut_median_live"))
    trend = NIRNAY_POLARITY == "trend"
    return [
        {"label": "Stocks Analysed", "value": f"{an:.0f}/{req:.0f}" if an is not None and req else "—",
         "subtext": "all fetched" if not failed else f"{plural(failed, 'stock')} failed",
         "color_class": "neutral" if not failed else "warning"},
        {"label": "History", "value": f"{num(n.get('trading_days')):,.0f}" if num(n.get("trading_days")) else "—",
         "subtext": (f"trading days · {pretty_date(span[0])} – {pretty_date(span[1])}" if len(span) == 2
                     else "trading days"), "color_class": "neutral"},
        {"label": "Live Today", "value": f"{live:.0f}/{uni:.0f}" if live is not None and uni else "—",
         "subtext": f"stocks with a reading on {pretty_date(lat.get('date'))}", "color_class": "neutral"},
        {"label": "Adaptive Cuts", "value": f"{signed(os_c)} / {signed(ob_c)}",
         "subtext": "oversold / overbought · median of live stocks", "color_class": "neutral",
         "tooltip": TOOLTIPS["cuts"]},
        {"label": "Polarity", "value": "Trend" if trend else "Contrarian",
         "subtext": "strength counts as bullish" if trend else "oversold counts as bullish",
         "color_class": "neutral", "tooltip": TOOLTIPS["polarity"]},
    ]


def _read_items(con: dict, rr: dict, v: dict) -> list[dict]:
    k = _d(con.get("nko"))
    comps = [_d(c) for c in _list(k.get("components")) if isinstance(c, dict)]
    per = [sum(1 for c in comps if c.get("pillar") == p) for p in PILLARS]
    w = _d(_d(v.get("kpi")).get("pillar_weights"))
    ws = [num(w.get(p)) for p in PILLARS]
    n = _d(rr.get("nirnay"))
    two, av = num(n.get("overlap_valid_dates")), num(n.get("aarambh_valid_dates"))
    words = ("top-down", "bottom-up", "agreement")
    equal = all(x is not None for x in ws) and max(ws) - min(ws) < 1e-6
    return [
        {"label": "Inputs → Pillars", "value": f"{len(comps)} → {len(PILLARS)}" if comps else "—",
         "subtext": " · ".join(f"{c} {wd}" for c, wd in zip(per, words)) if comps else "no read this run",
         "color_class": "neutral", "tooltip": TOOLTIPS["pillars"]},
        {"label": "Pillar Weights",
         "value": " · ".join(fmt(x, ".2f") for x in ws) if all(x is not None for x in ws) else "—",
         "subtext": " · ".join(words) + (" — equal, fixed" if equal else " — single-system today"),
         "color_class": "neutral", "tooltip": TOOLTIPS["weights"]},
        {"label": "Standardized On", "value": f"≥ {CONV2_ZMIN_PERIODS}",
         "subtext": "earlier sessions per input, never later", "color_class": "neutral",
         "tooltip": TOOLTIPS["standardize"]},
        {"label": "Two-System Dates", "value": f"{two:,.0f}" if two is not None else "—",
         "subtext": (("every forecast date had both systems" if two == av else
                      f"of {av:,.0f} forecast dates · the rest are single-system") if av and two is not None else
                     "dates both systems published"), "color_class": "neutral", "tooltip": TOOLTIPS["two_system"]},
    ]


def _render_model(con: dict, rr: dict, rc: dict, v: dict, h: int) -> None:
    forward = con.get("forward_mode") is not False
    a = _d(rr.get("aarambh"))
    n = _d(rr.get("nirnay"))
    wf = _d(a.get("wf_meta"))
    has_n = bool(num(n.get("stocks_analyzed")))
    parts = []
    if num(wf.get("min_train")) is not None:
        parts.append(f"Aarambh refits a four-model ensemble walk-forward on {fmt(wf.get('min_train'), ',.0f')}–"
                     f"{fmt(wf.get('max_train'), ',.0f')} rows")
    if has_n:
        parts.append(f"Nirnay reads {fmt(n.get('stocks_analyzed'), '.0f')} of {fmt(n.get('stocks_requested'), '.0f')} "
                     f"stocks, as {'trend' if NIRNAY_POLARITY == 'trend' else 'contrarian'}")
    comps = _list(_d(con.get("nko")).get("components"))
    if comps:
        parts.append(f"the read averages {len(comps)} inputs into {len(PILLARS)} pillars")
    desc = ("What each system is, as this run fitted it: " + "; ".join(parts) + "." if parts else
            "What each system is, as this run fitted it.")
    render_section_header("Model", desc, icon="cpu")
    render_sub_header("Aarambh — top-down")
    render_kpi_strip(_aarambh_items(rr, rc, forward, h), max_cols=5, key="system-model-aarambh")
    render_sub_header("Nirnay — bottom-up")
    if has_n:
        render_kpi_strip(_nirnay_items(rr), max_cols=5, key="system-model-nirnay")
    else:
        render_empty_state("No Nirnay stocks this run",
                           "There was no constituent data, so the read is top-down only and the bottom-up and "
                           "agreement pillars are left out.", eyebrow="Nirnay")
    render_sub_header("Nishkarsh — the read")
    render_kpi_strip(_read_items(con, rr, v), max_cols=4, key="system-model-read")


# ═══════════════════════════════════════════════════════════════════════════
#  4 · DATA LAYER HEALTH
# ═══════════════════════════════════════════════════════════════════════════

_PROV = {"live": ("LIVE", "success"), "cache": ("CACHED", "info"), "stale": ("STALE", "warning"),
         "fallback": ("FALLBACK", "warning"), "empty": ("EMPTY", "danger")}


def provenance_items(run_config) -> list[dict]:
    """One card per source in ``SOURCE_NAMES`` order: the provenance word (toned), as-of · the
    note's first clause; the whole note (links hidden) in the tooltip. Formats only."""
    prov = _d(_d(run_config).get("provenance"))
    items = []
    for key, name in SOURCE_NAMES.items():
        p = prov.get(key)
        if not isinstance(p, dict):
            items.append({"label": name, "value": "—", "subtext": "not used this run", "color_class": "neutral"})
            continue
        note = redact(_str(p.get("note")))
        word, tone = _PROV.get(_str(p.get("provenance")).lower(), ("—", "neutral"))
        if "SYNTHETIC" in note.upper():
            word, tone = "SYNTHETIC", "warning"
        missing = [str(m) for m in _list(p.get("missing")) if _str(m)]
        sub = [f"as of {pretty_date(p.get('as_of'))}" if _str(p.get("as_of")) else "no as-of date"]
        if missing:
            sub.append(f"{len(missing)} series missing")
        first = note.split(";")[0].strip()
        if first.upper().startswith("SYNTHETIC"):
            first = "seeded world, not the market"
        if first:
            sub.append(first)
        tip = note + (f" Missing: {', '.join(missing)}." if missing else "")
        items.append({"label": name, "value": word, "subtext": " · ".join(sub), "color_class": tone,
                      "tooltip": tip.strip()})
    return items


def _process_health_items() -> tuple[list[dict], list[dict]]:
    """This server's cache and circuit-breaker state (process-wide, shared by sessions)."""
    try:
        from data.cache import all_caches
        from data.circuit_breaker import all_circuits
    except Exception:  # noqa: BLE001 — diagnostics only
        return [], []
    caches, circuits = [], []
    now = time.time()
    for c in all_caches():
        s = c.stats()
        total = int(s.get("hits", 0)) + int(s.get("misses", 0))
        hit = float(s.get("hit_rate", 0.0)) * 100.0
        last = num(s.get("last_fetch_time"))
        caches.append({
            "label": f"Cache · {_str(s.get('namespace'), 'cache')}",
            "value": f"{hit:.0f}%" if total else "—",
            "subtext": f"{int(s.get('disk_entries', 0))} on disk · "
                       + (f"last fetch {(now - last) / 60:.0f}m ago" if last else "no fetch this session"),
            "color_class": ("neutral" if not total else "success" if hit >= 70 else "warning" if hit >= 30
                            else "danger"),
            "tooltip": (f"{s.get('hits', 0)} hits / {s.get('misses', 0)} misses · {s.get('stale_hits', 0)} "
                        f"stale-fallback · {s.get('writes', 0)} writes · TTL {s.get('ttl_seconds', 0)}s · "
                        f"version {s.get('version', '')}")})
    for cb in all_circuits():
        s = cb.get_state()
        state = _str(s.get("state")).lower()
        word, tone = {"closed": ("CLOSED", "success"), "half_open": ("HALF-OPEN", "warning")}.get(
            state, ("OPEN", "danger"))
        lf = num(s.get("last_failure"))
        circuits.append({
            "label": f"Circuit · {_str(s.get('name'), 'service').replace('_', ' ')}",
            "value": word,
            "subtext": (f"{s.get('failure_count', 0)} failures · last {(now - lf) / 60:.0f}m ago" if lf else
                        f"{s.get('success_count', 0)} successful calls"),
            "color_class": tone,
            "tooltip": (f"Opens after {plural(s.get('failure_threshold'), 'failure')}; retries after "
                        f"{fmt(s.get('recovery_timeout'), '.0f')}s. OPEN blocks calls; HALF-OPEN allows one test call.")})
    used = any(c["value"] != "—" for c in caches) or any(
        not c["subtext"].startswith("0 successful") for c in circuits)
    return (caches, circuits) if used else ([], [])


def _render_data_health(rc: dict) -> None:
    items = provenance_items(rc)
    used = [i for i in items if i["value"] != "—"]
    synthetic = is_synthetic(rc)
    if synthetic:
        desc = ("Where each input came from on this run and how current it is — every source is the seeded "
                "synthetic world, not the market.")
    else:
        live = sum(1 for i in used if i["value"] in ("LIVE", "CACHED"))
        desc = (f"Where each input came from on this run and how current it is — {live} of "
                f"{plural(len(used), 'source')} current" + ("." if live == len(used) else
                                                            "; the rest are named in the notices above."))
    render_section_header("Data Layer Health", desc, icon="database", accent="cyan")
    render_kpi_strip(items, max_cols=5, key="system-data")
    if not synthetic:
        caches, circuits = _process_health_items()
        if caches:
            render_kpi_strip(caches, max_cols=4, key="system-cache")
        if circuits:
            render_kpi_strip(circuits, max_cols=3, key="system-circuit")


# ═══════════════════════════════════════════════════════════════════════════
#  5 · METHOD & LIMITS
# ═══════════════════════════════════════════════════════════════════════════

def method_html() -> str:
    rows = "".join(f'<div class="lookback-row prose"><span class="lbl">{esc(k)}</span>'
                   f'<span class="val">{esc(v)}</span></div>' for k, v in METHOD_LINES)
    return f'<div class="panel-specs method-specs">{rows}</div>'


# ═══════════════════════════════════════════════════════════════════════════
#  THE PAGE
# ═══════════════════════════════════════════════════════════════════════════

def render_system_page() -> None:
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    rr, rc = _d(ss.get("run_report")), _d(ss.get("run_config"))
    v = _d(con.get("verdict"))
    h = int(num(con.get("horizon")) or 10)
    if not con:
        render_section_header("Walk-Forward Edge — Every Engine", "This run produced no conclusion to grade.",
                              icon="trending", accent="accent")
        render_empty_state("No read for this run", "The run produced no conclusion, so there is nothing to grade.",
                           eyebrow="System")
        return

    _render_edges(con, system_edges(con), h)

    rows = build_ledger_rows(con, ss.get("precedent"))
    graded = [r for r in rows if r.get("id")]
    held = sum(1 for r in graded if _list(r.get("status"))[1:2] == ["success"])
    read_status = {"Validated": "validated", "Unproven": "unproven"}.get(
        _str(_list(rows[0].get("status"))[0]) if rows else "", "not yet graded")
    held_txt = f"{held} of the {len(graded)} graded claims "
    render_section_header(
        "The Ledger",
        f"Every claim Nishkarsh makes, its status on this run's data and the one number that decides it — "
        f"{held_txt}{'holds' if held == 1 else 'hold'}; the read itself is {read_status}.",
        icon="shield", accent="accent")
    with panel("system-ledger", context=default_chart_context(f"{len(graded)} claims · IDs per docs/EDGE_LEDGER.md",
                                                             windowed=False),
               footer=esc(f"Graded at one-sided p < {EDGE_ALPHA:g} by rules written before each test; tradeable "
                          f"needs a significant HAC t net of {BACKTEST_REFERENCE_BPS:g} bps a trade.")):
        st.markdown(ledger_html(rows), unsafe_allow_html=True)

    _render_model(con, rr, rc, v, h)
    _render_data_health(rc)

    render_section_header(
        "Method & Limits",
        "How the read is built, what it is measured against, and what it cannot tell you.",
        icon="help-circle")
    with panel("system-method", context=f"{len(METHOD_LINES)} notes · they apply to every page"):
        st.markdown(method_html(), unsafe_allow_html=True)
