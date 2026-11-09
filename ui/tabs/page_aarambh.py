"""
Nishkarsh — Aarambh: the top-down forecast (≈ Tattva's Mūla page; hue: violet).

Reading order — the house convention every engine page follows:

  1 ANSWER    what does Aarambh say?          the top-down pillar · its forecast · its edge
  2 ANCHOR    what is it read against?        Forecast & the Nifty (the pillar over the index)
  3 INPUTS    what is the pillar made of?     its four inputs, each against its own past
  4 TRUST     can the forecast be believed?   model quality · walk-forward edge (+ its stages
                                              and the cost-aware backtest — Aarambh's own evidence)

The page reads the one run in session state and formats it: the pillar and its inputs from the
Nishkarsh Oscillator's decomposition (``conclusion["nko"]`` and ``conclusion_frame``), the
forecast from ``aarambh_ts``, the walk-forward facts from ``run_report["aarambh"]``, the edge in
the one edge contract (``nko["pillar_edge"]["top_down"]``). Sign: the pillar and every input are
+ = bullish, drawn green above zero and red below. Levels mode carries ``LEVELS_TAG``. Windowed
page: the window control sits in "Forecast & The Nifty" and frames every chart here.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from convergence.views import PILLAR_NO_LEAN, lean_phrase
from ui.components import (
    _d, _list, _str, esc, fmt, index_name, num, pretty_date, render_chart_panel, render_edge_panel,
    render_empty_state, render_kpi_strip, render_section_header, render_sub_header, render_table_panel, signed,
    tone_of,
)
from ui.copy import LEVELS_TAG
from ui.palette import UI_CHART_HEIGHT_STACKED, UI_CHART_HEIGHT_XLARGE
from ui.tabs.engine_common import (
    MIN_SESSIONS, add_sign_fill, by_date, ic_card, series_by_date, window_slice,
)
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, style_axes

#: The pillar's four inputs, in the oscillator's own order: (component id, card/axis label).
INPUTS = (("forecast", "Forecast"), ("extension", "Extension"), ("zone_breadth_a", "Zone breadth"),
          ("crossings_a", "Cut crossings"))
#: |z| beyond which a pillar reading is unusually stretched for itself (dotted on the chart).
DEEP_Z = 2.0
#: How the run's target column reads in a sentence.
_TARGET_WORDS = {"NIFTY50_PE": "the Nifty's P/E", "NIFTY50_PB": "the Nifty's P/B",
                 "NIFTY50_DY": "the Nifty's dividend yield"}
#: Rendered row height of the table primitive (27px nominal + cell padding/rule) — sizes the
#: iframe of a short table so its last row is not clipped.
_ROW_H = 30
#: The forecast's stages (``conclusion["chain"]``), in plain words.
_STAGE = {"forecast": "Aarambh forecast", "valuation_head": "Valuation head", "engine_conviction": "Engine conviction",
          "C": "Conclusion C", "kappa_shrunk": "After κ shrink"}

TOOLTIPS = {
    "pillar": (
        "The top-down pillar: the average of Aarambh's four inputs — its forecast, how stretched that "
        "forecast is across lookbacks, lookback zone breadth and lookback cut crossings — each "
        "standardized against its own past, then re-standardized. + = bullish. It is one of the three "
        "pillars the Nishkarsh read averages."
    ),
    "forecast": (
        "Aarambh's walk-forward forecast of the next sessions' change, standardized against its own "
        "earlier forecasts. Every forecast was made by models that had not seen its outcome: they are "
        "refitted on a schedule, on rows whose outcomes were already known, with a purge gap."
    ),
    "extension": (
        "How far the forecast stands outside its own recent range, averaged across Aarambh's lookbacks "
        "and sign-flipped so + = bullish. A forecast far from its usual range reads larger here."
    ),
    "zone_breadth": (
        "Of Aarambh's lookbacks, the share reading the forecast in its bullish zone minus the share in "
        "its bearish zone (points), standardized against its own past."
    ),
    "ic": (
        "Rank correlation of the top-down pillar with the Nifty's forward return over the horizon, "
        "scored on non-overlapping windows on dates both systems read. 0 = no skill."
    ),
    "windows": (
        "The forecast is refitted on a schedule; each refit trains only on rows whose outcomes were "
        "known by then, so no forecast is graded on data it was fitted to."
    ),
    "oos": "Rows that carry a genuine out-of-sample forecast. Everything before the first one is training history.",
    "skill": (
        "Out-of-sample R² of the forecast against simply predicting no change over the horizon. Below 0, "
        "predicting no change did better on squared error — a forecast can still rank days usefully "
        "(the walk-forward IC) while losing here."
    ),
    "spread": (
        "How far apart the ensemble's models (Ridge, Huber, ElasticNet and PCA-OLS, equal weights) put "
        "today's forecast. Wide = the models disagree about the forecast itself."
    ),
    "purge": (
        "Rows left out between each training window and the forecast it makes — the forecast horizon — "
        "so overlapping outcomes cannot leak from training into the forecast."
    ),
}


def _target_words(target) -> str:
    t = _str(target)
    return _TARGET_WORDS.get(t.upper(), t or "the target")


def _latest_forecast(ts: pd.DataFrame) -> tuple[float | None, float | None]:
    """``(forecast, model spread)`` on the latest Valid row of the published Aarambh frame."""
    if ts.empty or "Forecast" not in ts.columns:
        return None, None
    f = ts
    if "Valid" in f.columns:
        f = f[f["Valid"].fillna(False).astype(bool)]
    f = f[pd.to_numeric(f["Forecast"], errors="coerce").notna()]
    if f.empty:
        return None, None
    last = f.iloc[-1]
    return num(last.get("Forecast")), num(last.get("ModelSpread"))


def _comp_z(k: dict) -> dict:
    return {_str(_d(c).get("id")): num(_d(c).get("z")) for c in _list(k.get("components"))}


def _pct(log_change) -> float | None:
    """A log change as a percent move (display units)."""
    v = num(log_change)
    return None if v is None else (math.exp(v) - 1.0) * 100.0


# ═══════════════════════════════════════════════════════════════════════
#  1 · THE READ
# ═══════════════════════════════════════════════════════════════════════

def _answer(td, zf, fc_pct, forward: bool, h: int, target: str) -> str:
    if td is None:
        return ("Aarambh forecasts the Nifty from the top down; there is no top-down reading for the latest "
                "date of this run.")
    lean = lean_phrase(td)
    tail = (f"the top-down pillar reads {signed(td, '+.2f')}σ: "
            + (f"it {lean}." if lean != "no lean" else "it has no lean."))
    if not forward:
        return (f"Aarambh reads {target} against the level its inputs imply ({LEVELS_TAG}); against its own "
                f"past that reads {signed(zf, '+.2f')}σ, and with its extension and lookback breadth {tail}")
    head = (f"Aarambh forecasts {target} {signed(fc_pct, '+.2f')}% over the next {h} sessions"
            if fc_pct is not None else f"Aarambh forecasts the next {h} sessions")
    return (head + (f"; against its own past that is {signed(zf, '+.2f')}σ" if zf is not None else "")
            + f", and with its extension and lookback breadth {tail}")


def _read_items(td, z: dict, fr_last, fc_pct, forward, h, edge) -> list[dict]:
    zb = num(fr_last.get("in_zone_breadth_a")) if fr_last is not None else None
    ext = num(fr_last.get("in_extension")) if fr_last is not None else None
    lean = lean_phrase(td)
    return [
        {"label": "Top-Down Pillar", "value": f"{signed(td, '+.2f')}σ" if td is not None else "—",
         "subtext": f"{lean} · one of three pillars" if lean else "no reading for the latest date",
         "color_class": tone_of(td, 0.5), "tooltip": TOOLTIPS["pillar"]},
        {"label": "Forecast", "value": f"{signed(z.get('forecast'), '+.2f')}σ" if z.get("forecast") is not None
         else "—",
         "subtext": ((f"{signed(fc_pct, '+.2f')}% over the next {h} sessions" if fc_pct is not None
                      else f"next {h} sessions, standardized") if forward else LEVELS_TAG),
         "color_class": tone_of(z.get("forecast"), 0.5), "tooltip": TOOLTIPS["forecast"]},
        {"label": "Forecast Extension", "value": f"{signed(z.get('extension'), '+.2f')}σ"
         if z.get("extension") is not None else "—",
         "subtext": (f"raw {signed(ext, '+.2f')} · across lookbacks" if ext is not None else "across lookbacks"),
         "color_class": tone_of(z.get("extension"), 0.5), "tooltip": TOOLTIPS["extension"]},
        {"label": "Lookback Zone Breadth", "value": f"{signed(z.get('zone_breadth_a'), '+.2f')}σ"
         if z.get("zone_breadth_a") is not None else "—",
         "subtext": (f"{signed(zb, '+.0f')} pts · bullish − bearish lookbacks" if zb is not None
                     else "bullish − bearish lookbacks"),
         "color_class": tone_of(z.get("zone_breadth_a"), 0.5), "tooltip": TOOLTIPS["zone_breadth"]},
        ic_card(edge, "top-down pillar", h, TOOLTIPS["ic"]),
    ]


# ═══════════════════════════════════════════════════════════════════════
#  2 · FORECAST & THE NIFTY
# ═══════════════════════════════════════════════════════════════════════

def _forecast_chart(fr: pd.DataFrame, nifty: pd.Series, ts: pd.DataFrame, forward: bool, name: str,
                    target: str) -> None:
    """The Nifty (top, full ink) over the top-down pillar it is read against (bottom)."""
    last = fr.index.max()
    pw = window_slice(pd.to_numeric(fr["P_top_down"], errors="coerce"), last).dropna()
    if len(pw) < MIN_SESSIONS:
        render_empty_state(
            "Building top-down history",
            "The pillar is standardized against its own past, which needs a few weeks of readings in this "
            "window before it is drawn. Widen the chart window, or run again once more history exists.",
            eyebrow="Aarambh")
        return
    start = pw.index.min()
    nw = nifty[(nifty.index >= start) & (nifty.index <= last)] if len(nifty) else nifty
    context = (not forward) and {"Actual", "FairValue"} <= set(ts.columns)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.06,
                        specs=[[{"secondary_y": bool(context)}], [{"secondary_y": False}]])
    if len(nw):
        fig.add_trace(go.Scatter(x=list(nw.index), y=nw.to_numpy(dtype=float), mode="lines", name=name,
                                 line=dict(color=chart_color("accent"), width=1.6)), row=1, col=1)
    if context:
        tw = ts[(ts.index >= start) & (ts.index <= last)]
        fig.add_trace(go.Scatter(x=list(tw.index), y=pd.to_numeric(tw["Actual"], errors="coerce"), mode="lines",
                                 name=f"{target} (context)", line=dict(color=chart_rgba("slate", 0.8), width=1.1)),
                      row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=list(tw.index), y=pd.to_numeric(tw["FairValue"], errors="coerce"),
                                 mode="lines", name="Fair value (context)",
                                 line=dict(color=chart_color("violet"), width=1.1, dash="dot")),
                      row=1, col=1, secondary_y=True)

    x, y = list(pw.index), pw.to_numpy(dtype=float)
    fig.add_hrect(y0=-PILLAR_NO_LEAN, y1=PILLAR_NO_LEAN, fillcolor=grid_rgba(0.05), line_width=0, row=2, col=1)
    add_sign_fill(fig, x, y, above="emerald", below="rose", alpha=0.12, row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Top-down pillar",
                             line=dict(color=chart_color("violet"), width=1.4)), row=2, col=1)
    for lvl, nm in ((DEEP_Z, "emerald"), (-DEEP_Z, "rose")):
        fig.add_hline(y=lvl, line_dash="dot", line_color=chart_rgba(nm, 0.35), line_width=0.6, row=2, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=2, col=1)

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_XLARGE))
    style_axes(fig, y_title=name, row=1, col=1)
    style_axes(fig, y_title="Top-down (σ)", row=2, col=1)
    if context:
        fig.update_yaxes(title_text=target, showgrid=False, row=1, col=1, secondary_y=True)
    render_chart_panel(fig, "aarambh_forecast",
                       units="index level · top-down σ" + (" · context" if context else ""), window=True)


# ═══════════════════════════════════════════════════════════════════════
#  3 · ITS INPUTS
# ═══════════════════════════════════════════════════════════════════════

def _inputs_chart(fr: pd.DataFrame) -> None:
    """The four oriented inputs' z, one row each, on the pillar's dates."""
    cols = [f"z_{cid}" for cid, _ in INPUTS]
    if not set(cols) <= set(fr.columns):
        render_empty_state("No input history", "This run published no top-down inputs.", eyebrow="Aarambh")
        return
    w = window_slice(fr[cols].apply(pd.to_numeric, errors="coerce")).dropna(how="all")
    if len(w) < MIN_SESSIONS:
        render_empty_state("Building input history",
                           "Each input is standardized against its own past and needs a few weeks of readings in "
                           "this window before it is drawn.", eyebrow="Aarambh")
        return
    fig = make_subplots(rows=len(INPUTS), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    x = list(w.index)
    for i, (cid, lab) in enumerate(INPUTS, start=1):
        y = w[f"z_{cid}"].to_numpy(dtype=float)
        add_sign_fill(fig, x, y, above="emerald", below="rose", alpha=0.08, row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=lab, line=dict(color=chart_color("violet"), width=1.2)),
                      row=i, col=1)
        fig.add_hline(y=0, line_color=grid_rgba(0.08), line_width=0.5, row=i, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED - 80, show_legend=False))
    for i, (_, lab) in enumerate(INPUTS, start=1):
        style_axes(fig, y_title=lab, row=i, col=1)
    render_chart_panel(fig, "aarambh_inputs", units="input z · + = bullish",
                       footer=esc("Each input is standardized against its own past only; the pillar is their "
                                  "average, re-standardized."))


# ═══════════════════════════════════════════════════════════════════════
#  4 · MODEL QUALITY
# ═══════════════════════════════════════════════════════════════════════

def quality_items(aarambh: dict, spread_today, forward: bool, h: int) -> list[dict]:
    """The five Model Quality cards, from ``run_report["aarambh"]`` — formatted, never graded here."""
    a = _d(aarambh)
    wf, ms = _d(a.get("wf_meta")), _d(a.get("model_stats"))
    chunks, starved, fb = num(wf.get("chunks_total")), num(wf.get("chunks_skipped_starved")), num(wf.get("chunks_fallback"))
    wsub = (f"refit every {fmt(wf.get('refit'), '.0f')} rows · train {fmt(wf.get('min_train'), ',.0f')}–"
            f"{fmt(wf.get('max_train'), ',.0f')}")
    if (starved or 0) > 0 or (fb or 0) > 0:
        wsub = f"{fmt(starved, '.0f')} starved · {fmt(fb, '.0f')} fallback"
    valid, rows = num(a.get("valid")), num(a.get("rows"))
    skill = num(ms.get("r2_vs_rw"))
    base = "no change" if forward else "random walk"
    avg = num(ms.get("avg_model_spread"))
    sp = num(spread_today)
    purge = num(wf.get("purge"))
    if forward:
        sp_val = f"{sp * 1e4:.0f} bps" if sp is not None else "—"
        sp_sub = f"average {avg * 1e4:.0f} bps · four models" if avg is not None else "four models"
    else:
        sp_val = fmt(sp, ".2f")
        sp_sub = f"target units · average {fmt(avg, '.2f')}"
    return [
        {"label": "Walk-Forward Windows", "value": fmt(chunks, ",.0f"), "subtext": wsub,
         "color_class": "warning" if (starved or 0) > 0 or (fb or 0) > 0 else "neutral",
         "tooltip": TOOLTIPS["windows"]},
        {"label": "Out-of-Sample Rows", "value": fmt(valid, ",.0f"),
         "subtext": (f"of {fmt(rows, ',.0f')} · from {pretty_date(a.get('first_valid'))}" if rows is not None
                     else "rows with a genuine forecast"),
         "color_class": "neutral", "tooltip": TOOLTIPS["oos"]},
        {"label": f"Skill vs {base.title()}", "value": signed(skill, "+.3f"),
         "subtext": (f"below 0: {base} did better" if skill is not None and skill < 0
                     else f"beats predicting {base}" if skill is not None else "not measured"),
         "color_class": ("success" if skill is not None and skill > 0 else
                         "warning" if skill is not None and skill > -0.05 else
                         "danger" if skill is not None else "neutral"),
         "tooltip": TOOLTIPS["skill"]},
        {"label": "Ensemble Spread", "value": sp_val, "subtext": sp_sub, "color_class": "info",
         "tooltip": TOOLTIPS["spread"]},
        {"label": "Purge", "value": f"{purge:.0f} rows" if purge is not None else "—",
         "subtext": (f"= the {h}-session horizon" if forward and purge is not None and purge >= h
                     else "levels fit — no forward purge" if not forward else "gap before each forecast"),
         "color_class": "success" if forward and purge is not None and purge >= h else "neutral",
         "tooltip": TOOLTIPS["purge"]},
    ]


# ═══════════════════════════════════════════════════════════════════════
#  5 · STAGES · BACKTEST (Aarambh's own evidence, under its edge)
# ═══════════════════════════════════════════════════════════════════════

def stage_table(chain) -> pd.DataFrame:
    """``conclusion["chain"]`` as a table: Stage · IC · Windows (empty when absent)."""
    rows = [{"Stage": _STAGE.get(_str(_d(c).get("stage")), _str(_d(c).get("stage")) or "—"),
             "IC": num(_d(c).get("ic")), "Windows": int(num(_d(c).get("n")) or 0)}
            for c in _list(chain) if isinstance(c, dict)]
    df = pd.DataFrame(rows, columns=["Stage", "IC", "Windows"])
    df["IC"] = pd.to_numeric(df["IC"], errors="coerce")
    return df


def backtest_table(bt) -> pd.DataFrame:
    """``conclusion["backtest"]`` by cost (+ the separate-periods reference row) as a table."""
    b = _d(bt)
    by = _d(b.get("by_cost"))
    keys = sorted((k for k in by if num(k) is not None), key=lambda s: num(s))
    rows = [(f"{fmt(k, 'g')} bps", _d(by[k])) for k in keys]
    if _d(b.get("nonoverlap_reference")):
        rows.append((f"{fmt(b.get('reference_bps'), 'g')} bps · separate periods", _d(b["nonoverlap_reference"])))
    out = [{"Cost": lab, "Gross Sharpe": num(r.get("gross_sharpe")), "Net Sharpe": num(r.get("net_sharpe")),
            "HAC t": num(r.get("t")),
            "Return / yr %": (num(r.get("ann_return")) * 100.0 if num(r.get("ann_return")) is not None else None),
            "Turnover / yr": num(r.get("turnover_per_year")), "Break-even bps": num(r.get("breakeven_bps")),
            "Days": int(num(r.get("n")) or 0)} for lab, r in rows]
    cols = ["Cost", "Gross Sharpe", "Net Sharpe", "HAC t", "Return / yr %", "Turnover / yr", "Break-even bps", "Days"]
    df = pd.DataFrame(out, columns=cols)
    for c in cols[1:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def render_aarambh_page() -> None:
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    k = _d(con.get("nko"))
    h = int(num(con.get("horizon")) or 10)
    forward = con.get("forward_mode") is not False
    target = _target_words(_d(ss.get("run_config")).get("target"))
    ts = by_date(ss.get("aarambh_ts"))
    fr = by_date(ss.get("conclusion_frame"))
    td = num(_d(k.get("pillars")).get("top_down"))
    z = _comp_z(k)
    fc, sp_today = _latest_forecast(ts)
    fc_pct = _pct(fc) if forward else None
    edge = _d(_d(k.get("pillar_edge")).get("top_down")) or None
    as_of = pd.Timestamp(k["as_of"]).normalize() if _str(k.get("as_of")) else (fr.index.max() if len(fr) else None)
    fr_last = fr.loc[as_of] if as_of is not None and as_of in fr.index else None

    render_section_header("Aarambh — The Top-Down Forecast", _answer(td, z.get("forecast"), fc_pct, forward, h, target),
                          icon="trending", accent="violet")
    if not con or fr.empty:
        render_empty_state("No Aarambh read for this run", "The run published no top-down forecast to show.",
                           eyebrow="Aarambh")
        return
    render_kpi_strip(_read_items(td, z, fr_last, fc_pct, forward, h, edge), max_cols=5, key="aarambh-read")

    name = index_name(con)
    nifty = series_by_date(ss.get("label_levels"))
    lvl = num(nifty.iloc[-1]) if len(nifty) else None
    render_section_header(
        "Forecast & The Nifty",
        (f"Top: the {name} ({lvl:,.2f} on {pretty_date(nifty.index[-1])}). " if lvl is not None else
         f"Top: the {name}. ")
        + (f"Bottom: the top-down pillar ({signed(td, '+.2f')}σ today), + = bullish — green above zero, red "
           if td is not None else "Bottom: the top-down pillar, + = bullish — green above zero, red ")
        + f"below; inside the shaded ±{PILLAR_NO_LEAN:g}σ band it has no lean, beyond ±{DEEP_Z:g}σ (dotted) it "
          "is unusually stretched for itself."
        + ("" if forward else f" {target[:1].upper() + target[1:]} and its fair value are drawn as context "
                              f"(right axis): {LEVELS_TAG}."),
        icon="trending", accent="violet",
    )
    if "P_top_down" in fr.columns:
        _forecast_chart(fr, nifty, ts, forward, name, target)
    else:
        render_empty_state("No top-down pillar", "This run published no top-down pillar to draw.", eyebrow="Aarambh")

    have = [(lab, z.get(cid)) for cid, lab in INPUTS if z.get(cid) is not None]
    biggest = max(have, key=lambda t: abs(t[1])) if have else None
    render_section_header(
        "Its Inputs",
        ("The pillar is the average of four inputs, each standardized against its own past (+ = bullish): today "
         + ", ".join(f"{lab.lower()} {signed(v, '+.2f')}σ" for lab, v in have)
         + (f" — the largest is {biggest[0].lower()}." if biggest else ".")) if have else
        "The pillar is the average of four inputs — the forecast, its extension, lookback zone breadth and "
        "lookback cut crossings — each standardized against its own past (+ = bullish).",
        icon="layers", accent="violet",
    )
    _inputs_chart(fr)

    a = _d(_d(ss.get("run_report")).get("aarambh"))
    skill = num(_d(a.get("model_stats")).get("r2_vs_rw"))
    base = "predicting no change" if forward else "a random walk"
    render_section_header(
        "Model Quality",
        (f"Does the walk-forward forecast beat {base}? Its out-of-sample skill is {signed(skill, '+.3f')} — "
         + (f"{base} did better on squared error, so read its ranking (the edge below), not its size."
            if skill < 0 else f"it beats {base} out of sample.")) if skill is not None else
        f"Does the walk-forward forecast beat {base}? No out-of-sample skill was measured this run.",
        icon="cpu", accent="violet",
    )
    render_kpi_strip(quality_items(a, sp_today, forward, h), max_cols=5, key="aarambh-quality")

    ic = num(_d(edge).get("ic"))
    pos, nb = int(num(_d(edge).get("positive_blocks")) or 0), int(num(_d(edge).get("scored_blocks")) or 0)
    render_section_header(
        "Walk-Forward Edge",
        f"When the top-down pillar leaned one way, did the Nifty move that way over the next {h} sessions? "
        + (f"On {int(num(_d(edge).get('n')) or 0):,} non-overlapping windows its IC is {signed(ic, '+.3f')}"
           + (f", positive in {pos} of {nb} periods." if nb else ".") if ic is not None else
           "Not graded yet — it needs dates both systems read.")
        + " Below it, Aarambh's own stages and a cost-aware backtest.",
        icon="trending", accent="violet",
    )
    render_edge_panel(edge, "aarambh", what="the top-down pillar")
    stages = stage_table(con.get("chain"))
    if not stages.empty:
        render_sub_header("Stage by Stage")
        render_table_panel(stages, "aarambh-stages", units=f"non-overlapping {h}d IC", windowed=False,
                           col_precision={"IC": 3}, sign_color_cols={"IC"}, max_height=240, row_height=_ROW_H,
                           footer=esc("Accuracy at each stage of the forecast, on the same windows."))
    bt = backtest_table(con.get("backtest"))
    if not bt.empty:
        render_sub_header("Cost-Aware Backtest")
        render_table_panel(bt, "aarambh-backtest", units="position = C / 100 · staggered tranches", windowed=False,
                           sign_color_cols={"Net Sharpe", "Return / yr %"}, max_height=280, row_height=_ROW_H,
                           col_labels={"Return / yr %": "Return / yr %", "HAC t": "HAC t"},
                           footer=esc(f"A long/short book sized by the conclusion C, held {h} sessions in "
                                      "staggered tranches; costs are per unit of turnover."))
