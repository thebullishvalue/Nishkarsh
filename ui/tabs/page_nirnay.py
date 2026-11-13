"""
Nishkarsh — Nirnay: what the fifty stocks are doing (≈ Tattva's Swayam page; hue: cyan; the
unit everywhere is a STOCK — one Nifty 50 constituent read through its own oscillator).

Reading order — the house convention every engine page follows:

  1 ANSWER    what do the stocks say?          net breadth · the zones · live stocks · edge
  2 ANCHOR    how has that moved?              breadth over time
  3 TRUST     has it been worth anything?      walk-forward edge of the bottom-up pillar
  4 STATE     who carries it, and the regime   today's stocks by oscillator · regime · average
  5 DETAIL    the evidence behind it           one stock's readings

Polarity (``core.config.NIRNAY_POLARITY``, one switch): under "trend" a stock reading
overbought is bullish evidence and oversold bearish; net breadth is reported bullish-positive
(the bullish-side share minus the bearish-side share) and coloured by that meaning. Oscillators
keep their native orientation (+ = strong). The oriented views come from ``convergence.views``;
this page reads, sorts and windows them. Windowed page: the window control sits in "Breadth
Over Time" and frames every time chart here; "Which Stocks Carry the Breadth" is today only.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.config import NIRNAY_POLARITY
from convergence.views import ZONE_WORDS, avg_oscillator_view, breadth_view, stock_snapshot
from ui.components import (
    _d, _str, esc, fmt, num, pretty_date, render_chart_panel, render_edge_panel, render_empty_state,
    render_kpi_strip, render_section_header, render_table_panel, signed, tone_of,
)
from ui.copy import NIRNAY_READ_SENTENCE
from ui.palette import UI_CHART_HEIGHT_LARGE, UI_CHART_HEIGHT_MEDIUM
from ui.tabs.engine_common import MIN_SESSIONS, add_sign_fill, by_date, ic_card, window_slice
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, style_axes

_TREND = NIRNAY_POLARITY == "trend"
#: The drill-down's columns, in reading order (never the engine's crossing flags).
DRILL_COLS = ("Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition", "Regime", "Vol_Regime", "Change_Point",
              "Confidence")

TOOLTIPS = {
    "net": (
        "Share of live stocks in the bullish-side zone minus the share in the bearish-side zone, in "
        "points. Each stock's zone is set by its own adaptive cuts. "
        + ("Read as trend: overbought stocks count as bullish, oversold as bearish." if _TREND else
           "Read contrarian: oversold stocks count as bullish, overbought as bearish.")
    ),
    "overbought": "Share of live stocks whose oscillator is above its own adaptive overbought cut today.",
    "oversold": "Share of live stocks whose oscillator is below its own adaptive oversold cut today.",
    "avg": (
        "Mean of the live stocks' unified oscillator (MSF + MMR, ±10). Compared with the median stock's "
        "adaptive cuts on the same date — the cuts a typical stock is judged against."
    ),
    "live": (
        "Stocks with a genuine bar today, past their warm-up and with a finite oscillator. Missing or "
        "newly listed stocks are left out, never counted as neutral."
    ),
    "ic": (
        "Rank correlation of the bottom-up pillar (the five stock inputs, standardized) with the Nifty's "
        "forward return over the horizon, scored on non-overlapping windows. 0 = no skill."
    ),
}


def _stance(net) -> str:
    v = num(net)
    if v is None:
        return ""
    if _TREND:
        return "broad strength" if v > 10 else "broad weakness" if v < -10 else "no clear side"
    return ("a depressed market, which reads bullish" if v > 10 else
            "a stretched market, which reads bearish" if v < -10 else "no clear side")


def _zone_tone(share, bullish_side: bool) -> str:
    v = num(share)
    if v is None or v < 50:
        return "neutral"
    return "success" if bullish_side else "danger"


# ═══════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════

def _breadth_chart(bw: pd.DataFrame) -> None:
    """Net breadth (top, bullish-positive) over the two zone shares it is made of."""
    x = list(bw.index)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], shared_xaxes=True, vertical_spacing=0.07)
    net = bw["net"].to_numpy(dtype=float)
    add_sign_fill(fig, x, net, above="emerald", below="rose", alpha=0.12, row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=net, mode="lines", name="Net breadth",
                             line=dict(color=chart_color("cyan"), width=1.4)), row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=1, col=1)
    for col, word, hue in (("bull_pct", ZONE_WORDS[0], "emerald"), ("bear_pct", ZONE_WORDS[1], "rose")):
        fig.add_trace(go.Scatter(x=x, y=bw[col].to_numpy(dtype=float), mode="lines", name=f"Stocks {word.lower()}",
                                 fill="tozeroy", fillcolor=chart_rgba(hue, 0.12),
                                 line=dict(color=chart_color(hue), width=1.2)), row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE + 40))
    style_axes(fig, y_title="Net (pts)", y_range=[-100, 100], row=1, col=1)
    style_axes(fig, y_title="% of stocks", y_range=[0, 100], row=2, col=1)
    render_chart_panel(fig, "nirnay_breadth", units="% of stocks", window=True)


def _stocks_chart(snap: pd.DataFrame) -> None:
    """Today's oscillator of every live stock, sorted, coloured by its own zone, with its own cuts."""
    s = snap.sort_values("osc", ascending=False)
    syms = list(s.index)
    colors = [chart_color("emerald") if sd > 0 else chart_color("rose") if sd < 0 else chart_rgba("cyan", 0.45)
              for sd in s["side"]]
    fig = go.Figure(go.Bar(x=syms, y=s["osc"].to_numpy(dtype=float), marker=dict(color=colors), name="Oscillator",
                           customdata=[[z] for z in s["zone"]],
                           hovertemplate="%{x}: %{y:+.2f} · %{customdata[0]}<extra></extra>"))
    for col, word in (("ob_cut", "overbought cut"), ("os_cut", "oversold cut")):
        fig.add_trace(go.Scatter(x=syms, y=pd.to_numeric(s[col], errors="coerce").to_numpy(dtype=float),
                                 mode="markers", name=word.capitalize(),
                                 marker=dict(symbol="line-ew", size=10, line=dict(width=1.2, color=grid_rgba(0.45))),
                                 customdata=[[word]] * len(syms),
                                 hovertemplate="%{x} %{customdata[0]}: %{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6)
    many = len(syms) > 20
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False,
                                     margin=dict(t=12, l=60, r=24, b=70 if many else 38)))
    style_axes(fig, y_title="Oscillator")
    fig.update_xaxes(tickangle=-90 if many else 0, showgrid=False)
    n_bull, n_bear = int((s["side"] > 0).sum()), int((s["side"] < 0).sum())
    n_neu = len(s) - n_bull - n_bear
    counts = {ZONE_WORDS[0].lower(): n_bull, ZONE_WORDS[1].lower(): n_bear}
    render_chart_panel(
        fig, "nirnay_stocks", units="today's oscillator per stock", windowed=False,
        footer=esc(f"{counts['overbought']} overbought · {counts['oversold']} oversold · {n_neu} neutral. "
                   "Ticks mark each stock's own adaptive cuts; its zone is set by them, not by one shared line."))


def _regime_chart(nd: pd.DataFrame) -> None:
    """The stocks' average HMM state probabilities (top) and their regime composition (bottom)."""
    x = list(nd.index)
    bull_hue, bear_hue = ("emerald", "rose") if _TREND else ("rose", "emerald")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5], shared_xaxes=True, vertical_spacing=0.08)
    for col, name, hue in (("avg_hmm_bull", "P(bull state)", bull_hue), ("avg_hmm_bear", "P(bear state)", bear_hue)):
        if col in nd.columns:
            fig.add_trace(go.Scatter(x=x, y=pd.to_numeric(nd[col], errors="coerce").to_numpy(dtype=float),
                                     mode="lines", name=name, line=dict(color=chart_color(hue), width=1.4),
                                     fill="tozeroy", fillcolor=chart_rgba(hue, 0.08)), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color=grid_rgba(0.08), line_width=0.5, row=1, col=1)
    for col, name, hue in (("Regime_Bull_Pct", "% bull regime", bull_hue),
                           ("Regime_Bear_Pct", "% bear regime", bear_hue)):
        if col in nd.columns:
            fig.add_trace(go.Scatter(x=x, y=pd.to_numeric(nd[col], errors="coerce").to_numpy(dtype=float),
                                     mode="lines", name=name, line=dict(color=chart_rgba(hue, 0.85), width=1.1)),
                          row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
    style_axes(fig, y_title="Probability", y_range=[0, 1], row=1, col=1)
    style_axes(fig, y_title="% of stocks", y_range=[0, 100], row=2, col=1)
    render_chart_panel(fig, "nirnay_regime", units="probability · % of stocks")


def _avg_chart(av: pd.DataFrame) -> None:
    """The stocks' average oscillator against the median stock's adaptive cuts, date by date."""
    x = list(av.index)
    v = av["avg"].to_numpy(dtype=float)
    up, dn = ("emerald", "rose") if _TREND else ("rose", "emerald")
    colors = [chart_color("emerald") if sd > 0 else chart_color("rose") if sd < 0 else chart_rgba("slate", 0.6)
              for sd in av["side"].fillna(0)]
    fig = go.Figure()
    add_sign_fill(fig, x, v, above=up, below=dn, alpha=0.05)
    # The cut each marker was judged against, drawn as a step so the dots and the line always tell
    # the same story about the same date (no repainting with today's cut).
    for col, hue in (("ob_cut", up), ("os_cut", dn)):
        fig.add_trace(go.Scatter(x=x, y=av[col].to_numpy(dtype=float), mode="lines", line_shape="hv",
                                 line=dict(width=0.6, dash="dot", color=chart_rgba(hue, 0.35)),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=v, mode="lines+markers", name="Average oscillator",
                             line=dict(color=chart_color("cyan"), width=1.3), marker=dict(size=3, color=colors)))
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False))
    style_axes(fig, y_title="Avg oscillator")
    lo, hi = num(av["os_cut"].dropna().iloc[-1]) if av["os_cut"].notna().any() else None, \
        num(av["ob_cut"].dropna().iloc[-1]) if av["ob_cut"].notna().any() else None
    render_chart_panel(fig, "nirnay_avg", units=(f"oscillator · median cuts {signed(lo, '+.2f')} / {signed(hi, '+.2f')}"
                                                 if lo is not None and hi is not None else "oscillator"))


def drill_table(df) -> pd.DataFrame:
    """One stock's last 60 sessions, newest first, in reading order (flags as yes/no)."""
    f = by_date(df)
    if f.empty:
        return pd.DataFrame()
    cols = [c for c in DRILL_COLS if c in f.columns]
    out = f[cols].tail(60).iloc[::-1].copy()
    if "Change_Point" in out.columns:
        out["Change_Point"] = out["Change_Point"].map(lambda x: "yes" if x is True or x == 1 else "no")
    for c in ("Condition", "Regime", "Vol_Regime"):
        if c in out.columns:
            out[c] = out[c].map(lambda x: _str(x, "—"))
    out.index.name = "Date"
    return out


def _drilldown(dfs: dict) -> None:
    keys = sorted(k for k, v in dfs.items() if isinstance(v, pd.DataFrame) and len(v))
    if not keys:
        render_empty_state("No stock readings", "The run kept no per-stock readings to show.", eyebrow="Nirnay")
        return
    if st.session_state.get("nirnay_stock") not in keys:      # seed before the widget (options can change)
        st.session_state["nirnay_stock"] = keys[0]
    with st.container(key="filterbar"):
        sym = st.selectbox("Stock", keys, key="nirnay_stock", format_func=lambda s: str(s).split(".")[0])
    tbl = drill_table(dfs.get(sym))
    render_table_panel(
        tbl, "nirnay-stock-drilldown", units="last 60 sessions", windowed=False, index_label="Date",
        max_height=520, sign_color_cols={"MSF_Osc", "MMR_Osc", "Unified_Osc"} if _TREND else None,
        col_precision={"Confidence": 2}, col_labels={"Vol_Regime": "Vol Regime", "Change_Point": "Change Point"})


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def render_nirnay_page() -> None:
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    k = _d(con.get("nko"))
    h = int(num(con.get("horizon")) or 10)
    nd = by_date(ss.get("nirnay_daily"))
    bw = breadth_view(nd)
    dfs = ss.get("nirnay_constituent_dfs") if isinstance(ss.get("nirnay_constituent_dfs"), dict) else {}
    nr = _d(_d(ss.get("run_report")).get("nirnay"))
    edge = _d(_d(k.get("pillar_edge")).get("bottom_up")) or None

    if bw.empty or bw["net"].dropna().empty:
        render_section_header("Nirnay — What the Fifty Stocks Are Doing",
                              "Nirnay reads the Nifty's fifty stocks from the bottom up; this run has no stock data, "
                              "so the read is top-down only.", icon="activity", accent="cyan")
        render_empty_state(
            "No Nirnay breadth for this run",
            "Nirnay reads each Nifty 50 stock's own prices, and the run received no stock prices (the price feed "
            "and NSE's archives both came back empty). The read is Aarambh's alone; the notice above names the "
            "source that failed.", eyebrow="Nirnay")
        return

    last = bw.dropna(subset=["net"]).iloc[-1]
    day = bw.dropna(subset=["net"]).index[-1]
    net, bull, bear = num(last["net"]), num(last["bull_pct"]), num(last["bear_pct"])
    n_live = num(last["n_live"])
    n_uni = num(nd.loc[day].get("n_universe")) if "n_universe" in nd.columns else None
    ob, os_ = (bull, bear) if _TREND else (bear, bull)
    av = avg_oscillator_view(nd, dfs)
    av_last = av.loc[day] if day in av.index else None
    avg = num(av_last["avg"]) if av_last is not None else None
    side = num(av_last["side"]) if av_last is not None else None
    lo = num(av_last["os_cut"]) if av_last is not None else None
    hi = num(av_last["ob_cut"]) if av_last is not None else None
    n_txt = f"{int(n_live)}" if n_live is not None else "—"

    render_section_header(
        "Nirnay — What the Fifty Stocks Are Doing",
        f"Net breadth {signed(net, '+.0f')} pts: {fmt(ob, '.0f')}% of {n_txt} stocks read overbought and "
        f"{fmt(os_, '.0f')}% oversold; read as {NIRNAY_POLARITY}, that is {_stance(net)}.",
        icon="activity", accent="cyan",
    )
    cov = (n_live / n_uni) if n_live is not None and n_uni else None
    render_kpi_strip([
        {"label": "Net Breadth", "value": f"{signed(net, '+.0f')} pts" if net is not None else "—",
         "subtext": f"{ZONE_WORDS[0].lower()} − {ZONE_WORDS[1].lower()} · bullish-positive",
         "color_class": tone_of(net, 10.0), "tooltip": TOOLTIPS["net"]},
        {"label": "Stocks Overbought", "value": f"{fmt(ob, '.0f')}%" if ob is not None else "—",
         "subtext": f"of {n_txt} stocks", "color_class": _zone_tone(ob, _TREND), "tooltip": TOOLTIPS["overbought"]},
        {"label": "Stocks Oversold", "value": f"{fmt(os_, '.0f')}%" if os_ is not None else "—",
         "subtext": f"of {n_txt} stocks", "color_class": _zone_tone(os_, not _TREND),
         "tooltip": TOOLTIPS["oversold"]},
        {"label": "Average Oscillator", "value": signed(avg, "+.2f"),
         "subtext": (f"median cuts {signed(lo, '+.2f')} / {signed(hi, '+.2f')}" if lo is not None and hi is not None
                     else "mean of the live stocks, ±10"),
         "color_class": "success" if side and side > 0 else "danger" if side and side < 0 else "neutral",
         "tooltip": TOOLTIPS["avg"]},
        {"label": "Live Stocks", "value": (f"{int(n_live)}/{int(n_uni)}" if n_live is not None and n_uni
                                           else n_txt),
         "subtext": (f"{cov:.0%} coverage · {pretty_date(day)}" if cov is not None else pretty_date(day)),
         "color_class": "info", "tooltip": TOOLTIPS["live"]},
        ic_card(edge, "bottom-up pillar", h, TOOLTIPS["ic"]),
    ], max_cols=6, key="nirnay-read")

    render_section_header(
        "Breadth Over Time",
        f"Top: net breadth, bullish-positive — {signed(net, '+.0f')} pts today. Bottom: the share of stocks "
        f"{ZONE_WORDS[0].lower()} (green) and {ZONE_WORDS[1].lower()} (red) it is made of. {NIRNAY_READ_SENTENCE}",
        icon="layers", accent="cyan",
    )
    bwin = window_slice(bw.dropna(subset=["net"]))
    if len(bwin) < MIN_SESSIONS:
        render_empty_state("Building breadth history",
                           "Fewer than ten sessions of stock breadth fall in this window. Widen the chart window.",
                           eyebrow="Nirnay")
    else:
        _breadth_chart(bwin)

    ic = num(_d(edge).get("ic"))
    pos, nb = int(num(_d(edge).get("positive_blocks")) or 0), int(num(_d(edge).get("scored_blocks")) or 0)
    render_section_header(
        "Walk-Forward Edge",
        f"When the bottom-up pillar leaned one way, did the Nifty move that way over the next {h} sessions? "
        + (f"On {int(num(_d(edge).get('n')) or 0):,} non-overlapping windows its IC is {signed(ic, '+.3f')}"
           + (f", positive in {pos} of {nb} periods." if nb else ".") if ic is not None else
           "Not graded yet — it needs enough dates both systems read.")
        + " Graded exactly as the top-down pillar is.",
        icon="trending", accent="cyan",
    )
    render_edge_panel(edge, "nirnay", what="the bottom-up pillar")

    snap = stock_snapshot(dfs, day)
    render_section_header(
        "Which Stocks Carry the Breadth",
        (f"Every live stock's oscillator on {pretty_date(day)}, strongest first: {int((snap['zone'] == 'Overbought').sum())} "
         f"overbought, {int((snap['zone'] == 'Oversold').sum())} oversold. Each is judged against its own adaptive "
         "cuts, so a stock's zone is about its own history, not a shared line.") if len(snap) else
        "Every live stock's oscillator today, strongest first, coloured by its own zone.",
        icon="bar-chart", accent="cyan",
    )
    if len(snap):
        _stocks_chart(snap)
    else:
        render_empty_state("No stock readings today", "No stock was live on the latest breadth date.",
                           eyebrow="Nirnay")

    hb = num(nd.loc[day].get("avg_hmm_bull")) if "avg_hmm_bull" in nd.columns else None
    hr = num(nd.loc[day].get("avg_hmm_bear")) if "avg_hmm_bear" in nd.columns else None
    rb = num(nd.loc[day].get("Regime_Bull_Pct")) if "Regime_Bull_Pct" in nd.columns else None
    rr = num(nd.loc[day].get("Regime_Bear_Pct")) if "Regime_Bear_Pct" in nd.columns else None
    render_section_header(
        "Regime",
        (f"Top: the stocks' average probability of their bull and bear HMM states — {fmt(hb, '.0%')} bull, "
         f"{fmt(hr, '.0%')} bear today. Bottom: the share of stocks in a bull or bear regime — {fmt(rb, '.0f')}% "
         f"and {fmt(rr, '.0f')}%. Sustained P above 0.5 is a confident regime; frequent crossings mean no regime.")
        if hb is not None and hr is not None else
        "Top: the stocks' average probability of their bull and bear HMM states. Bottom: the share of stocks in a "
        "bull or bear regime.",
        icon="eye", accent="cyan",
    )
    ndw = window_slice(nd)
    if len(ndw) < MIN_SESSIONS:
        render_empty_state("Building regime history", "Fewer than ten sessions fall in this window.",
                           eyebrow="Nirnay")
    else:
        _regime_chart(ndw)

    render_section_header(
        "Average Oscillator",
        (f"The stocks' mean oscillator ({signed(avg, '+.2f')} today) against the median stock's adaptive cuts on "
         "each date (dotted). Beyond a cut = broad participation; whipsaws near zero = no consensus.")
        if avg is not None else
        "The stocks' mean oscillator against the median stock's adaptive cuts on each date (dotted).",
        icon="activity", accent="cyan",
    )
    avw = window_slice(av.dropna(subset=["avg"]))
    if len(avw) < MIN_SESSIONS:
        render_empty_state("Building oscillator history", "Fewer than ten sessions fall in this window.",
                           eyebrow="Nirnay")
    else:
        _avg_chart(avw)

    failed = len(_d(nr.get("failed")))
    render_section_header(
        "Individual Stocks",
        f"One stock's MSF, MMR and unified oscillator, its zone and regime — the readings the breadth is counted "
        f"from. {fmt(nr.get('stocks_analyzed'), '.0f')} of {fmt(nr.get('stocks_requested'), '.0f')} stocks analysed"
        + (f", {failed} failed." if failed else ".")
        + " Today's list is applied to the past, so stocks that left the index are missing (survivorship).",
        icon="database",
    )
    _drilldown(dfs)
