"""
Nishkarsh v2 — Overview tab: the finished-product read of the run.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Top to bottom: the market read + Agreement Oscillator gauge (hero), the oscillator's history
against the Nifty, one compact card per system, "How reliable is this?" (the Edge Ledger in
plain language) and low-key data notes. Everything here RENDERS the conclusion layer's
outputs (``conclusion``, ``conclusion_frame``, ``label_levels``, ``nirnay_daily``,
``run_report``, ``run_config``); nothing is graded or re-estimated. No internal symbols —
the technical figures live in the Evidence tab. Buy/sell wording is reserved for a VALIDATED
oscillator (``ui.components.BUY_SELL_RE``).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.config import BACKTEST_REFERENCE_BPS
from ui.components import (
    CALL_TIERS, _d, _str, build_market_read, chip_html, edge_status_code, esc, fmt, num, pretty_date, render_market_hero,
    signed, sparkline_svg,
)
from ui.copy import LEVELS_TAG, OSC_SCALE_TEXT, TRUST_PLAIN, TRUST_PLAIN_SENTENCE
from ui.theme import chart_layout

_PRICE = "rgba(226,232,240,0.78)"
_CRITICAL = ("Stale", "STALE", "FALLBACK", "EMPTY", "failed")
_PROV_WORD = {"live": "live", "cache": "cached", "stale": "stale", "fallback": "fallback source", "empty": "empty"}


def _since(obj, start):
    if obj is None or len(obj) == 0:
        return obj.iloc[0:0] if obj is not None else pd.DataFrame()
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    return obj if start is None else obj[pd.DatetimeIndex(obj.index) >= pd.Timestamp(start)]


def span_words(tf_label: str) -> str:
    return {"3M": "the last 3 months", "6M": "the last 6 months", "1Y": "the last year",
            "2Y": "the last 2 years"}.get(str(tf_label), "all history")


def _index_name(con: dict) -> str:
    src = _str(con.get("label_src")).upper()
    return "Nifty 50" if ("NSEI" in src or "NIFTY" in src or not src) else _str(con.get("label_src"))


# ── (3) oscillator history vs the index ──────────────────────────────────────
def _history_chart(fr: pd.DataFrame, price: pd.Series | None, index_name: str) -> None:
    sao = pd.to_numeric(fr["SAO"], errors="coerce") if "SAO" in fr.columns else pd.Series(dtype=float)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = fr.index
    # Contiguous bars (each as wide as the gap to the next date) read as a stepped area and break
    # cleanly on days without data. Display only — the values are the frame's SAO, unchanged.
    xs = pd.DatetimeIndex(x)
    gaps_ms = pd.Series(xs[1:] - xs[:-1]).dt.total_seconds().mul(1000).tolist() if len(xs) > 1 else []
    width = gaps_ms + [float(pd.Series(gaps_ms).median()) if gaps_ms else 86_400_000.0]
    vals = [None if pd.isna(v) else float(v) for v in sao]
    colors = ["rgba(52,211,153,0.55)" if (v or 0) > 0 else "rgba(251,113,133,0.55)" for v in vals]
    fig.add_trace(go.Bar(x=list(xs), y=vals, width=width, offset=0, marker=dict(color=colors, line=dict(width=0)),
                         name="Agreement", showlegend=False, hovertemplate="Agreement %{y:+.2f}<extra></extra>"),
                  secondary_y=False)
    if price is not None and len(price):
        fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name=index_name, showlegend=False,
                                 line=dict(color=_PRICE, width=1.5),
                                 hovertemplate=f"{index_name} " + "%{y:,.0f}<extra></extra>"), secondary_y=True)
    lay = chart_layout(height=330, show_legend=False, margin=dict(t=8, l=8, r=8, b=8))
    fig.update_layout(**lay, bargap=0)
    tick = dict(family="Inter, sans-serif", size=11, color="#7C879A")
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, tickfont=tick, ticks="", showspikes=True,
                     spikemode="across", spikethickness=0.5, spikedash="dot", spikecolor="rgba(148,163,184,0.35)")
    fig.update_yaxes(range=[-1.05, 1.05], tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["−1", "", "0", "", "+1"],
                     showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=True,
                     zerolinecolor="rgba(148,163,184,0.25)", zerolinewidth=1, tickfont=tick, secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickformat=",.0f", tickfont=tick, secondary_y=True)
    st.plotly_chart(fig, width="stretch", key="ov_history", config={"displayModeBar": False})


def _render_history(ss, start, tf_label: str, con: dict) -> None:
    frame = ss.get("conclusion_frame")
    if frame is None or len(frame) == 0 or "SAO" not in frame.columns:
        return
    fr = _since(frame, start)
    lv = ss.get("label_levels")
    price = _since(lv, start) if isinstance(lv, pd.Series) else None
    name = _index_name(con)
    span = span_words(tf_label)
    with st.container(border=True, key="nk_history"):
        st.markdown(
            f'<div class="nk-card-head"><div><h3>Agreement over time</h3>'
            f'<p>The oscillator each day over {esc(span)}, with the {esc(name)} for context.</p></div>'
            f'<div class="nk-legend"><span><i class="sw buy"></i>Buy-side agreement</span>'
            f'<span><i class="sw sell"></i>Sell-side agreement</span>'
            f'<span><i class="sw line"></i>{esc(name)} (right axis)</span></div></div>', unsafe_allow_html=True)
        if len(fr):
            _history_chart(fr, price, name)
        st.markdown(f'<div class="nk-card-foot">{esc(OSC_SCALE_TEXT)}. A gap means Nirnay had no data that '
                    'day; flat at 0 means the systems disagreed.</div>', unsafe_allow_html=True)


# ── (4) the two systems ──────────────────────────────────────────────────────
def _pill(text: str, cls: str = "") -> str:
    return f'<span class="nk-pill {esc(cls)}">{esc(text)}</span>'


def _system_cards(ss, start, con: dict, m: dict, tf_label: str) -> None:
    lat = _d(con.get("latest"))
    span = span_words(tf_label)
    frame = ss.get("conclusion_frame")
    fr = _since(frame, start) if frame is not None else pd.DataFrame()
    t_tier = m.get("trust_tier", "INSUFFICIENT_DATA")
    h = m.get("horizon")
    forward = m.get("forward_mode", True)
    c_val = num(lat.get("C"))
    a_spark = sparkline_svg([(fr["C"].tolist() if "C" in fr.columns else [], "#D4A853")], y_range=(-100, 100),
                            zero=True)
    a_txt = str(m.get("aarambh_text", "—"))
    if not forward:
        a_sentence = f"Valuation head: {esc(a_txt)} — {esc(LEVELS_TAG)}."
    elif a_txt.startswith("forecast leans"):
        a_sentence = f"Its {h}-day forecast {esc(a_txt[len('forecast '):])}."
    elif a_txt.startswith("forecast "):
        a_sentence = f"Its {h}-day forecast is {esc(a_txt[len('forecast '):])}."
    else:
        a_sentence = "No forecast for the latest date."
    # Colour only a proven directional call (trust tier DIRECTIONAL/TRADEABLE), never a lean.
    direction = _str(lat.get("direction")).upper()
    a_cls = (("buy" if direction == "BULLISH" else "sell" if direction == "BEARISH" else "")
             if t_tier in CALL_TIERS and _str(lat.get("tier")).upper() in ("STRONG", "MODERATE", "WEAK") else "")
    pos = 50.0 if c_val is None else max(0.0, min(100.0, 50.0 + c_val / 2.0))
    conv_bar = ('<div class="nk-convbar"><div class="mid"></div>'
                + ('' if c_val is None else f'<div class="mark" style="left:{pos:.1f}%"></div>') + '</div>')
    a_card = (
        f'<div class="nk-sys">'
        f'<div class="nk-sys-head"><div><h4>Aarambh</h4><span>Top-down forecast</span></div>'
        f'{_pill(m.get("aarambh_pill", "—"), a_cls)}</div>'
        f'<div class="nk-sys-figure"><span class="nk-big">{esc(signed(c_val, "+.0f"))}</span>'
        f'<span class="nk-unit">conviction, on a −100 to +100 scale</span></div>{conv_bar}'
        f'<p>{a_sentence}</p>'
        f'<div class="nk-spark-wrap">{a_spark}<div class="nk-spark-cap">Conviction over {esc(span)} · '
        f'above the dotted line leans higher</div></div>'
        f'<div class="nk-sys-foot">Track record {chip_html(TRUST_PLAIN[t_tier][0], TRUST_PLAIN[t_tier][1])}</div>'
        f'</div>')

    nl = _d(_d(_d(ss.get("run_report")).get("nirnay")).get("latest"))
    ob, os_ = num(nl.get("Overbought_Pct")), num(nl.get("Oversold_Pct"))
    n_spark = sparkline_svg([(fr["zN"].tolist() if "zN" in fr.columns else [], "#22D3EE")], y_range=(-3, 3),
                            zero=True)
    bar = ""
    if ob is not None or os_ is not None:
        o1, o2 = max(0.0, min(100.0, os_ or 0.0)), max(0.0, min(100.0, ob or 0.0))
        bar = (f'<div class="nk-breadth"><div class="seg os" style="width:{o1:.1f}%"></div>'
               f'<div class="seg mid" style="width:{max(0.0, 100 - o1 - o2):.1f}%"></div>'
               f'<div class="seg ob" style="width:{o2:.1f}%"></div></div>')
    n_live = num(nl.get("n_live"))
    n_sentence = (f"Of the {int(n_live)} Nifty stocks with live data, {fmt(os_, '.0f')}% are oversold and "
                  f"{fmt(ob, '.0f')}% overbought." if n_live else "No stock-level data for this date.")
    nir_state = "Context only" if not (_d(con.get("lift")).get("passed") is True) else "Measured"
    n_card = (
        f'<div class="nk-sys">'
        f'<div class="nk-sys-head"><div><h4>Nirnay</h4><span>Bottom-up stock breadth</span></div>'
        f'{_pill(m.get("nirnay_pill", "—"))}</div>'
        f'<div class="nk-sys-figure"><span class="nk-big os">{esc(fmt(os_, ".0f", suffix="%"))}</span>'
        f'<span class="nk-unit">oversold</span><span class="nk-big ob">{esc(fmt(ob, ".0f", suffix="%"))}</span>'
        f'<span class="nk-unit">overbought</span></div>{bar}'
        f'<p>{esc(n_sentence)}</p>'
        f'<div class="nk-spark-wrap">{n_spark}<div class="nk-spark-cap">Nirnay\'s reading over {esc(span)} · '
        f'above the dotted line leans bullish</div></div>'
        f'<div class="nk-sys-foot">Role in the conclusion {chip_html(nir_state, "muted")}</div>'
        f'</div>')
    c1, c2 = st.columns(2, gap="medium")
    c1.markdown(a_card, unsafe_allow_html=True)
    c2.markdown(n_card, unsafe_allow_html=True)


# ── (5) How reliable is this? — the Edge Ledger in plain language ────────────
def build_reliability_rows(conclusion) -> list[dict]:
    """Plain-language Edge Ledger rows (claim · status · meaning · numbers) from the conclusion's own
    flags. Pure; status words follow the conclusion layer's booleans, never a UI threshold."""
    con = _d(conclusion)
    h = int(num(con.get("horizon")) or 10)
    rows = []

    e = _d(con.get("edge"))
    code = edge_status_code(e)
    status = {"PROVEN": ("Proven", "good", f"Its direction has matched the next {h} trading days more often "
                         "than chance on this data."),
              "NOT_PROVEN": ("Not proven", "caution", "So far its direction has not matched the market often "
                             "enough to rule out luck."),
              "INVERTED": ("Wrong way", "bad", "It has pointed the wrong way more often than chance, so its "
                           "direction is not used."),
              "INSUFFICIENT": ("Not enough data", "muted", "There is not enough history yet to judge it.")}[code]
    rows.append({"claim": "Aarambh's forecast points the right way", "status": status[0], "class": status[1],
                 "meaning": status[2],
                 "numbers": f"accuracy score {signed(e.get('ic'), '+.3f')} · p {fmt(e.get('p'), '.2f')} · "
                            f"{fmt(e.get('n'), '.0f')} separate {h}-day periods"})

    lift = _d(con.get("lift"))
    if con.get("single_system") is True:
        st_ = ("Not running", "muted", "Nirnay has no data in this run, so the conclusion is Aarambh's alone.")
    elif lift.get("available") is not True:
        st_ = ("Not enough data", "muted", "Too few agreeing and disagreeing periods to measure this yet.")
    elif lift.get("passed") is True:
        st_ = ("Proven", "good", "Aarambh has been more accurate when Nirnay agrees, so disagreement can lower "
               "conviction.")
    else:
        st_ = ("Not proven", "caution", "Agreement has not been shown to make Aarambh more accurate, so Nirnay "
               "does not change the conclusion.")
    rows.append({"claim": "Agreement with Nirnay makes Aarambh more accurate", "status": st_[0],
                 "class": st_[1], "meaning": st_[2],
                 "numbers": f"accuracy when they agree {signed(lift.get('ic_confirm'), '+.3f')} vs disagree "
                            f"{signed(lift.get('ic_contradict'), '+.3f')} · p {fmt(lift.get('p_one_sided'), '.2f')}"})

    oos = _d(con.get("oos"))
    if oos.get("adds_value") is True:
        st_ = ("Improved", "good", "On history the models never saw, the combined conclusion beat Aarambh alone.")
    elif num(oos.get("n")) is None:
        st_ = ("Not enough data", "muted", "Not enough unseen history to compare yet.")
    else:
        st_ = ("No improvement", "caution", "On history the models never saw, combining the two did not beat "
               "Aarambh alone.")
    rows.append({"claim": "The combined conclusion beats Aarambh alone", "status": st_[0], "class": st_[1],
                 "meaning": st_[2],
                 "numbers": f"accuracy combined {signed(oos.get('ic_C'), '+.3f')} vs Aarambh alone "
                            f"{signed(oos.get('ic_A'), '+.3f')} · {fmt(oos.get('n'), '.0f')} periods"})

    bt = _d(con.get("backtest"))
    ref = num(bt.get("reference_bps"))
    ref = BACKTEST_REFERENCE_BPS if ref is None else ref
    by_cost = _d(bt.get("by_cost"))
    b = next((_d(v) for k, v in by_cost.items() if num(k) == ref), {})
    tier = _str(_d(con.get("trust")).get("tier")).upper()
    if tier == "TRADEABLE_EDGE":
        st_ = ("Survives costs", "good", "Trading the conclusion would have stayed profitable after realistic costs.")
    elif not b:
        st_ = ("Not enough data", "muted", "No cost-aware test could be run on this data.")
    else:
        st_ = ("Not shown", "caution", "Trading the conclusion has not been shown to stay profitable after "
               "realistic costs.")
    rows.append({"claim": "Trading it would pay after costs", "status": st_[0], "class": st_[1], "meaning": st_[2],
                 "numbers": f"risk-adjusted return (Sharpe) {signed(b.get('net_sharpe'))} after {ref:g} bps per "
                            f"trade · break-even cost {fmt(b.get('breakeven_bps'), '.1f')} bps"})

    osc = _d(con.get("oscillator"))
    ev = _d(osc.get("evidence"))
    status = _str(osc.get("status")).upper()
    st_ = {"VALIDATED": ("Validated", "good", f"Its readings have lined up with the Nifty's next {h} trading "
                         "days, clearing the bar set before it was tested."),
           "UNPROVEN": ("Unproven", "caution", "Its readings have not yet lined up with the Nifty's next moves "
                        "strongly or consistently enough to act on.")}.get(
        status, ("Not enough data", "muted", "Not enough history with both systems running to measure it."))
    yearly = _d(ev.get("yearly"))
    share = num(ev.get("yearly_positive_share"))
    yrs = f" · positive in {int(round(share * len(yearly)))} of {len(yearly)} years" if share is not None and yearly else ""
    rows.append({"claim": "The Agreement Oscillator anticipates the market", "status": st_[0], "class": st_[1],
                 "meaning": st_[2],
                 "numbers": f"accuracy score {signed(ev.get('ic'), '+.3f')} · p {fmt(ev.get('p'), '.2f')}{yrs}"})
    return rows


def _render_reliability(con: dict, m: dict) -> None:
    tier = m.get("trust_tier", "INSUFFICIENT_DATA")
    rows = "".join(
        f'<div class="nk-ledger-row"><div class="nk-claim">{esc(r["claim"])}</div>'
        f'<div class="nk-ledger-status">{chip_html(r["status"], r["class"])}</div>'
        f'<div class="nk-meaning">{esc(r["meaning"])}<div class="nk-ledger-nums">{esc(r["numbers"])}</div></div>'
        f'</div>' for r in build_reliability_rows(con))
    with st.expander("How reliable is this?", expanded=False):
        st.markdown(
            f'<div class="nk-ledger"><div class="nk-ledger-overall"><span>Overall grade</span>'
            f'{chip_html(TRUST_PLAIN[tier][0], TRUST_PLAIN[tier][1])}<p>{esc(TRUST_PLAIN_SENTENCE[tier])} '
            f'Each claim below is tested on history the models did not see when they were fitted.</p></div>'
            f'{rows}<div class="nk-ledger-foot">Exact figures, tests and charts: Evidence tab.</div></div>',
            unsafe_allow_html=True)


# ── (6) data notes ───────────────────────────────────────────────────────────
def _render_data_notes(rc: dict) -> None:
    prov = _d(rc.get("provenance"))
    warns = [str(w) for w in (rc.get("warnings") or []) if not str(w).upper().startswith("SYNTHETIC")]
    items = []
    for k, p in prov.items():
        p = _d(p)
        word = _PROV_WORD.get(_str(p.get("provenance")).lower(), _str(p.get("provenance")) or "—")
        miss = p.get("missing") or []
        items.append(f'<li><b>{esc(str(k).capitalize())}</b> {esc(word)}'
                     + (f' · as of {esc(pretty_date(p.get("as_of")))}' if _str(p.get("as_of")) else "")
                     + (f' · {len(miss)} missing' if miss else "") + "</li>")
    label = "Data notes" + (f" · {len(warns)} to review" if warns else "")
    with st.expander(label, expanded=False):
        if items:
            st.markdown(f'<ul class="nk-notes">{"".join(items)}</ul>', unsafe_allow_html=True)
        for w in warns:
            st.markdown(f'<div class="nk-note">{esc(w)}</div>', unsafe_allow_html=True)
        h0 = _str(rc.get("history_start"))
        st.markdown(f'<div class="nk-note-foot">History used from {esc(pretty_date(h0) if h0 else "—")} · '
                    f'{len(rc.get("features") or [])} model inputs · target {esc(_str(rc.get("target")) or "—")}</div>',
                    unsafe_allow_html=True)


def render_overview_tab(start, tf_label: str) -> None:
    """Overview — market read + oscillator gauge, history, system cards, reliability, data notes."""
    ss = st.session_state
    con, rc = _d(ss.get("conclusion")), _d(ss.get("run_config"))
    nl = _d(_d(_d(ss.get("run_report")).get("nirnay")).get("latest"))
    m = build_market_read(con, rc, nl)
    render_market_hero(m)
    crit = [w for w in (rc.get("warnings") or []) if any(k in str(w) for k in _CRITICAL)]
    if crit:
        st.markdown('<div class="nk-inline-warn">Some inputs are stale or came from a fallback source — '
                    'see Data notes at the bottom of this page.</div>', unsafe_allow_html=True)
    _render_history(ss, start, tf_label, con)
    _system_cards(ss, start, con, m, tf_label)
    st.markdown('<div class="nk-gap"></div>', unsafe_allow_html=True)
    _render_reliability(con, m)
    _render_data_notes(rc)
