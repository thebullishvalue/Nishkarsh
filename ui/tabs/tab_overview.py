"""
Nishkarsh v2 — Overview tab: the one read of the run, the Nishkarsh Oscillator (NKO).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Top to bottom: the hero (the read in words + the ±100 gauge, inferred from ``conclusion["nko"]``
and ``precedent["hero"]`` only), the read over time against the Nifty (with its three pillars as
a decomposition), "What drives today's read" (pillars → the ten inputs: forecast, extension,
zone breadth, cut crossings, HMM state, regime mix, agreement anatomy), "How reliable is this?"
and low-key data notes. Everything here RENDERS pipeline outputs (``conclusion``,
``conclusion_frame``, ``label_levels``, ``precedent``, ``run_config``); nothing is graded or
re-estimated. Buy/sell/signal/call wording is reserved for a VALIDATED read
(``ui.components.BUY_SELL_RE``).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from convergence.nishkarsh import COMPONENTS, PILLARS
from core.config import CONV2_TIER_MODERATE, CONV2_TIER_STRONG, CONV2_TIER_WEAK
from ui.components import (
    _d, _str, build_hero_verdict, chip_html, component_label, edge_status_code, esc, fmt, num, pretty_date,
    render_hero, signed, sparkline_svg, z_side, zbar_html,
)
from ui.copy import NKO_COMPONENT_PLAIN, NKO_PILLAR_PLAIN, NKO_SCALE_TEXT, NKO_STATUS
from ui.theme import chart_layout

_PRICE = "rgba(226,232,240,0.78)"
_CRITICAL = ("Stale", "STALE", "FALLBACK", "EMPTY", "failed")
_PROV_WORD = {"live": "live", "cache": "cached", "stale": "stale", "fallback": "fallback source", "empty": "empty"}
_SIDE_RGBA = {"bull": "rgba(52,211,153,0.55)", "bear": "rgba(251,113,133,0.55)", "flat": "rgba(148,163,184,0.45)",
              "none": "rgba(148,163,184,0.25)"}
PILLAR_COLOR = {"top_down": "#D4A853", "bottom_up": "#22D3EE", "agreement": "#A78BFA"}
_SINGLE_FILL = "rgba(148,163,184,0.14)"
_TICK = dict(family="Inter, sans-serif", size=11, color="#7C879A")
VIEWS = ("Nishkarsh read", "Its three pillars")


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


def single_system_runs(fr: pd.DataFrame) -> list[tuple]:
    """Contiguous stretches where the read exists but only the top-down pillar counted (two_system False)."""
    if fr is None or not len(fr) or "NKO" not in fr.columns or "two_system" not in fr.columns:
        return []
    mask = pd.to_numeric(fr["NKO"], errors="coerce").notna() & ~fr["two_system"].astype(bool)
    out, s0, prev = [], None, None
    for d, m in mask.items():
        if m and s0 is None:
            s0 = d
        elif not m and s0 is not None:
            out.append((s0, prev))
            s0 = None
        prev = d
    if s0 is not None:
        out.append((s0, prev))
    return out


def _bar_widths(xs: pd.DatetimeIndex) -> list[float]:
    gaps = pd.Series(xs[1:] - xs[:-1]).dt.total_seconds().mul(1000).tolist() if len(xs) > 1 else []
    return gaps + [float(pd.Series(gaps).median()) if gaps else 86_400_000.0]


def _axes(fig) -> None:
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, tickfont=_TICK, ticks="", showspikes=True,
                     spikemode="across", spikethickness=0.5, spikedash="dot", spikecolor="rgba(148,163,184,0.35)")


# ── (2) the read over time vs the index ──────────────────────────────────────
def _nko_chart(fr: pd.DataFrame, price: pd.Series | None, index_name: str) -> go.Figure:
    nko = pd.to_numeric(fr["NKO"], errors="coerce")
    xs = pd.DatetimeIndex(fr.index)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for x0, x1 in single_system_runs(fr):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=_SINGLE_FILL, line_width=0, layer="below")
    fig.add_hrect(y0=-CONV2_TIER_WEAK, y1=CONV2_TIER_WEAK, fillcolor="rgba(148,163,184,0.06)", line_width=0,
                  layer="below")
    vals = [None if pd.isna(v) else float(v) for v in nko]
    colors = [_SIDE_RGBA["bull" if (v or 0) > CONV2_TIER_WEAK else "bear" if (v or 0) < -CONV2_TIER_WEAK else "flat"]
              for v in vals]
    # Contiguous bars (each as wide as the gap to the next date) read as a stepped area and break on
    # days without a reading. Display only — the values are the frame's NKO, unchanged.
    fig.add_trace(go.Bar(x=list(xs), y=vals, width=_bar_widths(xs), offset=0,
                         marker=dict(color=colors, line=dict(width=0)), name="Nishkarsh read", showlegend=False,
                         hovertemplate="Nishkarsh read %{y:+.0f}<extra></extra>"), secondary_y=False)
    if price is not None and len(price):
        fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name=index_name, showlegend=False,
                                 line=dict(color=_PRICE, width=1.5),
                                 hovertemplate=f"{index_name} " + "%{y:,.0f}<extra></extra>"), secondary_y=True)
    fig.update_layout(**chart_layout(height=340, show_legend=False, margin=dict(t=8, l=8, r=8, b=8)), bargap=0)
    _axes(fig)
    ticks = [-100, -CONV2_TIER_STRONG, -CONV2_TIER_MODERATE, 0, CONV2_TIER_MODERATE, CONV2_TIER_STRONG, 100]
    fig.update_yaxes(range=[-105, 105], tickvals=ticks, ticktext=[signed(t, "+.0f") for t in ticks],
                     showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                     zerolinecolor="rgba(148,163,184,0.25)", zerolinewidth=1, tickfont=_TICK, secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickformat=",.0f", tickfont=_TICK, secondary_y=True)
    return fig


def _pillars_chart(fr: pd.DataFrame) -> go.Figure:
    labels = [NKO_PILLAR_PLAIN[p][0] + f" ({NKO_PILLAR_PLAIN[p][1]})" for p in PILLARS]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09, subplot_titles=labels)
    runs = single_system_runs(fr)
    for i, p in enumerate(PILLARS, start=1):
        col = f"P_{p}"
        y = pd.to_numeric(fr[col], errors="coerce") if col in fr.columns else pd.Series(index=fr.index, dtype=float)
        fig.add_trace(go.Scatter(x=fr.index, y=y, mode="lines", name=labels[i - 1], showlegend=False,
                                 line=dict(color=PILLAR_COLOR[p], width=1.4), fill="tozeroy",
                                 fillcolor=PILLAR_COLOR[p] + "22", connectgaps=False,
                                 hovertemplate=f"{NKO_PILLAR_PLAIN[p][0]} " + "%{y:+.2f}<extra></extra>"),
                      row=i, col=1)
        fig.update_yaxes(range=[-4.2, 4.2], tickvals=[-3, 0, 3], ticktext=["−3", "0", "+3"], showgrid=True,
                         gridcolor="rgba(255,255,255,0.04)", zeroline=True, zerolinecolor="rgba(148,163,184,0.3)",
                         tickfont=_TICK, row=i, col=1)
    for x0, x1 in runs:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=_SINGLE_FILL, line_width=0, layer="below", row="all", col=1)
    fig.update_layout(**chart_layout(height=420, show_legend=False, margin=dict(t=24, l=8, r=8, b=8)))
    fig.update_annotations(font=dict(family="Inter, sans-serif", size=12, color="#AEB6C4"), x=0, xanchor="left")
    _axes(fig)
    return fig


def _render_history(ss, start, tf_label: str, con: dict) -> None:
    frame = ss.get("conclusion_frame")
    if frame is None or len(frame) == 0 or "NKO" not in frame.columns:
        return
    fr = _since(frame, start)
    lv = ss.get("label_levels")
    price = _since(lv, start) if isinstance(lv, pd.Series) else None
    name = _index_name(con)
    span = span_words(tf_label)
    with st.container(border=True, key="nk_history"):
        left, right = st.columns([3, 1.2], vertical_alignment="bottom")
        left.markdown(f'<div class="nk-card-head"><div><h3>The read over time</h3><p>The Nishkarsh Oscillator '
                      f'each day over {esc(span)}, with the {esc(name)} for context.</p></div></div>',
                      unsafe_allow_html=True)
        with right:
            view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="ov_hist_view", required=True,
                                        label_visibility="collapsed", width="stretch") or VIEWS[0]
        pillars = view == VIEWS[1]
        legend = ("".join(f'<span><i class="sw" style="background:{PILLAR_COLOR[p]}"></i>'
                          f'{esc(NKO_PILLAR_PLAIN[p][0])}</span>' for p in PILLARS) if pillars else
                  f'<span><i class="sw bull"></i>Bullish read</span><span><i class="sw bear"></i>Bearish read</span>'
                  f'<span><i class="sw flat"></i>No lean</span><span><i class="sw line"></i>{esc(name)} (right axis)</span>')
        has_single = bool(single_system_runs(fr))
        st.markdown(f'<div class="nk-legend">{legend}' + ('<span><i class="sw single"></i>Single-system '
                    '(top-down only)</span>' if has_single else "") + '</div>', unsafe_allow_html=True)
        if len(fr):
            fig = _pillars_chart(fr) if pillars else _nko_chart(fr, price, name)
            st.plotly_chart(fig, width="stretch", key="ov_history", config={"displayModeBar": False})
        foot = ("Each pillar is a causal z-score against its own past (+ = bullish); the read is their average, "
                "rescaled to ±100." if pillars else f"Scale: {NKO_SCALE_TEXT}.")
        st.markdown(f'<div class="nk-card-foot">{esc(foot)} Shaded stretches had no Nirnay data, so only the '
                    'top-down pillar counted; a gap means no reading that day.</div>', unsafe_allow_html=True)


# ── (3) what drives today's read: pillars → components ───────────────────────
def _pillar_card(p: str, k: dict, fr: pd.DataFrame, span: str) -> str:
    name, system, desc = NKO_PILLAR_PLAIN[p]
    z = num(_d(k.get("pillars")).get(p))
    comps = [_d(c) for c in (k.get("components") or []) if _d(c).get("pillar") == p]
    if not comps:
        comps = [{"id": cid, "pillar": pil, "z": None} for pil, cid, _, _ in COMPONENTS if pil == p]
    drivers = {_str(_d(c).get("id")) for c in (k.get("drivers") or [])}
    rows = "".join(
        f'<div class="nk-comp{" driver" if _str(c.get("id")) in drivers else ""}">'
        f'<div class="nk-comp-top"><span class="nk-comp-label">{esc(component_label(c.get("id"), c.get("label")))}'
        + ('<span class="nk-driver-tag">driver</span>' if _str(c.get("id")) in drivers else "")
        + f'</span><span class="nk-comp-z {z_side(c.get("z"))}">{esc(signed(c.get("z")))}</span></div>'
        f'{zbar_html(c.get("z"))}'
        f'<div class="nk-comp-src">{esc(NKO_COMPONENT_PLAIN.get(_str(c.get("id")), ("", _str(c.get("source"))))[1])}'
        f'</div></div>' for c in comps)
    col = f"P_{p}"
    spark = sparkline_svg([(fr[col].tolist() if col in fr.columns else [], PILLAR_COLOR[p])], y_range=(-4, 4),
                          zero=True)
    n_in = len(comps)
    return (f'<div class="nk-pillar">'
            f'<div class="nk-pillar-head"><div><h4>{esc(name)}</h4><span>{esc(system)} · {n_in} input'
            f'{"s" if n_in != 1 else ""}</span></div><span class="nk-pillar-z {z_side(z)}">{esc(signed(z))}</span></div>'
            f'{zbar_html(z)}<p>{esc(desc[:1].upper() + desc[1:])}.</p>'
            f'<div class="nk-spark-wrap">{spark}<div class="nk-spark-cap">This pillar over {esc(span)} · '
            f'above the dotted line leans bullish</div></div>'
            f'<div class="nk-comp-list">{rows}</div></div>')


def _render_drivers(ss, start, tf_label: str, con: dict) -> None:
    k = _d(con.get("nko"))
    if not k:
        return
    frame = ss.get("conclusion_frame")
    fr = _since(frame, start) if frame is not None else pd.DataFrame()
    span = span_words(tf_label)
    single = k.get("two_system") is not True
    st.markdown('<div class="nk-section-head"><h3>What drives today\'s read</h3><p>The read is the average of '
                'three pillars, each a causal z-score against its own past (+ = bullish). Under each pillar, the '
                'inputs it averages, with today\'s z-score; the three largest are marked as drivers.'
                + (' Today is a single-system date: the bottom-up and agreement pillars have no reading.'
                   if single else '') + '</p></div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for c, p in zip(cols, PILLARS):
        c.markdown(_pillar_card(p, k, fr, span), unsafe_allow_html=True)


# ── (4) How reliable is this? — the read's measurement in plain language ─────
def build_reliability_rows(conclusion) -> list[dict]:
    """Plain-language rows (claim · status · meaning · numbers) from the run's own flags. Pure; status
    words follow the measurement's booleans, never a UI threshold."""
    con = _d(conclusion)
    h = int(num(con.get("horizon")) or 10)
    k = _d(con.get("nko"))
    status = _str(k.get("status")).upper()
    status = status if status in NKO_STATUS else "INSUFFICIENT_DATA"
    label, cls, _ = NKO_STATUS[status]
    meaning = {"VALIDATED": f"Its readings have lined up with the Nifty's next {h} trading days, clearing the "
                            "bar set before it was tested.",
               "UNPROVEN": "Its readings have not yet lined up with the Nifty's next moves strongly or "
                           "consistently enough to act on.",
               "INSUFFICIENT_DATA": "Not enough history with both systems running to measure it."}[status]
    ev = _d(k.get("evidence"))
    yearly = _d(ev.get("yearly"))
    share = num(ev.get("yearly_positive_share"))
    yrs = f" · positive in {int(round(share * len(yearly)))} of {len(yearly)} years" if share is not None and yearly else ""
    rows = [{"claim": "The Nishkarsh read anticipates the Nifty", "status": label, "class": cls, "meaning": meaning,
             "numbers": f"accuracy score {signed(ev.get('ic'), '+.3f')} · p {fmt(ev.get('p'), '.2f')} · "
                        f"{fmt(ev.get('n'), '.0f')} separate {h}-day periods{yrs}"}]
    pic = _d(k.get("pillar_ic"))
    for p in PILLARS:
        e = _d(pic.get(p))
        rows.append({"claim": f"{NKO_PILLAR_PLAIN[p][0]} pillar on its own", "status": "Reported", "class": "muted",
                     "meaning": "Measured on the same periods for context; it does not decide anything.",
                     "numbers": f"accuracy score {signed(e.get('ic'), '+.3f')} · p {fmt(e.get('p'), '.2f')} · "
                                f"{fmt(e.get('n'), '.0f')} periods"})
    e = _d(con.get("edge"))
    st_ = {"PROVEN": ("Proven", "good", f"Its direction has matched the next {h} trading days more often than "
                      "chance on this data."),
           "NOT_PROVEN": ("Not proven", "caution", "So far its direction has not matched the market often enough "
                          "to rule out luck."),
           "INVERTED": ("Wrong way", "bad", "It has pointed the wrong way more often than chance."),
           "INSUFFICIENT": ("Not enough data", "muted", "There is not enough history yet to judge it.")}[
        edge_status_code(e)]
    rows.append({"claim": "Aarambh's forecast input points the right way", "status": st_[0], "class": st_[1],
                 "meaning": st_[2] + " It is one of the ten inputs.",
                 "numbers": f"accuracy score {signed(e.get('ic'), '+.3f')} · p {fmt(e.get('p'), '.2f')} · "
                            f"{fmt(e.get('n'), '.0f')} separate {h}-day periods"})
    return rows


def _render_reliability(con: dict) -> None:
    k = _d(con.get("nko"))
    status = _str(k.get("status")).upper()
    label, cls, sentence = NKO_STATUS.get(status, NKO_STATUS["INSUFFICIENT_DATA"])
    h = int(num(con.get("horizon")) or 10)
    rows = "".join(
        f'<div class="nk-ledger-row"><div class="nk-claim">{esc(r["claim"])}</div>'
        f'<div class="nk-ledger-status">{chip_html(r["status"], r["class"])}</div>'
        f'<div class="nk-meaning">{esc(r["meaning"])}<div class="nk-ledger-nums">{esc(r["numbers"])}</div></div>'
        f'</div>' for r in build_reliability_rows(con))
    with st.expander("How reliable is this?", expanded=False):
        st.markdown(
            f'<div class="nk-ledger"><div class="nk-ledger-overall"><span>The Nishkarsh read</span>'
            f'{chip_html(label, cls)}<p>{esc(sentence.format(h=h))} Each figure is measured on history with the '
            f'rule fixed before the test.</p></div>'
            f'{rows}<div class="nk-ledger-foot">Exact figures, the rule and every component: Evidence tab.</div></div>',
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
    """Overview — the Nishkarsh read (hero + gauge), its history, what drives it, reliability, data notes."""
    ss = st.session_state
    con, rc = _d(ss.get("conclusion")), _d(ss.get("run_config"))
    render_hero(build_hero_verdict(con, _d(ss.get("precedent")).get("hero"), rc))
    crit = [w for w in (rc.get("warnings") or []) if any(k in str(w) for k in _CRITICAL)]
    if crit:
        st.markdown('<div class="nk-inline-warn">Some inputs are stale or came from a fallback source — '
                    'see Data notes at the bottom of this page.</div>', unsafe_allow_html=True)
    _render_history(ss, start, tf_label, con)
    _render_drivers(ss, start, tf_label, con)
    st.markdown('<div class="nk-gap"></div>', unsafe_allow_html=True)
    _render_reliability(con)
    _render_data_notes(rc)
