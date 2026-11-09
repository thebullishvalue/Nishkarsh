"""
Nishkarsh — Drivers: the decomposition of the one read. Three sections in the same grammar —
the pillar's name, today's value and what it measures; the pillar over time; its inputs as small
multiples — then, for the bottom-up pillar, every stock today, and for the agreement pillar the
agreement anatomy. Renders ``conclusion["nko"]``, ``conclusion_frame``, ``nirnay_daily``,
``nirnay_constituent_dfs`` and ``anatomy_df``; nothing is re-estimated (sorting is display).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from convergence.nishkarsh import COMPONENTS
from core.config import NIRNAY_SIGN
from ui.charts import AMBER, CONFIG, lines_figure, since, small_multiples, z_figure
from ui.components import _d, _list, _str, component_label, esc, fmt, num, signed, z_side, zbar_html
from ui.copy import LEVELS_TAG, NIRNAY_READ_SENTENCE, NKO_PILLAR_PLAIN

_ZONE_SIDE = ({"OVERBOUGHT": "bull", "OVERSOLD": "bear"} if NIRNAY_SIGN > 0
              else {"OVERBOUGHT": "bear", "OVERSOLD": "bull"})


def _col(fr: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(fr[c], errors="coerce") if c in fr.columns else pd.Series(float("nan"), index=fr.index)


def _pillar_header(p: str, k: dict, extra: str = "") -> str:
    name, title, desc = NKO_PILLAR_PLAIN[p]
    z = num(_d(k.get("pillars")).get(p))
    return (f'<div class="nk-pillar-sec"><div class="nk-pillar-top"><div><div class="nk-pillar-name">{esc(name)}</div>'
            f'<h3>{esc(title)}</h3></div><div class="nk-pillar-today"><span>today</span>'
            f'<b class="{z_side(z, 2)}">{esc(signed(z))}</b></div></div><p>{esc(desc)}'
            + (f' {esc(extra)}' if extra else "") + '</p></div>')


def _tiles(p: str, k: dict, fr: pd.DataFrame, daily: pd.DataFrame | None) -> list[dict]:
    comps = {_str(_d(c).get("id")): num(_d(c).get("z")) for c in _list(k.get("components"))}
    tiles = []
    for pil, cid, lab, _ in COMPONENTS:
        if pil != p:
            continue
        name = component_label(cid, lab)
        title = f"{name} · {signed(comps.get(cid), '+.1f')}"
        if cid == "zone_breadth_n" and daily is not None and len(daily):
            ob = (_col(daily, "Overbought_Pct"), "Overbought")
            os_ = (_col(daily, "Oversold_Pct"), "Oversold")
            tiles.append({"title": title, "kind": "mirror", "series": [ob, os_] if NIRNAY_SIGN > 0 else [os_, ob]})
        else:
            tiles.append({"title": title, "kind": "z", "series": [(_col(fr, f"z_{cid}"), name)]})
    return tiles


def _section_charts(p: str, k: dict, fr: pd.DataFrame, daily=None) -> None:
    st.markdown('<div class="nk-sub">The pillar over time</div>', unsafe_allow_html=True)
    st.plotly_chart(z_figure(_col(fr, f"P_{p}"), NKO_PILLAR_PLAIN[p][0], height=160), width="stretch",
                    key=f"drv_pillar_{p}", config=CONFIG)
    up = "overbought" if NIRNAY_SIGN > 0 else "oversold"
    down = "oversold" if NIRNAY_SIGN > 0 else "overbought"
    extra = (f"; zone breadth shows the % of stocks {up} above the line and {down} below it"
             if daily is not None and len(daily) else "")
    st.markdown(f'<div class="nk-sub">Its inputs, each against its own past (+ = bullish){esc(extra)}</div>',
                unsafe_allow_html=True)
    st.plotly_chart(small_multiples(_tiles(p, k, fr, daily)), width="stretch", key=f"drv_inputs_{p}", config=CONFIG)


def _stock_rows(dfs: dict, as_of) -> tuple[list[dict], int]:
    """One row per stock on the latest Nirnay date, sorted most bullish first (+ = bullish under the
    run's polarity). Stocks without a warm bar that day are counted, not drawn."""
    rows, missing = [], 0
    for sym, df in (dfs or {}).items():
        if df is None or not len(df):
            missing += 1
            continue
        d = df
        if isinstance(d.columns, pd.MultiIndex):
            d = d.copy()
            d.columns = [c[0] for c in d.columns]
        if as_of is not None and as_of not in d.index:
            missing += 1
            continue
        r = d.loc[as_of] if as_of is not None else d.iloc[-1]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[-1]
        osc = num(r.get("Unified_Osc"))
        if osc is None or bool(r.get("Warmup", False)):
            missing += 1
            continue
        zone = _str(r.get("Condition")).upper()
        rows.append({"sym": str(sym).replace(".NS", ""), "osc": NIRNAY_SIGN * osc,
                     "side": _ZONE_SIDE.get(zone, "flat"), "zone": zone.capitalize() or "—",
                     "regime": _str(r.get("Regime")).replace("_", " ").lower() or "—"})
    rows.sort(key=lambda x: -x["osc"])
    return rows, missing


def _strip_html(rows: list[dict]) -> str:
    scale = max([3.0] + [abs(r["osc"]) for r in rows])
    half = (len(rows) + 1) // 2

    def col(rs):
        return "".join(
            f'<div class="nk-srow" title="{esc(r["zone"])}"><span class="sym">{esc(r["sym"])}</span>'
            f'<div class="bar zone-{r["side"]}">{zbar_html(r["osc"], scale)}</div>'
            f'<span class="val {r["side"]}">{esc(signed(r["osc"], "+.1f"))}</span>'
            f'<span class="tag">{esc(r["regime"])}</span></div>' for r in rs)
    return f'<div class="nk-strip"><div>{col(rows[:half])}</div><div>{col(rows[half:])}</div></div>'


def _render_stocks(ss, daily: pd.DataFrame) -> None:
    dfs = ss.get("nirnay_constituent_dfs") or {}
    if not dfs or daily is None or not len(daily):
        return
    as_of = daily.index.max()
    rows, missing = _stock_rows(dfs, as_of)
    if not rows:
        return
    st.markdown(
        '<div class="nk-sub">Every stock today, most bullish first</div>'
        '<div class="nk-strip-legend"><span><i class="sw bull"></i>bullish zone</span><span><i class="sw bear"></i>'
        'bearish zone</span><span><i class="sw flat"></i>neutral</span><span>bar = the stock\'s oscillator · '
        'tag = its regime</span>' + (f'<span>{missing} without a live bar today</span>' if missing else "")
        + f'</div>{_strip_html(rows)}', unsafe_allow_html=True)


def render_drivers_tab(start, tf_label: str) -> None:
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    k = _d(con.get("nko"))
    frame = ss.get("conclusion_frame")
    if not k or frame is None or not len(frame):
        st.markdown('<div class="nk-empty">The Nishkarsh read was not built in this run.</div>', unsafe_allow_html=True)
        return
    fr = since(frame, start)
    st.markdown('<p class="nk-intro">The Nishkarsh read is the average of three pillars. Each is built from the '
                'inputs below, each standardized against its own past (+ = bullish), so every chart here shares '
                'one scale: above the line leans bullish, below it bearish.</p>', unsafe_allow_html=True)

    # Top-down
    levels = "" if con.get("forward_mode") is not False else f"In levels mode the forecast input is {LEVELS_TAG}."
    st.markdown(_pillar_header("top_down", k, levels), unsafe_allow_html=True)
    _section_charts("top_down", k, fr)

    # Bottom-up
    daily_full = ss.get("nirnay_daily")
    has_n = daily_full is not None and len(daily_full) > 0
    if has_n:
        daily_full = daily_full[~daily_full.index.duplicated(keep="last")].sort_index()
        last = daily_full.iloc[-1]
        cov = (f"{fmt(last.get('n_live'), '.0f')} of {fmt(last.get('n_universe'), '.0f')} stocks live today · "
               f"{fmt(last.get('Overbought_Pct'), '.0f', suffix='%')} overbought · "
               f"{fmt(last.get('Oversold_Pct'), '.0f', suffix='%')} oversold.")
        st.markdown(_pillar_header("bottom_up", k, cov), unsafe_allow_html=True)
        _section_charts("bottom_up", k, fr, since(daily_full, start))
        _render_stocks(ss, daily_full)
    else:
        st.markdown(_pillar_header("bottom_up", k) + '<div class="nk-empty">Not available this run — there is no '
                    'constituent data, so the read is top-down only.</div>', unsafe_allow_html=True)

    # Agreement
    st.markdown(_pillar_header("agreement", k, NIRNAY_READ_SENTENCE), unsafe_allow_html=True)
    if not has_n:
        st.markdown('<div class="nk-empty">Not available this run — agreement needs both systems.</div>',
                    unsafe_allow_html=True)
        return
    st.markdown('<div class="nk-sub">The pillar over time — its one input is the signed agreement: the two '
                'systems\' shared direction times how fully they agree</div>', unsafe_allow_html=True)
    st.plotly_chart(z_figure(_col(fr, "P_agreement"), "Agreement", height=160), width="stretch",
                    key="drv_pillar_agreement", config=CONFIG)
    an = since(ss.get("anatomy_df"), start)
    if not len(an):
        return
    lat = an.iloc[-1]
    dims = "".join(
        f'<div class="nk-dim"><span>{esc(lab)}</span><div class="nk-dimbar"><div style="width:'
        f'{max(0.0, min(1.0, num(lat.get(c)) or 0.0)) * 100:.0f}%"></div></div><b>{esc(fmt(lat.get(c), ".0%"))}</b></div>'
        for c, lab in (("dim_direction", "Direction"), ("dim_breadth", "Breadth"), ("dim_magnitude", "Magnitude"),
                       ("dim_regime", "Regime")) if c in an.columns)
    c1, c2 = st.columns([2.2, 1], gap="large")
    with c1:
        st.markdown(f'<div class="nk-sub">How fully they agree · {esc(fmt(lat.get("agreement_ratio"), ".0%"))} '
                    'today</div>', unsafe_allow_html=True)
        if "agreement_ratio" in an.columns:
            st.plotly_chart(lines_figure([(an["agreement_ratio"], "Agreement", AMBER, 1.6)], height=170,
                                         y_range=[0, 1.02], pct=True, legend=False), width="stretch",
                            key="drv_anatomy", config=CONFIG)
    with c2:
        st.markdown('<div class="nk-sub">Its four parts today</div>'
                    f'<div class="nk-dims">{dims}</div>', unsafe_allow_html=True)
