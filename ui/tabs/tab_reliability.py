"""
Nishkarsh — Reliability: everything about trust, plus method, limits and glossary, in place.

Status summary of the one read (status, four numbers, accuracy by year) · the Ledger (every claim
the system makes, its status and the one number that decides it, keyed to docs/EDGE_LEDGER.md) ·
the detailed tests (agreement lift, cost-aware backtest, stage-by-stage accuracy) · the models' facts · method
& limits and a glossary as plain text. Status words follow the measurement's own flags
(``conclusion`` payload); the UI grades nothing itself.
"""

from __future__ import annotations

import streamlit as st

from core.config import BACKTEST_REFERENCE_BPS, EDGE_ALPHA, EDGE_TRADEABLE_T
from ui.charts import CONFIG, yearly_bars
from ui.components import (
    PILLARS, _d, _list, _str, chip_html, edge_status_code, esc, fmt, num, plural, pretty_date, section_head, signed,
    stat_tiles,
)
from ui.copy import GLOSSARY, LEVELS_TAG, METHOD_LINES, NIRNAY_READ, NKO_PILLAR_PLAIN, NKO_STATUS

_READ_STATUS = {"VALIDATED": ("Validated", "good"), "UNPROVEN": ("Unproven", "caution"),
                "INSUFFICIENT_DATA": ("Not enough data", "muted")}
_STAGE = {"forecast": "Aarambh forecast", "engine_conviction": "Engine conviction", "C": "Conclusion C",
          "kappa_shrunk": "After κ shrink"}


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
    rows.append({"id": "", "claim": "Its three pillars on their own", "status": ("Context", "muted"),
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
                 "status": {"PROVEN": ("Proven", "good"), "NOT_PROVEN": ("Not proven", "caution"),
                            "INVERTED": ("Wrong way", "bad"), "INSUFFICIENT": ("Not enough data", "muted")}[code],
                 "number": _ic_num(e),
                 "meaning": {"PROVEN": f"Its direction has matched the next {h} days more often than chance.",
                             "NOT_PROVEN": "Its direction has not matched the market often enough to rule out luck.",
                             "INVERTED": "It has pointed the wrong way more often than chance.",
                             "INSUFFICIENT": "Not enough history to judge it."}[code]})

    lift, oos = _d(con.get("lift")), _d(con.get("oos"))
    if lift.get("available") is not True:
        ls = ("Not enough data", "muted")
    else:
        ls = ("Passed", "good") if lift.get("passed") is True else ("Not proven", "caution")
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
                 "status": ("Tradeable", "good") if tradeable else
                 (("Not tradeable", "caution") if ref else ("Not enough data", "muted")),
                 "number": f"net Sharpe {signed(ref.get('net_sharpe'))} · t {signed(ref.get('t'))} at "
                           f"{BACKTEST_REFERENCE_BPS:g} bps",
                 "meaning": (f"Net of {BACKTEST_REFERENCE_BPS:g} bps a trade, positioning on it has earned a "
                             "significant return." if tradeable else
                             f"Net of {BACKTEST_REFERENCE_BPS:g} bps a trade, positioning on it has not earned a "
                             "reliable return.")})

    ph = _d(_d(precedent).get("hero"))
    n, pos, u, p = (num(ph.get(x)) for x in ("n", "positive_pct", "unconditional_positive_pct", "binomial_p"))
    if not n or pos is None:
        ps, pnum = ("Not enough data", "muted"), "—"
    else:
        sig = p is not None and p < EDGE_ALPHA
        ps = ("Significant", "good") if sig else ("Descriptive", "muted")
        pnum = (f"{int(round(pos / 100 * n))} of {int(n)} rose vs {fmt(u, '.0f', suffix='%')} usually · "
                f"p {fmt(p, '.2f')}")
    rows.append({"id": "L-10", "claim": "Past moments like today point the way", "status": ps, "number": pnum,
                 "meaning": ("These moments' outcomes differ from the usual by more than chance." if ps[0] == "Significant"
                             else "A small sample; their outcomes are description, not evidence.")})
    return rows


def _ledger_html(rows: list[dict]) -> str:
    body = "".join(
        f'<div class="nk-lrow"><div class="nk-lclaim">{esc(r["claim"])}'
        + (f'<span class="nk-lid">{esc(r["id"])}</span>' if r["id"] else "")
        + f'</div><div>{chip_html(r["status"][0], r["status"][1])}</div>'
        f'<div class="nk-lnum">{esc(r["number"])}</div><div class="nk-lmean">{esc(r["meaning"])}</div></div>'
        for r in rows)
    head = ('<div class="nk-lrow head"><div>Claim</div><div>Status</div><div>The deciding number</div>'
            '<div>What it means</div></div>')
    return f'<div class="nk-ledger">{head}{body}</div>'


def _kv(rows: list[tuple]) -> str:
    return '<dl class="nk-kv">' + "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in rows) + "</dl>"


def _lift_panel(con: dict) -> str:
    lift = _d(con.get("lift"))
    if lift.get("available") is not True:
        body = _kv([("Status", "Not enough periods on each side to test.")])
    else:
        body = _kv([
            ("When Nirnay agrees", f"IC {signed(lift.get('ic_confirm'), '+.3f')} over "
                                   f"{plural(lift.get('n_confirm'), 'period')}"),
            ("When it disagrees", f"IC {signed(lift.get('ic_contradict'), '+.3f')} over "
                                  f"{plural(lift.get('n_contradict'), 'period')}"),
            ("Lift", f"{signed(lift.get('lift'), '+.3f')} · one-sided p {fmt(lift.get('p_one_sided'), '.2f')}"),
            ("Smallest detectable lift", fmt(lift.get("mde"), ".2f")),
            ("Across start days", f"{signed(lift.get('lift_phase_min'), '+.2f')} to "
                                  f"{signed(lift.get('lift_phase_max'), '+.2f')}"),
            ("κ applied", f"{fmt(lift.get('kappa'), '.2f')} · Intelligence "
                          f"{'on' if con.get('kappa_enabled') is True else 'off'}"),
        ])
    return f'<div class="nk-panel"><h4>Agreement lift</h4>{body}</div>'


def _chain_panel(con: dict) -> str:
    chain = [_d(c) for c in _list(con.get("chain")) if isinstance(c, dict)]
    rows = [(_STAGE.get(_str(c.get("stage")), _str(c.get("stage")) or "—"),
             f"IC {signed(c.get('ic'), '+.3f')} · {plural(c.get('n'), 'period')}") for c in chain]
    return (f'<div class="nk-panel"><h4>Aarambh, stage by stage</h4>{_kv(rows or [("Stages", "—")])}'
            '<p class="nk-panel-foot">Accuracy at each stage of the forecast, on the same periods.</p></div>')


def _backtest_html(con: dict) -> str:
    bt = _d(con.get("backtest"))
    by = _d(bt.get("by_cost"))
    keys = sorted([kk for kk in by if num(kk) is not None], key=lambda s: num(s))

    def row(label, b):
        b = _d(b)
        return (f'<tr><td>{esc(label)}</td><td class="num">{esc(signed(b.get("gross_sharpe")))}</td>'
                f'<td class="num">{esc(signed(b.get("net_sharpe")))}</td><td class="num">{esc(signed(b.get("t")))}</td>'
                f'<td class="num">{esc(fmt((num(b.get("ann_return")) or 0) * 100 if num(b.get("ann_return")) is not None else None, "+.1f", suffix="%"))}</td>'
                f'<td class="num">{esc(fmt(b.get("turnover_per_year"), ".1f"))}</td>'
                f'<td class="num">{esc(fmt(b.get("breakeven_bps"), ".1f"))}</td>'
                f'<td class="num">{esc(fmt(b.get("n"), ",.0f"))}</td></tr>')
    rows = [row(f"{fmt(kk, 'g')} bps", by[kk]) for kk in keys]
    if _d(bt.get("nonoverlap_reference")):
        rows.append(row(f"{fmt(bt.get('reference_bps'), 'g')} bps · separate periods", bt["nonoverlap_reference"]))
    if not rows:
        return '<div class="nk-empty">No backtest in this run.</div>'
    return ('<div class="nk-table-wrap"><table class="nk-table"><thead><tr><th>Cost per trade</th>'
            '<th class="num">Gross Sharpe</th><th class="num">Net Sharpe</th><th class="num">HAC t</th>'
            '<th class="num">Return / yr</th><th class="num">Turnover / yr</th><th class="num">Break-even bps</th>'
            f'<th class="num">Days</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _model_facts(rr: dict, rc: dict, signal: dict, forward: bool, h: int) -> str:
    a = _d(rr.get("aarambh"))
    wf, ms = _d(a.get("wf_meta")), _d(a.get("model_stats"))
    feats = a.get("feature_names") or rc.get("features") or []
    arows = [
        ("Mode", f"Forward — forecasts the next {h} trading days" if forward else f"Levels — {LEVELS_TAG}"),
        ("Walk-forward", f"train {fmt(wf.get('min_train'), ',.0f')}–{fmt(wf.get('max_train'), ',.0f')} rows · purge "
                         f"{fmt(wf.get('purge'), '.0f')} · refit every {fmt(wf.get('refit'), '.0f')} rows"),
        ("Refit windows", f"{fmt(wf.get('chunks_total'), ',.0f')} ({fmt(wf.get('chunks_skipped_starved'), '.0f')} "
                          f"starved, {fmt(wf.get('chunks_fallback'), '.0f')} fallback)"),
        ("Out-of-sample rows", f"{fmt(a.get('valid'), ',.0f')} of {fmt(a.get('rows'), ',.0f')}, from "
                               f"{pretty_date(a.get('first_valid'))}"),
        ("Skill vs " + ("no change" if forward else "random walk"),
         f"{signed(ms.get('r2_vs_rw'), '+.3f')} — below 0, " + ("predicting no change" if forward
                                                                  else "yesterday's level") + " did better"),
        ("Ensemble", f"Ridge, Huber, ElasticNet and PCA-OLS, equal weights · average spread "
                     f"{fmt(ms.get('avg_model_spread'), '.4f')}"),
        ("Inputs", f"{len(feats)} series: " + ", ".join(map(str, feats)) if feats else "—"),
    ]
    if not forward:
        meaningful = _d(signal).get("half_life_meaningful") is True
        arows += [("Mean-reversion half-life", fmt(_d(signal).get("ou_half_life"), ".0f", suffix=" days")
                   if meaningful else "not meaningful on this fit"),
                  ("Stationarity", f"ADF p {fmt(_d(signal).get('adf_pvalue'), '.3f')} · KPSS p "
                                   f"{fmt(_d(signal).get('kpss_pvalue'), '.3f')}")]
    n = _d(rr.get("nirnay"))
    if n.get("stocks_analyzed"):
        lat = _d(n.get("latest"))
        span = n.get("span") if isinstance(n.get("span"), (list, tuple)) and len(n.get("span")) == 2 else (None, None)
        regimes = " · ".join(f"{str(r).replace('_', ' ').lower()} {c}" for r, c in _d(n.get("regime_latest")).items())
        nrows = [
            ("Stocks", f"{fmt(n.get('stocks_analyzed'), '.0f')} analysed of {fmt(n.get('stocks_requested'), '.0f')} "
                       f"({len(_d(n.get('failed')))} failed)"),
            ("History", f"{fmt(n.get('trading_days'), ',.0f')} trading days, {pretty_date(span[0])} – "
                        f"{pretty_date(span[1])}"),
            ("Live today", f"{fmt(lat.get('n_live'), '.0f')} of {fmt(lat.get('n_universe'), '.0f')}"),
            ("Adaptive cuts", f"oversold {signed(n.get('os_cut_median_live'))} · overbought "
                              f"{signed(n.get('ob_cut_median_live'))} (median of live stocks)"),
            ("Regimes today", regimes or "—"),
            ("Polarity", NIRNAY_READ),
        ]
    else:
        nrows = [("Stocks", "No constituent data this run — the read is top-down only.")]
    return (f'<div class="nk-cols2"><div class="nk-panel"><h4>Aarambh — top-down</h4>{_kv(arows)}</div>'
            f'<div class="nk-panel"><h4>Nirnay — bottom-up</h4>{_kv(nrows)}</div></div>')


def render_reliability_tab(signal: dict | None = None) -> None:
    ss = st.session_state
    con, rc, rr = _d(ss.get("conclusion")), _d(ss.get("run_config")), _d(ss.get("run_report"))
    k = _d(con.get("nko"))
    h = int(num(con.get("horizon")) or 10)
    forward = con.get("forward_mode") is not False

    status = _str(k.get("status")).upper()
    label, cls, sentence = NKO_STATUS.get(status, NKO_STATUS["INSUFFICIENT_DATA"])
    ev = _d(k.get("evidence"))
    st.markdown(
        section_head("How far to trust the read", level="top") +
        f'<div class="nk-verdict">{chip_html(label, cls)}<p>{esc(sentence.format(h=h))}</p></div>'
        + stat_tiles([
            ("Accuracy (IC)", signed(ev.get("ic"), "+.3f"), f"vs the Nifty's next {h} trading days"),
            ("Periods tested", fmt(ev.get("n"), ",.0f"), f"separate {h}-day periods, both systems running"),
            ("p-value", fmt(ev.get("p"), ".2f"), f"needs to be below {EDGE_ALPHA:g}"),
            ("Smallest detectable", fmt(ev.get("mde"), ".2f"), "the least accuracy this history can confirm"),
        ]), unsafe_allow_html=True)
    yearly = {str(y): num(v) for y, v in _d(ev.get("yearly")).items() if num(v) is not None}
    if yearly:
        share = num(ev.get("yearly_positive_share"))
        pos = f" — positive in {int(round(share * len(yearly)))} of {len(yearly)}" if share is not None else ""
        st.markdown(f'<div class="nk-sub">Accuracy by calendar year{esc(pos)}</div>', unsafe_allow_html=True)
        st.plotly_chart(yearly_bars(yearly, height=180), width="stretch", key="rel_yearly", config=CONFIG)

    st.markdown(section_head("The ledger", "Every claim Nishkarsh makes, its status on this run's data, and the one "
                             "number that decides it. IDs refer to docs/EDGE_LEDGER.md.")
                + _ledger_html(build_ledger_rows(con, ss.get("precedent"))), unsafe_allow_html=True)

    st.markdown(section_head("Detailed tests", f"Graded at one-sided p < {EDGE_ALPHA:g}; tradeable needs HAC t ≥ "
                             f"{EDGE_TRADEABLE_T:g} net of {BACKTEST_REFERENCE_BPS:g} bps.")
                + f'<div class="nk-cols2">{_lift_panel(con)}{_chain_panel(con)}</div>'
                + '<div class="nk-sub">Cost-aware backtest of the Aarambh forecast (position = C / 100)</div>'
                + _backtest_html(con), unsafe_allow_html=True)

    st.markdown(section_head("The models") + _model_facts(rr, rc, signal or {}, forward, h), unsafe_allow_html=True)

    method = "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in METHOD_LINES)
    gloss = "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in GLOSSARY)
    st.markdown(section_head("Method and limits") + f'<dl class="nk-defs">{method}</dl>'
                + section_head("Glossary", level="sm") + f'<dl class="nk-defs gloss">{gloss}</dl>',
                unsafe_allow_html=True)
