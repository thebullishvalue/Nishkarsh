"""
Nishkarsh — System: how far to trust each part, the ledger, the models, the data, the method
(≈ Tattva's System page).

PLACEHOLDER (phase A): renders the page's first section header and an empty state. Phase B
fills it per DESIGN_SPEC_v4 §3 "System": "Walk-Forward Edge — Every Engine" (table panel +
grouped per-period bars from ``conclusion["nko"]["edge"]`` / ``["pillar_edge"]``) → "The Ledger"
(``build_ledger_rows`` below → ``render_table_panel``; status tones are ``render_chip`` tones)
→ "Model" → "Data Layer Health" (per source; notes through ``pipeline.shell.redact``) →
"Method & Limits" (``ui.copy.METHOD_LINES``; glossary terms go into card TOOLTIPS). Unwindowed.

``build_ledger_rows`` is the retired Reliability page's pure ledger (tested in
tests/test_ui_pages.py): one row per claim, status words from the payload's own flags.
"""

from __future__ import annotations

import streamlit as st

from core.config import BACKTEST_REFERENCE_BPS, EDGE_ALPHA
from pipeline.shell import redact  # noqa: F401 — the one link redactor (Data Layer Health notes)
from ui.components import (
    PILLARS, _d, _str, edge_status_code, fmt, num, plural, render_empty_state, render_section_header, signed,
)
from ui.copy import NKO_PILLAR_PLAIN

__all__ = ["build_ledger_rows", "redact", "render_system_page"]

_READ_STATUS = {"VALIDATED": ("Validated", "success"), "UNPROVEN": ("Unproven", "warning"),
                "INSUFFICIENT_DATA": ("Not enough data", "neutral")}
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


def render_system_page() -> None:
    v = _d(_d(st.session_state.get("conclusion")).get("verdict"))
    t = _d(v.get("trust"))
    desc = (f"Every engine graded one way — the read's own walk-forward edge is {_str(t.get('chip'), 'NO READ').lower()}"
            f" (IC {signed(t.get('oos_ic'), '+.3f')}) — with the claim ledger, the models' facts, the data layer and "
            "the method beside it." if num(t.get("oos_ic")) is not None else
            "Every engine graded one way, with the claim ledger, the models' facts, the data layer and the method.")
    render_section_header("Walk-Forward Edge — Every Engine", desc, icon="trending", accent="accent")
    render_empty_state("Being rebuilt", "This page is moving onto the new design system. The read's own edge is "
                       "on Overview's Inference Chain (the Edge row) meanwhile.", eyebrow="System")
