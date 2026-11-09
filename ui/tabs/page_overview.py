"""
Nishkarsh — Overview: the read, then why (Tattva's Overview grammar).

ANSWER → ANCHOR → SIGNAL → TRUST → STATE → DETAIL, compressed onto one screen: six numbers
across the top answer "what does it read and how far can I trust it"; the Inference Chain
under them answers "why", for the reader who wants it. Both are built from ONE object, the
read verdict (``conclusion["verdict"]``, built once per run by
``convergence.verdict.build_read_verdict``), so the strip, the card and every page's command
bar cannot disagree. This page formats only.
"""

from __future__ import annotations

import streamlit as st

from ui.components import (
    _d, _str, fmt, num, render_empty_state, render_hero_card, render_kpi_strip, render_section_header, signed,
    tone_of,
)

#: Long "how to read" text lives in the cards' tooltips, not on their face.
TOOLTIPS = {
    "read": ("The lean word follows the tier of the Nishkarsh Oscillator: STRONGLY at ±60, plain at ±30, "
             "LEANS beyond ±10; inside ±10 there is no lean. A lean describes what the systems see; it "
             "is not a forecast until the read's walk-forward record validates it."),
    "nko": ("One causal number in −100…+100 (+ = bullish): ten inputs, each standardized against its own "
            "past only, averaged into three pillars (top-down, bottom-up, agreement), then into one. "
            "No weight or sign is fitted to returns. The percentile compares today's strength with every "
            "earlier reading."),
    "edge": ("Rank correlation (IC) of the read with the Nifty's next move, scored on non-overlapping "
             "windows at the read's horizon. 0 = no skill; the SE says how much of the number is "
             "sampling noise."),
    "breadth": ("Share of the Nifty's stocks on the bullish side of their own oscillator cuts minus the "
                "share on the bearish side, in percentage points. Nirnay's polarity decides which side is "
                "bullish (trend: overbought = strength)."),
    "top_down": ("Aarambh's walk-forward forecast of the next sessions, with its extension and lookback "
                 "breadth, standardized against its own past (σ, + = bullish)."),
    "agreement": ("How fully Aarambh and Nirnay agree across direction, breadth, magnitude and regime "
                  "(0–100%). Agreement amplifies the read; disagreement mutes it."),
}


def _title(word: str) -> str:
    """'STRONGLY BULLISH' → 'Strongly Bullish' (the card value; CSS does not case values)."""
    return " ".join(w[:1].upper() + w[1:].lower() for w in _str(word, "—").split())


def kpi_items(v: dict) -> list[dict]:
    """The Overview KPI strip from the verdict — six cards, numbers formatted, nothing computed."""
    k = _d(v.get("kpi"))
    side = _str(v.get("side"))
    tier = _str(k.get("tier") or v.get("tier"))
    vt = _str(k.get("value_text") or v.get("value_text"), "—")
    if tier in ("STRONG", "MODERATE", "WEAK"):
        read_sub = f"{tier.lower()} tier · {vt}"
    elif tier == "NO LEAN":
        read_sub = f"inside ±10 · {vt}"
    else:
        read_sub = "no reading for the latest date"
    pct = num(k.get("strength_percentile"))
    e = _d(k.get("edge"))
    ic, se = num(e.get("ic")), num(e.get("se"))
    chip = _title(_str(e.get("chip"), "No read"))
    b = _d(k.get("breadth"))
    net = num(b.get("net_pts"))
    bull, bear = num(b.get("bull_pct")), num(b.get("bear_pct"))
    breadth_sub = (f"{bull:.0f}% {_str(b.get('bull_label'))} · {bear:.0f}% {_str(b.get('bear_label'))}"
                   if bull is not None and bear is not None else "no Nirnay breadth for this run")
    td = num(k.get("top_down_z"))
    ar = num(k.get("agreement_ratio"))
    return [
        {"label": "Read", "value": _title(v.get("signal") or "NO READ"), "subtext": read_sub,
         "color_class": "success" if side == "bull" else "danger" if side == "bear" else "neutral",
         "tooltip": TOOLTIPS["read"]},
        {"label": "Nishkarsh Oscillator", "value": vt,
         "subtext": (f"stronger than {pct:.0f}% of past readings" if pct is not None
                     else "no history to compare yet"),
         "color_class": "accent", "tooltip": TOOLTIPS["nko"]},
        {"label": "Walk-Forward Edge", "value": signed(ic, "+.3f"),
         "subtext": f"{chip} · ± {se:.3f}" if se is not None else chip,
         "color_class": "info", "tooltip": TOOLTIPS["edge"]},
        {"label": "Nirnay Net Breadth", "value": f"{signed(net, '+.0f')} pts" if net is not None else "—",
         "subtext": breadth_sub, "color_class": tone_of(net, 10.0), "tooltip": TOOLTIPS["breadth"]},
        {"label": "Top-Down Pillar", "value": f"{signed(td, '+.2f')}σ" if td is not None else "—",
         "subtext": "Aarambh's forecast, standardized", "color_class": tone_of(td, 0.5),
         "tooltip": TOOLTIPS["top_down"]},
        {"label": "Engine Agreement", "value": fmt(ar, ".0%") if ar is not None else "—",
         "subtext": "Aarambh × Nirnay, four dimensions", "color_class": "neutral",
         "tooltip": TOOLTIPS["agreement"]},
    ]


def read_verdict() -> dict:
    """The run's verdict (built in the convergence layer). An older cached bundle without one is
    re-worded from its own payload by the same pure builder — never re-estimated."""
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    v = _d(con.get("verdict"))
    if v:
        return v
    from convergence.verdict import build_read_verdict
    return build_read_verdict(con, _d(ss.get("precedent")).get("hero"))


def render_overview_page() -> None:
    v = read_verdict()
    if not _d(st.session_state.get("conclusion")):
        render_empty_state("No read for this run", "The run produced no conclusion to read.", eyebrow="Overview")
        return
    render_kpi_strip(kpi_items(v), max_cols=6)
    render_section_header(
        "Inference Chain",
        "One reading from three pillars — Aarambh's top-down forecast, Nirnay's bottom-up breadth and "
        "their agreement — each standardized against its own past. The read is their average; its "
        "walk-forward edge says how far to trust it, and the weakest link is named.",
        icon="target", accent="accent",
    )
    render_hero_card(v)
