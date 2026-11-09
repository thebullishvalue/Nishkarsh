"""
Nishkarsh — Precedent: what followed reads like today's (≈ Tattva's Precedent page).

Past dates whose Nishkarsh Oscillator state — the read, its three pillars and its 5- and
20-day change — most resembled today's, one per episode, and what the Nifty did next. It is
NOT independent of the read: the state it matches on is the read's own, so "the last times
the state looked like this" means "the last times the read stood here". That is what makes
it useful — it asks what this exact reading has historically preceded — and why it is one
row of the Inference Chain rather than a second opinion.

Reading order — the house convention every analysis page follows:

  1 ANSWER    what followed reads like today's?    Base rate per horizon + what it is
  2 ANCHOR    where did they happen?               The Nifty with every match marked
  3 TRUST     has the matcher been worth anything? Analog skill — not yet graded (L-10)
  4 DETAIL    the evidence behind it               The analog cards themselves

Renders ``session_state["precedent"]`` (computed ONCE per run by
``pipeline.run.compute_precedent``); this page recomputes nothing. A base rate is coloured
only when its exact binomial p (against the usual up-share of the same candidate pool) is
below EDGE_ALPHA (spec Override 3); otherwise it is description, in neutral. Realized
outcomes (the shading, the forward tiles) are coloured by their own sign — facts, not leans.
Unwindowed page: every panel is drawn on the full history.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from convergence.verdict import lean_word
from core.config import EDGE_ALPHA
from ui.components import (
    PILLARS, _d, _date, _list, _str, esc, index_name, nko_side, num, plural, render_chart_panel, render_chip,
    render_empty_state, render_interpretation_card, render_kpi_strip, render_section_header, signed, z_side,
)
from ui.copy import GLOSSARY, NKO_PILLAR_PLAIN
from ui.palette import UI_CHART_HEIGHT_MEDIUM
from ui.theme import chart_color, chart_layout, chart_rgba, panel_bg, style_axes

#: Analog cards rendered (the matcher returns at most this many; the base rate uses all of them).
MAX_ANALOG_CARDS = 10
_DEFAULT_HOLDS = (3, 5, 10, 20)
_GLOSS = dict(GLOSSARY)

# ── tooltips (long "how to read" text lives here, not on the face) ───────────
TOOLTIPS = {
    "base_rate": ("Median Nifty return over the next {h} sessions after the past matches, and how many of them "
                  "rose — against how often the Nifty rose over ANY {h}-session window in the same candidate pool "
                  "(\"usually\"). p-value: " + _GLOSS["p-value"] + " Here it is an exact binomial test of the up-count "
                  "against the usual share; only below " + f"{EDGE_ALPHA:g}" + " is the card coloured."),
}


def _title(word: str) -> str:
    return " ".join(w[:1].upper() + w[1:].lower() for w in _str(word, "—").split())


def _fwd(a: dict, h: int) -> float | None:
    f = _d(a.get("fwd"))
    return num(f.get(int(h), f.get(str(h))))


def _summary(prec) -> dict[int, dict]:
    out = {}
    for h, v in _d(_d(prec).get("summary")).items():
        if num(h) is not None and isinstance(v, dict):
            out[int(num(h))] = v
    return dict(sorted(out.items()))


def _analogs(prec) -> list[dict]:
    return [a for a in _list(_d(prec).get("analogs")) if isinstance(a, dict)]


def _holds(summary: dict, analogs: list[dict]) -> tuple[int, ...]:
    hs = tuple(summary) or tuple(sorted({int(num(h)) for a in analogs for h in _d(a.get("fwd")) if num(h) is not None}))
    if hs:
        return hs
    cfg = _d(_d(st.session_state.get("run_report")).get("config"))
    return tuple(int(num(h)) for h in _list(cfg.get("precedent_horizons")) if num(h) is not None) or _DEFAULT_HOLDS


def base_rate_items(summary: dict) -> list[dict]:
    """One metric card per horizon: median return, k of n rose vs usually, p. Neutral unless the
    payload's own binomial p is below EDGE_ALPHA; then the tone is the side the test found
    (rose more often than usual → success, less → danger). Formats only."""
    items = []
    for h, s in sorted(_d(summary).items(), key=lambda kv: num(kv[0]) or 0):
        s = _d(s)
        med, pos, n, u, p = (num(s.get(x)) for x in ("median", "positive_pct", "n", "unconditional_positive_pct",
                                                    "binomial_p"))
        k = int(round(pos / 100.0 * n)) if pos is not None and n else None
        rose = f"{k} of {int(n)} rose" if k is not None else "— rose"
        sub = " · ".join((rose, f"usually {u:.0f}%" if u is not None else "usually —",
                          f"p {p:.2f}" if p is not None else "p —"))
        sig = p is not None and p < EDGE_ALPHA and pos is not None and u is not None
        tone = ("success" if pos > u else "danger" if pos < u else "neutral") if sig else "neutral"
        items.append({"label": f"+{int(num(h))}D Median Return",
                      "value": f"{signed(med, '+.1f')}%" if med is not None else "—",
                      "subtext": sub, "color_class": tone,
                      "tooltip": TOOLTIPS["base_rate"].format(h=int(num(h)))})
    return items


# ── 2 · the chart ─────────────────────────────────────────────────────────────
def precedent_figure(price: pd.Series, index_label: str, matches: list[tuple], horizon: int,
                     today=None, height: int = UI_CHART_HEIGHT_MEDIUM) -> go.Figure:
    """The index with each past match marked where it happened, and the ``horizon`` sessions
    after it shaded by what the index did (emerald rose / rose fell — the match's own recorded
    outcome, a fact, not a lean). ``matches`` = [(date, label, fwd_pct or None)]; ``today`` = a
    date. Drawn on the design system: painted canvas, the app's axis grammar, colours resolved
    per render, hover strings pre-formatted (x-unified ignores d3 formats)."""
    price = pd.to_numeric(price, errors="coerce")
    price = price[~price.index.duplicated(keep="last")].sort_index().dropna()
    idx = pd.DatetimeIndex(price.index)
    fig = go.Figure()
    for d, _, fwd in matches:
        if fwd is None:
            continue
        i = idx.searchsorted(pd.Timestamp(d))
        if i >= len(idx):
            continue
        j = min(i + int(horizon), len(idx) - 1)
        fig.add_vrect(x0=idx[i], x1=idx[j], line_width=0, layer="below",
                      fillcolor=chart_rgba("emerald" if fwd > 0 else "rose", 0.12))
    fig.add_trace(go.Scatter(
        x=idx, y=price.values, mode="lines", name=index_label, showlegend=False,
        line=dict(color=chart_color("accent"), width=1.5),
        customdata=[[f"{v:,.0f}"] for v in price.values],
        hovertemplate="%{fullData.name}: %{customdata[0]}<extra></extra>"))
    pts = []
    for d, lab, fwd in matches:
        i = idx.searchsorted(pd.Timestamp(d))
        if i < len(idx):
            out = f"next {int(horizon)} sessions {signed(fwd, '+.1f')}%" if fwd is not None else "outcome —"
            pts.append((idx[i], float(price.iloc[i]), str(lab), out))
    if pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers+text", name="Match",
            text=[p[2] for p in pts], textposition="top center", showlegend=False,
            textfont=dict(size=9, family="JetBrains Mono, monospace", color=chart_color("slate")),
            marker=dict(size=7, color=chart_color("slate"), line=dict(color=panel_bg(), width=1)),
            customdata=[[p[3]] for p in pts],
            hovertemplate="Match %{text} · %{customdata[0]}<extra></extra>"))
    if today is not None and len(idx):
        k = min(idx.searchsorted(pd.Timestamp(today)), len(idx) - 1)
        fig.add_trace(go.Scatter(
            x=[idx[k]], y=[float(price.iloc[k])], mode="markers", name="Today", showlegend=False,
            marker=dict(size=10, color=chart_color("accent"), symbol="diamond",
                        line=dict(color=panel_bg(), width=1)),
            customdata=[[f"{float(price.iloc[k]):,.0f}"]],
            hovertemplate="Today: %{customdata[0]}<extra></extra>"))
    fig.update_layout(**chart_layout(height=height, show_legend=False))
    style_axes(fig, y_title=index_label)
    fig.update_yaxes(tickformat=",.0f")
    return fig


def _render_where(analogs: list[dict], horizon: int, label: str) -> None:
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    k = _d(con.get("nko"))
    frame = ss.get("conclusion_frame")
    levels = ss.get("label_levels")
    first = None
    if isinstance(frame, pd.DataFrame) and "NKO" in frame.columns:
        first = pd.to_numeric(frame["NKO"], errors="coerce").first_valid_index()
    price = levels if isinstance(levels, pd.Series) else None
    if price is not None and first is not None:
        price = price[price.index >= first]
    if price is None or not len(pd.to_numeric(price, errors="coerce").dropna()):
        render_empty_state("No index history to draw on",
                           f"This run has no {esc(label)} level series, so the matches cannot be placed on it.",
                           eyebrow="Precedent")
        return
    matches = []
    for i, a in enumerate(analogs[:MAX_ANALOG_CARDS], start=1):
        d = _date(a.get("date"))
        if d:
            matches.append((pd.Timestamp(d), str(i), _fwd(a, horizon)))
    as_of = _date(k.get("as_of"))
    fig = precedent_figure(price, label, matches, horizon, today=pd.Timestamp(as_of) if as_of else None)
    render_chart_panel(fig, "precedent-where", units=f"index level · next {int(horizon)} sessions shaded",
                       windowed=False)


# ── 4 · the analog cards ──────────────────────────────────────────────────────
_TIER_CLASS = {("bull", "STRONG"): "tier-strong-buy", ("bear", "STRONG"): "tier-caution"}


def _fwd_tile(h: int, v: float | None) -> str:
    cls = "neutral" if v is None else "pos" if round(v, 1) > 0 else "neg" if round(v, 1) < 0 else "flat"
    val = f"{signed(v, '+.1f')}%" if v is not None else "—"
    return (f'<div class="analog-fwd-tile {cls}"><span class="analog-fwd-label">+{int(h)}D</span>'
            f'<span class="analog-fwd-value">{esc(val)}</span></div>')


def _stat(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="analog-stat"><span class="analog-stat-label">{esc(label)}</span>'
            f'<span class="analog-stat-value{" " + cls if cls else ""}">{esc(value)}</span></div>')


def analog_card_html(rank: int, a: dict, holds: tuple[int, ...], label: str) -> str:
    """Tattva's ``.analog-card`` markup with Nishkarsh's stats. Every string escaped; the HTML is
    flush-left with no blank lines (CommonMark closes an HTML block on a blank line)."""
    word, side, tier = lean_word(a.get("nko"))
    tier_cls = _TIER_CLASS.get((side, tier)) or {"bull": "tier-buy", "bear": "tier-caution"}.get(side, "tier-hold")
    tone = {"bull": "success", "bear": "danger"}.get(side, "neutral")
    chip = render_chip(_title(word), tone, as_html=True) or ""
    if side in ("bull", "bear") and tier != "STRONG":
        chip = chip.replace('class="chip ', 'class="chip chip-outline ', 1)
    sim = num(a.get("similarity"))
    pil = _d(a.get("pillars"))
    v = num(a.get("nko"))
    side_cls = {"bull": "pos", "bear": "neg"}.get(nko_side(v), "")
    stats = [_stat("Similarity", f"{sim * 100:.1f}%" if sim is not None else "—", "accent"),
             _stat("Read Then", signed(v, "+.0f"), side_cls)]
    for p in PILLARS:
        z = num(pil.get(p))
        stats.append(_stat(NKO_PILLAR_PLAIN[p][0], f"{signed(z, '+.2f')}σ" if z is not None else "—",
                           {"bull": "pos", "bear": "neg"}.get(z_side(z, 2), "")))
    tiles = "".join(_fwd_tile(h, _fwd(a, h)) for h in holds)
    return (
        f'<div class="analog-card {tier_cls}">\n'
        f'<div class="analog-card-head">\n'
        f'<div class="analog-card-id">\n'
        f'<div class="analog-eyebrow">Analog {int(rank)} · Historical Match</div>\n'
        f'<div class="analog-symbol">{esc(_date(a.get("date")) or "—")}</div>\n'
        f'</div>\n'
        f'{chip}\n'
        f'</div>\n'
        f'<div class="analog-stat-row">{"".join(stats)}</div>\n'
        f'<div class="analog-fwd-block">\n'
        f'<div class="analog-fwd-block-label">Forward {esc(label)} Return</div>\n'
        f'<div class="analog-fwd-grid" style="grid-template-columns:repeat({max(1, len(holds))},1fr);">{tiles}</div>\n'
        f'</div>\n'
        f'</div>'
    )


# ── the page ──────────────────────────────────────────────────────────────────
def _hs_text(holds: tuple[int, ...]) -> str:
    hs = [str(int(h)) for h in holds]
    return (", ".join(hs[:-1]) + " and " + hs[-1]) if len(hs) > 1 else (hs[0] if hs else "—")


def render_precedent_page() -> None:
    ss = st.session_state
    con = _d(ss.get("conclusion"))
    prec = _d(ss.get("precedent"))
    v = _d(con.get("verdict"))
    label = index_name(con)
    summary, analogs = _summary(prec), _analogs(prec)
    holds = _holds(summary, analogs)
    horizon = int(num(_d(prec.get("hero")).get("horizon")) or num(con.get("horizon")) or 10)
    if horizon not in holds and holds:
        horizon = max(holds)

    method = (f"Past dates whose read, its three pillars and its 5- and 20-day change most resembled today's — one "
              f"match per episode — and what the {label} did over the next {_hs_text(holds)} sessions.")
    line = _str(_d(v.get("precedent")).get("text"))
    render_section_header("Precedent — What Followed Reads Like Today's",
                          f"{method} {line}" if analogs and line else method, icon="compass", accent="accent")

    if not analogs or not summary:
        render_empty_state(
            "Not enough history for a base rate",
            "Past matches need dates when both systems ran and whose longest outcome is already known; this run "
            "has too few of them (or no Nirnay stock data), so there is nothing to compare today against.",
            eyebrow="Precedent")
        return

    items = base_rate_items(summary)
    render_kpi_strip(items, max_cols=max(1, min(6, len(items))), key="precedent-base")
    sig = [f"+{h}D" for h, s in summary.items()
           if num(_d(s).get("binomial_p")) is not None and num(_d(s).get("binomial_p")) < EDGE_ALPHA]
    verdict_line = (f"Today {'only ' if len(sig) < len(summary) else ''}{', '.join(sig)} "
                    f"{'clears' if len(sig) == 1 else 'clear'} that bar." if sig else
                    "Today no horizon clears it, so every card above is description, in neutral.")
    render_interpretation_card(
        "A Base Rate, Conditioned on the Read's Own State",
        f"What the {esc(label)} actually did after the most similar past reads — no model is fitted to returns to "
        "produce it. The state is the read's own (its level, its three pillars and its 5- and 20-day change), so "
        "this asks what <strong>this</strong> reading has historically preceded; it is one row of the Inference "
        "Chain, read with the read rather than against it. Every match is a distinct episode (a Theiler window "
        "keeps adjacent days of one episode out), and “usually” is how often the "
        f"{esc(label)} rose over any window of the same length in the same history. A base rate counts as evidence "
        f"only when its binomial p is below {EDGE_ALPHA:g}. {esc(verdict_line)}",
        color="info")

    render_section_header(
        "Where They Happened",
        f"The {label} since the read began, each match numbered by closeness; the shading is the next {horizon} "
        f"sessions after it — green where the {label} rose, red where it fell. The diamond is today.",
        icon="trending", accent="accent")
    _render_where(analogs, horizon, label)

    render_section_header(
        "Analog Skill",
        "Has the matcher itself anticipated the Nifty, horizon by horizon? That is not measured yet — so the base "
        "rates above describe what followed, they do not forecast what follows.",
        icon="bar-chart", accent="cyan")
    render_empty_state("Not yet graded",
                       "The matcher's own walk-forward skill has not been measured (Edge Ledger L-10); read the base "
                       "rate as description.", eyebrow="Precedent")

    shown = analogs[:MAX_ANALOG_CARDS]
    render_section_header(
        "Top Analog Periods",
        f"The {plural(len(shown), 'distinct episode')} behind the base rate above, closest first — the read as it "
        f"stood then, and what the {label} did next. Numbers match the chart.",
        icon="layers")
    cols = st.columns(2, gap="small")
    for i, a in enumerate(shown):
        with cols[i % 2]:
            st.markdown(analog_card_html(i + 1, a, holds, label), unsafe_allow_html=True)
