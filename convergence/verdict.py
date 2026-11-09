"""
The read verdict — what the Overview hero card, the Overview KPI strip and every page's
command bar print about the Nishkarsh read. Built ONCE per run (pipeline stage 5) and
stored in the conclusion payload as ``conclusion["verdict"]``; the UI only formats it.

Anatomy (Tattva's decision-card grammar, Nishkarsh content):

  signal       the lean word — the word follows the tier: STRONGLY BULLISH (|NKO| ≥ 60),
               BULLISH (≥ 30), LEANS BULLISH (> 10), NO LEAN (inside ±10), NO READ.
  trust        the read's own walk-forward edge (``edge_tier`` of its IC) — the chip.
  headline     ONE sentence on why the read is what it is (its pillars, largest input).
  limit        the binding constraint, always named. Until the read is VALIDATED by its
               pre-registered rule, that is its track record.
  gates        TOP-DOWN · BOTTOM-UP · AGREEMENT · EDGE · PRECEDENT, each
               {name, title, value (bar width 0..1), tone, label, detail, support}.
  risk         divergence events in the window; the single-system note.

Honesty (C-UI rules, enforced by tests/test_hero_verdict.py): no buy/sell/signal/call
wording and no directional ``call`` unless ``nko.status == "VALIDATED"``; the track record is
always named; a single-system date says "top-down only"; the precedent votes (and is
coloured) only when its binomial p < EDGE_ALPHA; levels mode carries the falsified-as-a-
forecaster tag; SYNTHETIC runs carry the banner; no "nan"/"None" text — "—" instead.
Pure: never raises, reads only the dicts it is given.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

from core.config import (
    AARAMBH_FORWARD_SIGNAL,
    AARAMBH_FWD_HORIZON,
    CONV2_TIER_MODERATE,
    CONV2_TIER_STRONG,
    CONV2_TIER_WEAK,
    EDGE_ALPHA,
    NIRNAY_POLARITY,
    NIRNAY_SIGN,
)

__all__ = ["build_read_verdict", "edge_tier", "explain_read", "lean_word", "precedent_line", "track_record",
           "verdict_texts", "VERDICT_KEYS", "PILLARS"]

SCHEMA = "nishkarsh.verdict/1"
PILLARS = ("top_down", "bottom_up", "agreement")
_TREND = NIRNAY_POLARITY == "trend"

# ── wording (single-sourced here; ui.copy re-exports what the pages need) ────
LEVELS_TAG = "valuation context — falsified as a forecaster"
SYNTH_BANNER = "SYNTHETIC DEMO DATA — not market data; no conclusion here is about the real market."
NKO_TIER_WORD = {"STRONG": "strongly", "MODERATE": "moderately", "WEAK": "mildly"}
# component id → (plain label, what it reads). + = bullish for every component.
NKO_COMPONENT_PLAIN = {
    "forecast": ("Aarambh's forecast", "the forecast of the next trading days, standardized"),
    "extension": ("Forecast extension", "how stretched the forecast is against its recent range"),
    "zone_breadth_a": ("Lookback zone breadth", "lookbacks in the bullish zone minus the bearish zone"),
    "crossings_a": ("Lookback cut crossings", "fresh bullish minus bearish crossings across lookbacks"),
    "oscillator_n": ("Stock oscillator", "the stocks' average oscillator"),
    "zone_breadth_n": ("Stock zone breadth", "% of stocks overbought minus % oversold" if _TREND
                       else "% of stocks oversold minus % overbought"),
    "regime_mix": ("Stock regimes", "% of stocks in a bull regime minus a bear regime" if _TREND
                   else "% of stocks in a bear regime minus a bull regime"),
    "hmm_state": ("Bull vs bear odds", "the stocks' average bull minus bear state probability" if _TREND
                  else "the stocks' average bear minus bull state probability"),
    "cut_crossings_n": ("Stock cut crossings", "fresh overbought-side minus oversold-side crossings per stock"
                        if _TREND else "fresh oversold-side minus overbought-side crossings per stock"),
    "signed_agreement": ("Signed agreement", "both systems' direction times how fully they agree"),
}
# (pillar, component id) in the oscillator's own order (convergence.nishkarsh.COMPONENTS).
_PILLAR_INPUTS = {
    "top_down": ("forecast", "extension", "zone_breadth_a", "crossings_a"),
    "bottom_up": ("oscillator_n", "zone_breadth_n", "regime_mix", "hmm_state", "cut_crossings_n"),
    "agreement": ("signed_agreement",),
}
PILLAR_PHRASE = {
    "top_down": ("Aarambh's top-down view leaning up", "Aarambh's top-down view leaning down"),
    "bottom_up": ("broad strength in the Nifty's stocks", "broad weakness in the Nifty's stocks"),
    "agreement": ("both systems leaning bullish together", "both systems leaning bearish together"),
}
PILLAR_SUBJECT = {"top_down": "Aarambh's top-down view", "bottom_up": "the stocks", "agreement": "their agreement"}
GATE_TITLES = {"top_down": "Top-down", "bottom_up": "Bottom-up", "agreement": "Agreement", "edge": "Edge",
               "precedent": "Precedent"}
NKO_STATUSES = ("VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA")
STATUS_LABEL = {"VALIDATED": "Validated", "UNPROVEN": "Unproven", "INSUFFICIENT_DATA": "Not enough data"}
TRACK_WORD = {"VALIDATED": "validated", "UNPROVEN": "unproven", "INSUFFICIENT_DATA": "not enough data"}
FOOT = "The read is the average of the three pillars above; the edge decides how far to trust it."
_NO_CLAIM_TAIL = "it describes what the systems see, not where the Nifty goes."
_Z_LEAN = 0.25            # |pillar z| below this reads "no lean" on a gate row (display band only)

VERDICT_KEYS = ("schema", "eyebrow", "signal", "signal_class", "side", "value", "value_text", "tier", "direction",
                "strength", "strength_sub", "headline", "limit", "binding", "gates", "risk", "foot", "trust",
                "status", "status_label", "validated", "call", "action", "track", "precedent", "kpi",
                "two_system", "single_system", "as_of", "as_of_note", "horizon", "forward_mode", "levels_note",
                "synthetic", "banner")


# ── small pure helpers ───────────────────────────────────────────────────────
def _num(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else []


def _str(x, default: str = "") -> str:
    if x is None or isinstance(x, (float, int)) and _num(x) is None:
        return default
    s = str(x).strip()
    return s if s and s.lower() not in ("nan", "none", "inf", "-inf", "+inf") else default


def _date(x) -> str:
    if hasattr(x, "strftime"):
        try:
            return x.strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            return ""
    s = x.strip() if isinstance(x, str) else ""
    return s[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", s) else ""


def pretty_date(x) -> str:
    """'2026-11-09' → '9 Nov 2026'; anything unparseable → '—'."""
    d = _date(x)
    if not d:
        return "—"
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        return "—"
    return f"{dt.day} {dt:%b %Y}"


def _fmt(x, spec: str = "+.2f", dash: str = "—", suffix: str = "") -> str:
    v = _num(x)
    if v is None:
        return dash
    try:
        return format(v, spec) + suffix
    except (TypeError, ValueError):
        return dash


def signed(x, spec: str = "+.2f") -> str:
    """Typographic minus (−); a value that rounds to zero carries no sign; missing → '—'."""
    v, plain = _num(x), spec.replace("+", "")
    try:
        if v is not None and float(_fmt(v, plain).replace(",", "")) == 0:
            return _fmt(0.0, plain)
    except ValueError:
        pass
    return _fmt(x, spec).replace("-", "−")


def plural(n, word: str) -> str:
    v = _num(n)
    if v is None:
        return f"— {word}s"
    k = int(round(v))
    return f"{k:,} {word}" + ("" if k == 1 else "s")


def _gate(value, lo: float, hi: float) -> float:
    """Map a reading onto [0.02, 1] (Tattva's gate scale); missing → 0.5 (uninformative)."""
    v = _num(value)
    if v is None:
        return 0.5
    if hi == lo:
        return 1.0
    return float(min(1.0, max(0.02, (v - lo) / (hi - lo))))


# ── the shared scales ────────────────────────────────────────────────────────
def edge_tier(ic) -> tuple[str, str]:
    """``(tier, chip)`` for a walk-forward IC — the one scale every page uses (Tattva):
    NO READ · NO EDGE (≤ 0) · MARGINAL (> 0) · MODEST EDGE (≥ 0.05) · SOLID EDGE (≥ 0.10)."""
    v = _num(ic)
    if v is None:
        return "uncalibrated", "NO READ"
    if v >= 0.10:
        return "solid", "SOLID EDGE"
    if v >= 0.05:
        return "modest", "MODEST EDGE"
    if v > 0:
        return "marginal", "MARGINAL"
    return "no_edge", "NO EDGE"


def nko_tier_of(v) -> str:
    v = _num(v)
    if v is None:
        return "NO READ"
    a = abs(v)
    return ("STRONG" if a >= CONV2_TIER_STRONG else "MODERATE" if a >= CONV2_TIER_MODERATE
            else "WEAK" if a > CONV2_TIER_WEAK else "NO LEAN")


def lean_word(value) -> tuple[str, str, str]:
    """``(word, side, tier)`` — the word follows the tier. side ∈ bull / bear / flat / none."""
    v = _num(value)
    if v is not None and not -100.0 <= v <= 100.0:
        v = None
    tier = nko_tier_of(v)
    if v is None:
        return "NO READ", "none", tier
    if tier == "NO LEAN":
        return "NO LEAN", "flat", tier
    d = "BULLISH" if v > 0 else "BEARISH"
    prefix = {"STRONG": "STRONGLY ", "MODERATE": "", "WEAK": "LEANS "}[tier]
    return prefix + d, ("bull" if v > 0 else "bear"), tier


def _component_label(cid, fallback="") -> str:
    return NKO_COMPONENT_PLAIN.get(_str(cid), (_str(fallback) or "An input", ""))[0]


def _mid(label: str) -> str:
    return label if (label[:2].isupper() or label.startswith("Aarambh")) else label[:1].lower() + label[1:]


def _strength_word(z: float) -> str:
    a = abs(z)
    return "strongly" if a >= 2 else "clearly" if a >= 1 else "mildly" if a >= 0.5 else "slightly"


def _lead_input(k: dict, pillar: str) -> str:
    comps = [_d(c) for c in _list(k.get("components")) if _d(c).get("pillar") == pillar]
    if len(comps) < 2:
        return ""
    zs = [(c, _num(c.get("z"))) for c in comps]
    zs = [(c, z) for c, z in zs if z is not None]
    if not zs:
        return ""
    c, z = max(zs, key=lambda t: abs(t[1]))
    return f", led by {_mid(_component_label(c.get('id'), c.get('label')))} ({signed(z, '+.1f')})"


def explain_read(k) -> str:
    """ONE sentence on why today's read is what it is, from its pillars (and the largest input)."""
    k = _d(k)
    v = _num(k.get("value"))
    if v is None or not -100.0 <= v <= 100.0:
        return "No reading for the latest date."
    pil = _d(k.get("pillars"))
    z = {p: _num(pil.get(p)) for p in PILLARS}
    two = k.get("two_system") is True and z["bottom_up"] is not None
    if not two:
        td = z["top_down"]
        if td is None:
            return "Top-down only today, and Aarambh has no reading for this date."
        return (f"Top-down only today — Nirnay has no stock data for this date — so the read is Aarambh's "
                f"alone, its forecast leaning {'up' if td >= 0 else 'down'} (top-down {signed(td, '+.1f')})"
                f"{_lead_input(k, 'top_down')}.")
    parts = {p: f"{('top-down', 'bottom-up', 'agreement')[i]} {signed(z[p], '+.1f')}" for i, p in enumerate(PILLARS)
             if z[p] is not None}
    if nko_tier_of(v) == "NO LEAN":
        return "The pillars roughly cancel out — " + ", ".join(parts.values()) + "."
    have = [p for p in PILLARS if z[p] is not None]
    lead = max(have, key=lambda p: abs(z[p]))
    zl = z[lead]
    head = f"Driven by {PILLAR_PHRASE[lead][0 if zl >= 0 else 1]} ({parts[lead]}){_lead_input(k, lead)}"
    if lead == "agreement":
        rest = [parts[p] for p in ("top_down", "bottom_up") if p in parts]
        return head + ("; " + ", ".join(rest) if rest else "") + "."
    other = "bottom_up" if lead == "top_down" else "top_down"
    zo = z[other]
    if zo is None:
        return head + "."
    plural_subject = other == "bottom_up"
    agree = (zo >= 0) == (zl >= 0)
    verb = ("agree" if plural_subject else "agrees") if agree else ("disagree" if plural_subject else "disagrees")
    return head + f"; {PILLAR_SUBJECT[other]} {verb} {_strength_word(zo)} ({signed(zo, '+.1f')})."


def _streak_text(k: dict) -> str:
    st_ = _d(k.get("streak"))
    n = _num(st_.get("sessions"))
    side = _str(st_.get("side")).upper()
    if n is None or n < 1 or side not in ("BULLISH", "BEARISH", "NO LEAN"):
        return ""
    n_i = int(n)
    if side == "NO LEAN":
        return "no lean today" if n_i == 1 else f"no lean for {plural(n_i, 'session')}"
    word = side.lower()
    return f"turned {word} today" if n_i == 1 else f"{word} for {plural(n_i, 'session')}"


def track_record(k, h: int) -> dict:
    """The read's track record in one sentence (word, status, text). Wording unlocks only when VALIDATED."""
    k = _d(k)
    status = _str(k.get("status")).upper()
    status = status if status in NKO_STATUSES else "INSUFFICIENT_DATA"
    ev = _d(k.get("evidence"))
    n, ic, p = _num(ev.get("n")), _num(ev.get("ic")), _num(ev.get("p"))
    periods = f"{int(n):,} past {h}-day periods" if n else f"past {h}-day periods"
    if status == "VALIDATED":
        text = (f"on {periods} this read has lined up with the Nifty's next move (accuracy {signed(ic, '+.3f')}, "
                f"p {_fmt(p, '.2f')}), so its lean is a measured directional call.")
    elif status == "UNPROVEN":
        text = (f"on {periods} this read has not yet shown it anticipates the Nifty (accuracy {signed(ic, '+.3f')}). "
                "It describes what the systems see, not where the market goes.")
    else:
        text = "there is not yet enough history with both systems running to measure this read."
    return {"status": status, "word": TRACK_WORD[status], "text": text}


def precedent_line(ph, read_dir: int = 0) -> dict:
    """What followed reads like today's (``precedent["hero"]``). It VOTES only when its exact
    binomial p < EDGE_ALPHA; otherwise it is description."""
    ph = _d(ph)
    n, pos = _num(ph.get("n")), _num(ph.get("positive_pct"))
    if not n or n < 1 or pos is None:
        return {"state": "none", "votes": False, "n": 0, "k": 0, "p": None,
                "text": "No past moments like today were found for this run."}
    n_i = int(round(n))
    k = int(round(pos / 100.0 * n_i))
    u, p, med, hp = (_num(ph.get(x)) for x in ("unconditional_positive_pct", "binomial_p", "median", "horizon"))
    hz = f"{int(hp)} sessions" if hp is not None and hp >= 1 else "the horizon"
    text = (f"In {n_i} past moments like today, the Nifty rose {k} time{'s' if k != 1 else ''} over the next {hz} "
            f"(usually {_fmt(u, '.0f', suffix='%')}) — median {signed(med, '+.1f')}%")
    out = {"n": n_i, "k": k, "p": p, "usual_pct": u, "median": med, "horizon": int(hp) if hp else None}
    if not (p is not None and p < EDGE_ALPHA and u is not None):
        return out | {"state": "info", "votes": False, "text": text + f"; not significant (p {_fmt(p, '.2f')})."}
    lean = 1 if pos > u else -1 if pos < u else 0
    word = "a rise" if lean > 0 else "a fall" if lean < 0 else "neither way"
    if read_dir and lean:
        state, rel = (("confirm", ", in line with today's read") if lean == read_dir
                      else ("conflict", ", against today's read"))
    else:
        state, rel = "info", ""
    return out | {"state": state, "votes": True,
                  "text": text + f"; significant (p {_fmt(p, '.2f')}) — these moments lean to {word}{rel}."}


# ── the gates ────────────────────────────────────────────────────────────────
def _pillar_gate(name: str, k: dict, read_dir: int, two: bool, as_of: str, agreement_ratio) -> dict:
    z = _num(_d(k.get("pillars")).get(name))
    comps = {_str(_d(c).get("id")): _num(_d(c).get("z")) for c in _list(k.get("components"))}
    title = GATE_TITLES[name]
    if z is None:
        if name != "top_down" and not two:
            detail = (f"Nirnay has no stock data for {pretty_date(as_of)}; the read is top-down only."
                      if as_of else "Nirnay has no stock data for this date; the read is top-down only.")
            label = "no stock data"
        else:
            detail, label = "No reading for this date.", "no reading"
        return {"name": name, "title": title, "value": 0.02, "tone": "neutral", "label": label,
                "detail": detail, "support": 0.5, "z": None}
    tone = "long" if z >= _Z_LEAN else "short" if z <= -_Z_LEAN else "neutral"
    label = (f"leans bullish {signed(z, '+.1f')}σ" if z >= _Z_LEAN else
             f"leans bearish {signed(z, '+.1f')}σ" if z <= -_Z_LEAN else f"no lean {signed(z, '+.1f')}σ")
    bits = [f"{_mid(_component_label(cid))} {signed(comps.get(cid), '+.1f')}" for cid in _PILLAR_INPUTS[name]]
    if name == "agreement":
        ar = _num(agreement_ratio)
        detail = (f"Signed agreement {signed(comps.get('signed_agreement'), '+.1f')} — both systems' direction "
                  "times how fully they agree" + (f"; they agree {ar:.0%} across direction, breadth, magnitude "
                                                  "and regime." if ar is not None else "."))
    else:
        detail = " · ".join(bits)
        detail = detail[:1].upper() + detail[1:] + " (each against its own past; + = bullish)."
    aligned = z * read_dir if read_dir else abs(z)
    return {"name": name, "title": title, "value": min(1.0, abs(z) / 3.0), "tone": tone, "label": label,
            "detail": detail, "support": _gate(aligned, -1.0, 1.5), "z": z}


def _edge_gate(k: dict, h: int) -> dict:
    edge = _d(k.get("edge"))
    ev = _d(k.get("evidence"))
    ic = _num(edge.get("ic")) if edge else _num(ev.get("ic"))
    tier, chip = edge_tier(ic)
    if ic is None:
        return {"name": "edge", "title": "Edge", "value": 0.25, "tone": "neutral", "label": "not graded",
                "detail": "Not enough two-system history to grade the read walk-forward.", "support": 0.25}
    se = _num(edge.get("se"))
    n = _num(edge.get("n") if edge else ev.get("n"))
    p = _num(ev.get("p"))
    pos, nb = _num(edge.get("positive_blocks")), _num(edge.get("scored_blocks"))
    val = _gate(ic, -0.02, 0.15)
    tone = "short" if val < 0.35 else "caution" if val < 0.75 else "long"
    detail = (f"The read's non-overlapping {h}-day IC is {signed(ic, '+.3f')}"
              + (f" (± {se:.3f})" if se is not None else "")
              + (f" over {int(n):,} periods" if n else "")
              + (f", p {_fmt(p, '.2f')}" if p is not None else "")
              + (f"; {int(pos)} of {int(nb)} years positive." if nb else "."))
    return {"name": "edge", "title": "Edge", "value": val, "tone": tone, "label": chip.lower(),
            "detail": detail, "support": val}


def _precedent_gate(pr: dict, read_dir: int) -> dict:
    n, k = pr.get("n") or 0, pr.get("k") or 0
    if not n:
        return {"name": "precedent", "title": "Precedent", "value": 0.02, "tone": "neutral",
                "label": "no precedent", "detail": pr.get("text", "No past moments like today were found."),
                "support": 0.5}
    share = k / n
    aligned = share if read_dir >= 0 else 1.0 - share
    if pr.get("votes"):
        agrees = pr.get("state") == "confirm"
        label = ("precedent agrees" if agrees else "precedent disagrees" if pr.get("state") == "conflict"
                 else "precedent leans")
        tone = "long" if agrees else "short" if pr.get("state") == "conflict" else "neutral"
        support = 1.0 if agrees else 0.1 if pr.get("state") == "conflict" else 0.5
    else:
        label, tone, support = "descriptive", "neutral", 0.5
    u = pr.get("usual_pct")
    hz = pr.get("horizon")
    detail = (f"{k} of {n} past moments like today rose over the next "
              + (f"{hz} sessions" if hz else "horizon")
              + (f", against {u:.0f}% usually" if _num(u) is not None else "")
              + (f" (median {signed(pr.get('median'), '+.1f')}%; p {_fmt(pr.get('p'), '.2f')})"))
    if not pr.get("votes"):
        detail += f" — not significant at p < {EDGE_ALPHA:g}, so it describes rather than votes."
    else:
        detail += "."
    return {"name": "precedent", "title": "Precedent", "value": aligned, "tone": tone, "label": label,
            "detail": detail, "support": support}


def _pillar_weights(k: dict) -> dict:
    present = [p for p in PILLARS if _num(_d(k.get("pillars")).get(p)) is not None]
    return {p: (1.0 / len(present) if p in present else 0.0) for p in PILLARS} if present else {}


# ── the verdict ──────────────────────────────────────────────────────────────
def build_read_verdict(conclusion, precedent_hero=None, *, synthetic: bool = False, breadth=None,
                       agreement_ratio=None, forward_mode=None) -> dict:
    """The read verdict (see module docstring). ``conclusion`` is the conclusion payload (with
    ``nko``); ``precedent_hero`` is ``precedent["hero"]``; ``breadth`` is Nirnay's latest row
    ``{overbought_pct, oversold_pct, n_live, n_universe, as_of}``; ``agreement_ratio`` is the
    anatomy's latest agreement ratio (0..1). Pure; never raises; no NaN text."""
    try:
        return _build(conclusion, precedent_hero, synthetic, breadth, agreement_ratio, forward_mode)
    except Exception:  # noqa: BLE001 — a verdict must never take the page down
        return _build({}, None, bool(synthetic), None, None, forward_mode)


def _build(conclusion, precedent_hero, synthetic, breadth, agreement_ratio, forward_mode) -> dict:
    con = _d(conclusion)
    k = _d(con.get("nko"))
    fm = forward_mode if isinstance(forward_mode, bool) else con.get("forward_mode")
    forward = bool(fm) if isinstance(fm, bool) else bool(AARAMBH_FORWARD_SIGNAL)
    h_raw = _num(con.get("horizon"))
    h = int(h_raw) if h_raw is not None and h_raw >= 1 else int(AARAMBH_FWD_HORIZON)

    track = track_record(k, h)
    status = track["status"]
    validated = status == "VALIDATED"
    value = _num(k.get("value"))
    if value is not None and not -100.0 <= value <= 100.0:
        value = None
    word, side, tier = lean_word(value)
    direction = "BULLISH" if side == "bull" else "BEARISH" if side == "bear" else "NEUTRAL"
    read_dir = 1 if side == "bull" else -1 if side == "bear" else 0
    two = k.get("two_system") is True
    v_txt = signed(value, "+.0f")
    as_of = _date(k.get("as_of")) or _date(con.get("as_of"))
    run_as_of = _date(con.get("as_of"))

    # ── gates ────────────────────────────────────────────────────────────
    pr = precedent_line(precedent_hero, read_dir)
    gates = [_pillar_gate(p, k, read_dir, two, as_of, agreement_ratio) for p in PILLARS]
    gates.append(_edge_gate(k, h))
    gates.append(_precedent_gate(pr, read_dir))

    # ── the binding constraint ───────────────────────────────────────────
    ev = _d(k.get("evidence"))
    edge = _d(k.get("edge"))
    ic = _num(edge.get("ic")) if edge else _num(ev.get("ic"))
    if value is None:
        limit, binding = "No read for the latest date — nothing to grade.", None
    elif not validated:
        binding = "edge"
        n, p = _num(ev.get("n")), _num(ev.get("p"))
        if status == "INSUFFICIENT_DATA" or ic is None:
            limit = ("Limited by its track record: there is not yet enough two-system history to measure the "
                     f"read — {_NO_CLAIM_TAIL}")
        else:
            limit = (f"Limited by its track record: walk-forward IC {signed(ic, '+.3f')}"
                     + (f" over {int(n):,} periods" if n else "")
                     + (f" (p {_fmt(p, '.2f')})" if p is not None else "")
                     + f" — {_NO_CLAIM_TAIL}")
    else:
        weakest = min(gates, key=lambda g: g["support"])
        if weakest["support"] >= 0.75:
            limit, binding = "Nothing is materially limiting this read — every row supports it.", None
        else:
            limit, binding = f"Limited by {weakest['title'].lower()}: {weakest['detail']}", weakest["name"]

    # ── risk flags ───────────────────────────────────────────────────────
    risk = []
    dr = _d(con.get("divergence_recent"))
    win = _num(dr.get("window_days"))
    n_div = sum(int(_num(v) or 0) for kk, v in dr.items() if kk != "window_days")
    if n_div > 0:
        risk.append(f"{n_div} Aarambh/Nirnay divergence event{'s' if n_div != 1 else ''}"
                    + (f" in the last {int(win)} sessions." if win else "."))
    if value is not None and not two:
        risk.append(f"Top-down only on {pretty_date(as_of)}: Nirnay has no stock data, so the bottom-up and "
                    "agreement pillars are absent." if as_of else
                    "Top-down only: Nirnay has no stock data, so the bottom-up and agreement pillars are absent.")

    # ── the right-hand block ─────────────────────────────────────────────
    tier_word = {"STRONG": "STRONG", "MODERATE": "MODERATE", "WEAK": "WEAK", "NO LEAN": "NO LEAN",
                 "NO READ": "NO READ"}[tier]
    strength = f"{tier_word} · {v_txt}" if value is not None else "NO READ"
    pct = _num(k.get("strength_percentile"))
    sub_bits = []
    if value is not None and pct is not None:
        sub_bits.append(f"stronger than {pct:.0f}% of past readings")
    streak = _streak_text(k) if value is not None else ""
    if streak:
        sub_bits.append(streak)
    strength_sub = " · ".join(sub_bits)

    tier_name, chip = edge_tier(ic)
    trust = {"tier": tier_name, "chip": chip, "oos_ic": ic, "se": _num(edge.get("se")),
             "n": int(_num(edge.get("n") if edge else ev.get("n")) or 0), "p": _num(ev.get("p")),
             "positive_blocks": int(_num(edge.get("positive_blocks")) or 0),
             "n_blocks": int(_num(edge.get("scored_blocks")) or 0)}

    call = bool(validated and read_dir != 0)
    action = None
    if call:   # unreachable on current evidence: kept so the branch exists when the rule passes
        action = {"word": ("BUY" if read_dir > 0 else "SELL"),
                  "prose": "a validated directional read — sized by its tier"}

    # ── KPI inputs (numbers only; the page formats them) ─────────────────
    b = _d(breadth)
    ob, os_ = _num(b.get("overbought_pct")), _num(b.get("oversold_pct"))
    net = (NIRNAY_SIGN * (ob - os_)) if ob is not None and os_ is not None else None
    bull_lab, bear_lab = ("overbought", "oversold") if NIRNAY_SIGN > 0 else ("oversold", "overbought")
    kpi = {
        "read_word": word, "tier": tier, "value": value, "value_text": v_txt,
        "strength_percentile": pct,
        "edge": {"ic": ic, "se": trust["se"], "tier": tier_name, "chip": chip},
        "breadth": {"net_pts": net, "bull_label": bull_lab, "bear_label": bear_lab,
                    "bull_pct": ob if NIRNAY_SIGN > 0 else os_, "bear_pct": os_ if NIRNAY_SIGN > 0 else ob,
                    "n_live": _num(b.get("n_live")), "n_universe": _num(b.get("n_universe")),
                    "as_of": _date(b.get("as_of")) or None},
        "top_down_z": _num(_d(k.get("pillars")).get("top_down")),
        # The read is the plain mean of the pillars present today (convergence.nishkarsh):
        # each present pillar weighs 1/n, an absent one 0 — the rail's Model readout.
        "pillar_weights": _pillar_weights(k),
        "agreement_ratio": _num(agreement_ratio),
    }

    note = (f"Read from {pretty_date(as_of)}, the latest day with a reading."
            if as_of and run_as_of and as_of != run_as_of else "")
    synth = bool(synthetic)
    return {
        "schema": SCHEMA,
        "eyebrow": f"Nishkarsh · {h}-day read" + ("" if forward else f" · {LEVELS_TAG}"),
        "signal": word, "signal_class": side if side in ("bull", "bear") else "flat",
        "side": side, "value": value, "value_text": v_txt, "tier": tier, "direction": direction,
        "strength": strength, "strength_sub": strength_sub,
        "headline": explain_read(k) if value is not None else "No reading for the latest date.",
        "limit": limit, "binding": binding, "gates": gates, "risk": risk, "foot": FOOT,
        "trust": trust, "status": status, "status_label": STATUS_LABEL[status], "validated": validated,
        "call": call, "action": action, "track": track, "precedent": pr, "kpi": kpi,
        "two_system": two, "single_system": not two,
        "as_of": pretty_date(as_of) if as_of else "—", "as_of_note": note, "horizon": h,
        "forward_mode": forward, "levels_note": None if forward else LEVELS_TAG,
        "synthetic": synth, "banner": SYNTH_BANNER if synth else None,
    }


def verdict_texts(v) -> list[str]:
    """Every string the verdict writes itself (what the honesty tests scan)."""
    v = _d(v)
    out = [v.get(x, "") for x in ("eyebrow", "signal", "strength", "strength_sub", "headline", "limit", "foot",
                                   "as_of", "as_of_note", "value_text", "status_label", "levels_note")]
    out += [_d(v.get("track")).get("text", ""), _d(v.get("track")).get("word", ""),
            _d(v.get("precedent")).get("text", ""), _d(v.get("trust")).get("chip", "")]
    out += [str(r) for r in _list(v.get("risk"))]
    for g in _list(v.get("gates")):
        out += [_d(g).get("title", ""), _d(g).get("label", ""), _d(g).get("detail", "")]
    return [str(t) for t in out if t]
