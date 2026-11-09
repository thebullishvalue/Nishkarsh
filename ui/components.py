"""
Nishkarsh — shared UI pieces: escaping and number formatting, the Today hero (built from the
Nishkarsh Oscillator read-out only), the data-notes list behind the header pill, and small
inline-SVG/HTML primitives (gauge, diverging bar, chip).

Every data/user string that reaches ``unsafe_allow_html`` goes through ``esc()``. Nothing here
estimates or grades anything: values are read from the pipeline's session state and only
formatted, sorted or worded. Missing numbers render "—", never "nan".
"""

from __future__ import annotations

import html as html_mod
import math
import re
from datetime import datetime

import streamlit as st

from core.config import (
    AARAMBH_FORWARD_SIGNAL, AARAMBH_FWD_HORIZON, CONV2_TIER_MODERATE, CONV2_TIER_STRONG, CONV2_TIER_WEAK, EDGE_ALPHA,
)
from ui.copy import (
    LEVELS_TAG, NKO_COMPONENT_PLAIN, NKO_STATUS, NKO_TIER_WORD, PILLAR_PHRASE, PILLAR_SUBJECT, SYNTH_BANNER,
)

PILLARS = ("top_down", "bottom_up", "agreement")


# ── Escaping + number formatting ─────────────────────────────────────────────

def esc(x) -> str:
    """HTML-escape ANY data/user string before it reaches ``unsafe_allow_html``."""
    return html_mod.escape("" if x is None else str(x), quote=True)


def num(x) -> float | None:
    """``x`` as a finite float, else None (bool, None, NaN, ±inf, non-numeric → None)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, spec: str = "+.2f", dash: str = "—", suffix: str = "") -> str:
    """Format a number; every None / NaN / inf / garbage renders as ``dash`` (never "+nan")."""
    v = num(x)
    if v is None:
        return dash
    try:
        return format(v, spec) + suffix
    except (TypeError, ValueError):
        return dash


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else []


def _str(x, default: str = "") -> str:
    if x is None or isinstance(x, (float, int)) and num(x) is None:
        return default
    s = str(x).strip()
    return s if s and s.lower() not in ("nan", "none", "inf", "-inf", "+inf") else default


def _date(x) -> str:
    """'YYYY-MM-DD…' strings (or timestamps) only; anything else → ''."""
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


def signed(x, spec: str = "+.2f") -> str:
    """``fmt`` with a typographic minus (−) for display; a value that rounds to zero carries no sign."""
    v, plain = num(x), spec.replace("+", "")
    try:
        if v is not None and float(fmt(v, plain).replace(",", "")) == 0:
            return fmt(0.0, plain)
    except ValueError:
        pass
    return fmt(x, spec).replace("-", "−")


def plural(n, word: str) -> str:
    v = num(n)
    if v is None:
        return f"— {word}s"
    k = int(round(v))
    return f"{k:,} {word}" + ("" if k == 1 else "s")


def index_name(con) -> str:
    """The return label's index in plain words ("Nifty 50" for ^NSEI / the sheet's NIFTY column)."""
    src = _str(_d(con).get("label_src")).upper()
    return "Nifty 50" if ("NSEI" in src or "NIFTY" in src or not src) else _str(_d(con).get("label_src"))


def is_synthetic(run_config) -> bool:
    """True when the run's provenance (or a run warning) marks the data SYNTHETIC."""
    rc = _d(run_config)
    for v in _d(rc.get("provenance")).values():
        if "SYNTHETIC" in str(_d(v).get("note", "")).upper():
            return True
    return any(str(w).upper().startswith("SYNTHETIC") for w in _list(rc.get("warnings")))


def data_notes(run_config) -> list[str]:
    """The run's data notes: every run warning (bar the SYNTHETIC banner) plus any source whose
    provenance is stale / fallback / empty and has no warning of its own. The header pill counts
    these; the Data page lists them."""
    rc = _d(run_config)
    notes = [str(w) for w in _list(rc.get("warnings")) if _str(w) and not str(w).upper().startswith("SYNTHETIC")]
    for k, p in _d(rc.get("provenance")).items():
        word = _str(_d(p).get("provenance")).lower()
        if word in ("stale", "fallback", "empty") and not any(n.startswith(f"{k}:") for n in notes):
            notes.append(f"{k}: provenance {word.upper()}")
    return notes


# ── grading words (formatting the measurement's own flags; the UI grades nothing) ──

def edge_status_code(e) -> str:
    """PROVEN / INVERTED / NOT_PROVEN / INSUFFICIENT for an IC evidence block — PROVEN only when the
    conclusion layer's own ``significant`` flag says so."""
    e = _d(e)
    if not e or e.get("available") is not True:
        return "INSUFFICIENT"
    if e.get("significant") is True:
        return "PROVEN"
    ic, p = num(e.get("ic")), num(e.get("p"))
    if ic is not None and ic < 0 and p is not None and (1.0 - p) < EDGE_ALPHA:
        return "INVERTED"
    return "NOT_PROVEN"


def nko_tier_of(v) -> str:
    """The fixed NKO bands (same mapping as ``convergence.nishkarsh.nko_tier``)."""
    v = num(v)
    if v is None:
        return "NO READ"
    a = abs(v)
    return ("STRONG" if a >= CONV2_TIER_STRONG else "MODERATE" if a >= CONV2_TIER_MODERATE
            else "WEAK" if a > CONV2_TIER_WEAK else "NO LEAN")


def nko_side(v) -> str:
    """'bull' / 'bear' outside the no-lean band, 'flat' inside it, 'none' without a value (colour class)."""
    v = num(v)
    if v is None:
        return "none"
    return "bull" if v > CONV2_TIER_WEAK else "bear" if v < -CONV2_TIER_WEAK else "flat"


def z_side(z, dp: int | None = None) -> str:
    """Colour class by sign; with ``dp``, a value that rounds to zero at that precision is 'flat'."""
    z = num(z)
    if z is not None and dp is not None and round(z, dp) == 0:
        return "flat"
    return "none" if z is None else "bull" if z > 0 else "bear" if z < 0 else "flat"


def _mid(label: str) -> str:
    """A label as it reads mid-sentence ("Stock cut crossings" → "stock cut crossings"; "Aarambh's …" kept)."""
    return label if (label[:2].isupper() or label.startswith("Aarambh")) else label[:1].lower() + label[1:]


def component_label(cid, fallback="") -> str:
    return NKO_COMPONENT_PLAIN.get(_str(cid), (_str(fallback) or "An input", ""))[0]


# ── The Nishkarsh read (NKO): the Today hero (pure; renders, never grades) ────

NKO_STATUSES = ("VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA")
NKO_TIERS = ("STRONG", "MODERATE", "WEAK", "NO LEAN", "NO READ")
# Buy/sell/signal/call wording — allowed in the read's own words ONLY when its status is VALIDATED.
# ("buy-side" / "sell-side" describe agreement and are always allowed.)
BUY_SELL_RE = re.compile(r"\b(buy|sell)\b(?!-side)|\bsignals?\b|\bcalls?\b", re.I)
TRACK_WORD = {"VALIDATED": ("validated", "good"), "UNPROVEN": ("unproven", "caution"),
              "INSUFFICIENT_DATA": ("not enough data", "muted")}


def _strength_word(z: float) -> str:
    a = abs(z)
    return "strongly" if a >= 2 else "clearly" if a >= 1 else "mildly" if a >= 0.5 else "slightly"


def _lead_input(k: dict, pillar: str) -> str:
    """', led by <input> (z)' — the largest input of a multi-input pillar."""
    comps = [_d(c) for c in _list(k.get("components")) if _d(c).get("pillar") == pillar]
    if len(comps) < 2:
        return ""
    zs = [(c, num(c.get("z"))) for c in comps]
    zs = [(c, z) for c, z in zs if z is not None]
    if not zs:
        return ""
    c, z = max(zs, key=lambda t: abs(t[1]))
    return f", led by {_mid(component_label(c.get('id'), c.get('label')))} ({signed(z, '+.1f')})"


def explain_read(k) -> str:
    """ONE sentence on why today's read is what it is, from its pillars (and the largest input)."""
    k = _d(k)
    v = num(k.get("value"))
    if v is None or not -100.0 <= v <= 100.0:
        return "No reading for the latest date."
    pil = _d(k.get("pillars"))
    z = {p: num(pil.get(p)) for p in PILLARS}
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
    head = (f"Driven by {PILLAR_PHRASE[lead][0 if zl >= 0 else 1]} ({parts[lead]})"
            f"{_lead_input(k, lead)}")
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


def _short_date(d: str, ref: str) -> str:
    """'2026-10-21' → '21 Oct' when it shares ``ref``'s year, else '21 Oct 2025'."""
    full = pretty_date(d)
    return full[:-5] if full != "—" and ref[:4] == d[:4] else full


def context_chips(k) -> list[dict]:
    """The three stat chips under the headline: 5-day change, the streak on this side, strength."""
    k = _d(k)
    ch = num(k.get("change_5d"))
    st_ = _d(k.get("streak"))
    n = num(st_.get("sessions"))
    side = _str(st_.get("side")).upper()
    since = _date(st_.get("since"))
    pct = num(k.get("strength_percentile"))
    if n is None or n < 1:
        streak = "—"
    elif int(n) == 1:
        streak = "1 session · new today"
    else:
        streak = plural(n, "session") + (f" · since {_short_date(since, _date(k.get('as_of')))}" if since else "")
    return [
        {"label": "5-day change", "value": f"{signed(ch, '+.0f')} pts" if ch is not None else "—",
         "class": z_side(ch) if ch is not None and abs(ch) >= 1 else "flat"},
        {"label": "Without a lean for" if side == "NO LEAN" else "On this side for", "value": streak,
         "class": "flat"},
        {"label": "Stronger than", "value": f"{pct:.0f}% of past readings" if pct is not None else "—",
         "class": "flat"},
    ]


def track_record(k, h: int) -> dict:
    """The one inline reliability line on Today (word, colour class, sentence)."""
    k = _d(k)
    status = _str(k.get("status")).upper()
    status = status if status in NKO_STATUSES else "INSUFFICIENT_DATA"
    word, cls = TRACK_WORD[status]
    ev = _d(k.get("evidence"))
    n, ic, p = num(ev.get("n")), num(ev.get("ic")), num(ev.get("p"))
    periods = f"{int(n):,} past {h}-day periods" if n else f"past {h}-day periods"
    if status == "VALIDATED":
        text = (f"on {periods} this read has lined up with the Nifty's next move (accuracy {signed(ic)}, "
                f"p {fmt(p, '.2f')}), so its lean is a measured directional call.")
    elif status == "UNPROVEN":
        text = (f"on {periods} this read has not yet shown it anticipates the Nifty (accuracy {signed(ic)}). "
                "It describes what the systems see, not where the market goes.")
    else:
        text = "there is not yet enough history with both systems running to measure this read."
    return {"status": status, "word": word, "class": cls, "text": text + " Details on Reliability."}


def precedent_line(ph, read_dir: int = 0) -> dict:
    """The one-line 'after readings like today' from ``precedent["hero"]`` (basis: the NKO state).
    It VOTES only when its exact binomial p < EDGE_ALPHA; otherwise it is description."""
    ph = _d(ph)
    n, pos = num(ph.get("n")), num(ph.get("positive_pct"))
    if not n or n < 1 or pos is None:
        return {"state": "info", "votes": False, "n": 0,
                "text": "No past moments like today were found for this run."}
    n_i = int(round(n))
    k = int(round(pos / 100.0 * n_i))
    u, p, med, hp = (num(ph.get(x)) for x in ("unconditional_positive_pct", "binomial_p", "median", "horizon"))
    hz = f"{int(hp)} sessions" if hp is not None and hp >= 1 else "the horizon"
    text = (f"In {n_i} past moments like today, the Nifty rose {k} time{'s' if k != 1 else ''} over the next {hz} "
            f"(usually {fmt(u, '.0f', suffix='%')}) — median {signed(med, '+.1f')}%")
    if not (p is not None and p < EDGE_ALPHA and u is not None):
        return {"state": "info", "votes": False, "n": n_i, "text": text + f"; not significant (p {fmt(p, '.2f')})."}
    lean = 1 if pos > u else -1 if pos < u else 0
    word = "a rise" if lean > 0 else "a fall" if lean < 0 else "neither way"
    if read_dir and lean:
        state, rel = (("confirm", ", in line with today's read") if lean == read_dir
                      else ("conflict", ", against today's read"))
    else:
        state, rel = "info", ""
    return {"state": state, "votes": True, "n": n_i,
            "text": text + f"; significant (p {fmt(p, '.2f')}) — these moments lean to {word}{rel}."}


def build_hero_verdict(conclusion, precedent_hero=None, run_config=None) -> dict:
    """The Today hero, inferred from the Nishkarsh Oscillator ONLY (``conclusion["nko"]``) plus the
    precedent read of the same state (``precedent["hero"]``). Pure; never raises; no NaN text.

    Honesty rules: the headline describes the instrument ("Nishkarsh reads moderately bearish");
    buy/sell/signal/call wording (``BUY_SELL_RE``) and a directional ``call`` only when the read's
    measured status is VALIDATED; the track record is always named; a single-system date says
    "top-down only"; the precedent votes only when its binomial p < EDGE_ALPHA; levels mode carries
    the falsified-as-a-forecaster tag; SYNTHETIC runs carry the banner."""
    con, rc = _d(conclusion), _d(run_config)
    k = _d(con.get("nko"))
    fm = con.get("forward_mode")
    if not isinstance(fm, bool):
        fm = rc.get("forward_mode")
    forward = bool(fm) if isinstance(fm, bool) else bool(AARAMBH_FORWARD_SIGNAL)
    h_raw = num(con.get("horizon"))
    h = int(h_raw) if h_raw is not None and h_raw >= 1 else int(AARAMBH_FWD_HORIZON)

    track = track_record(k, h)
    status = track["status"]
    validated = status == "VALIDATED"
    value = num(k.get("value"))
    if value is not None and not -100.0 <= value <= 100.0:
        value = None
    tier = nko_tier_of(value)
    direction = ("BULLISH" if value > CONV2_TIER_WEAK else "BEARISH" if value < -CONV2_TIER_WEAK
                 else "NEUTRAL") if value is not None else "NEUTRAL"
    read_dir = 1 if direction == "BULLISH" else -1 if direction == "BEARISH" else 0
    call = validated and read_dir != 0
    side = nko_side(value)
    two = k.get("two_system") is True
    v_txt = signed(value, "+.0f")

    if value is None:
        headline, gauge_label = "No Nishkarsh read today", "No reading"
    elif tier == "NO LEAN":
        headline, gauge_label = "Nishkarsh has no lean today", "No lean"
    else:
        dword = direction.lower()
        headline = f"Nishkarsh reads {NKO_TIER_WORD[tier]} {dword}"
        gauge_label = f"{tier.capitalize()} · {dword}"
    if value is not None and not two:
        headline += " (top-down only today)"

    as_of = _date(k.get("as_of")) or _date(con.get("as_of")) or _date(rc.get("as_of"))
    run_as_of = _date(con.get("as_of"))
    note = (f"Read from {pretty_date(as_of)}, the latest day with a reading."
            if as_of and run_as_of and as_of != run_as_of else "")
    synth = is_synthetic(rc)
    return {
        "eyebrow": "Today's Nishkarsh read" + ("" if forward else f" · {LEVELS_TAG}"),
        "headline": headline, "side": side, "value": value, "value_text": v_txt,
        "tier": tier, "direction": direction, "call": bool(call), "validated": validated,
        "gauge_label": gauge_label, "two_system": two, "single_system": not two,
        "explain": explain_read(k) if value is not None else "No reading for the latest date.",
        "chips": context_chips(k) if value is not None else [],
        "track": track, "status": status, "status_label": NKO_STATUS[status][0],
        "precedent": precedent_line(precedent_hero, read_dir),
        "as_of": pretty_date(as_of) if as_of else "—", "as_of_note": note, "horizon": h,
        "forward_mode": forward, "levels_note": None if forward else LEVELS_TAG,
        "synthetic": synth, "banner": SYNTH_BANNER if synth else None,
    }


def hero_texts(v: dict) -> list[str]:
    """Every string the hero writes itself (what the honesty tests scan)."""
    v = _d(v)
    out = [v.get(x, "") for x in ("eyebrow", "headline", "gauge_label", "explain", "as_of", "as_of_note",
                                   "value_text", "status_label", "levels_note")]
    out += [_d(v.get("track")).get("text", ""), _d(v.get("track")).get("word", ""),
            _d(v.get("precedent")).get("text", "")]
    for c in _list(v.get("chips")):
        out += [_d(c).get("label", ""), _d(c).get("value", "")]
    return [str(t) for t in out if t]


# ── SVG / HTML primitives (inline, no external deps) ──────────────────────────

_SIDE_COLOR = {"bull": "#34D399", "bear": "#FB7185", "flat": "#94A3B8", "none": "#64748B"}
_CLASSES = ("good", "caution", "bad", "muted", "info")


def gauge_svg(value) -> str:
    """Semicircular −100…+100 gauge with the fixed NKO bands (±10 no lean, ±30 moderate, ±60 strong):
    bearish half red, bullish half green, deeper with strength; a filled arc from 0 to today's value
    and a marker on the arc. The number sits inside the arc (rendered by the caller). None → track only."""
    cx, cy, r = 150.0, 150.0, 118.0

    def pt(v, rr=r):
        th = math.pi * (1.0 - (v + 100.0) / 200.0)
        return cx + rr * math.cos(th), cy - rr * math.sin(th)

    def arc(v0, v1, rr=r):
        (x0, y0), (x1, y1) = pt(v0, rr), pt(v1, rr)
        return f"M {x0:.2f} {y0:.2f} A {rr:.0f} {rr:.0f} 0 0 1 {x1:.2f} {y1:.2f}"

    w, sw = CONV2_TIER_WEAK, 18
    bands = ((-100, -CONV2_TIER_STRONG, "251,113,133", 0.34), (-CONV2_TIER_STRONG, -CONV2_TIER_MODERATE, "251,113,133", 0.22),
             (-CONV2_TIER_MODERATE, -w, "251,113,133", 0.12), (-w, w, "148,163,184", 0.12),
             (w, CONV2_TIER_MODERATE, "52,211,153", 0.12), (CONV2_TIER_MODERATE, CONV2_TIER_STRONG, "52,211,153", 0.22),
             (CONV2_TIER_STRONG, 100, "52,211,153", 0.34))
    parts = [f'<path d="{arc(a + 0.6, b - 0.6)}" stroke="rgba({c},{o})" stroke-width="{sw}" fill="none"/>'
             for a, b, c, o in bands]
    for t in (-CONV2_TIER_STRONG, -CONV2_TIER_MODERATE, -w, 0, w, CONV2_TIER_MODERATE, CONV2_TIER_STRONG):
        (x0, y0), (x1, y1) = pt(t, r + 13), pt(t, r + (21 if t == 0 else 18))
        parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                     f'stroke="rgba(148,163,184,{0.6 if t == 0 else 0.4})" stroke-width="1.5" stroke-linecap="round"/>')
    v = num(value)
    if v is not None and -100.0 <= v <= 100.0:
        color = _SIDE_COLOR[nko_side(v)]
        if abs(v) >= 1.0:
            lo, hi = (0.0, v) if v > 0 else (v, 0.0)
            parts.append(f'<path d="{arc(lo, hi, r)}" stroke="{color}" stroke-width="5" fill="none" '
                         f'stroke-linecap="round" opacity="0.95"/>')
        mx, my = pt(v, r)
        parts.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="11" fill="{color}" stroke="#0B1020" stroke-width="4"/>')
    return ('<svg class="nk-gauge-svg" viewBox="0 0 300 160" role="img" aria-label="Nishkarsh Oscillator gauge '
            'from minus 100 (strongly bearish) to plus 100 (strongly bullish)">' + "".join(parts) + "</svg>")


def zbar_html(z, scale: float = 3.0) -> str:
    """A signed horizontal bar for a z-score (centre = 0, ends = ±``scale``); colour by sign."""
    v = num(z)
    if v is None:
        return '<div class="nk-zbar"><div class="mid"></div></div>'
    frac = max(-1.0, min(1.0, v / scale)) * 50.0
    left = 50.0 + min(0.0, frac)
    return (f'<div class="nk-zbar"><div class="mid"></div><div class="fill {z_side(v)}" '
            f'style="left:{left:.1f}%;width:{abs(frac):.1f}%"></div></div>')


def chip_html(label: str, cls: str = "muted") -> str:
    cls = cls if cls in _CLASSES else "muted"
    return f'<span class="nk-chip {cls}"><span class="nk-dot"></span>{esc(label)}</span>'


def section_head(title: str, sub: str = "", right: str = "", level: str = "") -> str:
    """A section heading (h3 + one plain line); ``right`` is escaped text shown at the right."""
    r = f'<span class="nk-head-right">{esc(right)}</span>' if right else ""
    p = f"<p>{esc(sub)}</p>" if sub else ""
    return f'<div class="nk-section-head {esc(level)}"><div class="nk-head-row"><h3>{esc(title)}</h3>{r}</div>{p}</div>'


def stat_tiles(tiles: list[tuple]) -> str:
    """A row of quiet stat tiles: [(label, value, note), …] — all escaped."""
    cells = "".join(f'<div class="nk-tile"><div class="nk-tile-label">{esc(lab)}</div>'
                    f'<div class="nk-tile-value">{esc(val)}</div><div class="nk-tile-note">{esc(note)}</div></div>'
                    for lab, val, note in tiles)
    return f'<div class="nk-tiles">{cells}</div>'


def render_hero(v: dict) -> None:
    """The Today hero: the read in words + three stat chips (left), the gauge with the number inside
    (right), and one full-width track-record line. Every string escaped."""
    v = _d(v)
    side = str(v.get("side", "none"))
    side = side if side in _SIDE_COLOR else "none"
    chips = "".join(
        f'<div class="nk-stat"><span>{esc(_d(c).get("label"))}</span>'
        f'<b class="{esc(_d(c).get("class") if _d(c).get("class") in _SIDE_COLOR else "flat")}">'
        f'{esc(_d(c).get("value"))}</b></div>' for c in _list(v.get("chips")))
    tr = _d(v.get("track"))
    t_cls = tr.get("class") if tr.get("class") in _CLASSES else "muted"
    note = f'<div class="nk-meta-note">{esc(v.get("as_of_note"))}</div>' if v.get("as_of_note") else ""
    st.markdown(
        f'<div class="nk-hero">'
        f'<div class="nk-hero-main">'
        f'<div class="nk-eyebrow">{esc(v.get("eyebrow") or "Today’s Nishkarsh read")}</div>'
        f'<h2 class="nk-headline">{esc(v.get("headline", "—"))}</h2>'
        f'<p class="nk-explain">{esc(v.get("explain", "—"))}</p>'
        + (f'<div class="nk-stats">{chips}</div>' if chips else "")
        + f'{note}</div>'
        f'<div class="nk-hero-gauge"><div class="nk-gauge-wrap">{gauge_svg(v.get("value"))}'
        f'<div class="nk-gauge-in"><div class="nk-gauge-value {side}">{esc(v.get("value_text", "—"))}</div>'
        f'<div class="nk-gauge-label">{esc(v.get("gauge_label", "No reading"))}</div></div></div>'
        f'<div class="nk-gauge-scale"><span>−100 bearish</span><span>+100 bullish</span></div></div>'
        f'<div class="nk-track {t_cls}"><span class="nk-dot"></span><p><b>Track record: '
        f'{esc(tr.get("word") or "not enough data")}</b> — {esc(tr.get("text") or "")}</p></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_error(name: str, e) -> None:
    st.markdown(f'<div class="nk-error"><b>Could not draw {esc(name)}</b><span>{esc(e)}</span></div>',
                unsafe_allow_html=True)
