"""
Nishkarsh v2 — Reusable UI components: metric cards, headers, the hero verdict, escaping.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

UI — Obsidian Quant Terminal design language. Every data/user string that reaches
``unsafe_allow_html`` goes through ``esc()`` (R7); the hero only RENDERS the conclusion
layer's verdict (C-V2) — it computes no statistics.
"""

from __future__ import annotations

import html as html_mod
import math
import re
from datetime import datetime

import streamlit as st

from core.config import AARAMBH_FORWARD_SIGNAL, AARAMBH_FWD_HORIZON, CONV2_TIER_WEAK, EDGE_ALPHA
from ui.copy import AGREEMENT_BANDS_TEXT as _AGREEMENT_BANDS_TEXT  # noqa: E402
from ui.copy import (
    AGREEMENT_BANDS, LEVELS_TAG, OSC_STATUS, SIGN_C, SIGN_ENGINE, SIGN_Z, SYNTH_BANNER, TIERS_TEXT, TRUST_PLAIN,
)


# ── SVG Icons (inline, no external deps) — with ARIA labels for accessibility

ICONS = {
    "chart":      '<svg aria-label="Chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "cube":       '<svg aria-label="Cube icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    "target":     '<svg aria-label="Target icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":     '<svg aria-label="Layers icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar-chart":  '<svg aria-label="Bar chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "activity":   '<svg aria-label="Activity icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "crosshair":  '<svg aria-label="Crosshair icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    "cpu":        '<svg aria-label="CPU icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "zap":        '<svg aria-label="Zap icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield":     '<svg aria-label="Shield icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "grid":       '<svg aria-label="Grid icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "database":   '<svg aria-label="Database icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "trending":   '<svg aria-label="Trending icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "eye":        '<svg aria-label="Eye icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "play":       '<svg aria-label="Play icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "chevron-right": '<svg aria-label="Expand icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>',
    "sun":        '<svg aria-label="Light mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    "moon":       '<svg aria-label="Dark mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    "download":   '<svg aria-label="Download icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "briefcase":  '<svg aria-label="Portfolio icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "compass":    '<svg aria-label="Regime icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "rocket":     '<svg aria-label="Strong Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-3 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4.5c1.62-1.63 5-2.5 5-2.5"/><path d="M12 15v5s3.03-.55 4.5-2c1.63-1.62 2.5-5 2.5-5"/></svg>',
    "trending-up": '<svg aria-label="Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "trending-down": '<svg aria-label="Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
    "arrow-up-right": '<svg aria-label="Weak Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>',
    "arrow-down-right": '<svg aria-label="Weak Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/></svg>',
    "arrow-up":   '<svg aria-label="Up" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow-down": '<svg aria-label="Down" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>',
    "move-horizontal": '<svg aria-label="Chop icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "alert-triangle": '<svg aria-label="Crisis icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "help-circle": '<svg aria-label="Unknown icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "circle":     '<svg aria-label="Circle" role="img" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
    "check-circle": '<svg aria-label="Check" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "scale":      '<svg aria-label="Weighting icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
}


def get_icon(name: str, size: int = 18, stroke_width: float = 1.5) -> str:
    """Return an SVG icon string with custom size and stroke width."""
    base_svg = ICONS.get(name, ICONS["chart"])

    # Clean existing attributes to avoid duplicates or stale values
    base_svg = re.sub(r'\s+width="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+height="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+stroke-width="[^"]*"', '', base_svg)

    # Inject standardized attributes
    return base_svg.replace('<svg', f'<svg width="{size}" height="{size}" stroke-width="{stroke_width}"')


def render_section_header(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
) -> None:
    """Render a styled section header with icon, title, and optional description.

    Args:
        title: Section title (rendered uppercase).
        description: Optional one-line description below title.
        icon: Key from ICONS dict.
        accent: CSS color class — "", "cyan", "emerald", "violet", "rose".
    """
    svg = get_icon(icon, size=16, stroke_width=1.8)
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text">'
        f'<h3>{html_mod.escape(title)}</h3>'
        f'{desc_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_gap() -> None:
    """Insert vertical spacing between major sections."""
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


# Tokens kept verbatim when an ALL-CAPS terminal label is turned into sentence case.
_KEEP_TOKENS = {"IC", "OOS", "MDE", "HAC", "DFA", "ADF", "KPSS", "OU", "HMM", "CSV", "CI", "PCA", "OLS", "DDM",
                "C", "R²", "P", "N", "IST", "NSE", "OB", "OS", "CUSUM", "GARCH", "MSF", "MMR", "NIFTY"}
_PROPER_TOKENS = {"AARAMBH": "Aarambh", "NIRNAY": "Nirnay", "NISHKARSH": "Nishkarsh", "HURST": "Hurst",
                  "SHARPE": "Sharpe", "KAPPA": "kappa"}


def label_case(text) -> str:
    """Sentence case for a terminal-style ALL-CAPS label ("OOS R² (MAGNITUDE)" → "OOS R² (magnitude)").
    Mixed-case text is returned unchanged; acronyms, symbols and proper names are kept."""
    s = "" if text is None else str(text)
    letters = [ch for ch in s if ch.isalpha()]
    if not letters or sum(ch.isupper() for ch in letters) < 0.7 * len(letters):
        return s
    out = []
    for w in s.split(" "):
        core = w.strip("()·,:;[]")
        if not core or core in _KEEP_TOKENS or "_" in core or any(ch.isdigit() for ch in core):
            out.append(w)
        elif core in _PROPER_TOKENS:
            out.append(w.replace(core, _PROPER_TOKENS[core]))
        else:
            out.append(w.lower())
    res = " ".join(out)
    for i, ch in enumerate(res):
        if ch.isalpha():
            return res[:i] + ch.upper() + res[i + 1:]
    return res


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
    tooltip: str = "",
    icon: str = "",
) -> None:
    """Render a terminal-styled metric card with optional tooltip.

    Args:
        label: Card label (rendered uppercase).
        value: Primary metric value.
        subtext: Optional secondary description below value.
        color_class: Semantic color — "neutral", "success", "danger", "warning", "info", "violet".
        tooltip: Optional hover explanation text.
        icon: Optional ICONS key — small icon inlined before the label.
    """
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="metric-tooltip" data-tooltip="{html_mod.escape(tooltip)}">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            f'<span class="metric-tooltip-text">{html_mod.escape(tooltip)}</span>'
            f'</div>'
        )

    sub_metric_html = f'<div class="sub-metric">{html_mod.escape(subtext)}</div>' if subtext else ""
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12, stroke_width=2)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f"<h4>{icon_html}{html_mod.escape(label_case(label))}</h4>"
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_metric_html}"
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_header(title: str, tagline: str) -> None:
    """Render the terminal masthead."""
    st.markdown(
        f'<div class="premium-header">'
        f"<h1>{html_mod.escape(title)}</h1>"
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_interpretation_card(
    title: str,
    body: str,
    color: str = "neutral",
) -> None:
    """Render a state-aware interpretation card — terminal readout style.

    Args:
        title: Short state label (e.g. "NEUTRAL", "STRONG OVERSOLD").
        body: One-paragraph explanation. RAW HTML: callers must pass every data value
            through ``esc()`` (only static markup may be unescaped).
        color: Semantic color — "neutral", "success", "danger", "warning", "info".
    """
    st.markdown(
        f'<div class="interp-card {html_mod.escape(color)}">'
        f'<div class="interp-title">{html_mod.escape(label_case(title))}</div>'
        f'<div class="interp-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Escaping + number formatting (A4-06 / A4-18) ─────────────────────────────

def esc(x) -> str:
    """HTML-escape ANY data/user string before it reaches ``unsafe_allow_html`` (R7)."""
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


def is_synthetic(run_config) -> bool:
    """True when the run's provenance (or a run warning) marks the data SYNTHETIC."""
    rc = _d(run_config)
    for v in _d(rc.get("provenance")).values():
        if "SYNTHETIC" in str(_d(v).get("note", "")).upper():
            return True
    return any(str(w).upper().startswith("SYNTHETIC") for w in (rc.get("warnings") or []))


# ── Hero verdict: pure (no Streamlit) — renders the conclusion, never recomputes it ──

CALL_TIERS = frozenset({"DIRECTIONAL_EDGE", "TRADEABLE_EDGE"})
TRUST_TIERS = ("INVERTED", "NO_PROVEN_EDGE", "DIRECTIONAL_EDGE", "TRADEABLE_EDGE", "INSUFFICIENT_DATA")
_C_TIERS = ("STRONG", "MODERATE", "WEAK", "NO LEAN")
_DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL")
_TRUST_DEFAULT_TEXT = {
    "INSUFFICIENT_DATA": "INSUFFICIENT DATA — no edge could be graded for this run.",
    "INVERTED": "INVERTED — no directional call.",
    "NO_PROVEN_EDGE": "NO PROVEN EDGE — direction is unvalidated.",
    "DIRECTIONAL_EDGE": "DIRECTIONAL EDGE.",
    "TRADEABLE_EDGE": "TRADEABLE EDGE.",
}
TRUST_CHIP_CLASS = {"TRADEABLE_EDGE": "solid", "DIRECTIONAL_EDGE": "modest", "NO_PROVEN_EDGE": "muted",
                    "INVERTED": "inverted", "INSUFFICIENT_DATA": "muted"}


def _precedent_row(ph: dict, call_dir: int, target: str) -> dict:
    n = num(ph.get("n"))
    pos = num(ph.get("positive_pct"))
    if not n or n < 1 or pos is None:
        return {"tag": "PRECEDENT", "state": "info", "word": "no data", "votes": False,
                "text": "No historical analogs for this run."}
    n_i = int(round(n))
    k = int(round(pos / 100.0 * n_i))
    u, p, med = num(ph.get("unconditional_positive_pct")), num(ph.get("binomial_p")), num(ph.get("median"))
    hp = num(ph.get("horizon"))
    hz = f"{int(hp)}d" if hp is not None else "the horizon"
    tgt = f" ({target})" if target else ""
    text = (f"{k}/{n_i} analogs rose over {hz} vs {fmt(u, '.0f', suffix='%')} unconditional "
            f"(binomial p={fmt(p, '.2f')}); median {fmt(med, '+.2f', suffix='%')}{tgt}")
    votes = p is not None and p < EDGE_ALPHA and u is not None
    if not votes:
        return {"tag": "PRECEDENT", "state": "info", "word": "descriptive", "votes": False,
                "text": text + f" — not significant at p < {EDGE_ALPHA:g}; descriptive only."}
    lean = 1 if pos > u else -1 if pos < u else 0
    word = "rise" if lean > 0 else "fall" if lean < 0 else "no lean"
    if call_dir and lean:
        state, w = ("confirm", "supports") if lean == call_dir else ("conflict", "against")
    else:
        state, w = "info", "context"
    return {"tag": "PRECEDENT", "state": state, "word": w, "votes": True,
            "text": text + f" — significant; analogs lean to a {word}."}


def build_hero_verdict(conclusion, aarambh_signal=None, precedent_hero=None, run_config=None,
                       nirnay_latest=None) -> dict:
    """Pure hero verdict from the C-V2 ``conclusion`` dict (+ engine signal, precedent hero, run config).

    Rules: the headline is the conclusion's latest C (+ = bullish) with its fixed tier/direction; the
    trust chip is ``conclusion["trust"]`` verbatim; a COLOURED directional call is made only when the
    trust tier is DIRECTIONAL_EDGE or TRADEABLE_EDGE (INVERTED → no call, ever); levels mode carries
    "valuation context — falsified as a forecaster"; the Nirnay row is ``conclusion["nirnay"]["text"]``
    verbatim; precedent votes only when its binomial p < EDGE_ALPHA. Never raises on missing keys or
    NaN; every missing number renders "—". ``market`` is the plain-language market read
    (``build_market_read``) built from the conclusion's oscillator + latest read-out."""
    con, sig, ph, rc = _d(conclusion), _d(aarambh_signal), _d(precedent_hero), _d(run_config)
    lat, trust, nir = _d(con.get("latest")), _d(con.get("trust")), _d(con.get("nirnay"))
    fm = con.get("forward_mode")
    if not isinstance(fm, bool):
        fm = rc.get("forward_mode")
    forward = bool(fm) if isinstance(fm, bool) else bool(AARAMBH_FORWARD_SIGNAL)
    h_raw = num(con.get("horizon"))
    h = int(h_raw) if h_raw is not None and h_raw >= 1 else int(AARAMBH_FWD_HORIZON)
    target = _str(rc.get("target"))

    C, Ca, zA, zN, kappa = (num(lat.get(k)) for k in ("C", "C_aarambh", "zA", "zN", "kappa"))
    tier = _str(lat.get("tier")).upper()
    tier = tier if tier in _C_TIERS else "NO LEAN"
    direction = _str(lat.get("direction")).upper()
    direction = direction if direction in _DIRECTIONS else "NEUTRAL"
    if C is None:
        tier, direction = "NO LEAN", "NEUTRAL"
    t_tier = _str(trust.get("tier")).upper()
    t_tier = t_tier if t_tier in TRUST_TIERS else "INSUFFICIENT_DATA"
    t_text = _str(trust.get("text")) or _TRUST_DEFAULT_TEXT[t_tier]

    has_dir = direction in ("BULLISH", "BEARISH") and tier != "NO LEAN"
    call = has_dir and t_tier in CALL_TIERS
    call_dir = (1 if direction == "BULLISH" else -1) if call else 0
    sig_class = ("buy" if call_dir > 0 else "sell") if call else "hold"
    dword = direction.lower()
    c_txt = fmt(C, "+.1f")

    if call:
        label = f"{direction} · {tier}"
        headline = (f"Conclusion C {c_txt} ({tier.lower()}) points {dword} over the next {h} trading days — "
                    f"{t_tier.replace('_', ' ').lower()} on this data.")
    elif t_tier == "INVERTED":
        label = "INVERTED · context only"
        headline = ("The directional head has pointed the wrong way on this data — no directional call "
                    f"is made (C {c_txt} is shown as context).")
    elif has_dir and t_tier == "NO_PROVEN_EDGE":
        label = f"LEAN {direction} ({tier}) · unvalidated"
        headline = f"The head leans {dword} (C {c_txt}), but no directional edge is proven on this data — no call."
    elif has_dir:
        label = f"LEAN {direction} ({tier}) · ungraded"
        headline = (f"The head leans {dword} (C {c_txt}), but there is too little out-of-sample history "
                    "to grade an edge — no call.")
    elif C is None:
        label = "NO CONCLUSION"
        headline = "No conclusion value for the latest date — no lean, no call."
    else:
        label = "NO LEAN"
        headline = f"C {c_txt} sits inside the ±{CONV2_TIER_WEAK:g} no-lean band — no directional lean."
    mode_label = (f"FORWARD · {h}d forecast head" if forward else f"LEVELS · {LEVELS_TAG}")
    if not forward:
        headline += f" Levels mode: {LEVELS_TAG}."

    # ── evidence rows ──
    conv = num(sig.get("conviction_score"))
    model_txt = (f"Aarambh {'forecast' if forward else 'valuation head'} zA {fmt(zA, '+.2f')} → "
                 f"C_aarambh {fmt(Ca, '+.1f')} · κ {fmt(kappa, '.2f')} → C {c_txt} ({SIGN_Z}); "
                 f"engine conviction {fmt(conv, '+.1f')} ({SIGN_ENGINE}).")
    if not forward:
        model_txt += f" {LEVELS_TAG[0].upper() + LEVELS_TAG[1:]}."
    rows = [{"tag": "MODEL", "state": "info", "word": "head", "votes": True, "text": model_txt}]

    n_state = _str(nir.get("state")).upper() or "NO_DATA"
    lift, oos = _d(con.get("lift")), _d(con.get("oos"))
    passed = lift.get("passed") is True
    if passed and n_state == "CONFIRM" and oos.get("adds_value") is True:
        n_row_state, n_word = "confirm", "measured"
    elif passed and n_state == "CONTRADICT" and con.get("kappa_enabled") is True and (kappa or 1.0) < 1.0:
        n_row_state, n_word = "conflict", "shrinks"
    else:
        n_row_state, n_word = "info", "context"
    rows.append({"tag": "NIRNAY", "state": n_row_state, "word": n_word, "votes": n_row_state != "info",
                 "text": _str(nir.get("text")) or "Nirnay: — (no read-out in this run)."})

    rows.append(_precedent_row(ph, call_dir, target) if ph else
                {"tag": "PRECEDENT", "state": "info", "word": "no data", "votes": False,
                 "text": "No historical analogs for this run."})

    div, agr = _d(con.get("divergence_recent")), _d(con.get("agreement"))
    if con.get("single_system") is True:
        risk = {"tag": "RISK", "state": "info", "word": "n/a", "votes": False,
                "text": "Single-system mode — no cross-system divergence monitoring."}
    else:
        w = num(div.get("window_days"))
        nc = num(div.get("CONTRADICTION"))
        risk_txt = (f"Last {fmt(w, '.0f')} sessions: {fmt(nc, '.0f')} contradiction-event days, "
                    f"{fmt(div.get('AARAMBH_LEADS'), '.0f')} Aarambh-leads, {fmt(div.get('NIRNAY_LEADS'), '.0f')} "
                    f"Nirnay-leads · 60-session same-sign rate "
                    f"{fmt(agr.get('same_sign_rate_60d'), '.0%')}.")
        flagged = nc is not None and nc > 0
        risk = {"tag": "RISK", "state": "conflict" if flagged else "info",
                "word": "divergence" if flagged else "calm", "votes": False, "text": risk_txt}
    rows.append(risk)

    as_of = _date(con.get("as_of")) or _date(lat.get("date")) or _date(rc.get("as_of")) or "—"
    synth = is_synthetic(rc)
    return {
        "signal": label, "signal_class": sig_class, "call": bool(call), "direction": direction, "tier": tier,
        "score": C, "score_text": f"C {c_txt}", "sign_note": f"{SIGN_C} · {TIERS_TEXT}",
        "horizon": h, "forward_mode": forward, "mode_label": mode_label, "headline": headline,
        "trust": {"tier": t_tier, "label": t_tier.replace("_", " "), "text": t_text,
                  "class": TRUST_CHIP_CLASS[t_tier]},
        "evidence": rows, "as_of": as_of, "synthetic": synth, "banner": SYNTH_BANNER if synth else None,
        "market": build_market_read(con, rc, nirnay_latest),
    }


# ── Market read: the Overview's plain-language read-out (pure; renders, never grades) ──

OSC_STATUSES = ("VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA")
# Buy/sell/signal wording — allowed for the oscillator ONLY when its status is VALIDATED.
# ("buy-side" / "sell-side" describe agreement and are always allowed.)
BUY_SELL_RE = re.compile(r"\b(buy|sell)\b(?!-side)|\bsignals?\b", re.I)
_SIDES = {"BUY-SIDE": "buy", "SELL-SIDE": "sell", "SPLIT": "split"}


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
    """``fmt`` with a typographic minus (−) for display; an exact zero carries no sign."""
    if num(x) == 0:
        return fmt(0.0, spec.replace("+", ""))
    return fmt(x, spec).replace("-", "−")


def agreement_word(strength) -> str:
    v = num(strength)
    if v is None or v <= 0:
        return ""
    for lo, word in AGREEMENT_BANDS:
        if v >= lo:
            return word
    return AGREEMENT_BANDS[-1][1]


def edge_status_code(e) -> str:
    """PROVEN / INVERTED / NOT_PROVEN / INSUFFICIENT for an IC evidence block — PROVEN only when the
    conclusion layer's own ``significant`` flag says so (the UI grades nothing itself)."""
    e = _d(e)
    if not e or e.get("available") is not True:
        return "INSUFFICIENT"
    if e.get("significant") is True:
        return "PROVEN"
    ic, p = num(e.get("ic")), num(e.get("p"))
    if ic is not None and ic < 0 and p is not None and (1.0 - p) < EDGE_ALPHA:
        return "INVERTED"
    return "NOT_PROVEN"


def _aarambh_read(lat: dict, forward: bool) -> tuple[str, str]:
    """(sub-line phrase, card pill) for Aarambh's latest read-out."""
    C = num(lat.get("C"))
    tier = _str(lat.get("tier")).upper()
    direction = _str(lat.get("direction")).upper()
    noun = "forecast" if forward else "valuation head"
    if C is None:
        return f"no {noun} reading", "No reading"
    if tier not in ("STRONG", "MODERATE", "WEAK") or direction not in ("BULLISH", "BEARISH"):
        tilt = ", tilting higher" if C > 0 else ", tilting lower" if C < 0 else ""
        return f"{noun} flat{tilt}", "Flat" + (" · tilts higher" if C > 0 else " · tilts lower" if C < 0 else "")
    word = "higher" if direction == "BULLISH" else "lower"
    return f"{noun} leans {word} ({tier.lower()} conviction)", f"Leans {word} · {tier.lower()}"


def _nirnay_read(lat: dict, nl: dict, single: bool) -> tuple[str, str]:
    """(sub-line phrase, card pill) for Nirnay: breadth in words + its lean from the conclusion state."""
    if single:
        return "not running (no stock data)", "Not running"
    n = num(nl.get("n_live"))
    ob, os_ = num(nl.get("Overbought_Pct")), num(nl.get("Oversold_Pct"))
    if not n or (ob is None and os_ is None):
        phrase = "no stock data today"
    else:
        names = f"the {int(round(n))} Nifty stocks"
        if (ob or 0) >= (os_ or 0) and (ob or 0) > 0:
            phrase = f"{fmt(ob, '.0f')}% of {names} overbought"
        elif (os_ or 0) > 0:
            phrase = f"{fmt(os_, '.0f')}% of {names} oversold"
        else:
            phrase = f"none of {names} at an extreme"
    zn, state = num(lat.get("zN")), _str(lat.get("state")).upper()
    if zn is None or state in ("", "NO_DATA"):
        pill = "No data"
    elif state == "NEUTRAL":
        pill = "Neutral" + (" · tilts bullish" if zn > 0 else " · tilts bearish" if zn < 0 else "")
    else:
        pill = "Leans bullish" if zn > 0 else "Leans bearish" if zn < 0 else "Neutral"
    return phrase, pill


def _osc_numbers(ev: dict, h: int) -> list[dict]:
    """The oscillator's measured evidence in plain labels (values only re-formatted, never re-derived)."""
    if not ev:
        return []
    yearly = _d(ev.get("yearly"))
    share = num(ev.get("yearly_positive_share"))
    years = (f"{int(round(share * len(yearly)))} of {len(yearly)} years" if share is not None and yearly
             else "—")
    ci = ev.get("boot_ci") if isinstance(ev.get("boot_ci"), (list, tuple)) and len(ev.get("boot_ci")) == 2 \
        else (None, None)
    n = num(ev.get("n"))
    return [
        {"label": "Accuracy score", "value": signed(ev.get("ic")),
         "note": f"how closely the gauge has lined up with the Nifty's next {h} trading days (0 = no link)"},
        {"label": "History tested", "value": f"{int(n)} periods" if n is not None else "—",
         "note": f"separate, non-overlapping {h}-day periods"},
        {"label": "Significance", "value": f"p = {fmt(ev.get('p'), '.2f')}",
         "note": f"needs to be below {EDGE_ALPHA:g} to count as evidence"},
        {"label": "Consistency", "value": years, "note": "calendar years with a positive accuracy score"},
        {"label": "Likely range", "value": f"{signed(ci[0])} to {signed(ci[1])}",
         "note": "95% range of the accuracy score (block bootstrap)"},
        {"label": "Detectable effect", "value": fmt(ev.get("mde"), ".2f"),
         "note": "smallest accuracy score this much history can confirm"},
    ]


def build_market_read(conclusion, run_config=None, nirnay_latest=None) -> dict:
    """Plain-language market read from ``conclusion["oscillator"]`` + ``conclusion["latest"]`` (+ the
    Nirnay breadth read-out). Pure; never raises; no NaN text. Honesty rules: buy/sell/signal wording
    only when the oscillator's status is VALIDATED (``BUY_SELL_RE``); colour shows the side of
    agreement only; levels mode carries the falsified-as-a-forecaster tag."""
    con, rc, nl = _d(conclusion), _d(run_config), _d(nirnay_latest)
    lat, osc = _d(con.get("latest")), _d(con.get("oscillator"))
    fm = con.get("forward_mode")
    if not isinstance(fm, bool):
        fm = rc.get("forward_mode")
    forward = bool(fm) if isinstance(fm, bool) else bool(AARAMBH_FORWARD_SIGNAL)
    h_raw = num(con.get("horizon"))
    h = int(h_raw) if h_raw is not None and h_raw >= 1 else int(AARAMBH_FWD_HORIZON)
    single = con.get("single_system") is True

    status = _str(osc.get("status")).upper()
    status = status if status in OSC_STATUSES else "INSUFFICIENT_DATA"
    validated = status == "VALIDATED"
    value = num(osc.get("value"))
    if value is not None and not -1.0 <= value <= 1.0:
        value = None
    side = _SIDES.get(_str(osc.get("side")).upper()) if value is not None else None
    if value is not None and side is None:
        side = "buy" if value > 0 else "sell" if value < 0 else "split"
    side = side or "none"
    strength = num(osc.get("strength"))
    strength = abs(value) if strength is None and value is not None else strength
    word = agreement_word(strength) if side in ("buy", "sell") else ""

    if side == "buy":
        headline, tag, gauge_label = "Both systems lean buy-side", f"{word} agreement", "Buy-side agreement"
    elif side == "sell":
        headline, tag, gauge_label = "Both systems lean sell-side", f"{word} agreement", "Sell-side agreement"
    elif side == "split":
        headline, tag, gauge_label = "The two systems don't agree today", "no shared side", "Split"
    else:
        headline, tag, gauge_label = "No agreement reading today", "no reading", "No reading"

    a_phrase, a_pill = _aarambh_read(lat, forward)
    n_phrase, n_pill = _nirnay_read(lat, nl, single)
    s_label, s_class, s_text = OSC_STATUS[status]
    as_of = _date(con.get("as_of")) or _date(lat.get("date")) or _date(rc.get("as_of"))
    osc_as_of = _date(osc.get("as_of"))
    as_of_note = ""
    if side == "none":
        as_of_note = ("Nirnay is not running, so there is no second system to agree with." if single
                      else "Nirnay has no stock data for this date.")
    elif osc_as_of and as_of and osc_as_of != as_of:
        as_of_note = f"Agreement reading from {pretty_date(osc_as_of)}, the latest day both systems had data."
    trust_tier = _str(_d(con.get("trust")).get("tier")).upper()
    trust_tier = trust_tier if trust_tier in TRUST_PLAIN else "INSUFFICIENT_DATA"
    ev = _d(osc.get("evidence"))
    return {
        "headline": headline, "tag": tag, "side": side, "validated": validated,
        "value": value, "value_text": signed(value), "strength": strength, "strength_word": word,
        "gauge_label": gauge_label,
        "aarambh_text": a_phrase, "aarambh_pill": a_pill, "nirnay_text": n_phrase, "nirnay_pill": n_pill,
        "subline": f"Aarambh: {a_phrase} · Nirnay: {n_phrase}",
        "as_of": pretty_date(as_of) if as_of else "—", "as_of_note": as_of_note, "horizon": h,
        "status": status, "status_label": s_label, "status_class": s_class, "status_text": s_text.format(h=h),
        "numbers": _osc_numbers(ev, h),
        "trust_tier": trust_tier, "trust_label": TRUST_PLAIN[trust_tier][0], "trust_class": TRUST_PLAIN[trust_tier][1],
        "forward_mode": forward, "levels_note": None if forward else LEVELS_TAG,
        "single_system": single, "synthetic": is_synthetic(rc),
    }


def market_texts(m: dict) -> list[str]:
    """Every string the market read writes itself (what the honesty tests scan)."""
    m = _d(m)
    out = [m.get(k, "") for k in ("headline", "tag", "gauge_label", "subline", "aarambh_text", "aarambh_pill",
                                   "nirnay_text", "nirnay_pill", "as_of", "as_of_note", "value_text",
                                   "status_label", "status_text", "trust_label")]
    for r in m.get("numbers") or []:
        out += [_d(r).get("label", ""), _d(r).get("value", ""), _d(r).get("note", "")]
    return [str(t) for t in out if t]


# ── SVG primitives (inline, no external deps) ────────────────────────────────

_SIDE_COLOR = {"buy": "#34D399", "sell": "#FB7185", "split": "#94A3B8", "none": "#64748B"}


def gauge_svg(value, side: str = "none") -> str:
    """Semicircular −1…+1 gauge: sell-side half tinted red, buy-side half green, a filled arc from 0
    to today's value and a needle. ``value`` None → track only."""
    cx, cy, r = 150.0, 150.0, 118.0

    def pt(v, rr=r):
        th = math.pi * (1.0 - (v + 1.0) / 2.0)
        return cx + rr * math.cos(th), cy - rr * math.sin(th)

    def arc(v0, v1, rr=r):
        (x0, y0), (x1, y1) = pt(v0, rr), pt(v1, rr)
        return f"M {x0:.2f} {y0:.2f} A {rr:.0f} {rr:.0f} 0 0 1 {x1:.2f} {y1:.2f}"

    parts = [
        f'<path d="{arc(-1, 1)}" stroke="rgba(148,163,184,0.10)" stroke-width="20" fill="none" stroke-linecap="round"/>',
        f'<path d="{arc(-1, -0.015)}" stroke="rgba(251,113,133,0.20)" stroke-width="20" fill="none" stroke-linecap="round"/>',
        f'<path d="{arc(0.015, 1)}" stroke="rgba(52,211,153,0.20)" stroke-width="20" fill="none" stroke-linecap="round"/>',
    ]
    for t in (-0.6, -0.3, 0, 0.3, 0.6):          # band edges (strong / moderate / weak) and the split
        (x0, y0), (x1, y1) = pt(t, r + 15), pt(t, r + (24 if t == 0 else 21))
        parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                     f'stroke="rgba(148,163,184,{0.6 if t == 0 else 0.4})" stroke-width="1.5" stroke-linecap="round"/>')
    v = num(value)
    if v is not None and -1.0 <= v <= 1.0:
        color = _SIDE_COLOR.get(side, _SIDE_COLOR["none"])
        if abs(v) >= 0.015:
            lo, hi = (0.0, v) if v > 0 else (v, 0.0)
            parts.append(f'<path d="{arc(lo, hi)}" stroke="{color}" stroke-width="20" fill="none" '
                         f'stroke-linecap="butt" opacity="0.95"/>')
        nx, ny = pt(v, r - 30)
        mx, my = pt(v, r)
        parts += [f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="#E2E8F0" stroke-width="3" '
                  f'stroke-linecap="round"/>',
                  f'<circle cx="{cx}" cy="{cy}" r="7" fill="#E2E8F0"/>',
                  f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="11" fill="{color}" stroke="#0B1020" stroke-width="4"/>']
    else:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="rgba(148,163,184,0.35)"/>')
    return ('<svg class="nk-gauge-svg" viewBox="0 0 300 168" role="img" aria-label="Agreement Oscillator gauge '
            'from minus one (sell-side agreement) to plus one (buy-side agreement)">' + "".join(parts) + "</svg>")


def sparkline_svg(series, y_range=None, zero: bool = False, width: int = 320, height: int = 64,
                  max_points: int = 320) -> str:
    """Tiny inline line chart. ``series`` = [(values, colour), ...]; None/NaN values break the line."""
    clean = []
    for vals, color in series or []:
        vs = [num(v) for v in (list(vals) if vals is not None else [])]
        if len(vs) > max_points:
            step = math.ceil(len(vs) / max_points)
            vs = vs[::step] + ([vs[-1]] if (len(vs) - 1) % step else [])
        clean.append((vs, str(color)))
    finite = [v for vs, _ in clean for v in vs if v is not None]
    if not finite:
        return f'<svg class="nk-spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none"></svg>'
    lo, hi = (y_range if y_range else (min(finite), max(finite)))
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 4

    def y(v):
        return pad + (height - 2 * pad) * (1.0 - (v - lo) / (hi - lo))

    parts = []
    if zero and lo < 0 < hi:
        parts.append(f'<line x1="0" y1="{y(0):.1f}" x2="{width}" y2="{y(0):.1f}" stroke="rgba(148,163,184,0.25)" '
                     f'stroke-width="1" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>')
    for vs, color in clean:
        n = len(vs)
        segs, cur = [], []
        for i, v in enumerate(vs):
            if v is None:
                if cur:
                    segs.append(cur)
                cur = []
                continue
            x = width * (i / (n - 1)) if n > 1 else width / 2
            cur.append(f"{x:.1f},{y(min(max(v, lo), hi)):.1f}")
        if cur:
            segs.append(cur)
        for seg in segs:
            parts.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{esc(color)}" stroke-width="1.6" '
                         'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>')
    return (f'<svg class="nk-spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">'
            + "".join(parts) + "</svg>")


def chip_html(label: str, cls: str = "muted") -> str:
    cls = cls if cls in ("good", "caution", "bad", "muted", "info") else "muted"
    return f'<span class="nk-chip {cls}"><span class="nk-dot"></span>{esc(label)}</span>'


def render_market_hero(m: dict) -> None:
    """The Overview hero: market read (left) + the Agreement Oscillator gauge (right). Every string escaped."""
    m = _d(m)
    side = str(m.get("side", "none"))
    side = side if side in _SIDE_COLOR else "none"
    nums = "".join(
        f'<div class="nk-num-row"><div class="nk-num-label">{esc(_d(r).get("label"))}</div>'
        f'<div class="nk-num-value">{esc(_d(r).get("value"))}</div>'
        f'<div class="nk-num-note">{esc(_d(r).get("note"))}</div></div>' for r in (m.get("numbers") or []))
    details = (f'<details class="nk-details"><summary>See the numbers</summary><div class="nk-num-grid">{nums}</div>'
               f'<div class="nk-num-foot">Strength words: {esc(_AGREEMENT_BANDS_TEXT)}. Full measurement: '
               f'Evidence tab.</div></details>') if nums else ""
    levels = (f'<div class="nk-levels-note">Levels mode — {esc(m.get("levels_note"))}.</div>'
              if m.get("levels_note") else "")
    note = f'<div class="nk-meta-note">{esc(m.get("as_of_note"))}</div>' if m.get("as_of_note") else ""
    tag = esc(m.get("tag", ""))
    st.markdown(
        f'<div class="nk-hero">'
        f'<div class="nk-hero-main">'
        f'<div class="nk-eyebrow">Today\'s market read</div>'
        f'<h2 class="nk-headline">{esc(m.get("headline", "—"))}</h2>'
        f'<div class="nk-tagline {side}">{tag[:1].upper() + tag[1:] if tag else ""}</div>'
        f'<div class="nk-subline"><div><b>Aarambh</b> {esc(m.get("aarambh_text", "—"))}</div>'
        f'<div><b>Nirnay</b> {esc(m.get("nirnay_text", "—"))}</div></div>'
        f'<div class="nk-meta">As of {esc(m.get("as_of", "—"))} · {esc(m.get("horizon", ""))}-trading-day view</div>'
        f'{note}{levels}'
        f'<div class="nk-status">{chip_html(m.get("status_label", "Not enough data"), m.get("status_class", "muted"))}'
        f'<p>{esc(m.get("status_text", ""))}</p></div>{details}'
        f'</div>'
        f'<div class="nk-hero-gauge">'
        f'<div class="nk-gauge-title">Agreement Oscillator</div>'
        f'{gauge_svg(m.get("value"), side)}'
        f'<div class="nk-gauge-readout"><div class="nk-gauge-value {side}">{esc(m.get("value_text", "—"))}</div>'
        f'<div class="nk-gauge-label">{esc(m.get("gauge_label", "No reading"))}</div></div>'
        f'<div class="nk-gauge-scale"><span>−1 sell-side</span><span>0 split</span><span>+1 buy-side</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


_CHIP_STYLE = {
    "solid": ("#34D399", "rgba(52,211,153,0.18)"),
    "modest": ("#34D399", "rgba(52,211,153,0.10)"),
    "inverted": ("#FB7185", "rgba(251,113,133,0.12)"),
    "muted": ("var(--ink-tertiary)", "rgba(148,163,184,0.12)"),
}


def render_hero_card(verdict: dict) -> None:
    """Render a ``build_hero_verdict`` dict. Every text field is escaped here; tolerant of missing keys."""
    v = _d(verdict)
    trust = _d(v.get("trust"))
    fg, bg = _CHIP_STYLE.get(str(trust.get("class", "muted")), _CHIP_STYLE["muted"])
    cls = str(v.get("signal_class", "hold"))
    cls = cls if cls in ("buy", "sell", "hold") else "hold"
    rows_html = "".join(
        f'<div class="hero-evidence-row {esc(_d(r).get("state", "info"))}">'
        f'<span class="hero-tag"><span class="dot"></span>{esc(label_case(_d(r).get("tag", "")))}'
        f' · {esc(_d(r).get("word", ""))}</span>'
        f'<span class="hero-text">{esc(_d(r).get("text", ""))}</span></div>'
        for r in (v.get("evidence") or []))
    banner = (f'<div class="hero-source" style="color:#FB7185;font-weight:700;">{esc(v.get("banner"))}</div>'
              if v.get("banner") else "")
    st.markdown(
        f'<div class="signal-card {cls}">'
        f'<div class="label">Nishkarsh conclusion · as of {esc(v.get("as_of", "—"))}</div>{banner}'
        f'<div class="hero-source">{esc(v.get("mode_label", ""))} · {esc(v.get("sign_note", SIGN_C))}</div>'
        f'<div class="value">{esc(v.get("signal", "—"))} <span class="hero-score">{esc(v.get("score_text", "C —"))}'
        f'</span></div>'
        f'<div class="hero-chip-row"><span class="hero-chip" style="background:{bg};color:{fg};">'
        f'{esc(label_case(trust.get("label", "INSUFFICIENT DATA")))}</span></div>'
        f'<div class="hero-source">{esc(trust.get("text", ""))}</div>'
        f'<div class="hero-headline">{esc(v.get("headline", ""))}</div>'
        f'<div class="hero-evidence">{rows_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_warning_box(title: str, content: str) -> None:
    """Render a themed alert/warning box."""
    st.markdown(
        f"""
        <div class="warning-box">
            <div class="icon"></div>
            <div>
                <div class="title">{html_mod.escape(title)}</div>
                <div class="content">{html_mod.escape(content)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
