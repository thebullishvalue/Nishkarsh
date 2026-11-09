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

from core.config import (
    AARAMBH_FORWARD_SIGNAL, AARAMBH_FWD_HORIZON, CONV2_TIER_MODERATE, CONV2_TIER_STRONG, CONV2_TIER_WEAK, EDGE_ALPHA,
)
from ui.copy import (
    LEVELS_TAG, NKO_COMPONENT_PLAIN, NKO_PILLAR_PLAIN, NKO_SCALE_TEXT, NKO_STATUS, NKO_TIER_WORD, SIGN_C, SIGN_ENGINE,
    SIGN_Z, SYNTH_BANNER, TIERS_TEXT,
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


# ── Aarambh forecast input (C, κ): pure read-out, shown in Evidence as one component ──

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


def build_forecast_readout(conclusion, aarambh_signal=None, run_config=None) -> dict:
    """Pure read-out of the Aarambh forecast input (the C/κ conclusion) — one component of the
    Nishkarsh read, shown in Evidence. Rules: the figure is the latest C (+ = bullish) with its fixed
    tier/direction; the trust chip is ``conclusion["trust"]`` verbatim; a COLOURED direction only when
    the trust tier is DIRECTIONAL_EDGE or TRADEABLE_EDGE (INVERTED → never); levels mode carries
    "valuation context — falsified as a forecaster"; the Nirnay row is ``conclusion["nirnay"]["text"]``
    verbatim. Never raises on missing keys or NaN; every missing number renders "—"."""
    con, sig, rc = _d(conclusion), _d(aarambh_signal), _d(run_config)
    lat, trust, nir = _d(con.get("latest")), _d(con.get("trust")), _d(con.get("nirnay"))
    fm = con.get("forward_mode")
    if not isinstance(fm, bool):
        fm = rc.get("forward_mode")
    forward = bool(fm) if isinstance(fm, bool) else bool(AARAMBH_FORWARD_SIGNAL)
    h_raw = num(con.get("horizon"))
    h = int(h_raw) if h_raw is not None and h_raw >= 1 else int(AARAMBH_FWD_HORIZON)

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
        headline = ("The directional head has pointed the wrong way on this data — its direction is not "
                    f"used (C {c_txt} is shown as context).")
    elif has_dir and t_tier == "NO_PROVEN_EDGE":
        label = f"LEAN {direction} ({tier}) · unvalidated"
        headline = f"The head leans {dword} (C {c_txt}), but no directional edge is proven on this data — context only."
    elif has_dir:
        label = f"LEAN {direction} ({tier}) · ungraded"
        headline = (f"The head leans {dword} (C {c_txt}), but there is too little out-of-sample history "
                    "to grade an edge — context only.")
    elif C is None:
        label = "NO CONCLUSION"
        headline = "No forecast value for the latest date — no lean."
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
    }


# ── The Nishkarsh read (NKO): the hero every surface infers from (pure; renders, never grades) ──

NKO_STATUSES = ("VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA")
NKO_TIERS = ("STRONG", "MODERATE", "WEAK", "NO LEAN", "NO READ")
# Buy/sell/signal/call wording — allowed in the read's own words ONLY when its status is VALIDATED.
# ("buy-side" / "sell-side" describe agreement and are always allowed.)
BUY_SELL_RE = re.compile(r"\b(buy|sell)\b(?!-side)|\bsignals?\b|\bcalls?\b", re.I)


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


def z_side(z) -> str:
    z = num(z)
    return "none" if z is None else "bull" if z > 0 else "bear" if z < 0 else "flat"


def _lean_word(z) -> str:
    z = num(z)
    return "has no reading" if z is None else "leans bullish" if z > 0 else "leans bearish" if z < 0 else "is flat"


def _mid(label: str) -> str:
    """A label as it reads mid-sentence ("Stock cut crossings" → "stock cut crossings"; "HMM state" kept)."""
    return label if (label[:2].isupper() or label.startswith("Aarambh")) else label[:1].lower() + label[1:]


def component_label(cid, fallback="") -> str:
    return NKO_COMPONENT_PLAIN.get(_str(cid), (_str(fallback) or "An input", ""))[0]


def _drivers_text(drivers) -> str:
    items = [(component_label(_d(c).get("id"), _d(c).get("label")), num(_d(c).get("z")))
             for c in (drivers if isinstance(drivers, (list, tuple)) else [])]
    items = [(lab, z) for lab, z in items if z is not None][:3]
    if not items:
        return "No input has a reading for this date."
    names = [f"{_mid(lab)} ({signed(z, '+.1f')})" for lab, z in items]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    signs = {1 if z > 0 else -1 if z < 0 else 0 for _, z in items}
    tail = ("all on the bullish side" if signs == {1} else "all on the bearish side" if signs == {-1}
            else "pulling in different directions")
    return f"Driven most by {joined} — {tail}."


def _agreement_text(pillars: dict, two_system: bool) -> str:
    td, bu = num(pillars.get("top_down")), num(pillars.get("bottom_up"))
    if not two_system or bu is None:
        tail = f" Top-down {_lean_word(td)} ({signed(td)})." if td is not None else ""
        return ("Single-system read — Nirnay has no data for this date, so only the top-down pillar is "
                "counted." + tail)
    if td is None:
        return f"Single-system read — no top-down reading for this date; bottom-up {_lean_word(bu)} ({signed(bu)})."
    if td * bu > 0:
        return (f"Both systems lean {'bullish' if td > 0 else 'bearish'} — top-down {signed(td)}, bottom-up "
                f"{signed(bu)} — so their agreement strengthens the read.")
    return (f"The two systems disagree — top-down {_lean_word(td)} ({signed(td)}), bottom-up "
            f"{_lean_word(bu)} ({signed(bu)}) — so they partly cancel.")


def _precedent_row(ph, read_dir: int) -> dict:
    """The hero's precedent line from ``precedent["hero"]`` (basis: the NKO state). It VOTES only when its
    exact binomial p < EDGE_ALPHA; otherwise it is description."""
    ph = _d(ph)
    n, pos = num(ph.get("n")), num(ph.get("positive_pct"))
    if not n or n < 1 or pos is None:
        return {"tag": "PRECEDENT", "state": "info", "word": "no data", "votes": False,
                "text": "No past moments like today were found for this run."}
    n_i = int(round(n))
    k = int(round(pos / 100.0 * n_i))
    u, p, med, hp = (num(ph.get(x)) for x in ("unconditional_positive_pct", "binomial_p", "median", "horizon"))
    hz = f"{int(hp)} trading days" if hp is not None and hp >= 1 else "the horizon"
    text = (f"{k}/{n_i} past moments like today rose over the next {hz}, vs {fmt(u, '.0f', suffix='%')} "
            f"unconditional (binomial p={fmt(p, '.2f')}); median {signed(med, '+.2f')}% for the Nifty")
    if not (p is not None and p < EDGE_ALPHA and u is not None):
        return {"tag": "PRECEDENT", "state": "info", "word": "descriptive", "votes": False,
                "text": text + f" — not significant at p < {EDGE_ALPHA:g}, so descriptive only."}
    lean = 1 if pos > u else -1 if pos < u else 0
    word = "rise" if lean > 0 else "fall" if lean < 0 else "neither way"
    if read_dir and lean:
        state, w, rel = (("confirm", "in line", ", in line with today's read") if lean == read_dir
                         else ("conflict", "against", ", against today's read"))
    else:
        state, w, rel = "info", "context", ""
    return {"tag": "PRECEDENT", "state": state, "word": w, "votes": True,
            "text": text + f" — significant; these moments lean to a {word}{rel}."}


def _evidence_numbers(ev: dict, h: int) -> list[dict]:
    """The read's measured evidence in plain labels (values only re-formatted, never re-derived)."""
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
        {"label": "Accuracy score", "value": signed(ev.get("ic"), "+.3f"),
         "note": f"how closely the read has lined up with the Nifty's next {h} trading days (0 = no link)"},
        {"label": "History tested", "value": f"{int(n)} periods" if n is not None else "—",
         "note": f"separate, non-overlapping {h}-day periods with both systems running"},
        {"label": "Significance", "value": f"p = {fmt(ev.get('p'), '.2f')}",
         "note": f"needs to be below {EDGE_ALPHA:g} to count as evidence"},
        {"label": "Consistency", "value": years, "note": "calendar years with a positive accuracy score"},
        {"label": "Likely range", "value": f"{signed(ci[0])} to {signed(ci[1])}",
         "note": "95% range of the accuracy score (block bootstrap)"},
        {"label": "Detectable effect", "value": fmt(ev.get("mde"), ".2f"),
         "note": "smallest accuracy score this much history can confirm"},
    ]


def build_hero_verdict(conclusion, precedent_hero=None, run_config=None) -> dict:
    """The Overview hero, inferred from the Nishkarsh Oscillator ONLY (``conclusion["nko"]``) plus the
    precedent read of the same state (``precedent["hero"]``). Pure; never raises; no NaN text.

    Honesty rules: the headline describes the instrument ("Nishkarsh reads moderately bearish");
    buy/sell/signal/call wording (``BUY_SELL_RE``) and a directional ``call`` only when the read's
    measured status is VALIDATED; the status is always named and ``nko["text"]`` is carried verbatim;
    a single-system date says so; the precedent votes only when its binomial p < EDGE_ALPHA; levels
    mode carries the falsified-as-a-forecaster tag; SYNTHETIC runs carry the banner."""
    con, rc = _d(conclusion), _d(run_config)
    k = _d(con.get("nko"))
    fm = con.get("forward_mode")
    if not isinstance(fm, bool):
        fm = rc.get("forward_mode")
    forward = bool(fm) if isinstance(fm, bool) else bool(AARAMBH_FORWARD_SIGNAL)
    h_raw = num(con.get("horizon"))
    h = int(h_raw) if h_raw is not None and h_raw >= 1 else int(AARAMBH_FWD_HORIZON)

    status = _str(k.get("status")).upper()
    status = status if status in NKO_STATUSES else "INSUFFICIENT_DATA"
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
        headline, tag, gauge_label = "No Nishkarsh read today", "No reading for the latest date", "No reading"
    elif tier == "NO LEAN":
        headline = "Nishkarsh has no lean today"
        tag = f"{v_txt} — inside the ±{CONV2_TIER_WEAK:g} no-lean band"
        gauge_label = "No lean"
    else:
        dword = direction.lower()
        headline = f"Nishkarsh reads {NKO_TIER_WORD[tier]} {dword}"
        tag = f"{v_txt} — a {tier.lower()} {dword} read" + (" · validated directional call" if call else "")
        gauge_label = f"{tier.capitalize()} · {dword}"
    if value is not None and not two:
        tag += " · single-system"

    s_label, s_class, s_text = NKO_STATUS[status]
    as_of = _date(k.get("as_of")) or _date(con.get("as_of")) or _date(rc.get("as_of"))
    run_as_of = _date(con.get("as_of"))
    note = (f"Read from {pretty_date(as_of)}, the latest day with a reading."
            if as_of and run_as_of and as_of != run_as_of else "")
    synth = is_synthetic(rc)
    return {
        "headline": headline, "tag": tag, "side": side, "value": value, "value_text": v_txt,
        "tier": tier, "direction": direction, "call": bool(call), "validated": validated,
        "gauge_label": gauge_label, "two_system": two, "single_system": not two,
        "drivers_text": _drivers_text(k.get("drivers")) if value is not None else "No reading for this date.",
        "agreement_text": _agreement_text(_d(k.get("pillars")), two),
        "precedent": _precedent_row(precedent_hero, read_dir),
        "status": status, "status_label": s_label, "status_class": s_class, "status_text": s_text.format(h=h),
        "measured_text": _str(k.get("text")) or "Not measured in this run.",
        "numbers": _evidence_numbers(_d(k.get("evidence")), h),
        "as_of": pretty_date(as_of) if as_of else "—", "as_of_note": note, "horizon": h,
        "forward_mode": forward, "mode_label": (f"Forward · {h}-day view" if forward
                                                else f"Levels · the Aarambh forecast input is {LEVELS_TAG}"),
        "levels_note": None if forward else LEVELS_TAG,
        "synthetic": synth, "banner": SYNTH_BANNER if synth else None,
    }


def hero_texts(v: dict) -> list[str]:
    """Every string the hero writes itself (what the honesty tests scan; ``measured_text`` is upstream)."""
    v = _d(v)
    out = [v.get(x, "") for x in ("headline", "tag", "gauge_label", "drivers_text", "agreement_text", "as_of",
                                   "as_of_note", "value_text", "status_label", "status_text", "mode_label",
                                   "levels_note")]
    out.append(_d(v.get("precedent")).get("text", ""))
    for r in v.get("numbers") or []:
        out += [_d(r).get("label", ""), _d(r).get("value", ""), _d(r).get("note", "")]
    return [str(t) for t in out if t]


# ── SVG primitives (inline, no external deps) ────────────────────────────────

_SIDE_COLOR = {"bull": "#34D399", "bear": "#FB7185", "flat": "#94A3B8", "none": "#64748B"}


def gauge_svg(value) -> str:
    """Semicircular −100…+100 gauge with the fixed NKO bands (±10 no lean, ±30 moderate, ±60 strong):
    bearish half red, bullish half green, deeper with strength; a filled arc from 0 to today's value and
    a needle. ``value`` None → track only."""
    cx, cy, r = 150.0, 150.0, 118.0

    def pt(v, rr=r):
        th = math.pi * (1.0 - (v + 100.0) / 200.0)
        return cx + rr * math.cos(th), cy - rr * math.sin(th)

    def arc(v0, v1, rr=r):
        (x0, y0), (x1, y1) = pt(v0, rr), pt(v1, rr)
        return f"M {x0:.2f} {y0:.2f} A {rr:.0f} {rr:.0f} 0 0 1 {x1:.2f} {y1:.2f}"

    w, sw = CONV2_TIER_WEAK, 20
    bands = ((-100, -CONV2_TIER_STRONG, "251,113,133", 0.34), (-CONV2_TIER_STRONG, -CONV2_TIER_MODERATE, "251,113,133", 0.22),
             (-CONV2_TIER_MODERATE, -w, "251,113,133", 0.12), (-w, w, "148,163,184", 0.12),
             (w, CONV2_TIER_MODERATE, "52,211,153", 0.12), (CONV2_TIER_MODERATE, CONV2_TIER_STRONG, "52,211,153", 0.22),
             (CONV2_TIER_STRONG, 100, "52,211,153", 0.34))
    parts = [f'<path d="{arc(a + 0.6, b - 0.6)}" stroke="rgba({c},{o})" stroke-width="{sw}" fill="none"/>'
             for a, b, c, o in bands]
    for t in (-CONV2_TIER_STRONG, -CONV2_TIER_MODERATE, -w, 0, w, CONV2_TIER_MODERATE, CONV2_TIER_STRONG):
        (x0, y0), (x1, y1) = pt(t, r + 14), pt(t, r + (23 if t == 0 else 20))
        parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                     f'stroke="rgba(148,163,184,{0.6 if t == 0 else 0.4})" stroke-width="1.5" stroke-linecap="round"/>')
    v = num(value)
    if v is not None and -100.0 <= v <= 100.0:
        color = _SIDE_COLOR[nko_side(v)]
        if abs(v) >= 1.0:
            lo, hi = (0.0, v) if v > 0 else (v, 0.0)
            parts.append(f'<path d="{arc(lo, hi, r)}" stroke="{color}" stroke-width="6" fill="none" '
                         f'stroke-linecap="round" opacity="0.95"/>')
        nx, ny = pt(v, r - 30)
        mx, my = pt(v, r)
        parts += [f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="#E2E8F0" stroke-width="3" '
                  f'stroke-linecap="round"/>',
                  f'<circle cx="{cx}" cy="{cy}" r="7" fill="#E2E8F0"/>',
                  f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="11" fill="{color}" stroke="#0B1020" stroke-width="4"/>']
    else:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="rgba(148,163,184,0.35)"/>')
    return ('<svg class="nk-gauge-svg" viewBox="0 0 300 168" role="img" aria-label="Nishkarsh Oscillator gauge '
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


def render_hero(v: dict) -> None:
    """The Overview hero: the Nishkarsh read (left) + the oscillator gauge (right). Every string escaped."""
    v = _d(v)
    side = str(v.get("side", "none"))
    side = side if side in _SIDE_COLOR else "none"
    pr = _d(v.get("precedent"))
    p_cls = {"confirm": "good", "conflict": "bad"}.get(str(pr.get("state")), "muted")
    nums = "".join(
        f'<div class="nk-num-row"><div class="nk-num-label">{esc(_d(r).get("label"))}</div>'
        f'<div class="nk-num-value">{esc(_d(r).get("value"))}</div>'
        f'<div class="nk-num-note">{esc(_d(r).get("note"))}</div></div>' for r in (v.get("numbers") or []))
    details = (f'<details class="nk-details"><summary>How it was measured</summary>'
               f'<p class="nk-measured">{esc(v.get("measured_text", ""))}</p>'
               + (f'<div class="nk-num-grid">{nums}</div>' if nums else "")
               + f'<div class="nk-num-foot">Bands: {esc(NKO_SCALE_TEXT)}. Rule and full measurement: Evidence tab.'
                 f'</div></details>')
    levels = (f'<div class="nk-levels-note">Levels mode — the Aarambh forecast input is '
              f'{esc(v.get("levels_note"))}.</div>' if v.get("levels_note") else "")
    note = f'<div class="nk-meta-note">{esc(v.get("as_of_note"))}</div>' if v.get("as_of_note") else ""
    st.markdown(
        f'<div class="nk-hero">'
        f'<div class="nk-hero-main">'
        f'<div class="nk-eyebrow">Today\'s Nishkarsh read</div>'
        f'<h2 class="nk-headline">{esc(v.get("headline", "—"))}</h2>'
        f'<div class="nk-tagline {side}">{esc(v.get("tag", ""))}</div>'
        f'<div class="nk-subline"><div><b>Drivers</b> {esc(v.get("drivers_text", "—"))}</div>'
        f'<div><b>Systems</b> {esc(v.get("agreement_text", "—"))}</div>'
        f'<div><b>Precedent</b> {chip_html(str(pr.get("word") or "no data").capitalize(), p_cls)} '
        f'{esc(pr.get("text", "—"))}</div></div>'
        f'<div class="nk-meta">As of {esc(v.get("as_of", "—"))} · {esc(v.get("horizon", ""))}-trading-day view</div>'
        f'{note}{levels}'
        f'<div class="nk-status">{chip_html(v.get("status_label", "Not enough data"), v.get("status_class", "muted"))}'
        f'<p>{esc(v.get("status_text", ""))}</p></div>{details}'
        f'</div>'
        f'<div class="nk-hero-gauge">'
        f'<div class="nk-gauge-title">Nishkarsh Oscillator</div>'
        f'{gauge_svg(v.get("value"))}'
        f'<div class="nk-gauge-readout"><div class="nk-gauge-value {side}">{esc(v.get("value_text", "—"))}</div>'
        f'<div class="nk-gauge-label">{esc(v.get("gauge_label", "No reading"))}</div></div>'
        f'<div class="nk-gauge-scale"><span>−100 bearish</span><span>0</span><span>+100 bullish</span></div>'
        f'<div class="nk-gauge-bands">±{CONV2_TIER_WEAK:g} no lean · ±{CONV2_TIER_MODERATE:g} moderate · '
        f'±{CONV2_TIER_STRONG:g} strong</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_pillar_banner(conclusion, pillar: str) -> None:
    """One line opening a drill-down tab: which pillar of the one read this tab feeds, and today's value."""
    k = _d(_d(conclusion).get("nko"))
    name = NKO_PILLAR_PLAIN.get(pillar, (pillar,))[0]
    z = num(_d(k.get("pillars")).get(pillar))
    today = (f"today {signed(z)}" if z is not None else
             "no reading today" + (" (single-system run)" if k.get("two_system") is not True else ""))
    read = num(k.get("value"))
    tail = f" · the read itself is {signed(read, '+.0f')}" if read is not None else ""
    st.markdown(f'<div class="nk-feeds"><span class="nk-feeds-dot {z_side(z)}"></span><span>Feeds the '
                f'<b>{esc(name)}</b> pillar of the Nishkarsh read — {esc(today)}{esc(tail)}. The Overview shows '
                f'the one read; this tab is its drill-down.</span></div>', unsafe_allow_html=True)


_CHIP_STYLE = {
    "solid": ("#34D399", "rgba(52,211,153,0.18)"),
    "modest": ("#34D399", "rgba(52,211,153,0.10)"),
    "inverted": ("#FB7185", "rgba(251,113,133,0.12)"),
    "muted": ("var(--ink-tertiary)", "rgba(148,163,184,0.12)"),
}


def render_forecast_card(verdict: dict) -> None:
    """Render a ``build_forecast_readout`` dict. Every text field is escaped here; tolerant of missing keys."""
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
        f'<div class="label">Aarambh forecast input · as of {esc(v.get("as_of", "—"))}</div>{banner}'
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
