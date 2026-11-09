"""
Nishkarsh — the chart palette and chart heights (UI constants; vendored from Tattva 2.8.0).

Kept in the UI layer rather than ``core.config``: nothing outside ``ui/`` draws, and the
engines never need a colour. Every hex here EQUALS its CSS token in ``ui/theme.css``
(dark ramp) — a green line in a plot is the same green as the value in the card beside
it. The light (Paper) ramp lives beside the chart theme in ``ui/theme.py``.

Token hexes are mirrored in five places that must stay in sync: theme.css ``:root``,
``ui.theme.LIGHT_TOKENS``, this module + ``ui.theme._PALETTE_LIGHT``, the table token
sets in ``ui.components`` and ``.streamlit/config.toml``.
"""

from __future__ import annotations

# ── Chart palette — SINGLE SOURCE OF TRUTH (dark / Slate ramp) ───────────────
# Every hue clears WCAG AA (>= 4.5:1) on the chart surface — these are thin 1.5px
# lines, not filled shapes.
_PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    "emerald": (44, 163, 107),   # #2CA36B - Long / Bullish
    "rose":    (221, 90, 90),    # #DD5A5A - Short / Bearish
    "accent":  (76, 125, 240),   # #4C7DF0 - Primary / Interactive (brand, active nav, the read)
    "cyan":    (78, 159, 196),   # #4E9FC4 - Info (Nirnay's hue)
    "amber":   (215, 154, 60),   # #D79A3C - Caution / Warning (caution ONLY, never brand)
    "violet":  (155, 143, 212),  # #9B8FD4 - Secondary / Attribution (Aarambh's hue)
    "slate":   (126, 135, 151),  # #7E8797 - Neutral / Muted
}


def _palette_hex(name: str) -> str:
    r, g, b = _PALETTE_RGB[name]
    return f"#{r:02X}{g:02X}{b:02X}"


def rgba(name: str, alpha) -> str:
    """Semantic chart colour → ``rgba()`` (dark ramp). Prefer ``ui.theme.chart_rgba``,
    which follows the active appearance."""
    r, g, b = _PALETTE_RGB[name]
    return f"rgba({r},{g},{b},{alpha})"


COLOR_GREEN = _palette_hex("emerald")   # #2CA36B
COLOR_RED = _palette_hex("rose")        # #DD5A5A
COLOR_CYAN = _palette_hex("cyan")       # #4E9FC4
COLOR_AMBER = _palette_hex("amber")     # #D79A3C
COLOR_ACCENT = _palette_hex("accent")   # #4C7DF0 — primary/interactive; never warning semantics
COLOR_PURPLE = _palette_hex("violet")   # #9B8FD4

# ── Chart heights (one constant per role; no mobile clamp) ───────────────────
UI_CHART_HEIGHT_MEDIUM = 340
UI_CHART_HEIGHT_LARGE = 380
UI_CHART_HEIGHT_XLARGE = 540     # 2-row level / oscillator
UI_CHART_HEIGHT_STACKED = 680    # 3-row stacked
UI_CHART_HEIGHT_EDGE = 240       # per-period edge bars
UI_CHART_HEIGHT_REGIME = 300     # regime probabilities
