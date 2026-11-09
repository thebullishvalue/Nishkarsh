"""
Nishkarsh — user-facing copy for the pages, single-sourced. Nothing here computes anything.

Seven pages (Overview · Convergence · Aarambh · Nirnay · Precedent · System · Data) read ONE
inference, the Nishkarsh Oscillator (NKO). The read's own wording (lean words, pillar and input
names, the levels-mode tag, the synthetic banner) is single-sourced in ``convergence.verdict``
— the layer that builds the verdict — and re-exported here for the pages. Long "how to read"
text belongs in metric-card TOOLTIPS and section descriptions, not on the face.

Honesty: buy/sell/signal/call wording appears in a read's own words only when its measured
status is VALIDATED (``ui.components.BUY_SELL_RE``). Levels mode carries ``LEVELS_TAG``.
"""

from __future__ import annotations

import re

from core.config import (
    AARAMBH_FWD_HORIZON,
    CONV2_TIER_MODERATE,
    CONV2_TIER_STRONG,
    CONV2_TIER_WEAK,
    CONV2_ZMIN_PERIODS,
    EDGE_ALPHA,
    NIRNAY_POLARITY,
)
from convergence.verdict import (  # noqa: F401 — re-exported: the read's wording lives with the verdict
    LEVELS_TAG,
    NKO_COMPONENT_PLAIN,
    NKO_TIER_WORD,
    PILLAR_PHRASE,
    PILLAR_SUBJECT,
    SYNTH_BANNER,
)

# ── Nirnay polarity in words (core.config.NIRNAY_POLARITY — ONE switch) ──────
__all__ = [
    "NIRNAY_READ", "NIRNAY_READ_SENTENCE", "FORBIDDEN_FORWARD", "EDGE_WORDS", "NKO_BANDS_TEXT", "NKO_STATUS",
    "NKO_PILLAR_PLAIN", "PRECEDENT_METHOD", "METHOD_LINES", "GLOSSARY", "DATA_COLS_FORWARD", "DATA_COLS_LEVELS",
    "data_cols", "forbidden_hits",
    # re-exported from convergence.verdict (the read's own wording)
    "LEVELS_TAG", "NKO_COMPONENT_PLAIN", "NKO_TIER_WORD", "PILLAR_PHRASE", "PILLAR_SUBJECT", "SYNTH_BANNER",
]

_TREND = NIRNAY_POLARITY == "trend"
NIRNAY_READ = "read as trend (strength = bullish)" if _TREND else "read contrarian (oversold = bullish)"
NIRNAY_READ_SENTENCE = (
    "Nirnay is read as trend: strength in the stocks counts as bullish, weakness as bearish."
    if _TREND else
    "Nirnay is read contrarian: oversold stocks count as bullish, overbought as bearish.")


# Words that describe a valuation LEVEL model; they are false about the forward forecast.
FORBIDDEN_FORWARD = ("fair value", "fair-value", "residual", "cheap", "expensive",
                     "primary trade signal", "trust the signal", "undervalued", "overvalued")
# Edge words the UI must never emit on its own authority.
EDGE_WORDS = ("calibrated", "confirms", "adds value", "durable", "validated")

# ── the Nishkarsh read ───────────────────────────────────────────────────────
NKO_BANDS_TEXT = (f"±{CONV2_TIER_WEAK:g} no lean · ±{CONV2_TIER_MODERATE:g} moderate · "
                  f"±{CONV2_TIER_STRONG:g} strong")
# status → (label, chip class, one plain sentence)
NKO_STATUS = {
    "VALIDATED": ("Validated", "good",
                  "Its readings have lined up with the Nifty's next {h} trading days on this data, clearing the "
                  "bar that was set before it was tested."),
    "UNPROVEN": ("Unproven", "caution",
                 "It describes what the two systems see today, not where the market goes next: its readings have "
                 "not yet lined up with the Nifty's next {h} trading days strongly or consistently enough to "
                 "rely on."),
    "INSUFFICIENT_DATA": ("Not enough data", "muted",
                          "There is not enough history with both systems running to measure it yet."),
}
# pillar → (name, plain title, one sentence of what it measures)
NKO_PILLAR_PLAIN = {
    "top_down": ("Top-down", "Aarambh's forecast of the Nifty",
                 f"Aarambh's walk-forward forecast of the next {AARAMBH_FWD_HORIZON} trading days, and how "
                 "stretched that forecast is across lookbacks."),
    "bottom_up": ("Bottom-up", "What the Nifty's stocks are doing",
                  "Where the Nifty's stocks stand on their own oscillators: zones, regimes, bull-versus-bear "
                  "odds and fresh crossings."),
    "agreement": ("Agreement", "Do the two systems lean the same way",
                  "Whether the top-down and bottom-up reads point the same way, and how fully they agree."),
}
# ── Precedent ────────────────────────────────────────────────────────────────
PRECEDENT_METHOD = ("Matches today's read, its pillars and its 5- and 20-day change against past dates whose "
                    "outcomes were already known; one match per episode. A base rate counts as evidence only "
                    f"when its binomial p is below {EDGE_ALPHA:g}.")

# ── Reliability: method & limits, glossary ───────────────────────────────────
METHOD_LINES = (
    ("How the read is built",
     "Ten inputs — four from Aarambh, five from Nirnay's stocks, one for their agreement — are each "
     f"standardized against their own past only (at least {CONV2_ZMIN_PERIODS} earlier sessions), averaged "
     "into three pillars, and the pillars averaged into one number rescaled to ±100. No weight or sign is "
     "fitted to returns."),
    ("Nirnay's polarity",
     f"{NIRNAY_READ_SENTENCE} This is an owner decision (version 2.3.0), recorded with its evidence in "
     "docs/research/NIRNAY_POLARITY_CHANGE.md."),
    ("Measured before promoted",
     "Every claim on this page is graded by a rule written before the test. A reading is only described as "
     "directional once it passes; until then it describes what the systems see, not where the market goes."),
    ("Out-of-sample",
     "Aarambh is refitted walk-forward with a purge gap, so every forecast was made without seeing its own "
     "outcome. Accuracy is measured on separate, non-overlapping periods."),
    ("Survivorship",
     "The stock universe is today's Nifty 50 applied to the past: stocks that left the index are missing "
     "from history, which flatters the bottom-up read."),
    ("Data",
     "Index and stock prices come from Yahoo Finance, falling back to NSE's official archives; your sheet "
     "supplies the valuation and macro series. The Data page shows each source's freshness."),
    ("Not investment advice",
     "Nishkarsh describes and measures; it does not recommend trades."),
)
GLOSSARY = (
    ("Accuracy (IC)", "Rank correlation between a reading and the Nifty's next move: 0 = no link, +1 = perfect."),
    ("p-value", f"How surprising the result would be if there were no real link; below {EDGE_ALPHA:g} counts "
                "as evidence."),
    ("Smallest detectable", "The smallest accuracy this much history could confirm (80% power)."),
    ("Causal", "Computed from earlier data only — no value uses information from its own future."),
    ("Pillar", "Top-down, bottom-up or agreement: the average of its inputs, each standardized."),
    ("Single-system", "A date without stock data: only the top-down pillar counts, and it is left out of "
                      "the measurement."),
    ("Lift", "Accuracy on days the systems agree minus accuracy on days they disagree."),
    ("κ (kappa)", "Shrink factor on the Aarambh input: 1 = unchanged; below 1 only on disagreement days, and "
                  "only after the lift test passes."),
    ("HAC t", "A t-statistic corrected for overlapping, autocorrelated days."),
    ("Break-even", "The trading cost per trade that would erase a strategy's return."),
)

# ── Data page: engine columns shown per Aarambh mode ─────────────────────────
DATA_COLS_FORWARD = ("Date", "Valid", "Forecast", "Label", "ForecastZ", "ConvictionScore", "AvgZ",
                     "OversoldBreadth", "OverboughtBreadth", "ModelSpread", "Regime")
DATA_COLS_LEVELS = ("Date", "Valid", "Actual", "FairValue", "Residual", "ModelSpread", "AvgZ",
                    "OversoldBreadth", "OverboughtBreadth", "ConvictionScore", "ConvictionUpper",
                    "ConvictionLower", "Regime", "BullishDiv", "BearishDiv")


def data_cols(forward_mode: bool) -> tuple[str, ...]:
    return DATA_COLS_FORWARD if bool(forward_mode) else DATA_COLS_LEVELS


def forbidden_hits(text: str, forward_mode: bool) -> list[str]:
    """Forward-mode forbidden phrases present in ``text`` (empty in levels mode)."""
    if not forward_mode:
        return []
    t = str(text).lower()
    return [w for w in FORBIDDEN_FORWARD if re.search(r"\b" + re.escape(w) + r"\b", t)]
