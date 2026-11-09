"""
Nishkarsh v2 — user-facing copy, single-sourced (A4-08, A6-08, ruling R3/R8).

One table per Aarambh mode: FORWARD (the product path — Aarambh forecasts the h-day forward
change) and LEVELS (AARAMBH_FORWARD_SIGNAL=False — a valuation regression, labelled
"valuation context — falsified as a forecaster" wherever shown). Tabs read their wording
from ``mode_copy(forward_mode)``; nothing here computes anything.

Sign conventions (R3), stated on every axis/card that shows them:
  * conclusion C, zA, zN, ForecastZ:  + = bullish
  * Aarambh engine ConvictionScore / ConvictionBounded / AvgZ / Z_*:  − = bullish
"""

from __future__ import annotations

import re

from core.config import (
    NIRNAY_POLARITY,
    AARAMBH_FWD_HORIZON,
    AARAMBH_FWD_MOM_K,
    BACKTEST_REFERENCE_BPS,
    CONV2_TAU,
    CONV2_TIER_MODERATE,
    CONV2_TIER_STRONG,
    CONV2_TIER_WEAK,
    EDGE_ALPHA,
    EDGE_TRADEABLE_T,
)

# ── Nirnay polarity in words (core.config.NIRNAY_POLARITY — ONE switch) ──────
_TREND = NIRNAY_POLARITY == "trend"
NIRNAY_READ = "read as trend (strength = bullish)" if _TREND else "read contrarian (oversold = bullish)"
NIRNAY_READ_SHORT = "trend: strength = +" if _TREND else "contrarian: oversold = +"

# ── sign conventions ─────────────────────────────────────────────────────────
SIGN_C = "C in ±100 · + = bullish"
SIGN_Z = "+ = bullish"
SIGN_ENGINE = "− = bullish"
LEVELS_TAG = "valuation context — falsified as a forecaster"
TIERS_TEXT = (f"tiers |C| ≥ {CONV2_TIER_STRONG:g} STRONG · ≥ {CONV2_TIER_MODERATE:g} MODERATE · "
              f"≥ {CONV2_TIER_WEAK:g} WEAK · else NO LEAN (fixed)")
EDGE_SCALE_TEXT = (f"One threshold scale (core/config): significant = one-sided p < {EDGE_ALPHA:g}; "
                   f"tradeable = HAC t and non-overlap t ≥ {EDGE_TRADEABLE_T:g} net of "
                   f"{BACKTEST_REFERENCE_BPS:g} bps.")
SYNTH_BANNER = "SYNTHETIC DEMO DATA — not market data; no conclusion here is about the real market."

# Words that describe a valuation LEVEL model; they are false about the forward forecast.
FORBIDDEN_FORWARD = ("fair value", "fair-value", "residual", "cheap", "expensive",
                     "primary trade signal", "trust the signal", "undervalued", "overvalued")
# Edge words (R4) the UI must never emit on its own authority.
EDGE_WORDS = ("calibrated", "confirms", "adds value", "durable", "validated")

# ── masthead / landing ───────────────────────────────────────────────────────
TAGLINE = "Two systems. One conclusion — a measured claim, with the evidence shown."
TAGLINE_EXPLAINER = (
    "Aarambh (top-down forecast) is the directional head. Nirnay (bottom-up constituent breadth) "
    "can only shrink conviction on days it contradicts — and only once a walk-forward Convergence-"
    "Lift test shows agreement improves accuracy. Until the data prove that, Nirnay is shown as "
    "context only and the conclusion is Aarambh's.")

LANDING_CARDS = (
    ("aarambh", "AARAMBH",
     f"Purged walk-forward ensemble forecasting the {AARAMBH_FWD_HORIZON}-day forward change of the "
     f"target from {AARAMBH_FWD_MOM_K}-day predictor momentum. Its causally z-scored forecast is the "
     "directional head.",
     (("Ensemble", "Ridge + Huber + ElasticNet + PCA-OLS (equal weight)"),
      ("Validation", "purged walk-forward, out-of-sample only"),
      ("Sign", f"engine conviction {SIGN_ENGINE}"))),
    ("nirnay", "NIRNAY",
     "Per-constituent MSF + MMR oscillator with online HMM, GARCH and CUSUM regime read-outs, "
     "aggregated each day over the names with live, warmed-up data.",
     (("Head", f"mean oscillator, {NIRNAY_READ}"),
      ("Role", "context only until the lift test passes"),
      ("Coverage", "every % is of live names that day"))),
    ("convergence", "CONCLUSION",
     "Aarambh leads. Nirnay can only shrink conviction (×κ ≤ 1) on days it contradicts, and only "
     "when a walk-forward lift test passes — otherwise it is context only.",
     (("Scale", SIGN_C),
      ("Tiers", f"{CONV2_TIER_STRONG:g} / {CONV2_TIER_MODERATE:g} / {CONV2_TIER_WEAK:g} (fixed)"),
      ("Evidence", "Edge Ledger: IC, HAC t, bootstrap CI, cost-aware backtest"))),
)


# ── product copy: Overview, landing, sidebar (plain language — no internal symbols) ──
PRODUCT_TITLE = "Nishkarsh"
PRODUCT_TAGLINE = "Two independent reads of the Nifty 50, folded into one number."
LANDING_INTRO = (
    "Nishkarsh reads the market two ways: a top-down forecast (Aarambh) and a bottom-up scan of the "
    "Nifty's stocks (Nirnay). Both reads, and how far they agree, are folded into one series: the "
    "Nishkarsh Oscillator. Every surface shows that one read, what drives it, and how far it has "
    "earned your trust on past data.")
LANDING_STEPS = (
    ("Choose your data", "Connect your Google Sheet, upload a CSV or Excel file, or try the built-in demo."),
    ("Run the analysis", "Press Run analysis in the sidebar. Real data takes about a minute."),
    ("Read the overview", "Today's Nishkarsh read, what drives it, and how reliable it has been."),
)
LANDING_FOOTNOTE = "Nothing here is called a buy or sell signal until the data prove it. Not investment advice."
ABOUT_LINES = (
    "Nishkarsh (निष्कर्ष, “conclusion”) reads the Nifty 50 in two independent ways.",
    f"Aarambh is top-down: a model that forecasts the next {AARAMBH_FWD_HORIZON} trading days from the "
    "series in your sheet.",
    "Nirnay is bottom-up: it checks every Nifty stock for overbought and oversold conditions.",
    "The Nishkarsh Oscillator folds both reads, and how far they agree, into one number from −100 to +100.",
    "Every claim is graded on history the models never saw; the Evidence tab shows the figures.",
    "Not investment advice.",
)
INTEL_HELP_PLAIN = (
    "When on, days where Nirnay disagrees with Aarambh can reduce the Aarambh forecast input's conviction — "
    "but only if this run's data show that agreement improves accuracy. Otherwise on and off give the same "
    "result. Nirnay can never add conviction or flip the direction.")

# ── the Nishkarsh Oscillator (NKO): the one read every surface infers from ──
NKO_SCALE_TEXT = (f"−100 to +100, + = bullish · within ±{CONV2_TIER_WEAK:g} no lean · "
                  f"±{CONV2_TIER_MODERATE:g} moderate · ±{CONV2_TIER_STRONG:g} strong (fixed bands)")
NKO_TIER_WORD = {"STRONG": "strongly", "MODERATE": "moderately", "WEAK": "mildly"}
# status → (chip label, chip class, one plain sentence). Buy/sell/signal/call words are reserved for VALIDATED.
NKO_STATUS = {
    "VALIDATED": ("Validated", "good",
                  "Its readings have lined up with the Nifty's next {h} trading days on this data, clearing "
                  "the bar that was set before it was tested."),
    "UNPROVEN": ("Unproven", "caution",
                 "This describes what the two systems read today, not where the market goes next. Its track "
                 "record is not yet strong enough to act on."),
    "INSUFFICIENT_DATA": ("Not enough data", "muted",
                          "There is not enough history with both systems running to measure this read yet."),
}
NKO_PILLAR_PLAIN = {
    "top_down": ("Top-down", "Aarambh", "the forecast and how stretched it is across lookbacks"),
    "bottom_up": ("Bottom-up", "Nirnay", "the Nifty's stocks: oscillator, zones, regimes, HMM and cut crossings"),
    "agreement": ("Agreement", "Both", "whether the two systems lean the same way, and how fully"),
}
# component id → (plain label, what it reads). + = bullish for every component.
NKO_COMPONENT_PLAIN = {
    "forecast": ("Aarambh's forecast", "the conclusion over time: Aarambh's causal forecast z"),
    "extension": ("Forecast extension", "how stretched the forecast is across lookbacks (−AvgZ)"),
    "zone_breadth_a": ("Lookback zone breadth", "lookbacks in the bullish zone minus the bearish zone"),
    "crossings_a": ("Lookback cut crossings", "bullish minus bearish cut crossings across lookbacks"),
    "oscillator_n": ("Stock oscillator", f"the stocks' mean oscillator ({NIRNAY_READ_SHORT})"),
    "zone_breadth_n": ("Stock zone breadth", "% of stocks overbought minus % oversold" if _TREND
                       else "% of stocks oversold minus % overbought"),
    "regime_mix": ("Regime composition", "% of stocks in a bull regime minus a bear regime" if _TREND
                   else "% of stocks in a bear regime minus a bull regime, read contrarian"),
    "hmm_state": ("HMM state", "mean bull minus bear HMM probability" if _TREND
                  else "mean bear minus bull HMM probability, read contrarian"),
    "cut_crossings_n": ("Stock cut crossings", "overbought-side minus oversold-side cut crossings, per live stock"
                        if _TREND else "oversold-side minus overbought-side cut crossings, per live stock"),
    "signed_agreement": ("Signed agreement", "consensus × agreement ratio (the agreement anatomy)"),
}
NKO_DOCS = ("docs/research/NISHKARSH_OSCILLATOR_PREREG.md", "docs/research/NISHKARSH_OSCILLATOR_RESULTS.md")

# The signed agreement (former Agreement Oscillator) — now the Agreement pillar's input; Evidence only.
OSC_SCALE_TEXT = ("−1: both systems lean sell-side, fully aligned · 0: they disagree · "
                  "+1: both lean buy-side, fully aligned")
OSC_STATUS = {
    "VALIDATED": ("Validated", "good", ""),
    "UNPROVEN": ("Unproven", "caution", ""),
    "INSUFFICIENT_DATA": ("Not enough data", "muted", ""),
}
# ── per-mode tables ──────────────────────────────────────────────────────────
_FORWARD = {
    "key": "forward",
    "badge": f"Forward mode · the {AARAMBH_FWD_HORIZON}-day forecast is the directional head",
    "head_name": "forecast",
    "quality_desc": ("Magnitude fit of the out-of-sample forecast. A return forecast's R² sits near "
                     "zero by nature; direction is graded in Evidence → Edge ledger."),
    "r2_label": "OOS R² (MAGNITUDE)",
    "r2_sub": "near 0 is expected for a return forecast — not the verdict",
    "r2_tip": ("Share of the realized forward change's variance explained by the forecast on "
               "out-of-sample rows. Structurally near zero even for a useful directional signal."),
    "r2rw_label": "SKILL vs ZERO FORECAST",
    "r2rw_sub": "1 − SSE / SSE(zero-change forecast); < 0 = predicting no change did better",
    "r2rw_tip": ("1 − SSE(forecast) / SSE(naive), naive = 0 (a martingale: no change expected), "
                 "on every purge-th labelled out-of-sample row. Magnitude only — not the edge verdict."),
    "hurst_sub": "DFA Hurst of the forecast-extension increments · descriptive only",
    "spread_label": "ENSEMBLE SPREAD",
    "spread_sub": "std of member forecasts, in σ of the forecast itself",
    "edge_title": "DIRECTIONAL EDGE — GRADED BY THE CONCLUSION LAYER",
    "anchor_title": "Forecast vs Realized Forward Change",
    "anchor_desc": (f"Out-of-sample {AARAMBH_FWD_HORIZON}d forecast vs the realized {AARAMBH_FWD_HORIZON}d "
                    "change (the label). The unrealized tail has no label yet."),
    "anchor_a": "Realized change (label)",
    "anchor_b": "Forecast (out-of-sample)",
    "anchor_lower": f"Forecast z (causal) · {SIGN_Z}",
    "anchor_y": f"{AARAMBH_FWD_HORIZON}d forward Δlog",
    "conv_title": f"Engine Conviction ({SIGN_ENGINE})",
    "conv_desc": ("−100·tanh(ForecastZ/2): the forecast z-score mapped to ±100. No band is drawn — "
                  "the engine does not produce a calibrated band in forward mode."),
    "ext_title": "Forecast Extension Diagnostics",
    "ext_desc": ("Conformal z-scores of the (sign-flipped) forecast across lookbacks — how extended the "
                 f"forecast is vs its own recent range. Extension diagnostics, not the verdict · {SIGN_ENGINE}."),
    "breadth_bull": "Bullish-extended windows %",
    "breadth_bear": "Bearish-extended windows %",
    "avgz_desc": f"Mean conformal z across lookbacks ({SIGN_ENGINE}); beyond ±2 is extended.",
    "lookback_desc": f"Per-lookback conformal z and zone of the forecast extension ({SIGN_ENGINE}).",
    "freq_desc": "Threshold crossings of the extension z per lookback (descriptive).",
    "regime_title": "Extension State",
    "regime_desc": "Where today's forecast extension sits across lookbacks; history share is descriptive.",
    "zone_bull": "Bullish ext.",
    "zone_bear": "Bearish ext.",
    "zone_mid": "Neutral",
    "regime_bull": "BULLISH-EXTENDED",
    "regime_bear": "BEARISH-EXTENDED",
    "feature_title": "Feature Impact on the Forecast",
    "feature_desc": (f"How much each predictor's {AARAMBH_FWD_MOM_K}-day momentum moves the forecast now "
                     "(engine attribution)."),
    "levels_only_note": ("Stationarity / mean-reversion tests and the engine's hit-rate table describe a "
                         "LEVELS regression and are shown in levels mode only."),
    "data_cols": ("Date", "Valid", "Forecast", "Label", "ForecastZ", "ConvictionScore", "AvgZ",
                  "OversoldBreadth", "OverboughtBreadth", "ModelSpread", "Regime"),
}

_LEVELS = {
    "key": "levels",
    "badge": f"Levels mode · {LEVELS_TAG}",
    "head_name": "valuation head",
    "quality_desc": (f"Fit of the walk-forward fair-value regression — {LEVELS_TAG}. Level fit says "
                     "nothing about forward direction; see Evidence → Edge ledger."),
    "r2_label": "OOS R² (LEVEL FIT)",
    "r2_sub": "variance of the level explained out-of-sample — not a forecast grade",
    "r2_tip": ("Share of the target LEVEL's variance explained on out-of-sample rows. A high level R² "
               "is easy for persistent series and is not evidence of directional skill."),
    "r2rw_label": "SKILL vs RANDOM WALK",
    "r2rw_sub": "1 − SSE / SSE(yesterday's level); < 0 = a random walk did better",
    "r2rw_tip": ("1 − SSE(model) / SSE(naive), naive = the previous day's level (random walk), on "
                 "out-of-sample rows."),
    "hurst_sub": "DFA Hurst of the residual's increments · descriptive only",
    "spread_label": "MODEL SPREAD",
    "spread_sub": "std of member fair values, in σ of the residual",
    "edge_title": f"EDGE — {LEVELS_TAG.upper()}",
    "anchor_title": "Actual vs Walk-Forward Fair Value",
    "anchor_desc": f"Actual level vs the out-of-sample fair-value estimate — {LEVELS_TAG}.",
    "anchor_a": "Actual",
    "anchor_b": "Fair value (out-of-sample)",
    "anchor_lower": f"Residual (actual − fair value) · {SIGN_ENGINE}",
    "anchor_y": "",
    "conv_title": f"DDM-Filtered Conviction ({SIGN_ENGINE})",
    "conv_desc": (f"Evidence-accumulated conviction with its uncertainty band (not a calibrated "
                  f"confidence interval) — {LEVELS_TAG}."),
    "ext_title": "Valuation Extension Diagnostics",
    "ext_desc": f"Conformal z-scores of the residual across lookbacks ({SIGN_ENGINE}) — {LEVELS_TAG}.",
    "breadth_bull": "Undervalued windows %",
    "breadth_bear": "Overvalued windows %",
    "avgz_desc": f"Mean conformal z across lookbacks ({SIGN_ENGINE}); beyond ±2 is extended.",
    "lookback_desc": f"Per-lookback conformal z and zone of the residual ({SIGN_ENGINE}).",
    "freq_desc": "Threshold crossings of the residual z per lookback (descriptive).",
    "regime_title": "Valuation State",
    "regime_desc": f"Where today's residual sits across lookbacks — {LEVELS_TAG}.",
    "zone_bull": "Undervalued",
    "zone_bear": "Overvalued",
    "zone_mid": "Fair",
    "regime_bull": "OVERSOLD",
    "regime_bear": "OVERBOUGHT",
    "feature_title": "Feature Impact on Fair Value",
    "feature_desc": "How much each model input shifts the fair-value estimate now (engine attribution).",
    "levels_only_note": "",
    "data_cols": ("Date", "Valid", "Actual", "FairValue", "Residual", "ModelSpread", "AvgZ",
                  "OversoldBreadth", "OverboughtBreadth", "ConvictionScore", "ConvictionUpper",
                  "ConvictionLower", "Regime", "BullishDiv", "BearishDiv"),
}

CONVERGENCE = {
    "title": "Component: the Aarambh forecast input (C, κ)",
    "desc": (f"C = 100·tanh(zA/2)·κ · {SIGN_C}. Aarambh's causal z (zA) is the head; Nirnay's zN can only "
             f"shrink it (κ ≤ 1) on CONTRADICT days, and only after the lift test passes."),
    "c_axis": f"C ({SIGN_Z})",
    "z_axis": f"z ({SIGN_Z})",
    "kappa_axis": "κ (1 = no shrink)",
    "state_desc": (f"State: CONFIRM / CONTRADICT when |zN| ≥ {CONV2_TAU:g} with the same / opposite sign "
                   "as zA; NEUTRAL when |zN| < τ; NO_DATA when Nirnay has no live data that day."),
    "anatomy_title": "Agreement Anatomy",
    "anatomy_desc": ("Four fixed-weight agreement dimensions, computed only on dates where Nirnay has warm "
                     "data. Their ratio × the consensus is the signed agreement, the Agreement pillar's input."),
}

NIRNAY = {
    "avg_tip": ("Mean Unified oscillator over the names with live data that day (±10 scale). The "
                f"Nirnay head is {NIRNAY_READ}: zN = causal z of ±Avg_Signal (+ = bullish)."),
    "hmm_desc": ("Mean of the per-stock online-HMM probabilities across live names — a cross-sectional "
                 "average, not an index-level regime model. Gaps = no live data."),
    "zone_desc": ("Share of LIVE names whose oscillator is beyond its own adaptive cut (OS_Cut / OB_Cut). "
                  "Descriptive breadth."),
    "signals_desc": "Per-stock oscillator cut crossings counted across live names (descriptive).",
    "regime_desc": "Per-stock regime labels as a share of live names (WEAK_* folded into Bull / Bear).",
    "coverage_desc": "Live (warm, real-bar) names vs the constituent universe — every % uses the live count.",
}

PRECEDENT = {
    "desc": ("The past dates whose Nishkarsh state — the read, its three pillars and its 20-day change — most "
             "resembled today's (covariance-aware Mahalanobis matching, one match per window), and what the "
             "Nifty did next. Computed once per run."),
    "base_rate": ("A descriptive base rate: the share of these moments followed by a rise, against the "
                  "unconditional rise rate over the same history, with an exact two-sided binomial p. "
                  f"It counts as evidence only when p < {EDGE_ALPHA:g}; until then it is shown in neutral colours."),
    "similarity_note": "relative within this candidate pool (1 = the closest possible)",
}


def mode_copy(forward_mode: bool) -> dict:
    """The copy table for the displayed run's Aarambh mode."""
    return _FORWARD if bool(forward_mode) else _LEVELS


def zone_label(zone, forward_mode: bool) -> str:
    """Engine zone strings ('Undervalued' / 'Overvalued' / 'Fair') in the mode's words."""
    c = mode_copy(forward_mode)
    z = str(zone or "")
    if "Under" in z:
        return c["zone_bull"]
    if "Over" in z:
        return c["zone_bear"]
    return c["zone_mid"] if z and z.lower() not in ("nan", "none") else "—"


def regime_label(regime, forward_mode: bool) -> str:
    """Engine regime strings (STRONGLY OVERSOLD … OVERBOUGHT) in the mode's words."""
    c = mode_copy(forward_mode)
    r = str(regime or "").upper()
    strong = "STRONGLY " if "STRONG" in r else ""
    if "OVERSOLD" in r:
        return strong + c["regime_bull"]
    if "OVERBOUGHT" in r:
        return strong + c["regime_bear"]
    return "NEUTRAL" if r and r not in ("N/A", "NAN", "NONE") else "N/A"


def forbidden_hits(text: str, forward_mode: bool) -> list[str]:
    """Forward-mode forbidden phrases present in ``text`` (empty in levels mode)."""
    if not forward_mode:
        return []
    t = str(text).lower()
    return [w for w in FORBIDDEN_FORWARD if re.search(r"\b" + re.escape(w) + r"\b", t)]
