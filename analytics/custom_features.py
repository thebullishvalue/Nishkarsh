"""
Nishkarsh — Custom engineered predictors (causal, stationary).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Builds economically-motivated derived features (yield spreads, real
rates, credit/commodity ratios, FX momentum, cross-asset composites) from a
combined panel of sheet columns + macro tickers. Every feature is STATIONARY
(native spread, causal rolling z-score, or momentum) and CAUSAL (rolling stats
use rows ≤ t only — no future), so the output is backtest-safe and won't make a
level regression extrapolate. See docs/CUSTOM_PREDICTORS.md for the full spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features that embed the Aarambh target (PE) or its sibling valuation ratios.
# Feeding these into Aarambh is TARGET LEAKAGE (a model "predicting" PE from
# 1/PE), not signal — so callers exclude them from the Aarambh predictor set.
# They are valid for Nirnay/MMR (whose target is stock price, not index PE).
AARAMBH_LEAKAGE_EXCLUDE = frozenset({"EQ_RISK_PREMIUM", "IMPLIED_ROE", "DY_SPREAD"})

_Z_WINDOW = 252


def _z(s: pd.Series, w: int = _Z_WINDOW) -> pd.Series:
    """Causal rolling z-score (rows ≤ t only), clipped ±5."""
    mp = max(w // 4, 20)
    m = s.rolling(w, min_periods=mp).mean()
    sd = s.rolling(w, min_periods=mp).std()
    return ((s - m) / sd.replace(0, np.nan)).clip(-5.0, 5.0)


def _lz(num: pd.Series, den: pd.Series, w: int = _Z_WINDOW) -> pd.Series:
    """z-score of log(num/den) — stationary ratio in z-space."""
    r = np.log(num.replace(0, np.nan) / den.replace(0, np.nan))
    return _z(r, w)


def build_custom_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute every custom predictor whose inputs are present in ``panel``.

    ``panel`` is a date-indexed frame that may contain sheet columns
    (IN10Y, REPO, INIRYY, REL_BREADTH, NIFTY50_PE, …) and/or macro tickers
    (HYG, LQD, GC=F, HG=F, DX-Y.NYB, INR=X, …). Missing-input features are
    skipped silently. Returns a date-indexed frame of the computed features.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(index=getattr(panel, "index", None))

    p = panel.apply(pd.to_numeric, errors="coerce")
    out: dict[str, pd.Series] = {}

    def has(*cols: str) -> bool:
        return all(c in p.columns for c in cols)

    # ── A. Yield curve / term structure ──────────────────────────────────
    if has("IN10Y", "IN02Y"):
        out["TERM_SPREAD_IN"] = p["IN10Y"] - p["IN02Y"]
    if has("US10Y", "US02Y"):
        out["TERM_SPREAD_US"] = p["US10Y"] - p["US02Y"]
    if has("IN30Y", "IN02Y"):
        out["CURVE_FULL_IN"] = p["IN30Y"] - p["IN02Y"]
    if has("IN30Y", "IN10Y", "IN02Y"):
        out["CURVE_TWIST_IN"] = (p["IN30Y"] - p["IN10Y"]) - (p["IN10Y"] - p["IN02Y"])
    if has("IN30Y", "IN10Y"):
        out["LONG_END_IN"] = p["IN30Y"] - p["IN10Y"]

    # ── B. Real rates / monetary stance ──────────────────────────────────
    if has("REPO", "INIRYY"):
        out["REAL_REPO"] = p["REPO"] - p["INIRYY"]
    if has("IN10Y", "INIRYY"):
        out["IN_REAL_10Y"] = p["IN10Y"] - p["INIRYY"]
    if has("IN02Y", "REPO"):
        out["POLICY_EXPECT"] = p["IN02Y"] - p["REPO"]
    if has("IN10Y", "REPO"):
        out["TERM_PREMIUM_IN"] = p["IN10Y"] - p["REPO"]

    # ── C. Cross-country differentials / flows ───────────────────────────
    if has("US10Y", "IN10Y"):
        out["US_IN_10Y_DIFF"] = p["US10Y"] - p["IN10Y"]
    if has("US02Y", "IN02Y"):
        out["US_IN_2Y_DIFF"] = p["US02Y"] - p["IN02Y"]
    if has("IN10Y", "IN02Y", "US10Y", "US02Y"):
        out["REL_CURVE_STEEP"] = (p["IN10Y"] - p["IN02Y"]) - (p["US10Y"] - p["US02Y"])

    # ── D. Inflation regime ──────────────────────────────────────────────
    if has("INIRYY"):
        out["INFLATION_MOM"] = p["INIRYY"].diff(63)
        out["INFLATION_Z"] = _z(p["INIRYY"])

    # ── E. Breadth / internal momentum ───────────────────────────────────
    if has("REL_BREADTH"):
        out["BREADTH_MOM"] = p["REL_BREADTH"].diff(10)
        out["BREADTH_Z"] = _z(p["REL_BREADTH"])
    if has("AD_RATIO"):
        out["AD_MOM"] = p["AD_RATIO"].diff(21)
    if has("REL_AD_RATIO", "REL_BREADTH"):
        out["BREADTH_DIVERGENCE"] = p["REL_AD_RATIO"] - p["REL_BREADTH"]

    # ── F. Equity valuation / risk premium (Aarambh-leakage) ─────────────
    if has("NIFTY50_PE", "IN10Y"):
        out["EQ_RISK_PREMIUM"] = (1.0 / p["NIFTY50_PE"].replace(0, np.nan)) * 100.0 - p["IN10Y"]
    if has("NIFTY50_DY", "IN10Y"):
        out["DY_SPREAD"] = p["NIFTY50_DY"] - p["IN10Y"]
    if has("NIFTY50_PB", "NIFTY50_PE"):
        out["IMPLIED_ROE"] = p["NIFTY50_PB"] / p["NIFTY50_PE"].replace(0, np.nan)

    # ── G. Credit risk appetite ──────────────────────────────────────────
    if has("HYG", "LQD"):
        out["CREDIT_HY_IG"] = _lz(p["HYG"], p["LQD"])
        out["CREDIT_MOM"] = (p["HYG"] / p["LQD"].replace(0, np.nan)).pct_change(21)
    if has("EMHY", "EMB"):
        out["EM_CREDIT"] = _lz(p["EMHY"], p["EMB"])
    if has("FALN", "LQD"):
        out["FALLEN_ANGELS"] = _lz(p["FALN"], p["LQD"])
    if has("PFF", "LQD"):
        out["PREFERRED_STRESS"] = _lz(p["PFF"], p["LQD"])

    # ── H. Duration / rate-direction via ETFs ────────────────────────────
    if has("TLT", "SHY"):
        out["DURATION_BID"] = _lz(p["TLT"], p["SHY"])
    if has("TIP", "IEF"):
        out["US_INFL_EXPECT"] = _lz(p["TIP"], p["IEF"])
    if has("BNDX", "BND"):
        out["GLOBAL_RATES"] = _lz(p["BNDX"], p["BND"])

    # ── I. Currency / FX ─────────────────────────────────────────────────
    if has("DX-Y.NYB"):
        out["DXY_MOM"] = p["DX-Y.NYB"].pct_change(63)
    if has("INR=X"):
        out["USDINR_MOM"] = p["INR=X"].pct_change(21)
    if has("INR=X", "DX-Y.NYB"):
        out["INR_IDIOSYNCRATIC"] = _z(p["INR=X"]) - _z(p["DX-Y.NYB"])
    if has("INR=X", "EURINR=X", "GBPINR=X", "JPYINR=X"):
        out["INR_BASKET"] = _z(p[["INR=X", "EURINR=X", "GBPINR=X", "JPYINR=X"]].mean(axis=1))

    # ── J. Commodities / growth-inflation ────────────────────────────────
    if has("GC=F", "HG=F"):
        out["GROWTH_FEAR"] = _lz(p["GC=F"], p["HG=F"])
    if has("CL=F"):
        out["OIL_SHOCK"] = p["CL=F"].pct_change(21)
    if has("HG=F"):
        out["COPPER_MOM"] = p["HG=F"].pct_change(21)
    if has("GC=F", "INR=X"):
        out["GOLD_INR"] = _z(np.log((p["GC=F"] * p["INR=X"]).replace(0, np.nan)))
    if has("ZW=F", "ZC=F", "ZS=F", "SB=F"):
        agri = pd.concat(
            [p[c].pct_change(21) for c in ("ZW=F", "ZC=F", "ZS=F", "SB=F")], axis=1
        ).mean(axis=1)
        out["AGRI_INFLATION"] = _z(agri)
    if has("GC=F", "SI=F"):
        out["PRECIOUS_RATIO"] = _lz(p["GC=F"], p["SI=F"])
    if has("CL=F", "HG=F"):
        out["ENERGY_METALS"] = _lz(p["CL=F"], p["HG=F"])

    # ── K. EM / China ────────────────────────────────────────────────────
    if has("EMLC", "EMB"):
        out["EM_FX_CARRY"] = _lz(p["EMLC"], p["EMB"])
    if has("CNYB.L"):
        out["CHINA_RATES"] = p["CNYB.L"].pct_change(21)

    base = pd.DataFrame(out, index=p.index)

    # ── L. Cross-asset composites (depend on the base features) ──────────
    def _avail_z(specs):
        cols = [(sgn, n) for sgn, n in specs if n in base.columns]
        if len(cols) < 2:
            return None
        return pd.concat([sgn * _z(base[n]) for sgn, n in cols], axis=1).mean(axis=1)

    roo = _avail_z([(-1, "CREDIT_HY_IG"), (-1, "GROWTH_FEAR"),
                    (-1, "DXY_MOM"), (-1, "DURATION_BID")])
    if roo is not None:
        base["RISK_ON_OFF"] = roo
    fci = _avail_z([(-1, "REAL_REPO"), (-1, "CREDIT_HY_IG"),
                    (-1, "DXY_MOM"), (+1, "TERM_SPREAD_IN")])
    if fci is not None:
        base["FIN_CONDITIONS"] = fci

    return base
