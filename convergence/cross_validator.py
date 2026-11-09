"""
Nishkarsh — Agreement anatomy (descriptive, NOT used in the conclusion).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Four descriptive dimensions of how Aarambh and Nirnay line up on a date. They
describe the agreement; they do not drive the conclusion (that is
``convergence/conclusion.py``: Aarambh's forecast is the directional head, Nirnay
can only shrink it after a measured Convergence-Lift test). The 8-parameter
calibrated weighting of these dimensions was retired in 2.0.0 — its weights were
not identified by the data (audit A3-03, Edge Ledger L-06).

Semantics (fixed, stated):

* ONE Nirnay polarity in every dimension (audit A2-02: v1 mixed a trend-following
  regime dimension with a contrarian direction dimension, so they disagreed on ~75%
  of days). The polarity is ``core.config.NIRNAY_POLARITY`` — "trend" since 2.3.0:
  overbought breadth / a bull-regime majority read BULLISH ("contrarian" reverses it).
* Aarambh's sign convention is the engine's: ``ConvictionBounded`` < 0 = bullish.
* Fixed base weights ``CONV_WEIGHT_*`` — no adaptive shift, no calibrated path.
* Computed ONLY on dates where Nirnay has warm data (dates absent from the Nirnay
  daily frame, or with ``n_live == 0``, are skipped — never neutral-filled) and
  Aarambh has a valid, finite conviction.

Dimensions (each in [0, 1], 1 = full agreement; NaN when an input is missing):

1. ``dim_direction`` — sign(−ConvictionBounded) vs Nirnay's direction
   (trend: sign(Overbought_Pct − Oversold_Pct); contrarian: the reverse):
   1 same non-zero sign, 0.5 either side flat, 0 opposite.
2. ``dim_breadth``   — 1 − |Aarambh bullish-zone breadth − Nirnay bullish breadth| / 100
   (Nirnay bullish breadth: Overbought_Pct under trend, Oversold_Pct under contrarian).
3. ``dim_magnitude`` — 1 − | |ConvictionBounded|/100 − |Avg_Signal|/10 |  (unsigned).
4. ``dim_regime``    — Aarambh OVERSOLD/OVERBOUGHT regime vs Nirnay bull/bear majority
   (trend: bull-majority = bullish; contrarian: the reverse): 1 aligned, 0.8 both neutral
   (Aarambh NEUTRAL & |bull − bear| < 20pp), 0.2 otherwise.

``agreement_ratio`` = CONV_WEIGHT_*-weighted mean of the finite dimensions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import (
    NIRNAY_SIGN,
    CONV_WEIGHT_BREADTH,
    CONV_WEIGHT_DIRECTION,
    CONV_WEIGHT_MAGNITUDE,
    CONV_WEIGHT_REGIME,
)

ANATOMY_COLUMNS = ("dim_direction", "dim_breadth", "dim_magnitude", "dim_regime", "agreement_ratio")
ANATOMY_WEIGHTS = {
    "dim_direction": CONV_WEIGHT_DIRECTION,
    "dim_breadth": CONV_WEIGHT_BREADTH,
    "dim_magnitude": CONV_WEIGHT_MAGNITUDE,
    "dim_regime": CONV_WEIGHT_REGIME,
}
# Regime dimension: "both neutral" band on the Nirnay bull − bear spread (pp), and
# the descriptive scores (inherited v1 scale, kept so the panel reads the same).
REGIME_NEUTRAL_SPREAD_PP = 20.0
REGIME_SCORE_ALIGNED, REGIME_SCORE_BOTH_NEUTRAL, REGIME_SCORE_OTHER = 1.0, 0.8, 0.2


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _normalize_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce")).normalize()


def compute_agreement_anatomy(aarambh_ts: pd.DataFrame, nirnay_daily: pd.DataFrame) -> pd.DataFrame:
    """Per-date descriptive agreement anatomy (see module docstring).

    Parameters
    ----------
    aarambh_ts : DataFrame (DatetimeIndex)
        ``engine.ts_data``: uses ``Valid``, ``ConvictionBounded`` (falls back to
        ``ConvictionScore``), ``OversoldBreadth``, ``Regime``.
    nirnay_daily : DataFrame (date index)
        C-NIR daily aggregate: ``Oversold_Pct``, ``Overbought_Pct``, ``Avg_Signal``,
        ``Regime_Bull_Pct``, ``Regime_Bear_Pct``, ``n_live`` (optional).

    Returns
    -------
    DataFrame indexed by date (DatetimeIndex, Aarambh's timestamps) with columns
    ``dim_direction, dim_breadth, dim_magnitude, dim_regime, agreement_ratio``;
    only dates with warm Nirnay data and a finite Aarambh conviction. Empty frame
    with those columns when there is no such date.
    """
    empty = pd.DataFrame(columns=list(ANATOMY_COLUMNS), index=pd.DatetimeIndex([], name="Date"), dtype="float64")
    if aarambh_ts is None or len(aarambh_ts) == 0 or nirnay_daily is None or len(nirnay_daily) == 0:
        return empty

    a = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
    conv_col = "ConvictionBounded" if "ConvictionBounded" in a.columns else "ConvictionScore"
    conv = _num(a, conv_col)
    keep = conv.notna() & np.isfinite(conv)
    if "Valid" in a.columns:
        keep &= a["Valid"].fillna(False).astype(bool)
    a = a[keep]
    conv = conv[keep]
    if len(a) == 0:
        return empty

    n = nirnay_daily.copy()
    n.index = _normalize_index(n.index)
    n = n[~n.index.isna()]
    n = n[~n.index.duplicated(keep="last")]
    if "n_live" in n.columns:
        n = n[pd.to_numeric(n["n_live"], errors="coerce").fillna(0) > 0]
    a_key = _normalize_index(a.index)
    present = a_key.isin(n.index)
    if not present.any():
        return empty
    a = a[present]
    conv = conv[present]
    nn = n.reindex(a_key[present])
    nn.index = a.index

    os_n = _num(nn, "Oversold_Pct")
    ob_n = _num(nn, "Overbought_Pct")
    avg_n = _num(nn, "Avg_Signal")
    bull_n = _num(nn, "Regime_Bull_Pct")
    bear_n = _num(nn, "Regime_Bear_Pct")
    os_a = _num(a, "OversoldBreadth")

    # 1. Direction (Aarambh: −conviction = bullish; Nirnay per NIRNAY_POLARITY)
    dir_a = np.sign(-conv)
    dir_n = NIRNAY_SIGN * np.sign(ob_n - os_n)
    d_dir = pd.Series(np.where(dir_a * dir_n > 0, 1.0, np.where(dir_a * dir_n < 0, 0.0, 0.5)), index=a.index)
    d_dir[dir_n.isna()] = np.nan

    # 2. Breadth (Aarambh's bullish-zone share vs Nirnay's bullish-side share)
    bull_side_n = ob_n if NIRNAY_SIGN > 0 else os_n
    d_breadth = (1.0 - (os_a - bull_side_n).abs() / 100.0).clip(0.0, 1.0)

    # 3. Magnitude (unsigned strengths on their own scales)
    mag_a = (conv.abs() / 100.0).clip(upper=1.0)
    mag_n = (avg_n.abs() / 10.0).clip(upper=1.0)
    d_mag = (1.0 - (mag_a - mag_n).abs()).clip(0.0, 1.0)

    # 4. Regime (per NIRNAY_POLARITY: trend → a BULL-regime majority reads bullish)
    if "Regime" in a.columns:
        reg = a["Regime"].astype("string").fillna("N/A").str.upper()
        a_bull = reg.str.contains("OVERSOLD", regex=False).fillna(False).to_numpy(bool)
        a_bear = reg.str.contains("OVERBOUGHT", regex=False).fillna(False).to_numpy(bool)
        a_neut = (reg == "NEUTRAL").fillna(False).to_numpy(bool)
        a_known = a_bull | a_bear | a_neut
    else:
        a_bull = a_bear = a_neut = a_known = np.zeros(len(a), dtype=bool)
    n_bullish = ((bull_n > bear_n) if NIRNAY_SIGN > 0 else (bear_n > bull_n)).to_numpy()
    n_bearish = ((bear_n > bull_n) if NIRNAY_SIGN > 0 else (bull_n > bear_n)).to_numpy()
    n_flat = ((bull_n - bear_n).abs() < REGIME_NEUTRAL_SPREAD_PP).to_numpy()
    d_reg = np.full(len(a), REGIME_SCORE_OTHER)
    d_reg[(a_bull & n_bullish) | (a_bear & n_bearish)] = REGIME_SCORE_ALIGNED
    d_reg[a_neut & n_flat & ~((a_bull & n_bullish) | (a_bear & n_bearish))] = REGIME_SCORE_BOTH_NEUTRAL
    d_reg = pd.Series(d_reg, index=a.index)
    d_reg[~a_known | bull_n.isna().to_numpy() | bear_n.isna().to_numpy()] = np.nan

    out = pd.DataFrame({"dim_direction": d_dir, "dim_breadth": d_breadth,
                        "dim_magnitude": d_mag, "dim_regime": d_reg}, index=a.index).astype("float64")
    w = pd.Series(ANATOMY_WEIGHTS)
    dims = out[list(ANATOMY_WEIGHTS)]
    wmat = dims.notna().astype("float64") * w
    wsum = wmat.sum(axis=1)
    out["agreement_ratio"] = (dims.fillna(0.0) * w).sum(axis=1) / wsum.where(wsum > 0)
    out.index.name = "Date"
    return out[list(ANATOMY_COLUMNS)]
