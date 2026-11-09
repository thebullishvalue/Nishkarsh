"""
The Nishkarsh Oscillator (NKO) — the system's single inference series.

Every read-out the app used to show separately — the conclusion over time, the forecast
extension diagnostics, Aarambh's zone breadth and cut crossings, Nirnay's HMM state, zone
breadth, regime composition and cut crossings, and the agreement anatomy — is folded into
ONE causal oscillator in [−100, +100] (+ = bullish). The hero card and the Precedent read
infer from it; everything else is its decomposition.

Construction (fixed a priori — NO weight or sign is fitted to returns):

  1. Every input is oriented by its documented meaning so that + = bullish. Aarambh is read
     as published (its forecast and extension). Nirnay follows the ONE configured polarity
     (``core.config.NIRNAY_POLARITY``; "trend" since 2.3.0 — strong oscillator / overbought
     breadth / bull regime / bull HMM / overbought-side crossings = bullish; "contrarian"
     reverses every Nirnay input).
  2. Each oriented input is causally z-scored (``analytics.causal_stats.causal_z``:
     strictly-prior mean/std, ≥ 126 observations, clip ±4).
  3. Three PILLARS, each the equal-weight mean of its inputs' z-scores, re-standardized
     causally so no pillar dominates by its input count:
       • TOP-DOWN  (Aarambh): forecast z · −extension (AvgZ) · zone breadth · crossing breadth
       • BOTTOM-UP (Nirnay):  −oscillator · zone breadth · regime mix · HMM state · cut crossings
       • AGREEMENT:           consensus × agreement-anatomy ratio (the signed agreement)
  4. ``raw = mean(available pillars)``; ``NKO = 100·tanh(causal_z(raw)/2)``.

  When the two systems disagree, their pillars cancel and the agreement pillar is zero, so
  NKO collapses toward 0 — agreement amplifies, disagreement mutes, by construction.
  Dates without Nirnay data carry the TOP-DOWN pillar only and are flagged
  ``two_system = False``; the measurement uses two-system dates only.

Whether NKO forecasts anything is MEASURED (``nko_readout``) with the pre-registered rule in
``docs/research/NISHKARSH_OSCILLATOR_PREREG.md``; until it passes, the product presents it as
the system's read, never as a buy/sell signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.causal_stats import causal_z
from core.config import (
    NIRNAY_SIGN,
    CONV2_TIER_MODERATE,
    CONV2_TIER_STRONG,
    CONV2_TIER_WEAK,
    CONV2_ZCLIP,
    CONV2_ZMIN_PERIODS,
    EDGE_ALPHA,
)

__all__ = ["COMPONENTS", "PILLARS", "build_nishkarsh_oscillator", "nko_readout", "nko_tier"]

# (pillar, component id, human label, source)
COMPONENTS = (
    ("top_down", "forecast", "Aarambh forecast", "conclusion frame zA (the conclusion over time)"),
    ("top_down", "extension", "Forecast extension", "−AvgZ: conformal z across lookbacks"),
    ("top_down", "zone_breadth_a", "Lookback zone breadth", "OversoldBreadth − OverboughtBreadth"),
    ("top_down", "crossings_a", "Lookback cut crossings", "BuySignalBreadth − SellSignalBreadth"),
    ("bottom_up", "oscillator_n", "Constituent oscillator", "±Avg_Signal (NIRNAY_POLARITY)"),
    ("bottom_up", "zone_breadth_n", "Constituent zone breadth", "±(Overbought_Pct − Oversold_Pct)"),
    ("bottom_up", "regime_mix", "Regime composition", "±(Regime_Bull_Pct − Regime_Bear_Pct)"),
    ("bottom_up", "hmm_state", "HMM state", "±(avg_hmm_bull − avg_hmm_bear)"),
    ("bottom_up", "cut_crossings_n", "Constituent cut crossings", "±(Sell_Signals − Buy_Signals) / n_live"),
    ("agreement", "signed_agreement", "Signed agreement", "consensus × agreement ratio (anatomy)"),
)
PILLARS = ("top_down", "bottom_up", "agreement")
PILLAR_LABELS = {"top_down": "Top-down (Aarambh)", "bottom_up": "Bottom-up (Nirnay)", "agreement": "Agreement"}


def _z(s: pd.Series) -> pd.Series:
    return causal_z(s, min_periods=CONV2_ZMIN_PERIODS, clip=CONV2_ZCLIP)


def _col(df: pd.DataFrame | None, name: str, index: pd.Index) -> pd.Series:
    if df is None or not len(df) or name not in df.columns:
        return pd.Series(np.nan, index=index, dtype="float64")
    s = pd.to_numeric(df[name], errors="coerce")
    s.index = pd.DatetimeIndex(df.index)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(index)          # NO forward fill: a missing day is missing


def oriented_inputs(frame: pd.DataFrame, aarambh_ts: pd.DataFrame | None,
                    nirnay_daily: pd.DataFrame | None) -> pd.DataFrame:
    """The ten inputs on the conclusion frame's dates, oriented so + = bullish."""
    idx = frame.index
    a = lambda c: _col(aarambh_ts, c, idx)            # noqa: E731
    n = lambda c: _col(nirnay_daily, c, idx)           # noqa: E731
    out = pd.DataFrame(index=idx)
    out["forecast"] = pd.to_numeric(frame.get("zA"), errors="coerce")
    out["extension"] = -a("AvgZ")
    out["zone_breadth_a"] = a("OversoldBreadth") - a("OverboughtBreadth")
    out["crossings_a"] = a("BuySignalBreadth") - a("SellSignalBreadth")
    # Nirnay inputs: written in the TREND orientation and multiplied by NIRNAY_SIGN
    # (+1 trend, −1 contrarian) — one switch flips every bottom-up input together.
    S = NIRNAY_SIGN
    out["oscillator_n"] = S * n("Avg_Signal")
    out["zone_breadth_n"] = S * (n("Overbought_Pct") - n("Oversold_Pct"))
    out["regime_mix"] = S * (n("Regime_Bull_Pct") - n("Regime_Bear_Pct"))
    out["hmm_state"] = S * (n("avg_hmm_bull") - n("avg_hmm_bear"))
    live = n("n_live").where(lambda s: s > 0)
    out["cut_crossings_n"] = S * (n("Sell_Signals") - n("Buy_Signals")) / live    # Sell = overbought-side
    out["signed_agreement"] = pd.to_numeric(frame.get("SAO"), errors="coerce") if "SAO" in frame.columns \
        else pd.Series(np.nan, index=idx)
    return out


def build_nishkarsh_oscillator(frame: pd.DataFrame, aarambh_ts: pd.DataFrame | None,
                               nirnay_daily: pd.DataFrame | None) -> pd.DataFrame:
    """NKO and its full decomposition on the conclusion frame's dates (all causal).

    Columns: ``in_<id>`` oriented inputs, ``z_<id>`` their causal z, ``P_<pillar>`` pillar
    scores (re-standardized), ``NKO_raw``, ``NKO`` (±100), ``two_system`` (bool)."""
    raw = oriented_inputs(frame, aarambh_ts, nirnay_daily)
    out = pd.DataFrame(index=frame.index)
    zs = {}
    for _, cid, _, _ in COMPONENTS:
        out[f"in_{cid}"] = raw[cid]
        zs[cid] = _z(raw[cid])
        out[f"z_{cid}"] = zs[cid]
    for p in PILLARS:
        ids = [cid for pil, cid, _, _ in COMPONENTS if pil == p]
        m = pd.concat([zs[c] for c in ids], axis=1).mean(axis=1, skipna=True)
        out[f"P_{p}"] = _z(m) if p != "agreement" else _z(zs["signed_agreement"])
    P = out[[f"P_{p}" for p in PILLARS]]
    two = out["P_bottom_up"].notna() & out["P_agreement"].notna()
    out["two_system"] = two
    out["NKO_raw"] = P.mean(axis=1, skipna=True).where(out["P_top_down"].notna())
    out["NKO"] = 100.0 * np.tanh(_z(out["NKO_raw"]) / 2.0)
    return out


def nko_tier(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "NO READ"
    a = abs(v)
    return ("STRONG" if a >= CONV2_TIER_STRONG else "MODERATE" if a >= CONV2_TIER_MODERATE
            else "WEAK" if a > CONV2_TIER_WEAK else "NO LEAN")


def nko_readout(nko: pd.DataFrame, price_levels: pd.Series | None, h: int,
                alpha: float = EDGE_ALPHA) -> dict:
    """Latest reading, decomposition, measured status and pillar evidence.

    VALIDATED ⇔ (on two-system dates) one-sided p < alpha for IC(NKO, h-day forward
    return) > 0 AND the block-bootstrap CI excludes 0 AND ≥ 60% of calendar years
    positive — the pre-registered rule. Otherwise UNPROVEN (or INSUFFICIENT_DATA)."""
    from convergence.conclusion import _fmt, align_log_price, ic_evidence
    s = nko["NKO"].dropna()
    latest_date = s.index[-1] if len(s) else None
    row = nko.loc[latest_date] if latest_date is not None else None
    val = float(s.iloc[-1]) if len(s) else None
    pillars = {p: (float(row[f"P_{p}"]) if row is not None and np.isfinite(row[f"P_{p}"]) else None) for p in PILLARS}
    comps = []
    for pil, cid, label, src in COMPONENTS:
        z = row[f"z_{cid}"] if row is not None else np.nan
        comps.append({"id": cid, "pillar": pil, "label": label, "source": src,
                      "z": float(z) if np.isfinite(z) else None})
    tier = nko_tier(val)
    direction = ("BULLISH" if val and val > CONV2_TIER_WEAK else "BEARISH" if val and val < -CONV2_TIER_WEAK
                 else "NEUTRAL") if val is not None else "NEUTRAL"
    base = {"value": val, "as_of": str(latest_date.date()) if latest_date is not None else None,
            "tier": tier, "direction": direction,
            "two_system": bool(row["two_system"]) if row is not None else False,
            "pillars": pillars, "pillar_labels": PILLAR_LABELS, "components": comps,
            "drivers": sorted([c for c in comps if c["z"] is not None], key=lambda c: -abs(c["z"]))[:3]}
    sub = nko[nko["two_system"]]["NKO"]
    logp = align_log_price(price_levels, sub.index) if price_levels is not None and len(price_levels) and len(sub) else None
    if logp is None or sub.dropna().empty:
        return base | {"status": "INSUFFICIENT_DATA", "evidence": None, "pillar_ic": {},
                       "text": "Not enough two-system history to measure the oscillator."}
    fwd = logp.shift(-h) - logp
    e = ic_evidence(sub, fwd, h, signal_name="NKO", alpha=alpha)
    yearly = e.get("yearly") or {}
    yr_pos = (sum(v > 0 for v in yearly.values()) / len(yearly)) if yearly else None
    boot = e.get("boot_ci")
    if not e.get("available"):
        status = "INSUFFICIENT_DATA"
    elif np.isfinite(e["p"]) and e["p"] < alpha and boot and boot[0] > 0 and yr_pos is not None and yr_pos >= 0.6:
        status = "VALIDATED"
    else:
        status = "UNPROVEN"
    pillar_ic = {}
    for p in PILLARS:
        pe = ic_evidence(nko.loc[sub.index, f"P_{p}"], fwd, h, signal_name=p, alpha=alpha, bootstrap=False)
        pillar_ic[p] = {"ic": pe.get("ic"), "n": pe.get("n"), "p": pe.get("p")}
    num = (f"accuracy (rank IC) {_fmt(e.get('ic'), '+.3f')} over {int(e.get('n') or 0)} non-overlapping "
           f"{h}-day windows (p {_fmt(e.get('p'), '.2f')}; positive in "
           f"{int(round((yr_pos or 0) * len(yearly)))}/{len(yearly)} years)")
    text = (f"Validated directional read — {num}." if status == "VALIDATED" else
            f"Not yet a validated buy/sell signal — {num}; this sample can only confirm an IC of "
            f"{_fmt(e.get('mde'), '.2f')} or more." if status == "UNPROVEN" else
            f"Not enough two-system history to measure it ({int(e.get('n') or 0)} windows).")
    ev = {k: e.get(k) for k in ("ic", "n", "p", "mde", "hac_t", "boot_ci", "yearly")}
    ev["yearly_positive_share"] = yr_pos
    return base | {"status": status, "evidence": ev, "pillar_ic": pillar_ic, "text": text}
