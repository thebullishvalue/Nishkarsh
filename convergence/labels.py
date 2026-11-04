"""
Return-label resolution for the conclusion layer — pure functions, no IO.

The conclusion layer grades every edge claim against ONE forward-return label,
resolved ONCE per run (before the conclusion is computed) and named in the result
(``label_src``). Preference order (``core.config.CALIBRATION_RETURN_LABEL`` picks
the first choice; the rest follow the canonical order):

  1. ``"nsei"``   — NIFTY 50 index (^NSEI) price levels: survivorship-correct, the
                    asset actually traded.
  2. ``"target"`` — the selected Aarambh target's own level (e.g. NIFTY50_PE): a
                    valuation proxy, ≈ price over 3–20d horizons, not tradeable.
  3. ``"basket"`` — equal-weighted basket of TODAY's constituents: a real return but
                    SURVIVORSHIP-BIASED (optimistic); last resort, always warned.

Every fallback is reported in ``warnings`` — never silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_KINDS = ("nsei", "target", "basket")


def build_index_return_levels(constituent_ohlcv: dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weighted index price-level series from constituent closes.

    Averages daily simple returns across whichever constituents are available on
    each day (robust to ragged histories / listings), then compounds to a level.
    A genuine *price* return, but built from TODAY's index members, so it is
    survivorship-biased — use only as the last-resort label (see
    ``resolve_return_levels``).

    Returns an empty Series if no usable closes are found.
    """
    if not constituent_ohlcv:
        return pd.Series(dtype="float64")
    ret_cols: list[pd.Series] = []
    for _sym, df in constituent_ohlcv.items():
        if df is None or len(df) == 0:
            continue
        col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if col is None:
            continue
        s = pd.Series(df[col].to_numpy(dtype="float64"), index=pd.to_datetime(df.index, errors="coerce"))
        s = s[~s.index.isna()].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if len(s) > 1:
            # fill_method=None: a missing close is a missing return, never a fabricated 0
            ret_cols.append(s.pct_change(fill_method=None))
    if not ret_cols:
        return pd.Series(dtype="float64")
    rets = pd.concat(ret_cols, axis=1).sort_index()
    idx_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    level = (1.0 + idx_ret).cumprod()
    level.name = "index_level"
    return level


def _clean_levels(levels) -> pd.Series | None:
    """DatetimeIndex-sorted, de-duplicated, strictly positive finite levels, or None."""
    if levels is None:
        return None
    try:
        s = pd.Series(levels).astype("float64")
    except (TypeError, ValueError):
        return None
    if len(s) == 0:
        return None
    idx = pd.to_datetime(s.index, errors="coerce")
    s = pd.Series(s.to_numpy(), index=idx)
    s = s[~s.index.isna()]
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 2:
        return None
    s.index.name = "Date"
    return s


def resolve_return_levels(
    kind: str,
    nsei_levels: pd.Series | None,
    target_levels: pd.Series | None,
    basket_levels: pd.Series | None,
    *,
    target_name: str = "target",
) -> tuple[pd.Series, str, list[str]]:
    """Resolve the forward-return label used by the conclusion layer.

    Parameters
    ----------
    kind : {"nsei", "target", "basket"}
        Preferred label (``core.config.CALIBRATION_RETURN_LABEL``). The remaining
        kinds are tried in the canonical order nsei → target → basket.
    nsei_levels, target_levels, basket_levels : pd.Series | None
        Candidate PRICE/LEVEL series (DatetimeIndex). Empty / None / <2 usable
        points means unavailable.
    target_name : str
        Display name of the Aarambh target (used in ``label_src``).

    Returns
    -------
    (levels, label_src, warnings)
        ``levels`` — cleaned level series (empty float Series if nothing usable);
        ``label_src`` — human-readable label name ("" when nothing usable);
        ``warnings`` — one string per fallback / caveat, in order.
    """
    if kind not in LABEL_KINDS:
        raise ValueError(f"resolve_return_levels: kind must be one of {LABEL_KINDS}, got {kind!r}")
    candidates = {
        "nsei": (_clean_levels(nsei_levels), "NIFTY 50 index (^NSEI) fwd return"),
        "target": (_clean_levels(target_levels), f"Target fwd change ({target_name})"),
        "basket": (_clean_levels(basket_levels), "Constituent EW basket (survivorship-biased)"),
    }
    order = [kind] + [k for k in LABEL_KINDS if k != kind]
    warnings: list[str] = []
    for k in order:
        levels, src = candidates[k]
        if levels is None:
            warnings.append(f"Return label '{k}' unavailable (no usable levels).")
            continue
        if k != kind:
            warnings.append(f"Return label fell back from '{kind}' to '{k}' ({src}).")
        if k == "target":
            warnings.append(f"Label is the target level ({target_name}) — a valuation proxy, "
                            "not a tradeable price; backtest P&L is indicative only.")
        if k == "basket":
            warnings.append("Label is the equal-weight basket of TODAY's constituents — "
                            "SURVIVORSHIP-BIASED; edge and backtest statistics are optimistic.")
        return levels, src, warnings
    warnings.append("No return label available — edge, lift and backtest cannot be measured.")
    return pd.Series(dtype="float64"), "", warnings
