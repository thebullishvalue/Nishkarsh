"""
Nishkarsh — Math utilities: pure, stateless functions.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Side-effect free functions on NumPy arrays and pandas structures.
This module is the SINGLE source of the causal oscillator helpers used by the
Nirnay engine (``sigmoid``, ``zscore_clipped``, ``calculate_atr``,
``causal_gram_schmidt_orthogonalize``) and of the classification / significance
helpers used by the Aarambh engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ─── Transformations ─────────────────────────────────────────────────────────


def sigmoid(x: np.ndarray | float, scale: float = 1.0) -> np.ndarray | float:
    """Sigmoid transformation bounding values to [-1, 1].

    Uses the original Nirnay formula: ``2 / (1 + exp(-x/scale)) - 1``.

    Parameters
    ----------
    x : np.ndarray | float
        Input values.
    scale : float
        Divisor controlling the curve steepness. Larger = gentler slope.
    """
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0


def zscore_clipped(series: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
    """Causal rolling z-score with outlier clipping.

    The mean / std at row ``t`` are computed over the ``window`` rows BEFORE
    ``t`` (``rolling(...).shift(1)``), so today's value never enters its own
    normalisation and an outlier cannot shrink its own z. The series is
    forward-filled first (then a leading gap becomes 0.0). No backfill: row 0
    has no prior statistic and stays neutral (0.0), as does any row whose
    prior-window std is 0 or undefined.

    Parameters
    ----------
    series : pd.Series
        Input time-series.
    window : int
        Rolling window size (``min_periods=1`` on the prior window).
    clip : float
        Maximum absolute z-score before clipping.
    """
    series_filled = series.ffill().fillna(0)
    roll_mean = series_filled.rolling(window=window, min_periods=1).mean().shift(1)
    roll_std = series_filled.rolling(window=window, min_periods=1).std().shift(1)
    z = (series_filled - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)


def causal_gram_schmidt_orthogonalize(matrix: np.ndarray) -> np.ndarray:
    """Causal, unique-magnitude Gram-Schmidt orthogonalization of columns.

    Two deliberate properties, both fixing defects that a naive in-sample,
    unit-renormalized Gram-Schmidt has when used to build a *signal*:

    1. **Causal basis.** The projection coefficient of column ``j`` onto an
       earlier orthogonal component ``p`` at row ``t`` is estimated only from
       rows ``< t`` (expanding cumulative dot-products, shifted by one). No
       future information enters the value at ``t``, so historical composite
       values do not change as new bars arrive — the series is backtest-safe.

    2. **No renormalization.** Each orthogonal column keeps its *natural*
       magnitude, i.e. the size of its unique contribution. A column that is
       largely redundant with earlier ones collapses to a small residual and
       therefore contributes little to the composite — instead of being
       rescaled back up to unit RMS, which would promote its noise to full
       weight. The composite ``Σ uⱼ`` is thus a genuine sum of *independent*
       contributions at their true scale.

    Parameters
    ----------
    matrix : np.ndarray
        Shape ``(n_obs, k)``. Columns ordered by decreasing primacy (the first
        is preserved). Non-finite entries are treated as 0.

    Returns
    -------
    np.ndarray
        Same shape; column ``0`` unchanged, each later column the causal
        residual after removing its projection onto earlier components.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("causal_gram_schmidt_orthogonalize expects a 2-D matrix")
    M = np.where(np.isfinite(M), M, 0.0)
    n, k = M.shape
    U = np.zeros_like(M)
    if k == 0:
        return U
    U[:, 0] = M[:, 0]
    eps = 1e-12
    for i in range(1, k):
        resid = M[:, i].copy()
        for p in range(i):
            up = U[:, p]
            cross = np.cumsum(M[:, i] * up)
            denom = np.cumsum(up * up)
            # Shift by one row → coefficient at t uses only rows < t (causal).
            cross_sh = np.concatenate(([0.0], cross[:-1]))
            denom_sh = np.concatenate(([0.0], denom[:-1]))
            safe_denom = np.where(denom_sh > eps, denom_sh, 1.0)
            beta = np.where(denom_sh > eps, cross_sh / safe_denom, 0.0)
            resid = resid - beta * up
        U[:, i] = resid
    return U


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range — Wilder/EMA variant (``alpha = 1/length``).

    Causal: the true range at ``t`` uses only bar ``t`` and the prior close.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``High``, ``Low``, ``Close`` columns.
    length : int
        Smoothing period.
    """
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]
    close = df["Close"] if "Close" in df.columns else df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ─── Classification helpers ──────────────────────────────────────────────────


def _classify_zones(
    z_scores: np.ndarray, z_threshold: float = 1.0, z_extreme: float = 2.0
) -> np.ndarray:
    """Map z-scores to valuation zone labels.

    Parameters
    ----------
    z_scores : np.ndarray
        Raw z-scores.
    z_threshold : float
        Boundary between fair value and over/undervalued.
    z_extreme : float
        Boundary between over/undervalued and extreme zones.
    """
    condlist = [
        z_scores > z_extreme,
        z_scores > z_threshold,
        z_scores > -z_threshold,
        z_scores > -z_extreme,
    ]
    choicelist = ["Extreme Over", "Overvalued", "Fair Value", "Undervalued"]
    zones = np.select(condlist, choicelist, default="Extreme Under")
    np.putmask(zones, np.isnan(z_scores), "N/A")
    return zones


def _detect_crossover_signals(
    z_scores: np.ndarray, threshold: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Detect z-score threshold crossings as boolean signal arrays.

    A buy signal fires when z crosses below ``-threshold``.
    A sell signal fires when z crosses above ``+threshold``.
    """
    n = len(z_scores)
    if n < 2:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    z_curr = z_scores[1:]
    z_prev = z_scores[:-1]
    valid = np.isfinite(z_curr) & np.isfinite(z_prev)
    buy_cond = valid & (z_curr < -threshold) & (z_prev >= -threshold)
    sell_cond = valid & (z_curr > threshold) & (z_prev <= threshold)
    buy_signals = np.zeros(n, dtype=bool)
    sell_signals = np.zeros(n, dtype=bool)
    buy_signals[1:] = buy_cond
    sell_signals[1:] = sell_cond
    return buy_signals, sell_signals


def _compute_significance(values: list[float]) -> dict[str, float]:
    """Compute t-statistic and p-value for a list of values.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``t_stat``, ``p_value``, ``n``.
    """
    n = len(values)
    if n < 3:
        return {"mean": 0.0, "std": 0.0, "t_stat": 0.0, "p_value": 1.0, "n": n}
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    if std_val < 1e-10:
        return {"mean": mean_val, "std": std_val, "t_stat": np.inf, "p_value": 0.0, "n": n}
    se = std_val / np.sqrt(n)
    t_stat = mean_val / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    return {
        "mean": mean_val,
        "std": std_val,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": n,
    }


def _apply_conviction_bounds(score: np.ndarray | float, max_bound: float = 100.0) -> np.ndarray | float:
    """Apply soft bounds to conviction score via tanh transformation.

    Parameters
    ----------
    score : np.ndarray | float
        Raw conviction score(s).
    max_bound : float
        Asymptotic bound (default ±100).
    """
    return max_bound * np.tanh(np.asarray(score, dtype=np.float64) / max_bound)
