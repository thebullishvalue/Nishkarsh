"""
Nishkarsh — Rolling conformal (empirical-quantile) z-scores.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Causal, quantile-based standardization of a residual / forecast series.
Pure function, no IO, no state.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.stats import norm


@njit(cache=True)
def _percentile_linear(sorted_v: np.ndarray, q: float) -> float:
    """Linear-interpolated percentile of a pre-sorted array — matches numpy's
    default ``np.percentile(..., method='linear')`` (and hence nanpercentile /
    nanmedian once NaNs are removed)."""
    k = sorted_v.shape[0]
    if k == 1:
        return sorted_v[0]
    rank = q * (k - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    frac = rank - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


@njit(cache=True)
def _conformal_njit(series, window, min_periods, alpha, width_in_sigmas):
    n = series.shape[0]
    z_scores = np.full(n, np.nan)
    lower_bounds = np.full(n, np.nan)
    upper_bounds = np.full(n, np.nan)
    scratch = np.empty(window, dtype=np.float64)
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    for i in range(window, n):
        # Collect finite values from series[i-window:i] (excludes current point).
        cnt = 0
        for j in range(i - window, i):
            v = series[j]
            if np.isfinite(v):
                scratch[cnt] = v
                cnt += 1
        if cnt < min_periods:
            continue
        sv = np.sort(scratch[:cnt])
        ql = _percentile_linear(sv, q_lo)
        qu = _percentile_linear(sv, q_hi)
        qm = _percentile_linear(sv, 0.5)
        width = qu - ql
        if width > 1e-10:
            # Robust sigma: the central (1-alpha) interval of a normal spans
            # 2·Φ⁻¹(1-alpha/2) sigmas (3.92σ at alpha=0.05).
            z_scores[i] = (series[i] - qm) / (width / width_in_sigmas)
        lower_bounds[i] = ql
        upper_bounds[i] = qu
    return z_scores, lower_bounds, upper_bounds


def compute_conformal_zscores(
    series: np.ndarray,
    window: int,
    min_periods: int = 5,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal rolling z-score standardized by empirical quantiles.

    ``z_t = (x_t − median(W_t)) / σ̂_t`` where ``W_t`` is the finite values of
    ``series[t−window : t]`` (the CURRENT point is excluded — no look-ahead) and
    ``σ̂_t = (q_{1−α/2}(W_t) − q_{α/2}(W_t)) / (2·Φ⁻¹(1−α/2))`` — the width of the
    empirical central ``1−α`` interval expressed in normal-equivalent sigmas
    (``/3.92`` at ``α=0.05``). On Gaussian data ``z`` is therefore on the ordinary
    σ scale (std ≈ 1, ``P(|z|>1) ≈ 0.32``) once the window is long enough for the
    tail quantiles to be estimated; short windows (≲20) under-estimate the tail
    width and give over-dispersed ``z``. (Before 2.0.0 the width was divided by
    1.35 — the IQR constant applied to a 95% width — which shrank ``z`` 2.7–2.9×.)

    Parameters
    ----------
    series : np.ndarray
        Input time-series (NaN allowed; skipped by the window statistics).
    window : int
        Rolling window size (prior rows).
    min_periods : int
        Minimum finite observations required within the window.
    alpha : float
        Tail mass of the central interval used for the scale (default 0.05).

    Returns
    -------
    z_scores : np.ndarray
        Quantile-standardized z-scores (NaN where not computable).
    lower_bounds, upper_bounds : np.ndarray
        Empirical ``α/2`` and ``1−α/2`` quantiles of the prior window
        (descriptive; no coverage guarantee is claimed).
    """
    n = len(series)
    if n <= window:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    arr = np.ascontiguousarray(series, dtype=np.float64)
    width_in_sigmas = 2.0 * float(norm.ppf(1.0 - float(alpha) / 2.0))
    return _conformal_njit(arr, int(window), int(min_periods), float(alpha), width_in_sigmas)
