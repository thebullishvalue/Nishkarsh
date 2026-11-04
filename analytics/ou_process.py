"""
Nishkarsh — Ornstein-Uhlenbeck estimation from a discretely sampled series.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — AR(1) regression with the Kendall (1954) / Orcutt & Winokur (1969)
first-order bias correction of the autoregressive coefficient. Pure function.
"""

from __future__ import annotations

import numpy as np


def ornstein_uhlenbeck_estimate(
    series: np.ndarray,
    dt: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate OU parameters ``dx = θ(μ − x)dt + σdW`` via a bias-corrected AR(1).

    1. OLS fit ``x_t = a·x_{t−1} + b + ε``.
    2. Kendall / Orcutt-Winokur correction (OLS ``a`` is biased DOWN by ≈(1+3a)/n,
       so the correction is ADDITIVE), with a second-order term above a=0.95.
    3. The intercept is re-estimated CONSISTENTLY with the corrected slope,
       ``b = mean(x_t) − a_c·mean(x_{t−1})``, so ``μ = b/(1−a_c)`` is the sample
       level of the series. (Before 2.0.0 the uncorrected OLS intercept was
       paired with the corrected slope, biasing μ away from the true mean —
       median 7.68 vs a true 5 at a=0.97, n=300.)
    4. ``θ = −ln(a_c)/dt``; ``σ² = Var(ε)·2θ/(1−a_c²)`` (denominator floored).

    Returns ``(theta, mu, sigma)``. Series shorter than 20 finite points return
    the documented fallback ``(0.05, mean(x), std(x))``.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 20:
        if len(x) > 1:
            return 0.05, float(np.mean(x)), max(float(np.std(x)), 1e-6)
        if len(x) == 1:
            return 0.05, float(x[0]), 1.0
        return 0.05, 0.0, 1.0

    x_lag = x[:-1]
    x_curr = x[1:]
    n = len(x_lag)

    sx = np.sum(x_lag)
    sy = np.sum(x_curr)
    sxx = np.dot(x_lag, x_lag)
    sxy = np.dot(x_lag, x_curr)

    denom = n * sxx - sx**2
    if abs(denom) < 1e-12:
        return 0.05, float(np.mean(x)), max(float(np.std(x)), 1e-6)

    a = (n * sxy - sx * sy) / denom
    a = np.clip(a, 1e-6, 0.999)

    if a > 0.95:
        a_corrected = a + (1 + 3 * a) / n + 3 * (1 + 3 * a) / (n**2)
    else:
        a_corrected = a + (1 + 3 * a) / n
    a = float(np.clip(a_corrected, 0.0, 0.999))

    # Intercept consistent with the corrected slope (A1-07).
    b = float(np.mean(x_curr) - a * np.mean(x_lag))

    theta = -np.log(a) / dt if a > 1e-6 else 0.05
    mu = b / (1 - a) if abs(1 - a) > 1e-6 else float(np.mean(x))

    residuals = x_curr - a * x_lag - b
    sigma_sq = np.var(residuals)
    sigma = np.sqrt(max(sigma_sq * 2 * theta / max(1 - a**2, 0.001), 1e-12))

    return max(float(theta), 1e-4), float(mu), max(float(sigma), 1e-6)
