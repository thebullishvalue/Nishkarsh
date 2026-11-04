"""
Nishkarsh — Hurst exponent via Detrended Fluctuation Analysis (DFA).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Long-memory / mean-reversion diagnostic. Pure function.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def hurst_dfa(
    series: np.ndarray,
    min_scale: int = 8,
    max_scale: int | None = None,
) -> float:
    """DFA-1 scaling exponent α of ``series``, read as a Hurst exponent.

    INPUT CONVENTION: pass INCREMENTS (``np.diff`` of a level series, or returns).
    For increments, α ≈ H: ``H < 0.5`` = anti-persistent (the level mean-reverts),
    ``H ≈ 0.5`` = uncorrelated (random-walk level), ``H > 0.5`` = persistent
    (trending). Feeding a LEVEL series instead measures the level's own scaling
    (α ≈ 0.9–1.0 for a strongly mean-reverting AR(1)), which inverts the reading.

    Box sizes are log-spaced from ``min_scale`` to ``max_scale`` (default n//4)
    (Peng et al. 1994; Kantelhardt et al. 2001). The estimate is clipped to
    [0.01, 0.99].

    Returns NaN when not computable (fewer than 30 finite points, a degenerate
    scale range, or fewer than 3 usable scales) — never a fabricated 0.5.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 30:
        return float("nan")

    if max_scale is None:
        max_scale = n // 4

    if min_scale >= max_scale:
        return float("nan")

    y = np.cumsum(x - np.mean(x))

    scales = np.unique(np.round(np.geomspace(min_scale, max_scale, 20)).astype(int))
    log_scales: list[float] = []
    log_fluctuations: list[float] = []

    for scale in scales:
        n_windows = n // scale
        if n_windows < 2:
            continue

        rms_errors: list[float] = []
        x_seg = np.arange(scale)
        for i in range(n_windows):
            segment = y[i * scale : (i + 1) * scale]
            slope, intercept = np.polyfit(x_seg, segment, 1)
            trend = slope * x_seg + intercept
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            if rms > 1e-10:
                rms_errors.append(rms)

        if rms_errors:
            fluctuation = np.sqrt(np.mean(np.array(rms_errors) ** 2))
            if fluctuation > 1e-10:
                log_scales.append(np.log(scale))
                log_fluctuations.append(np.log(fluctuation))

    if len(log_scales) < 3:
        return float("nan")

    slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
    return float(np.clip(slope, 0.01, 0.99))
