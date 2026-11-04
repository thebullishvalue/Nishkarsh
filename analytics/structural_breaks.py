"""
Nishkarsh — Rolling-mean change-point HEURISTIC (descriptive diagnostic only).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Pure function. This is NOT a structural-break test: it has no null
distribution and no significance level, and it returns up to ``max_breaks``
candidate locations on ANY input (3 "breaks" on 100% of iid N(0,1) series of
length 2000). It is not Bai-Perron. It must never gate a model decision or be
presented as "a break was detected"; the engine exposes it only as
``changepoints_heuristic`` (where the rolling mean moved most), and no longer
uses it in the walk-forward.
"""

from __future__ import annotations

import logging

import numpy as np


def detect_structural_breaks(
    series: np.ndarray,
    max_breaks: int = 3,
    trim: float = 0.15,
) -> list[int]:
    """Locations where a rolling mean moves most (heuristic change points).

    Parameters
    ----------
    series : np.ndarray
        Input time-series (non-finite values contribute no movement).
    max_breaks : int
        Maximum number of candidate locations returned.
    trim : float
        Fraction of the series excluded at each end, also the rolling window
        length as a fraction of ``n``.

    Returns
    -------
    list[int]
        Sorted candidate indices (relative to the series start); ``[]`` for
        series shorter than 50.
    """
    if len(series) < 50:
        return []
    try:
        return _rolling_mean_breaks(np.asarray(series, dtype=np.float64), max_breaks, trim)
    except Exception as e:  # pragma: no cover - defensive
        logging.warning("Change-point heuristic failed: %s", e)
        return []


def _rolling_mean_breaks(
    series: np.ndarray,
    max_breaks: int,
    trim: float,
) -> list[int]:
    """Top-k peaks of |Δ rolling mean| outside the trimmed ends, separated by a window."""
    n = len(series)
    window = max(int(n * trim), 5)
    trim_n = int(n * trim)

    rolling_mean = np.convolve(series, np.ones(window) / window, mode="valid")
    diffs = np.nan_to_num(np.abs(np.diff(rolling_mean)), nan=0.0)

    pad = window // 2
    edge = max(trim_n - pad, 0)
    if edge > 0:
        diffs[:edge] = 0
        diffs[-edge:] = 0

    break_indices = []
    diffs_copy = diffs.copy()
    for _ in range(max_breaks):
        peak = int(np.argmax(diffs_copy))
        if diffs_copy[peak] == 0:
            break
        break_indices.append(peak + pad)
        lo = max(0, peak - window)
        hi = min(len(diffs_copy), peak + window)
        diffs_copy[lo:hi] = 0

    return sorted(break_indices)
