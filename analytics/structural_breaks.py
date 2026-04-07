"""
Structural break detection via Bai-Perron multiple breakpoint test.

Extracted from correl.py lines 592-615.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    from statsmodels.tsa.regime_switching.bai_perron import BaiPerronTest

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def detect_structural_breaks(
    series: np.ndarray,
    max_breaks: int = 3,
    trim: float = 0.15,
) -> list[int]:
    """Bai-Perron multiple breakpoint detection.

    Parameters
    ----------
    series : np.ndarray
        Input time-series.
    max_breaks : int
        Maximum number of structural breaks to detect.
    trim : float
        Trim fraction for each segment (default 15%).

    Returns
    -------
    list[int]
        Break indices relative to the series start.
    """
    if not _HAS_STATSMODELS or len(series) < 50:
        return []

    try:
        bp_test = BaiPerronTest(series)
        result = bp_test.test_breaks(max_breaks, trim=trim)

        if hasattr(result, "break_dates") and result.break_dates is not None:
            return [int(bd) for bd in result.break_dates]
    except Exception as e:
        logging.warning("Bai-Perron test failed: %s", e)

    return []
