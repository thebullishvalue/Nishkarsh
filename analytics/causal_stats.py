"""
Causal statistics primitives — pure functions, no IO, no state.

Every function here is CAUSAL (the value at row t uses only rows < t, or ≤ t where
stated) and therefore PREFIX-INVARIANT: appending rows never changes an earlier
output. These are the shared building blocks of the conclusion layer
(``convergence/conclusion.py``) and of Aarambh's forward-mode conviction, so the
two layers z-score a forecast in exactly the same way.

Ownership: ``causal_z`` is a frozen contract (signature and behaviour) shared by
``engines/aarambh.py`` and ``convergence/conclusion.py``; extend this module with
new functions rather than changing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_z(
    x: pd.Series | np.ndarray,
    min_periods: int = 126,
    clip: float = 4.0,
    window: int | None = None,
) -> pd.Series:
    """Causal z-score: ``z_t = (x_t − mean(x_<t)) / std(x_<t)``, clipped to ±clip.

    The mean/std use only STRICTLY PRIOR finite observations (expanding, or rolling
    over the last ``window`` rows when given), with ``ddof=1``. ``z_t`` is NaN until
    ``min_periods`` prior finite observations exist, where ``x_t`` is NaN, or where
    the prior std is ~0. NaN rows in ``x`` are skipped by the statistics (they do not
    count towards ``min_periods``) and stay NaN in the output.

    Prefix-invariant by construction: ``causal_z(x[:n]) == causal_z(x)[:n]``.
    """
    s = pd.Series(x, dtype="float64")
    roll = (s.expanding(min_periods=min_periods) if window is None
            else s.rolling(window, min_periods=min_periods))
    mu = roll.mean().shift(1)
    sd = roll.std(ddof=1).shift(1)
    z = (s - mu) / sd.where(sd > 1e-12)
    return z.clip(-clip, clip)
