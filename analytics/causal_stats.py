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

Inference helpers (``spearman``, ``fisher_z``, ``newey_west_t``,
``stationary_block_bootstrap_ic``, ``nonoverlap_index``) are summary statistics, not
per-row transforms; they carry no state and every random draw is seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sstats

from core.config import EDGE_ALPHA, EDGE_BOOTSTRAP_REPS, EDGE_BOOTSTRAP_SEED


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


# ─── Inference helpers (summary statistics; seeded where random) ─────────────

def spearman(x, y, min_obs: int = 8) -> float:
    """Spearman rank correlation over the pairwise-finite rows of ``x`` and ``y``.

    Returns NaN when fewer than ``min_obs`` finite pairs exist or either side is
    constant. Ties get average ranks (``pandas.Series.rank``)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"spearman: shape mismatch {a.shape} vs {b.shape}")
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < max(int(min_obs), 3):
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy(dtype=np.float64, copy=True)
    rb = pd.Series(b[m]).rank().to_numpy(dtype=np.float64, copy=True)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float((ra @ rb) / den)


def fisher_z(ic: float, n: int) -> tuple[float, float]:
    """Fisher transform of a Spearman correlation and its sampling variance.

    Returns ``(atanh(ic), 1.06 / (n − 3))`` (Fieller, Hartley & Pearson 1957 variance
    for Spearman's rho). ``ic`` is clipped to ±(1 − 1e-6) so the transform stays
    finite; NaN in → ``(nan, nan)``; ``n ≤ 3`` → variance NaN."""
    ic = float(ic)
    if not np.isfinite(ic):
        return float("nan"), float("nan")
    z = float(np.arctanh(np.clip(ic, -1.0 + 1e-6, 1.0 - 1e-6)))
    v = 1.06 / (n - 3) if n > 3 else float("nan")
    return z, float(v)


def newey_west_t(x, lags: int) -> float:
    """t-statistic of the mean of ``x`` with a Newey–West (Bartlett-kernel) HAC
    standard error using ``lags`` lags. Non-finite entries are dropped first.

    NaN when fewer than ``lags + 5`` finite values exist or the long-run variance
    is not positive."""
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = len(v)
    lags = max(int(lags), 0)
    if n < lags + 5:
        return float("nan")
    u = v - v.mean()
    s = float(u @ u) / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * float(u[L:] @ u[:-L]) / n
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    return float(v.mean() / np.sqrt(s / n))


def nonoverlap_index(index, h: int, phase: int = 0):
    """Every ``h``-th element of ``index`` starting at ``phase`` (0 ≤ phase < h).

    Applied to rows whose h-step forward labels are realized, consecutive selected
    rows are ≥ h rows apart, so their forward windows do not overlap. Works on a
    ``pd.Index``, a ``np.ndarray`` or an int ``n`` (→ positional ``np.arange``)."""
    h = int(h)
    if h < 1:
        raise ValueError("nonoverlap_index: h must be >= 1")
    phase = int(phase) % h
    if isinstance(index, (int, np.integer)):
        return np.arange(phase, int(index), h)
    return index[phase::h]


def stationary_block_bootstrap_ic(
    x,
    y,
    mean_block: float,
    reps: int = EDGE_BOOTSTRAP_REPS,
    seed: int = EDGE_BOOTSTRAP_SEED,
    alpha: float = EDGE_ALPHA,
    min_obs: int = 30,
) -> tuple[float, float]:
    """Two-sided ``1 − alpha`` percentile CI of the rank IC of ``(x, y)`` under the
    Politis–Romano (1994) stationary block bootstrap (geometric block lengths with
    mean ``mean_block``, circular wrap), deterministic for a given ``seed``.

    The pairs are rank-transformed ONCE on the full sample and each replicate's IC
    is the Pearson correlation of the resampled ranks (the standard rank-IC
    bootstrap; avoids re-ranking ties created by resampling). Returns
    ``(nan, nan)`` with fewer than ``min_obs`` finite pairs."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < max(int(min_obs), 3) or int(reps) < 10:
        return float("nan"), float("nan")
    ra = pd.Series(a[m]).rank().to_numpy(dtype=np.float64, copy=True)
    rb = pd.Series(b[m]).rank().to_numpy(dtype=np.float64, copy=True)
    p_new = 1.0 / max(float(mean_block), 1.0)
    rng = np.random.default_rng(int(seed))
    t_idx = np.arange(n)
    out = np.empty(int(reps), dtype=np.float64)
    chunk = max(1, min(int(reps), 2_000_000 // max(n, 1)))
    done = 0
    while done < reps:
        k = min(chunk, int(reps) - done)
        new_block = rng.random((k, n)) < p_new
        new_block[:, 0] = True
        starts = rng.integers(0, n, size=(k, n))
        # position at which the block covering t started (row-wise running max)
        s_pos = np.maximum.accumulate(np.where(new_block, t_idx, 0), axis=1)
        idx = (np.take_along_axis(starts, s_pos, axis=1) + (t_idx - s_pos)) % n
        A = ra[idx]
        B = rb[idx]
        A -= A.mean(axis=1, keepdims=True)
        B -= B.mean(axis=1, keepdims=True)
        den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[done:done + k] = (A * B).sum(axis=1) / den
        done += k
    out = out[np.isfinite(out)]
    if len(out) < 10:
        return float("nan"), float("nan")
    lo, hi = np.quantile(out, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def norm_sf(z: float) -> float:
    """Upper-tail standard-normal probability ``P(Z > z)`` (NaN-propagating)."""
    z = float(z)
    return float(_sstats.norm.sf(z)) if np.isfinite(z) else float("nan")
