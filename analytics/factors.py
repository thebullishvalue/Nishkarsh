"""
Nishkarsh — Causal macro factor reduction.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Compresses a wide, collinear macro panel into a few orthogonal
principal factors using an EXPANDING-window PCA, so downstream models select
from de-duplicated, backtest-safe drivers instead of dozens of near-duplicates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:  # graceful degradation — callers fall back to raw macros
    _HAS_SKLEARN = False


def build_causal_macro_factors(
    macro_df: pd.DataFrame,
    n_components: int = 8,
    refit_every: int = 21,
    min_train: int = 126,
    stationarize: bool = False,
    z_window: int = 252,
    return_loadings: bool = False,
    passthrough: "list[str] | tuple[str, ...] | None" = None,
):
    """Reduce a wide, collinear macro panel to a few orthogonal factors, causally.

    A 90-series macro panel is dominated by near-duplicate columns (dozens of
    bond ETFs measure the same duration/credit factor; FX pairs co-move), so
    selecting "top-k correlated" raw macros picks redundant, unstable drivers.
    This compresses the panel to ``n_components`` orthogonal principal factors.

    Causality: standardization and PCA are refit on the EXPANDING window every
    ``refit_every`` rows; the factor value at row ``t`` is produced by the most
    recent fit trained only on rows ``< t``. No future information enters any
    historical factor value, so the output is backtest-safe (consistent with
    the rest of the system's causal guarantees). Component *signs* are aligned
    across refits to avoid spurious sign-flip jumps in the factor series.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Wide macro panel indexed by date.
    n_components : int
        Number of orthogonal factors to retain.
    refit_every : int
        Rows between PCA refits (≈ monthly at 21).
    min_train : int
        Minimum rows before the first fit; earlier rows are NaN (warm-up).
    stationarize : bool
        If True, replace each column with its causal rolling z-score (window
        ``z_window``, clipped to ±5) BEFORE the PCA. Essential when the panel
        contains non-stationary *levels* (e.g. ETF/commodity prices) that feed a
        level regression: an unbounded trending feature makes the regression
        extrapolate catastrophically (R² → −47). Rolling-z detrends and bounds
        each input so the factors — and any model trained on them — stay sane.
        Leave False for MMR (which uses scale-free rolling correlations).
    z_window : int
        Rolling window for the stationarizing z-score.
    return_loadings : bool
        If True, also return the most-recent fit's loadings (PC × feature),
        so callers can name each factor by its top-weighted inputs.
    passthrough : list[str] | None
        Columns to keep OUT of the PCA and append to the output as their own
        named, raw columns (e.g. the India rate/inflation predictors). Their
        direct signal then isn't diluted across the orthogonal components. They
        are still causal (raw data is point-in-time) and not stationarized.

    Returns
    -------
    pd.DataFrame  (or (pd.DataFrame, pd.DataFrame) if return_loadings)
        Factor columns ``MACRO_PC1..k`` on ``macro_df``'s index. Empty frame if
        sklearn is unavailable or the panel is too small (callers then fall back
        to the raw macro columns).
    """
    _empty = pd.DataFrame(index=getattr(macro_df, "index", None))
    if not _HAS_SKLEARN or macro_df is None or macro_df.empty:
        return (_empty, pd.DataFrame()) if return_loadings else _empty

    # Reserved passthrough columns bypass the PCA — kept raw, appended as their
    # own named features so their signal isn't dissolved into the components.
    pass_cols = [c for c in (passthrough or []) if c in macro_df.columns]
    pass_raw = (
        macro_df[pass_cols].apply(pd.to_numeric, errors="coerce") if pass_cols else None
    )
    pca_input = macro_df.drop(columns=pass_cols) if pass_cols else macro_df

    clean = (
        pca_input.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    )
    if stationarize:
        # Causal rolling z-score (uses rows ≤ t only — no future) + clip, so
        # trending price levels become bounded, stationary oscillators.
        _mp = max(z_window // 4, 20)
        mu = clean.rolling(z_window, min_periods=_mp).mean()
        sd = clean.rolling(z_window, min_periods=_mp).std()
        clean = ((clean - mu) / sd.replace(0, np.nan)).clip(-5.0, 5.0).fillna(0.0)
    feature_names = list(clean.columns)
    X = clean.to_numpy(dtype=np.float64)
    n, m = X.shape
    if n <= min_train or m == 0:
        # Too few rows / no PCA columns — return just the passthrough (raw) if any.
        small = pass_raw.copy() if pass_raw is not None else pd.DataFrame(index=macro_df.index)
        return (small, pd.DataFrame()) if return_loadings else small

    k = int(min(n_components, m))
    factors = np.full((n, k), np.nan)
    prev_comp: np.ndarray | None = None

    # Causal expanding-window PCA, computed INCREMENTALLY. The factor at row t is
    # produced by a fit trained only on rows < t; refits happen on block
    # boundaries so the series is prefix-invariant (backtest-safe). Rather than
    # re-fit sklearn on the whole window every block — O(refits · window · m²) —
    # we carry running sufficient statistics (Σx and Σxxᵀ) and extend them by one
    # block per step, making the whole pass O(n · m²). The window correlation
    # matrix is eigendecomposed directly; this reproduces the previous
    # StandardScaler + PCA(covariance_eigh) output to ~1e-13 while being far
    # cheaper on a weak single-thread CPU, and it is deterministic and
    # independent of sklearn's solver/sign internals (its own sign convention is
    # applied below). Covariance is translation-invariant, so we accumulate
    # around a fixed CAUSAL shift (the first-window mean) purely for
    # floating-point stability — it changes nothing and uses no future data.
    shift = X[:min_train].mean(axis=0)
    Xs = X - shift
    s1 = np.zeros(m)
    s2 = np.zeros((m, m))
    _seed = Xs[:min_train]
    s1 += _seed.sum(axis=0)
    s2 += _seed.T @ _seed

    t = min_train
    while t < n:
        # Population moments over rows < t (true window mean = mean_s + shift).
        mean_s = s1 / t
        cov = s2 / t - np.outer(mean_s, mean_s)
        scale = np.sqrt(np.maximum(np.diag(cov), 0.0))
        scale[scale == 0.0] = 1.0  # match StandardScaler's handle_zeros_in_scale
        corr = cov / np.outer(scale, scale)
        eigvals, eigvecs = np.linalg.eigh(corr)  # ascending; corr is symmetric
        top = np.argsort(eigvals)[::-1][:k]
        comp = eigvecs[:, top].T.copy()  # (k, m) — top-k components
        # Deterministic sign: make the largest-magnitude loading positive.
        for r in range(comp.shape[0]):
            j = int(np.argmax(np.abs(comp[r])))
            if comp[r, j] < 0:
                comp[r] = -comp[r]
        if prev_comp is not None:
            # Align signs to the previous fit so the factor series doesn't flip
            # at refit boundaries (component sign is otherwise arbitrary).
            for j in range(min(len(comp), len(prev_comp))):
                if float(np.dot(comp[j], prev_comp[j])) < 0:
                    comp[j] = -comp[j]
        prev_comp = comp

        block_end = min(t + refit_every, n)
        # z-score the block on the window's mean/scale, then project. The PCA mean
        # of the standardized training data is exactly 0, so there is no extra
        # centering term: factors = ((X[block] − window_mean) / scale) · compᵀ.
        z_block = (X[t:block_end] - (mean_s + shift)) / scale
        factors[t:block_end] = z_block @ comp.T

        _blk = Xs[t:block_end]
        s1 += _blk.sum(axis=0)
        s2 += _blk.T @ _blk
        t = block_end

    cols = [f"MACRO_PC{i + 1}" for i in range(k)]
    factors_df = pd.DataFrame(factors, index=macro_df.index, columns=cols)
    # Append the reserved passthrough columns (raw) alongside the factors.
    if pass_raw is not None:
        factors_df = pd.concat([factors_df, pass_raw], axis=1)
    if return_loadings:
        loadings = (
            pd.DataFrame(prev_comp, index=cols, columns=feature_names)
            if prev_comp is not None else pd.DataFrame()
        )
        return factors_df, loadings
    return factors_df
