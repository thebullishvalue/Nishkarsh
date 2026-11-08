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

from importlib.util import find_spec

# The PCA is computed directly from the running covariance (numpy eigh); sklearn
# is only probed so callers keep the historical "PCA unavailable → raw macros"
# degradation path when the scientific stack is incomplete.
_HAS_SKLEARN = find_spec("sklearn") is not None


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
    recent fit trained only on rows ``< t``. Columns enter the fit only once
    they have ``min_train`` real (non-imputed) rows, and the fit uses the
    common window on which all its columns are real. No future information enters any
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
        Minimum REAL rows a column needs (before row t) to enter the fit at t;
        rows before the first fit are NaN (warm-up). A column's leading NaN
        prefix is never imputed. Components beyond the window's effective rank
        are NaN (never null-space directions).
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

    # ffill only — NO bfill and NO zero-fill. A column's leading never-observed
    # prefix is NOT data: it stays NaN and the column is simply absent from the
    # PCA until it has `min_train` real rows (A2-11). Zero-filling it made the
    # expanding standardization see a 0 → price-level step (|PC| up to ~116)
    # and gave zero-variance columns null-space eigenvectors.
    clean = pca_input.apply(pd.to_numeric, errors="coerce").ffill()
    if stationarize:
        # Causal rolling z-score (uses rows ≤ t only — no future) + clip, so
        # trending price levels become bounded, stationary oscillators. Leading
        # warm-up rows stay NaN (not real); an inner flat window (σ = 0) is z = 0.
        _mp = max(z_window // 4, 20)
        mu = clean.rolling(z_window, min_periods=_mp).mean()
        sd = clean.rolling(z_window, min_periods=_mp).std()
        z = ((clean - mu) / sd.replace(0, np.nan)).clip(-5.0, 5.0)
        started = z.notna().cummax()
        clean = z.where(~started | z.notna(), 0.0)
    feature_names = list(clean.columns)
    X = clean.to_numpy(dtype=np.float64)
    n, m = X.shape
    if n <= min_train or m == 0:
        # Too few rows / no PCA columns — return just the passthrough (raw) if any.
        small = pass_raw.copy() if pass_raw is not None else pd.DataFrame(index=macro_df.index)
        return (small, pd.DataFrame()) if return_loadings else small

    k = int(min(n_components, m))   # output width: a function of the panel's columns only
    factors = np.full((n, k), np.nan)
    prev_comp = np.zeros((k, m))     # last fit's loadings embedded in all-m coordinates
    have_prev = np.zeros(k, dtype=bool)

    # First real row per column (after ffill a column is real from there on).
    real = np.isfinite(X)
    first_real = np.where(real.any(axis=0), real.argmax(axis=0), n)

    # Causal expanding-window PCA over the COMMON real window. At refit row t
    # the eligible set E = {columns with ≥ min_train real rows before t}; the
    # window is [s, t) with s = max first_real over E, on which every column
    # of E is real. The factor at row t uses a fit on rows < t only, refits
    # happen on fixed block boundaries, and eligibility is decided from rows
    # < t, so the series is prefix-invariant (backtest-safe). Sufficient
    # statistics (Σx, Σxxᵀ around a fixed causal shift, for fp stability) are
    # extended block by block — O(n·m²); when a column joins E the window
    # start moves to its first real row and the sums are rebuilt over
    # [s, t) (that span is < min_train + refit_every rows).
    elig = np.zeros(0, dtype=np.int64)
    s_win = 0
    shift = np.zeros(0)
    s1 = np.zeros(0)
    s2 = np.zeros((0, 0))
    last_comp: np.ndarray | None = None

    t = min_train
    while t < n:
        block_end = min(t + refit_every, n)
        new_elig = np.flatnonzero(first_real <= t - min_train)
        if new_elig.size != elig.size:
            elig = new_elig
            s_win = int(first_real[elig].max())
            W = X[s_win:t][:, elig]
            shift = W.mean(axis=0)
            Ws = W - shift
            s1 = Ws.sum(axis=0)
            s2 = Ws.T @ Ws
        if elig.size:
            nw = t - s_win
            mean_s = s1 / nw
            cov = s2 / nw - np.outer(mean_s, mean_s)
            var = np.maximum(np.diag(cov), 0.0)
            mean_true = mean_s + shift
            # Zero-variance columns (pegged / flat over the window) carry no
            # information and would only add null-space directions: drop them.
            keep = var > (1e-9 * np.maximum(1.0, np.abs(mean_true))) ** 2
            if keep.any():
                kc = np.flatnonzero(keep)
                scale = np.sqrt(var[kc])
                corr = cov[np.ix_(kc, kc)] / np.outer(scale, scale)
                eigvals, eigvecs = np.linalg.eigh(corr)  # ascending; symmetric
                order = np.argsort(eigvals)[::-1]
                # Effective rank: never extract a component from the null space.
                rank = int(np.sum(eigvals > max(eigvals.max(), 0.0) * 1e-10))
                k_t = min(k, rank)
                cols_t = elig[kc]
                comp = np.zeros((k_t, m))
                comp[:, cols_t] = eigvecs[:, order[:k_t]].T
                for r in range(k_t):
                    # Deterministic sign: largest-magnitude loading positive,
                    # then align to the previous fit of the same component so
                    # the factor series doesn't flip at refit boundaries.
                    j = int(np.argmax(np.abs(comp[r])))
                    if comp[r, j] < 0:
                        comp[r] = -comp[r]
                    if have_prev[r] and float(np.dot(comp[r], prev_comp[r])) < 0:
                        comp[r] = -comp[r]
                    prev_comp[r] = comp[r]
                    have_prev[r] = True
                # factors = ((X[block] − window_mean) / scale) · compᵀ
                z_block = (X[t:block_end][:, cols_t] - mean_true[kc]) / scale
                factors[t:block_end, :k_t] = z_block @ comp[:, cols_t].T
                last_comp = comp

        if elig.size:
            _blk = X[t:block_end][:, elig] - shift
            s1 = s1 + _blk.sum(axis=0)
            s2 = s2 + _blk.T @ _blk
        t = block_end

    cols = [f"MACRO_PC{i + 1}" for i in range(k)]
    factors_df = pd.DataFrame(factors, index=macro_df.index, columns=cols)
    # Append the reserved passthrough columns (raw) alongside the factors.
    if pass_raw is not None:
        factors_df = pd.concat([factors_df, pass_raw], axis=1)
    if return_loadings:
        # Most recent fit's loadings (PC × feature; 0 for columns not in that
        # fit). PCs beyond that fit's effective rank are omitted.
        loadings = (
            pd.DataFrame(last_comp, index=cols[: len(last_comp)], columns=feature_names)
            if last_comp is not None else pd.DataFrame()
        )
        return factors_df, loadings
    return factors_df
