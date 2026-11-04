"""
Nishkarsh — Analog (Similar-Period) matcher.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Covariance-aware historical analog matching, ported from Arthagati's
``find_similar_periods`` (Mahalanobis + trajectory cosine + recency) and inherited
from Tattva. The matcher itself is unchanged; only its INPUTS are adapted to
Nishkarsh:

  • Feature vector is built from the quantities Nishkarsh already computes per day
    (``engine.ts_data``) — conformal extension (AvgZ), net internal breadth, target
    momentum, realized volatility, and rolling Hurst — instead of Arthagati's mood
    features.
  • Forward-return horizons follow the Aarambh forecast lens (AARAMBH_FWD_HORIZON
    primary; AARAMBH_FWD_MOM_K momentum window), not a fixed 5/20/60/90.

It answers an empirical, non-parametric question that complements the model
forecast: "when the target's state looked statistically like today, what did the
target do next?" — a base rate, descriptive not predictive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sstats

from analytics.hurst import hurst_dfa

# ── Blend weights (ported verbatim from Arthagati) ───────────────────────────
# Blend re-tuned for Tattva (2026-06-20, research/analog_tuning_study.py + research/analog_confirm.py:
# 13 targets, non-overlapping OOS IC full + recent-half). The ported Arthagati blend
# (.55/.35/.10) was actively HURTING: trajectory adds ~nothing and recency degrades
# the recent regime. PURE Mahalanobis state-matching is the clear winner — it
# recovers the decayed recent edge (10d recent IC −0.010 → +0.079; 20d −0.083 →
# +0.095) while holding full-sample IC. So trajectory + recency are dropped (weight
# 0; their dead code paths were removed in 2.0.0 — the score is the Mahalanobis
# similarity alone, bit-identical to the old 1/0/0 blend).


# ════════════════════════════════════════════════════════════════════════════
# Core matcher — ported verbatim from Arthagati (arthagati.py)
# ════════════════════════════════════════════════════════════════════════════

def _oas_shrinkage(S: np.ndarray, n: int) -> np.ndarray:
    """Oracle Approximating Shrinkage estimator (Chen, Wiesel, Eldar & Hero
    2010, IEEE Trans. Signal Processing 58(10), eq. 23, with the O(1/p)
    ``2/p`` correction term omitted — negligible for large p, and that
    omission is what makes this match the reference implementation used to
    verify it (``sklearn.covariance.OAS``). Verified to agree with sklearn's
    OAS to ~1e-15 on random SPD inputs, including the small-p (3-4 feature)
    regime this module actually runs in.
    Σ* = ρ·F + (1−ρ)·S  where F = (tr(S)/p)·I  (scaled identity target).
    Returns the shrunk covariance matrix — always well-conditioned.

    (Renamed from ``_ledoit_wolf_shrinkage`` in 2.0.0 — this is OAS, not Ledoit &
    Wolf 2004, a related but distinct shrinkage intensity; the old name is kept
    as an alias for import-site compatibility. The previous formula
    was mis-transcribed — it had ‖S − μI‖²_F in the numerator and tr(S²) in
    the denominator, the reverse of the OAS closed form — which under-shrinks
    exactly where shrinkage matters most: near-isotropic S, where the true ρ
    should approach 1. Measured max deviation from sklearn's OAS: 0.28.)
    """
    p = S.shape[0]
    if p == 0 or n < 2:
        return S
    trace_S = np.trace(S)
    mu = trace_S / p                       # target = μ·I
    alpha = np.mean(S * S)                 # tr(S²)/p² via the Frobenius identity
    mu_sq = mu ** 2
    rho_num = alpha + mu_sq
    rho_den = (n + 1.0) * (alpha - mu_sq / p)
    rho = min(max(rho_num / rho_den, 0.0), 1.0) if rho_den != 0 else 1.0
    return (1.0 - rho) * S + rho * mu * np.eye(p)


_ledoit_wolf_shrinkage = _oas_shrinkage  # backward-compatible alias (audit A4-13)


def select_analogs_theiler(
    scores: np.ndarray, top_n: int, gap: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy top-N selection under a Theiler exclusion window.

    Theiler (1986, Phys. Rev. A 34), adopted for analog/nearest-neighbor
    forecasting by Farmer & Sidorowich (1987, Phys. Rev. Lett. 59): candidates
    within ``gap`` rows of an already-accepted analog are excluded, so the
    returned indices are drawn from genuinely distinct episodes rather than a
    run of adjacent days whose rolling-window state (and h-day forward
    outcome) is nearly identical. Plain top-N-by-score (``nlargest``) does not
    have this property and can return "top_n analogs" that are really 1-3
    independent observations repeated.

    ``positions``: the TEMPORAL row position of each candidate (same length as
    ``scores``). Required when the candidate pool has been filtered (e.g. the
    engine warm-up rows removed) — array offsets then no longer measure time,
    and the exclusion window must be applied on the original row positions.
    Defaults to 0..n-1 (unfiltered pool: offset == time).

    Returns up to ``top_n`` integer offsets INTO ``scores``, best-first.
    """
    if positions is None:
        positions = np.arange(len(scores))
    order = np.argsort(scores)[::-1]
    accepted: list[int] = []
    accepted_time: list[int] = []
    for pos in order:
        p = int(pos)
        t = int(positions[p])
        if all(abs(t - a) >= gap for a in accepted_time):
            accepted.append(p)
            accepted_time.append(t)
            if len(accepted) >= top_n:
                break
    return np.array(accepted, dtype=np.int64)


def mahalanobis_distance_batch(features: np.ndarray, center: np.ndarray,
                               cov_matrix: np.ndarray) -> np.ndarray:
    """Mahalanobis distance: d_M = √((x−μ)ᵀ Σ⁻¹ (x−μ)).
    Uses OAS (Oracle Approximating Shrinkage) for a well-conditioned covariance
    inverse, replacing ad-hoc diagonal regularization.
    """
    diff = features - center
    n_samples = features.shape[0]
    shrunk_cov = _oas_shrinkage(cov_matrix, n_samples)
    try:
        cov_inv = np.linalg.inv(shrunk_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(shrunk_cov)
    left = diff @ cov_inv
    d_sq = np.maximum(np.sum(left * diff, axis=1), 0)
    return np.sqrt(d_sq)


# ════════════════════════════════════════════════════════════════════════════
# Tattva adaptation — feature vector from engine.ts_data
# ════════════════════════════════════════════════════════════════════════════

def _rolling_hurst(price: np.ndarray, window: int = 120, step: int = 5) -> np.ndarray:
    """Rolling Hurst (DFA) over a trailing window, computed every ``step`` points
    and forward-filled — a regime persistence/mean-reversion feature. Mirrors
    Arthagati's stepped rolling-Hurst approach to keep it cheap on ~2k-row series.

    Off-grid rows carry the latest on-grid estimate (causal forward fill; 0.5 only
    before the first estimate). The v1 placeholder was 0.5 on every row, so the
    ``.replace(0.0, nan).ffill()`` never fired and 4 of every 5 rows read 0.5 —
    the analog match keyed on (row mod 5) (audit A4-01).
    """
    n = len(price)
    if n < window:
        return np.full(n, 0.5, dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    log_p = np.log(np.where(price > 0, price, np.nan))
    rets = np.diff(log_p)
    for i in range(window, n, step):
        seg = rets[i - window:i]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 30:
            out[i] = hurst_dfa(seg)
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=np.float64, copy=True)


def _build_feature_frame(ts: pd.DataFrame, mom_window: int) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the per-day analog state matrix from Tattva's engine.ts_data.

    Feature set re-tuned for Tattva (research/analog_tuning_study.py): AvgZ was DROPPED —
    it degraded the recent regime (10d recent IC −0.010 → +0.034 without it) while
    NetBreadth proved critical and the candidate extras (ModelSpread, ExtremeBreadth,
    SignalBreadth, ConvictionRaw, MomentumLong) added nothing. Kept (availability-
    guarded):
      • Momentum     — trailing ``mom_window``-day log-return of the target Price
      • Realized Vol — rolling σ of daily log-returns over ``mom_window``
      • NetBreadth   — OversoldBreadth − OverboughtBreadth, if present (key feature)
      • Hurst        — rolling DFA Hurst of the target Price

    Returns ``(frame, feature_cols)`` where ``frame`` carries the feature columns
    plus ``Price`` and ``Date`` (forward returns and recency are derived from these).
    """
    if ts is None or ts.empty or "Price" not in ts.columns:
        return pd.DataFrame(), []

    df = ts.reset_index(drop=True).copy()
    price = pd.to_numeric(df["Price"], errors="coerce").to_numpy(dtype=np.float64)
    log_ret = pd.Series(np.log(np.where(price > 0, price, np.nan))).diff()

    feat = pd.DataFrame(index=df.index)
    feat["Momentum"] = log_ret.rolling(mom_window, min_periods=mom_window).sum()
    feat["RealizedVol"] = log_ret.rolling(mom_window, min_periods=mom_window).std()

    if {"OversoldBreadth", "OverboughtBreadth"}.issubset(df.columns):
        feat["NetBreadth"] = (pd.to_numeric(df["OversoldBreadth"], errors="coerce")
                              - pd.to_numeric(df["OverboughtBreadth"], errors="coerce"))
        # The engine's own [0, MIN_TRAIN_SIZE) warm-up (no genuine walk-forward
        # forecast — see the B4 audit fix) has no real breadth reading; force
        # NetBreadth to NaN there so the analog pool can exclude the warm-up
        # instead of matching against a fabricated "neutral" state.
        if "Valid" in df.columns:
            feat.loc[~df["Valid"].astype(bool), "NetBreadth"] = np.nan
    feat["Hurst"] = _rolling_hurst(price, window=max(60, mom_window * 3))

    feature_cols = [c for c in ("Momentum", "RealizedVol", "NetBreadth", "Hurst")
                    if c in feat.columns]
    feat["Price"] = price
    feat["Date"] = df["Date"].to_numpy() if "Date" in df.columns else df.index.to_numpy()
    # Carried DISPLAY-ONLY columns (not in feature_cols, never matched on):
    #   • AvgZ — the engine's extension z-score at the analog's date. It is
    #     not a matching feature, but the Precedent tab's analog cards display
    #     "Extension (Z)" and key their tier badge/color off it; without
    #     carrying it every card silently read the dict default 0.0 →
    #     permanently "Neutral" badges (audit finding B14).
    #   • ValidRow — the engine's Valid flag (False through the walk-forward
    #     warm-up). Lets find_similar_periods exclude rows whose NetBreadth
    #     would be median-filled fabrication from the candidate pool.
    if "AvgZ" in df.columns:
        feat["AvgZ"] = pd.to_numeric(df["AvgZ"], errors="coerce").to_numpy(dtype=np.float64)
    if "Valid" in df.columns:
        feat["ValidRow"] = df["Valid"].astype(bool).to_numpy()
    return feat, feature_cols


def find_similar_periods(
    ts: pd.DataFrame,
    target_col: str,
    hold_horizons: tuple[int, ...] = (3, 5, 10, 20),
    *,
    mom_window: int = 20,
    top_n: int = 10,
) -> list[dict]:
    """Analog finder — covariance-aware Mahalanobis state match.

    Score = Mahalanobis similarity (1 − d/max_d; RELATIVE to the farthest
    candidate, not an absolute probability). Trajectory/recency terms were
    dropped (weight 0, see the module header) and their code removed. Candidates are drawn from genuinely-forecast
    history only (engine warm-up rows excluded) and selected under a Theiler
    exclusion window so each returned analog is a distinct episode.

    Returns top-N analogs as dicts with the target's forward returns at each
    ``hold_horizons`` value (% Price change t→t+h).
    """
    feat, feature_cols = _build_feature_frame(ts, mom_window)
    if feat.empty or len(feature_cols) < 2:
        return []

    n = len(feat)
    purge = int(max(hold_horizons)) if hold_horizons else 20
    # Exclude the tail: those rows have no realized forward path yet.
    historical = feat.iloc[:n - purge].copy()
    # Exclude the engine's walk-forward WARM-UP rows from the candidate pool
    # (ValidRow False, see _build_feature_frame): their NetBreadth is genuinely
    # missing, and matching against a median-filled fabrication is not a state
    # match. The frame keeps its original RangeIndex labels after this filter,
    # so `historical.index` remains the TEMPORAL row position — the forward-
    # return lookups and the Theiler gap below both rely on that.
    if "ValidRow" in historical.columns:
        historical = historical[historical["ValidRow"]]
    if len(historical) < 30:
        return []

    latest = feat.iloc[-1]
    current_vec = latest[feature_cols].to_numpy(dtype=np.float64)
    # .copy() is load-bearing: on Streamlit Cloud DataFrame.to_numpy() can return a
    # READ-ONLY view (consolidated/cached buffers), and the median-fill below writes
    # in place → "assignment destination is read-only" without a writable copy.
    hist_matrix = historical[feature_cols].to_numpy(dtype=np.float64).copy()

    # Clean NaN/Inf → column medians (port of Arthagati's cleaning)
    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        median_val = np.median(col_data[valid]) if valid.any() else 0.0
        hist_matrix[~valid, col] = median_val
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)

    # ── Part 1: Mahalanobis (the signal — covariance-aware state match) ─────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(float(cov_matrix), 1e-6)]])
    maha_dist = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    max_dist = maha_dist.max() if maha_dist.max() > 0 else 1.0
    maha_sim = 1.0 - (maha_dist / max_dist)

    # ── Combined (pure Mahalanobis; trajectory/recency removed at weight 0) ─
    combined = maha_sim
    price_all = feat["Price"].to_numpy(dtype=np.float64)
    historical = historical.copy()
    historical["similarity"] = combined

    # Theiler exclusion window (see select_analogs_theiler docstring): without
    # it, top-N-by-similarity typically returns a RUN of adjacent days from
    # 1-3 historical episodes whose h-day forward outcomes overlap almost
    # completely — "10 analogs" that are really 1-3 independent observations,
    # inflating summarize_forward's apparent median/positive_pct precision.
    # gap = the longer of the momentum window (how far back the STATE feature
    # vector looks) and the longest requested forward horizon (how far the
    # OUTCOME extends) — below that, either the state or the outcome (or both)
    # would overlap a neighbor's. positions= carries the ORIGINAL temporal row
    # positions (RangeIndex labels) because the pool above may be filtered
    # (warm-up rows removed) — the gap must be measured in trading days.
    gap = max(int(mom_window), int(max(hold_horizons)) if hold_horizons else 0, 1)
    accepted = select_analogs_theiler(
        combined, top_n, gap, positions=historical.index.to_numpy(),
    )
    top = historical.iloc[accepted]

    def _num(row: dict, key: str, default: float) -> float:
        """Finite-or-default coercion — carried columns can be NaN."""
        v = row.get(key)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return default
        return fv if np.isfinite(fv) else default

    results: list[dict] = []
    for pos, row in zip(top.index, top.to_dict("records")):
        price_at = float(row["Price"]) if row["Price"] and row["Price"] > 0 else None
        fwd: dict[int, float | None] = {}
        for h in hold_horizons:
            fi = int(pos) + h
            if fi < len(price_all) and price_at:
                fwd[h] = (price_all[fi] / price_at - 1) * 100
            else:
                fwd[h] = None
        date_val = row["Date"]
        date_str = (pd.Timestamp(date_val).strftime("%Y-%m-%d")
                    if not isinstance(date_val, (int, np.integer)) else f"t={int(date_val)}")
        results.append({
            "date": date_str,
            "similarity": float(row["similarity"]),
            "price": price_at or 0.0,
            "momentum": _num(row, "Momentum", 0.0) * 100,  # → %
            "realized_vol": _num(row, "RealizedVol", 0.0) * 100,
            "avgz": _num(row, "AvgZ", 0.0),
            "net_breadth": _num(row, "NetBreadth", 0.0),
            "hurst": _num(row, "Hurst", 0.5),
            "fwd": {int(h): fwd[h] for h in hold_horizons},
        })
    return results


def unconditional_positive_pct(ts: pd.DataFrame, hold_horizons: tuple[int, ...]) -> dict[int, float]:
    """Unconditional h-day positive-return rate (%) over the SAME candidate pool that
    ``find_similar_periods`` draws analogs from (Valid rows with a realized path at
    the longest horizon) — the base rate an analog hit rate must beat."""
    if ts is None or len(ts) == 0 or "Price" not in ts.columns or not hold_horizons:
        return {}
    df = ts.reset_index(drop=True)
    price = pd.to_numeric(df["Price"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(price)
    purge = int(max(hold_horizons))
    pool = np.arange(max(n - purge, 0))
    if "Valid" in df.columns:
        pool = pool[df["Valid"].astype(bool).to_numpy()[pool]]
    out: dict[int, float] = {}
    for h in hold_horizons:
        h = int(h)
        p0, p1 = price[pool], price[pool + h]
        ok = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0)
        if ok.any():
            out[h] = float(np.mean(p1[ok] > p0[ok]) * 100.0)
    return out


def summarize_forward(periods: list[dict], hold_horizons: tuple[int, ...],
                      ts: pd.DataFrame | None = None) -> dict[int, dict]:
    """Per-horizon base-rate summary across the analogs.

    Keys per horizon: ``median`` (% return), ``positive_pct``, ``n``, and (audit
    A4-14) ``unconditional_positive_pct`` — the h-day positive rate over the whole
    candidate pool (needs ``ts``, the frame passed to ``find_similar_periods``) — and
    ``binomial_p``: two-sided exact binomial p of ``k/n`` positives against that
    unconditional rate (analogs are Theiler-separated by ≥ the longest horizon, so
    their outcomes do not overlap). Both are None when ``ts`` is not given.
    "7/10 up vs 58% unconditional, p = 0.35" is the honest read."""
    base = unconditional_positive_pct(ts, hold_horizons) if ts is not None else {}
    out: dict[int, dict] = {}
    for h in hold_horizons:
        vals = [p["fwd"].get(int(h)) for p in periods]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            k = sum(1 for v in vals if v > 0)
            u = base.get(int(h))
            bp = None
            if u is not None:
                bp = float(_sstats.binomtest(k, len(vals), min(max(u / 100.0, 0.0), 1.0)).pvalue)
            out[int(h)] = {
                "median": float(np.median(vals)),
                "positive_pct": k / len(vals) * 100.0,
                "n": len(vals),
                "unconditional_positive_pct": u,
                "binomial_p": bp,
            }
    return out
