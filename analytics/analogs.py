"""
Precedent — historical analogs of the NISHKARSH OSCILLATOR state.

Answers one descriptive question, from the system's single inference series
(``convergence/nishkarsh.py``): "on the past dates whose Nishkarsh Oscillator state
looked most like today's, what did the Nifty do next — and is that different from what
it usually does?"

  • State vector: NKO level, its three pillars (top-down, bottom-up, agreement) and its
    5- and 20-day change — i.e. where the read is, what drives it, and where it is going.
  • Match: covariance-aware Mahalanobis distance with OAS shrinkage (Chen, Wiesel, Eldar
    & Hero 2010) over CAUSAL candidates only — dates whose longest forward outcome was
    realized before today.
  • Distinct episodes: greedy Theiler exclusion (Theiler 1986; Farmer & Sidorowich 1987)
    in trading days, so ten analogs are ten episodes, not one episode ten times.
  • Honest base rate: the analogs' up-share is reported against the unconditional
    up-share of the same candidate pool, with an exact binomial p. Descriptive only — it
    votes in the hero only when that p is below EDGE_ALPHA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sstats

STATE_FEATURES = ("NKO", "P_top_down", "P_bottom_up", "P_agreement", "dNKO_5", "dNKO_20")


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


_ledoit_wolf_shrinkage = _oas_shrinkage  # backward-compatible alias (audit A4-13)


def nko_state_frame(nko: pd.DataFrame, price: pd.Series, hold_horizons: tuple[int, ...]) -> pd.DataFrame:
    """Per-date state vector + forward index returns (%) on the NKO frame's dates.

    ``pos`` is the trading-day position (the NKO frame is on the trading calendar), used
    for the Theiler gap and for causality. ``fwd_<h>`` is the % change of ``price`` from
    the date to h trading days later — NaN until realized."""
    df = pd.DataFrame(index=nko.index)
    for c in ("NKO", "P_top_down", "P_bottom_up", "P_agreement"):
        df[c] = pd.to_numeric(nko.get(c), errors="coerce")
    df["dNKO_5"] = df["NKO"] - df["NKO"].shift(5)
    df["dNKO_20"] = df["NKO"] - df["NKO"].shift(20)
    df["two_system"] = nko.get("two_system", pd.Series(True, index=nko.index)).astype(bool)
    df["pos"] = np.arange(len(df), dtype=np.int64)
    px = pd.to_numeric(price, errors="coerce")
    px.index = pd.DatetimeIndex(px.index)
    px = px[~px.index.duplicated(keep="last")].sort_index().reindex(df.index, method="ffill", limit=5)
    for h in hold_horizons:
        df[f"fwd_{int(h)}"] = (px.shift(-int(h)) / px - 1.0) * 100.0
    return df


def find_state_analogs(state: pd.DataFrame, hold_horizons: tuple[int, ...], *, top_n: int = 10,
                       gap: int | None = None, features: tuple[str, ...] = STATE_FEATURES,
                       min_pool: int = 60) -> tuple[list[dict], dict[int, float]]:
    """Top-N distinct past analogs of today's NKO state, and the unconditional up-share.

    Candidates: two-system dates whose state is complete and whose LONGEST forward
    outcome is realized before today (``pos ≤ today − max(h)``) — so no analog's outcome
    overlaps the present. Returns ``(analogs, unconditional_positive_pct_by_h)``."""
    if state is None or state.empty or not hold_horizons:
        return [], {}
    maxh = int(max(hold_horizons))
    gap = int(gap if gap is not None else max(maxh, 20))
    feats = [f for f in features if f in state.columns]
    live = state[state["two_system"] & state[feats].notna().all(axis=1)]
    if live.empty:
        return [], {}
    today = live.iloc[-1]
    fwd_cols = [f"fwd_{int(h)}" for h in hold_horizons]
    pool = live[(live["pos"] <= int(today["pos"]) - maxh) & live[fwd_cols].notna().all(axis=1)]
    if len(pool) < min_pool:
        return [], {}
    X = pool[feats].to_numpy(dtype=np.float64).copy()
    c = today[feats].to_numpy(dtype=np.float64)
    d = mahalanobis_distance_batch(X, c, np.cov(X, rowvar=False))
    sim = 1.0 - d / (d.max() if d.max() > 0 else 1.0)
    acc = select_analogs_theiler(sim, top_n, gap, positions=pool["pos"].to_numpy())
    out = []
    for i in acc:
        r = pool.iloc[int(i)]
        out.append({
            "date": str(pd.Timestamp(pool.index[int(i)]).date()),
            "similarity": float(sim[int(i)]),
            "nko": float(r["NKO"]),
            "pillars": {"top_down": float(r["P_top_down"]), "bottom_up": float(r["P_bottom_up"]),
                        "agreement": float(r["P_agreement"])},
            "dnko_20": float(r["dNKO_20"]),
            "fwd": {int(h): float(r[f"fwd_{int(h)}"]) for h in hold_horizons},
        })
    uncond = {int(h): float((pool[f"fwd_{int(h)}"] > 0).mean() * 100.0) for h in hold_horizons}
    return out, uncond


def summarize_forward(periods: list[dict], hold_horizons: tuple[int, ...],
                      unconditional: dict[int, float] | None = None) -> dict[int, dict]:
    """Per-horizon base rate across the analogs: ``median`` (% return), ``positive_pct``,
    ``n``, ``unconditional_positive_pct`` (the same candidate pool's up-share) and the exact
    two-sided ``binomial_p`` of k/n positives against it. Analogs are Theiler-separated by
    ≥ the longest horizon, so their outcomes do not overlap. "7/10 up vs 58% usually,
    p = 0.35" is the honest read."""
    out: dict[int, dict] = {}
    for h in hold_horizons:
        vals = [p["fwd"].get(int(h)) for p in periods]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        k = sum(1 for v in vals if v > 0)
        u = (unconditional or {}).get(int(h))
        bp = float(_sstats.binomtest(k, len(vals), min(max(u / 100.0, 0.0), 1.0)).pvalue) if u is not None else None
        out[int(h)] = {"median": float(np.median(vals)), "positive_pct": k / len(vals) * 100.0, "n": len(vals),
                       "unconditional_positive_pct": u, "binomial_p": bp}
    return out
