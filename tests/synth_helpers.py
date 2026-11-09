"""Seeded synthetic generators shared by the test-suite (no network, no IO).

Ported verbatim (same seeds / DGPs) from the council reproducers so every test is the
permanent form of a verified claim:

- ``make_levels`` / ``predictive_xy`` — council/a1/common.py (Aarambh predictive inputs,
  built exactly like app.py's forward branch).
- ``make_ohlcv`` / ``make_macro`` / ``make_market`` — council/a2/synth.py (Nirnay).
- ``conclusion_world`` — council/b3/synth_b3.py ``world`` (conclusion layer, C-AAR/C-NIR shaped).
- ``make_ts`` — council/a4/synth.py (analog matcher input).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

H = 10   # AARAMBH_FWD_HORIZON
K = 20   # AARAMBH_FWD_MOM_K
P = 14   # len(DEFAULT_PREDICTORS)


# ── Aarambh (a1/common.py) ───────────────────────────────────────────────────

def make_levels(seed: int, n_rows: int = 4900, p: int = P, signal: float = 0.0,
                vol: float = 0.01) -> pd.DataFrame:
    """p random-walk predictors + a strictly-positive PE (independent RW when signal=0)."""
    rng = np.random.default_rng(seed)
    lev = np.cumsum(rng.standard_normal((n_rows, p)), axis=0)
    mom0 = pd.Series(lev[:, 0]).diff(K).to_numpy()
    z0 = np.nan_to_num(mom0 / np.sqrt(K))
    eps = rng.standard_normal(n_rows)
    r = np.zeros(n_rows)
    r[1:] = vol * (signal * z0[:-1] + eps[1:])
    pe = 20.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame(lev, columns=[f"F{i}" for i in range(p)])
    df.insert(0, "PE", pe)
    return df


def predictive_xy(df: pd.DataFrame, h: int = H, k: int = K):
    """Predictive representation: X = k-day momentum, y = h-day forward log change."""
    target = "PE"
    feats = [c for c in df.columns if c != target]
    lvl = df[[target] + feats].astype(float)
    tlog = np.log(lvl[target].where(lvl[target] > 0))
    fwd = tlog.shift(-h) - tlog
    mom = lvl[feats].diff(k).replace([np.inf, -np.inf], np.nan)
    valid = mom.notna().all(axis=1).to_numpy()
    data = df.loc[valid].reset_index(drop=True)
    X = mom.loc[valid].to_numpy(dtype=np.float64)
    fwd_valid = fwd.loc[valid].to_numpy(dtype=np.float64)
    y = np.nan_to_num(fwd_valid, nan=0.0)
    mask = np.isfinite(fwd_valid)
    price = data[target].to_numpy(dtype=np.float64)
    return X, y, mask, price, feats


def fit_engine(X, y, m, p, mn: int, mx: int, purge: int = H, fwd: bool = True):
    from engines.aarambh import FairValueEngine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairValueEngine().fit(X, y, forward_signal=fwd, purge=purge, price=p if fwd else None,
                                     label_mask=m if fwd else None, min_train=mn, max_train=mx)


def rank_ic(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    return float(stats.spearmanr(a[m], b[m])[0])


def ic_nonovl(sig, y, ok, h: int = H) -> tuple[float, int]:
    """Mean over the h phases of the non-overlapping rank IC."""
    idx = np.where(ok & np.isfinite(sig))[0]
    return float(np.nanmean([rank_ic(sig[idx[o::h]], y[idx[o::h]]) for o in range(h)])), len(idx)


def frame_diff(a: pd.DataFrame, b: pd.DataFrame, cols) -> dict:
    """{column: n_mismatched_rows} (NaN == NaN; exact equality otherwise)."""
    bad = {}
    for c in cols:
        x, z = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fc":
            d = ~((x == z) | (np.isnan(x) & np.isnan(z)))
        else:
            d = x != z
        if np.any(d):
            bad[c] = int(np.sum(d))
    return bad


# ── Nirnay (a2/synth.py) ─────────────────────────────────────────────────────

def make_ohlcv(n: int = 1750, seed: int = 0, start: str = "2019-01-01",
               mkt: np.ndarray | None = None, beta: float = 1.0,
               idx: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if idx is None:
        idx = pd.bdate_range(start, periods=n)
    n = len(idx)
    idio = rng.normal(0.0003, 0.015, n)
    r = idio + (beta * mkt[:n] if mkt is not None else 0.0)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    vol = rng.lognormal(13, 0.4, n)
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)


def make_macro(idx: pd.DatetimeIndex, k: int = 11, seed: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(idx)
    common = np.cumsum(rng.normal(0, 0.01, n))
    return pd.DataFrame({f"M{j}": 50 * np.exp(0.5 * common + np.cumsum(rng.normal(0, 0.008, n)))
                         for j in range(k)}, index=idx)


def make_market(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0015, 0.0, 0.0015], size=n // 120 + 1), 120)[:n]
    return drift + rng.normal(0, 0.009, n)


def run_nirnay(d: pd.DataFrame, macro_cols) -> pd.DataFrame:
    """run_full_analysis with the B2 verification parameters."""
    from engines.nirnay import run_full_analysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_full_analysis(d.copy(), 20, 14, 1.5, 0.6, macro_columns=list(macro_cols), num_vars=5)[0]


# ── Conclusion layer (b3/synth_b3.py) ────────────────────────────────────────

def _ar1(n, phi, rng):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x * np.sqrt(1 - phi ** 2)


def conclusion_world(seed=0, n=4900, n_nir=1750, invalid=150, k_a=0.045, k_lift=0.0, sigma=0.01, noise_a=0.3,
                     nirnay_tail_gap=0, nirnay_holes=0):
    """(aarambh_ts, nirnay_daily, price_levels). f AR(.97) latent forecast, g AR(.90) Nirnay state,
    r_t = σ(k_a f_t + k_lift(2·agree_t − 1) f_t + ε_t), agree = sign(f) == sign(NIRNAY_SIGN·g):
    the planted lift is defined in the configured Nirnay polarity, so the world tests the
    mechanism whichever polarity is active (core.config.NIRNAY_POLARITY)."""
    from analytics.causal_stats import causal_z
    from core.config import NIRNAY_SIGN
    rng = np.random.default_rng(seed)
    f = _ar1(n, .97, rng)
    g = _ar1(n, .90, rng)
    eps = rng.standard_normal(n)
    agree = (np.sign(f) == np.sign(NIRNAY_SIGN * g)).astype(float)
    r = sigma * (k_a * f + k_lift * (2 * agree - 1) * f + eps)
    logp = np.concatenate([[0.0], np.cumsum(r)])[:n]
    dates = pd.bdate_range(end="2026-10-30", periods=n)
    pred = f + noise_a * rng.standard_normal(n)
    valid = np.arange(n) >= invalid
    fc = np.where(valid, pred * 0.01, np.nan)
    fz = causal_z(pd.Series(fc), 126, 4.0).to_numpy()
    cb = -100 * np.tanh(fz / 2)
    ats = pd.DataFrame({"Valid": valid, "Forecast": fc, "ForecastZ": fz, "ConvictionRaw": -fz * 10,
                        "ConvictionScore": cb, "ConvictionBounded": cb,
                        "OversoldBreadth": np.clip(50 + 20 * fz, 0, 100),
                        "Regime": np.where(fz > 1, "OVERSOLD", np.where(fz < -1, "OVERBOUGHT", "NEUTRAL")),
                        "Price": np.exp(logp)}, index=dates)
    ats.index.name = "Date"
    avg = 4.0 * g
    nd = pd.DataFrame({"Avg_Signal": avg, "n_live": 50, "n_universe": 50, "coverage": 1.0,
                       "Oversold_Pct": 100 * norm.cdf((-2 - avg) / 3), "Overbought_Pct": 100 * norm.cdf((avg - 2) / 3),
                       "Regime_Bull_Pct": 100 * norm.cdf(avg / 4) * .8, "Regime_Bear_Pct": 100 * norm.cdf(-avg / 4) * .8},
                      index=dates).iloc[n - n_nir: n - nirnay_tail_gap]
    if nirnay_holes:
        drop = rng.choice(len(nd) - 10, nirnay_holes, replace=False)
        nd = nd.drop(nd.index[drop])
    price = pd.Series(np.exp(logp) * 100, index=dates, name="^NSEI")
    return ats, nd, price


# ── Analogs (a4/synth.py) ────────────────────────────────────────────────────

def make_ts(n=2000, seed=7, warmup=30, drift=0.0002, vol=0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    logp = np.cumsum(rng.normal(drift, vol, n)) + np.log(20.0)
    price = np.exp(logp)
    os_ = np.zeros(n)
    ob = np.zeros(n)
    a = b = 0.0
    for i in range(n):
        a = 0.95 * a + rng.normal(0, 8)
        b = 0.95 * b + rng.normal(0, 8)
        os_[i] = np.clip(30 + a, 0, 100)
        ob[i] = np.clip(30 + b, 0, 100)
    valid = np.arange(n) >= warmup
    avgz = np.where(valid, rng.normal(0, 1, n), np.nan)
    dates = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"Date": dates, "Price": price, "OversoldBreadth": os_,
                         "OverboughtBreadth": ob, "AvgZ": avgz, "Valid": valid})
