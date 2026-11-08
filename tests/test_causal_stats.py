"""analytics/causal_stats.py — causal_z is a FROZEN contract; inference helpers are pure."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics.causal_stats import (causal_z, fisher_z, newey_west_t, nonoverlap_index, spearman,
                                    stationary_block_bootstrap_ic)


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.standard_normal(1500)) * 0.1 + rng.standard_normal(1500))
    x.iloc[[5, 300, 301, 900]] = np.nan
    return x


def test_causal_z_prefix_invariant(series):
    full = causal_z(series)
    for T in (130, 400, 1000, 1499):
        pd.testing.assert_series_equal(causal_z(series.iloc[:T]), full.iloc[:T])


def test_causal_z_strictly_prior(series):
    """z_t uses only x_<t: perturbing x_t (or any later row) never changes z_<=t-1,
    and z_t equals the hand-computed strictly-prior statistic."""
    z = causal_z(series, min_periods=126, clip=4.0)
    t = 700
    x2 = series.copy()
    x2.iloc[t:] += 50.0
    z2 = causal_z(x2, min_periods=126, clip=4.0)
    pd.testing.assert_series_equal(z.iloc[:t], z2.iloc[:t])
    prior = series.iloc[:t].dropna()
    expect = np.clip((series.iloc[t] - prior.mean()) / prior.std(ddof=1), -4, 4)
    assert z.iloc[t] == pytest.approx(expect, rel=1e-10)


def test_causal_z_warmup_nan_clip_and_window():
    x = pd.Series(np.arange(300, dtype=float))
    z = causal_z(x, min_periods=126)
    assert z.iloc[:126].isna().all() and np.isfinite(z.iloc[126])
    assert z.max() <= 4.0                       # a trending series saturates at the clip
    zc = causal_z(pd.Series(np.r_[np.zeros(200), 1.0]), min_periods=126)
    assert zc.isna().all()                      # zero prior std → NaN, not inf
    zw = causal_z(x, min_periods=20, window=50)
    assert np.isfinite(zw.iloc[60])


def test_spearman_matches_scipy_and_guards():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(200), rng.standard_normal(200)
    b = 0.3 * a + b
    a[3] = np.nan
    m = np.isfinite(a) & np.isfinite(b)
    assert spearman(a, b) == pytest.approx(stats.spearmanr(a[m], b[m])[0], abs=1e-12)
    assert np.isnan(spearman(a[:5], b[:5]))            # < min_obs
    assert np.isnan(spearman(np.ones(50), b[:50]))     # constant side
    with pytest.raises(ValueError):
        spearman(a, b[:10])


def test_fisher_z():
    z, v = fisher_z(0.2, 103)
    assert z == pytest.approx(np.arctanh(0.2)) and v == pytest.approx(1.06 / 100)
    assert np.isfinite(fisher_z(1.0, 50)[0])
    assert all(np.isnan(fisher_z(np.nan, 50)))
    assert np.isnan(fisher_z(0.1, 3)[1])


def test_newey_west_t():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(5000)
    t0 = newey_west_t(x, 0)
    assert t0 == pytest.approx(x.mean() / (x.std(ddof=0) / np.sqrt(len(x))), rel=1e-10)
    assert abs(newey_west_t(x, 10)) < 4
    # overlapping (MA(9)) sums: NW t with 9 lags ≈ iid t of the underlying; plain t is inflated
    e = rng.standard_normal(20000) + 0.02
    ma = np.convolve(e, np.ones(10), "valid")
    t_nw, t_iid = newey_west_t(ma, 9), newey_west_t(e, 0)
    assert abs(t_nw - t_iid) < 0.25 * abs(t_iid) + 0.5
    assert np.isnan(newey_west_t(x[:8], 5))


def test_bootstrap_ic_deterministic_by_seed():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(300)
    b = 0.3 * a + rng.standard_normal(300)
    c1 = stationary_block_bootstrap_ic(a, b, 20, reps=500, seed=11)
    c2 = stationary_block_bootstrap_ic(a, b, 20, reps=500, seed=11)
    c3 = stationary_block_bootstrap_ic(a, b, 20, reps=500, seed=12)
    assert c1 == c2 and c1 != c3
    lo, hi = c1
    assert lo < spearman(a, b) < hi
    assert all(np.isnan(stationary_block_bootstrap_ic(a[:10], b[:10], 5, reps=100)))


def test_nonoverlap_index():
    idx = pd.date_range("2020-01-01", periods=25)
    assert list(nonoverlap_index(idx, 10)) == [idx[0], idx[10], idx[20]]
    assert list(nonoverlap_index(idx, 10, phase=3)) == [idx[3], idx[13], idx[23]]
    assert nonoverlap_index(25, 10, phase=12).tolist() == [2, 12, 22]   # phase taken mod h
    assert nonoverlap_index(np.arange(7), 3).tolist() == [0, 3, 6]
    with pytest.raises(ValueError):
        nonoverlap_index(idx, 0)
