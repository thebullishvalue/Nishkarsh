"""analytics/analogs.py — A4-01 (Hurst sawtooth), A4-13 (OAS), A4-14 (base rate), candidate-pool causality.
Ports council/b3/t_analogs.py and council/a4/{exp_hurst_sawtooth,exp_oas,exp_analog_causality}.py."""
import numpy as np
import pytest
from scipy import stats

import analytics.analogs as AN
from synth_helpers import make_ts

HOLD = (3, 5, 10, 20)


def test_hurst_feature_no_sawtooth():
    ts = make_ts(n=1500, seed=11)
    h = AN._rolling_hurst(ts["Price"].to_numpy(), window=60, step=5)
    post = h[61:]
    assert float(np.mean(post == 0.5)) == 0.0          # v1: 4 of every 5 rows read the 0.5 placeholder
    assert np.corrcoef(post[:-1], post[1:])[0, 1] > 0.6
    changes = np.flatnonzero(np.diff(post) != 0) + 1
    assert len(changes) > 0 and np.all(np.diff(changes) % 5 == 0)   # updates only on the step grid
    assert np.all(AN._rolling_hurst(np.ones(30) * 20, window=60) == 0.5)  # too short → documented 0.5


def test_oas_matches_sklearn():
    OAS = pytest.importorskip("sklearn.covariance").OAS
    rng = np.random.default_rng(0)
    for p, n in [(3, 40), (4, 200), (8, 60), (20, 30)]:
        for kind in ("random", "isotropic"):
            A = rng.standard_normal((p, p)) if kind == "random" else np.eye(p)
            X = rng.standard_normal((n, p)) @ A.T
            S = np.cov(X, rowvar=False, bias=True)
            ref = OAS().fit(X).covariance_
            np.testing.assert_allclose(AN._oas_shrinkage(S, n), ref, rtol=0, atol=1e-10, err_msg=f"{p}x{n} {kind}")


def test_oas_alias_kept():
    assert AN._ledoit_wolf_shrinkage is AN._oas_shrinkage


@pytest.fixture(scope="module")
def full_ts():
    return make_ts(n=2600, seed=3)


def test_candidate_pool_causal_and_theiler(full_ts):
    dates = full_ts["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    gap = max(20, max(HOLD))
    for T in range(400, 2600, 311):
        t = full_ts.iloc[:T].reset_index(drop=True)
        an = AN.find_similar_periods(t, "PE", HOLD, mom_window=20)
        assert an, f"no analogs at T={T}"
        pos = sorted(int(np.flatnonzero(dates[:T] == a["date"])[0]) for a in an)
        assert all(p + max(HOLD) <= T - 1 for p in pos), "analog's forward window reaches today"
        assert all(b - a >= gap for a, b in zip(pos, pos[1:])), "Theiler gap violated"
        assert all(t["Valid"].iloc[p] for p in pos), "warm-up row in the candidate pool"
        assert all(all(v is not None and np.isfinite(v) for v in a["fwd"].values()) for a in an)


def test_summarize_forward_base_rate(full_ts):
    ts = full_ts.iloc[:1500].reset_index(drop=True)
    per = AN.find_similar_periods(ts, "PE", HOLD)
    s = AN.summarize_forward(per, HOLD, ts=ts)
    for h in HOLD:
        row = s[h]
        assert {"median", "positive_pct", "n", "unconditional_positive_pct", "binomial_p"} <= set(row)
        assert 0 <= row["unconditional_positive_pct"] <= 100 and 0 <= row["binomial_p"] <= 1
        k = round(row["positive_pct"] * row["n"] / 100)
        ref = stats.binomtest(k, row["n"], row["unconditional_positive_pct"] / 100).pvalue
        assert row["binomial_p"] == pytest.approx(ref)
    base = AN.unconditional_positive_pct(ts, HOLD)
    assert s[10]["unconditional_positive_pct"] == base[10]
    s0 = AN.summarize_forward(per, (10,))
    assert s0[10]["unconditional_positive_pct"] is None and s0[10]["binomial_p"] is None
