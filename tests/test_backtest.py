"""analytics/backtest.py — A3 v2-8 test 6 (analytic checks), from council/b3/t_det.py [6]."""
import numpy as np
import pandas as pd
import pytest

from analytics.backtest import backtest

KEYS = {"mode", "h", "cost_bps", "gross_sharpe", "net_sharpe", "t", "ann_return", "turnover_per_year",
        "breakeven_bps", "n"}


@pytest.fixture(scope="module")
def drift_world():
    idx = pd.bdate_range("2010-01-01", periods=2520)
    rng = np.random.default_rng(5)
    lp = pd.Series(np.cumsum(0.0004 + 0.01 * rng.standard_normal(len(idx))), index=idx)
    alt = pd.Series(np.where((np.arange(len(idx)) // 10) % 2 == 0, 1.0, -1.0), index=idx)
    return idx, lp, alt


def test_constant_long_under_drift(drift_world):
    idx, lp, _ = drift_world
    b = backtest(pd.Series(1.0, index=idx), lp, 10, 0.0, "tranche")
    assert KEYS <= set(b)
    r1 = lp.diff().shift(-1).dropna()
    assert b["ann_return"] == pytest.approx(252 * r1.iloc[9:].mean(), rel=0.02)
    assert b["gross_sharpe"] == pytest.approx(r1.mean() / r1.std() * np.sqrt(252), abs=0.02)
    assert b["turnover_per_year"] == pytest.approx(252 / len(r1), abs=1e-3)   # one build-up, then hold


def test_alternating_turnover_and_cost(drift_world):
    idx, lp, alt = drift_world
    g0 = backtest(alt, lp, 10, 0.0, "nonoverlap")
    g5 = backtest(alt, lp, 10, 5.0, "nonoverlap")
    per_year = 252 / 10
    assert g5["turnover_per_year"] / per_year == pytest.approx(2.0, abs=0.01)       # |Δp| = 2 per rebalance
    assert (g0["ann_return"] - g5["ann_return"]) / per_year * 1e4 == pytest.approx(10.0, abs=0.05)   # 2c
    assert g5["net_sharpe"] < g0["net_sharpe"]
    ta = backtest(alt, lp, 10, 5.0, "tranche")
    assert ta["turnover_per_year"] == pytest.approx(50.4, abs=0.5)                  # |p_t − p_{t−h}|/h = .2/day


def test_cost_monotone_and_breakeven(drift_world):
    idx, lp, alt = drift_world
    pos = pd.Series(np.sign(lp.diff(20).fillna(0.0)), index=idx)
    nets = [backtest(pos, lp, 10, c, "tranche")["net_sharpe"] for c in (0.0, 2.0, 5.0, 10.0)]
    assert all(a >= b for a, b in zip(nets, nets[1:]))
    b = backtest(pos, lp, 10, 5.0, "tranche")
    assert np.isfinite(b["breakeven_bps"]) and b["n"] > 2000


def test_empty_and_misaligned_inputs(drift_world):
    idx, lp, _ = drift_world
    e = backtest(pd.Series(dtype=float), lp, 10, 5.0, "tranche")
    assert e["n"] == 0 and e["t"] is None
    flat = backtest(pd.Series(np.nan, index=idx), lp, 10, 5.0, "nonoverlap")   # NaN position → flat
    assert flat["turnover_per_year"] == 0 and flat["ann_return"] == 0
    with pytest.raises(ValueError):
        backtest(pd.Series(1.0, index=idx), lp, 10, 5.0, "bogus")
