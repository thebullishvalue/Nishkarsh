"""engines/nirnay.py + analytics/regime.py (contract C-NIR). Ports council/b2/verify.py at small sizes."""
import numpy as np
import pandas as pd
import pytest

from analytics.regime import (AdaptiveKalmanFilter, CUSUMDetector, _prepare_inputs, _regime_loop_njit,
                              regime_loop_reference)
from core.config import NIRNAY_HMM_STD_FLOOR as FLOOR
from core.config import NIRNAY_WARMUP_BARS
from engines.nirnay import aggregate_constituent_timeseries
from synth_helpers import make_macro, make_market, make_ohlcv, run_nirnay

PCT_COLS = ("Oversold_Pct", "Overbought_Pct", "Regime_Bull_Pct", "Regime_Bear_Pct",
            "Regime_Neutral_Pct", "Regime_Transition_Pct")
DAILY_REQUIRED = ("n_universe", "n_live", "coverage", *PCT_COLS, "Avg_Signal", "Dispersion", "Buy_Signals",
                  "Sell_Signals", "Change_Points", "Total_Analyzed", "avg_hmm_bull", "avg_hmm_bear")


def _repaint(a: pd.DataFrame, b: pd.DataFrame) -> list:
    bad = []
    for c in a.columns:
        x, y = a[c], b[c].reindex(a.index)
        if (not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y))
                or pd.api.types.is_bool_dtype(x)):
            if (x.astype(str) != y.astype(str)).any():
                bad.append(c)
        else:
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if (np.isnan(xv) != np.isnan(yv)).any() or np.nanmax(np.abs(xv - yv), initial=0) > 1e-12:
                bad.append(c)
    return bad


# ── append invariance ────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["plain", "late+gap"])
def test_run_full_analysis_append_invariant(variant):
    N, K = 600, 60
    full = make_ohlcv(N + K, seed=1, mkt=make_market(N + K))
    macro = make_macro(full.index, k=6)
    full = full.join(macro)
    if variant == "late+gap":
        full.iloc[:100, :5] = np.nan       # NaN-Close pre-listing prefix
        full.iloc[300:303, 3] = np.nan     # inner missing bars
    a = run_nirnay(full.iloc[:N], macro.columns)
    b = run_nirnay(full, macro.columns)
    assert len(a) > 400
    assert _repaint(a, b) == []
    assert "Warmup" in a.columns and int(a["Warmup"].sum()) == NIRNAY_WARMUP_BARS
    if variant == "late+gap":
        assert len(a) == N - 100 - 3       # NaN-Close rows dropped from the output


@pytest.mark.parametrize("stationarize", [False, True])
def test_macro_factors_append_invariant(stationarize):
    from analytics.factors import build_causal_macro_factors
    idx = pd.bdate_range("2019-01-01", periods=700)
    pan = make_macro(idx, k=12, seed=5)
    pan["YOUNG"] = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, .003, len(pan))))
    pan.iloc[:300, pan.columns.get_loc("YOUNG")] = np.nan
    pan["PEG"] = 10.0
    fa = build_causal_macro_factors(pan.iloc[:600], 5, stationarize=stationarize)
    fb = build_causal_macro_factors(pan, 5, stationarize=stationarize)
    assert _repaint(fa, fb.iloc[:600]) == []


# ── kernel vs class oracle ───────────────────────────────────────────────────

def test_kernel_matches_class_oracle():
    n = 700
    mkt = make_market(n)
    idx = pd.bdate_range("2019-01-01", periods=n)
    mac = make_macro(idx, k=6)
    for seed in range(3):
        r = run_nirnay(make_ohlcv(n, seed=seed, mkt=mkt).join(mac), mac.columns)
        u, c = r.Unified.to_numpy(float), r.Close.to_numpy(float)
        sg, lr = _prepare_inputs(u, c)
        for name, a, b in zip(("regime", "hmm_bull", "hmm_bear", "vol", "change", "conf"),
                              _regime_loop_njit(sg, lr, FLOOR), regime_loop_reference(u, c)):
            np.testing.assert_allclose(a.astype(float), b.astype(float), rtol=0, atol=1e-12, err_msg=name)


def test_regime_kernel_start_slide_invariant():
    """Same Unified input started 1/5/21 rows later → identical regimes over the last 250 rows."""
    n = 800
    r = run_nirnay(make_ohlcv(n, seed=4, mkt=make_market(n)).join(make_macro(pd.bdate_range("2019-01-01", periods=n), k=6)),
                   [f"M{j}" for j in range(6)])
    sg, cl = r.Unified.to_numpy(float), r.Close.to_numpy(float)
    base = _regime_loop_njit(*_prepare_inputs(sg, cl), FLOOR)[0]
    for S in (1, 5, 21):
        slid = _regime_loop_njit(*_prepare_inputs(sg[S:], cl[S:]), FLOOR)[0]
        assert np.array_equal(base[-250:], slid[-250:])


# ── late lister / aggregate (C-NIR) ──────────────────────────────────────────

N_STOCKS, N_ROWS, LIST_AT = 12, 800, 300
LATE = {f"S{s}" for s in range(9, 12)}


@pytest.fixture(scope="module")
def late_world():
    mkt = make_market(N_ROWS)
    idx = pd.bdate_range("2019-01-01", periods=N_ROWS)
    mac = make_macro(idx, k=6)
    res, ref = {}, {}
    for s in range(N_STOCKS):
        d = make_ohlcv(N_ROWS, seed=s, mkt=mkt, beta=np.random.default_rng(s).uniform(.6, 1.4))
        name = f"S{s}"
        ref[name] = run_nirnay((d.iloc[LIST_AT:] if name in LATE else d).join(mac), mac.columns)
        if name in LATE:
            d.iloc[:LIST_AT] = np.nan                   # union-indexed download: NaN pre-listing rows
        res[name] = run_nirnay(d.join(mac), mac.columns)
    return idx, res, ref, aggregate_constituent_timeseries(res), aggregate_constituent_timeseries(ref)


def test_late_lister_prelisting_excluded(late_world):
    idx, res, ref, g, gr = late_world
    s = res["S10"]
    assert len(s) == N_ROWS - LIST_AT and s.index[0] == idx[LIST_AT]
    assert int(s["Warmup"].sum()) == NIRNAY_WARMUP_BARS
    early = set(g.n_live.iloc[: LIST_AT - NIRNAY_WARMUP_BARS].unique())
    warm = set(g.n_live.loc[idx[LIST_AT]:idx[LIST_AT + NIRNAY_WARMUP_BARS - 1]].unique())
    late = set(g.n_live.loc[idx[LIST_AT + NIRNAY_WARMUP_BARS]:].unique())
    assert early == {N_STOCKS - len(LATE)} and warm == {N_STOCKS - len(LATE)} and late == {N_STOCKS}
    assert (g.n_universe == N_STOCKS).all()
    assert g.index[0] == idx[NIRNAY_WARMUP_BARS]        # first live day = bar 51


def test_late_lister_no_breadth_bias(late_world):
    """Padding a late lister with NaN pre-listing rows must not move any breadth pct
    vs running it on its genuine bars only (v1: Oversold_Pct biased −6pp)."""
    _, _, _, g, gr = late_world
    assert g.index.equals(gr.index)
    for c in ("Oversold_Pct", "Overbought_Pct", "Neutral_Pct", "Regime_Bull_Pct", "Avg_Signal", "n_live"):
        np.testing.assert_allclose(g[c].to_numpy(float), gr[c].to_numpy(float), atol=1e-12, err_msg=c)


def test_aggregate_schema(late_world):
    _, _, _, g, _ = late_world
    assert isinstance(g.index, pd.DatetimeIndex) and g.index.name == "Date"
    assert g.index.is_monotonic_increasing and not g.index.has_duplicates
    for c in DAILY_REQUIRED:
        assert c in g.columns, c
    assert "Regime_Neutral" not in g.columns
    assert (g.n_live > 0).all()
    np.testing.assert_allclose(g.coverage, g.n_live / g.n_universe)
    assert (g.Total_Analyzed == g.n_live).all()
    regime_sum = g[["Regime_Bull_Pct", "Regime_Bear_Pct", "Regime_Neutral_Pct", "Regime_Transition_Pct"]].sum(axis=1)
    np.testing.assert_allclose(regime_sum, 100.0)
    for c in PCT_COLS:
        assert g[c].between(0, 100).all(), c
    assert g.Avg_Signal.abs().max() <= 10 + 1e-9


def test_aggregate_drops_days_with_no_live_names(late_world):
    idx, res, *_ = late_world
    gap = idx[500:503]
    holed = {k: v.drop(index=gap, errors="ignore") for k, v in res.items()}
    g = aggregate_constituent_timeseries(holed)
    assert not g.index.isin(gap).any()
    assert len(g) == len(late_world[3]) - 3
    empty = aggregate_constituent_timeseries({})
    assert isinstance(empty.index, pd.DatetimeIndex) and len(empty) == 0


# ── CUSUM null fire rate ─────────────────────────────────────────────────────

def _cusum_rate(z: np.ndarray) -> float:
    c = CUSUMDetector()
    fired = np.array([c.update(v) for v in z])
    return float(fired[50:].mean())


def test_cusum_null_fire_rate_bounded(late_world):
    r0 = np.random.default_rng(0)
    design = _cusum_rate(r0.normal(0, 1, 50000))
    assert 0.001 < design < 0.015                      # ARL ~170 by design (v1 fired 5.6% on filtered iid)
    y = np.cumsum(r0.normal(0, .1, 50000)) + r0.normal(0, .3, 50000)   # local-level null
    kf = AdaptiveKalmanFilter()
    nu = []
    for v in y:
        kf.update(v)
        nu.append(kf.last_standardized_innovation)
    assert _cusum_rate(np.array(nu)) < 0.02
    live = pd.concat([r.iloc[NIRNAY_WARMUP_BARS:] for r in late_world[1].values()])
    assert live.Change_Point.mean() < 0.07             # v1 shipped 0.126 on real Unified
