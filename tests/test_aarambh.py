"""engines/aarambh.py (contract C-AAR) + its analytics primitives.

Ports council/b1/{v_leak,v_causal,v_prims}.py at small walk-forward windows
(min_train/max_train passed explicitly; config defaults are 1500/2000)."""
import numpy as np
import pytest

from analytics.causal_stats import causal_z
from core.config import AARAMBH_WF_REFIT, CONV2_ZCLIP, CONV2_ZMIN_PERIODS
from synth_helpers import H, fit_engine, frame_diff, ic_nonovl, make_levels, predictive_xy, rank_ic

MN, MX = 120, 200
CAUSAL_SKIP = {"Actual", "Label", "FwdChg_5", "FwdChg_10", "FwdChg_20"}   # realized-label columns
WF_META_KEYS = {"min_train", "max_train", "purge", "refit", "first_forecast_row", "n_oos",
                "chunks_total", "chunks_skipped_starved", "chunks_fallback"}


@pytest.fixture(scope="module")
def base():
    X, y, m, p, _ = predictive_xy(make_levels(3, n_rows=520))
    return X, y, m, p, fit_engine(X, y, m, p, MN, MX)


# ── purge / leakage ──────────────────────────────────────────────────────────

def test_purge_invariant(base):
    """Perturbing every label y[s], s ≥ t − h, never changes Forecast[≤ t]."""
    X, y, m, p, e0 = base
    F0 = e0.ts_data["Forecast"].to_numpy()
    rng = np.random.default_rng(0)
    for t in rng.choice(np.flatnonzero(np.isfinite(F0)), 3, replace=False):
        y2 = y.copy()
        y2[t - H:] += rng.standard_normal(len(y2) - (t - H)) * 0.5
        F1 = fit_engine(X, y2, m, p, MN, MX).ts_data["Forecast"].to_numpy()
        a, b = F0[:t + 1], F1[:t + 1]
        assert np.array_equal(a, b, equal_nan=True), f"forecast at ≤{t} depends on labels ≥ t−h"


@pytest.mark.parametrize("seed", [1000, 1001])
def test_null_leakage_tiny_window(seed):
    """At a deliberately tiny window (30/60, where v1 silently dropped the purge and
    showed IC ≈ +0.5 on pure noise) the null forecast IC must be ≈ 0."""
    X, y, m, p, _ = predictive_xy(make_levels(seed, n_rows=900))
    e = fit_engine(X, y, m, p, 30, 60)
    assert e.wf_meta["purge"] == H and e.wf_meta["chunks_skipped_starved"] == 0
    ok = m & e.ts_data["Valid"].to_numpy()
    f = e.ts_data["Forecast"].to_numpy()
    ic, n = ic_nonovl(f, y, ok)
    assert n > 700
    assert abs(ic) < 0.2, f"null IC {ic:+.3f} — purge leak?"
    assert abs(rank_ic(np.where(ok, f, np.nan), y)) < 0.2


def test_starved_chunks_are_nan_and_counted():
    """A chunk with < min_train label-complete rows is skipped (NaN) — the purge is never dropped."""
    X, y, m, p, _ = predictive_xy(make_levels(5, n_rows=420))
    m = m.copy()
    m[:150] = False                                   # first 150 labels unrealized/missing
    e = fit_engine(X, y, m, p, MN, MX)
    meta = e.wf_meta
    assert meta["chunks_skipped_starved"] > 0
    # first forecast needs MN label-complete rows strictly before t − purge
    assert meta["first_forecast_row"] >= 150 + MN + H
    assert e.ts_data["Valid"].iloc[:150 + MN + H].sum() == 0


# ── prefix invariance / determinism ──────────────────────────────────────────

@pytest.mark.parametrize("fwd, n_short", [(True, 600), (True, 603), (False, 600)],
                         ids=["fwd", "fwd-mid-refit-chunk", "levels"])
def test_prefix_invariance(fwd, n_short):
    """Appending rows never repaints an earlier causal column. n_short=603 cuts the
    series in the middle of a refit chunk (refit boundary crossed by the extension)."""
    k = 40
    df = make_levels(7, n_rows=n_short + k, signal=0.08)
    Xs, ys, ms, ps, _ = predictive_xy(df.iloc[:n_short].reset_index(drop=True))
    Xl, yl, ml, pl, _ = predictive_xy(df)
    if fwd:
        es, el = fit_engine(Xs, ys, ms, ps, MN, MX), fit_engine(Xl, yl, ml, pl, MN, MX)
    else:
        es = fit_engine(Xs, ps, None, None, MN, MX, purge=0, fwd=False)
        el = fit_engine(Xl, pl, None, None, MN, MX, purge=0, fwd=False)
    if n_short == 603:
        assert (len(Xs) - (MN + H)) % AARAMBH_WF_REFIT != 0   # really cuts a chunk
    a, b = es.ts_data, el.ts_data.iloc[:len(es.ts_data)]
    cols = [c for c in a.columns if c not in CAUSAL_SKIP]
    assert len(cols) > 30 and a["Valid"].sum() > 400
    assert frame_diff(a, b, cols) == {}


def test_determinism(base):
    X, y, m, p, e1 = base
    e2 = fit_engine(X, y, m, p, MN, MX)
    assert frame_diff(e1.ts_data, e2.ts_data, e1.ts_data.columns) == {}
    assert e1.wf_meta == e2.wf_meta
    assert str(e1.get_current_signal()) == str(e2.get_current_signal())


# ── contract C-AAR fields ────────────────────────────────────────────────────

def test_wf_meta_fields(base):
    X, y, m, p, e = base
    meta = e.wf_meta
    assert set(meta) == WF_META_KEYS
    assert (meta["min_train"], meta["max_train"], meta["purge"], meta["refit"]) == (MN, MX, H, AARAMBH_WF_REFIT)
    assert meta["first_forecast_row"] == MN + H
    assert meta["n_oos"] == int(e.ts_data["Valid"].sum())
    assert meta["chunks_total"] == len(range(MN + H, len(y), AARAMBH_WF_REFIT))
    assert meta["chunks_skipped_starved"] == 0 and meta["chunks_fallback"] == 0


def test_forward_columns_and_conviction(base):
    X, y, m, p, e = base
    ts = e.ts_data
    for c in ("Valid", "Forecast", "Label", "ForecastZ", "ConvictionScore", "ConvictionBounded"):
        assert c in ts.columns
    valid = ts["Valid"].to_numpy(bool)
    fc = ts["Forecast"].to_numpy(float)
    assert np.array_equal(valid, np.isfinite(fc))
    assert np.isnan(ts["Label"].to_numpy(float)[~m]).all()          # unrealized tail is NaN
    fz_expect = causal_z(ts["Forecast"], CONV2_ZMIN_PERIODS, CONV2_ZCLIP).to_numpy()
    assert np.array_equal(ts["ForecastZ"].to_numpy(float), fz_expect, equal_nan=True)
    conv = -100.0 * np.tanh(ts["ForecastZ"].to_numpy(float) / 2.0)
    assert np.array_equal(ts["ConvictionScore"].to_numpy(float), conv, equal_nan=True)
    assert np.array_equal(ts["ConvictionBounded"].to_numpy(float), conv, equal_nan=True)
    assert ts["ConvictionUpper"].isna().all() and ts["ConvictionLower"].isna().all()
    assert (ts.loc[~valid, "Regime"] == "N/A").all()
    # sign convention: an expected rise (ForecastZ > 0) is bullish = NEGATIVE engine conviction
    pos = ts["ForecastZ"] > 0.5
    assert pos.any() and (ts.loc[pos, "ConvictionBounded"] < 0).all()


def test_forward_signal_defers_edge(base):
    sig = base[4].get_current_signal()
    assert sig["has_forward_edge"] is False and sig["inverted"] is False and sig["tradeable"] is False
    assert sig["edge_source"] == "conclusion" and sig["edge_assessment"].startswith("DEFERRED")
    assert isinstance(sig["changepoints_heuristic"], list)
    assert "break_detected" not in sig
    for k in ("forecast", "forecast_z", "conviction_score", "regime", "signal"):
        assert k in sig


def test_model_stats_honour_label_mask(base):
    X, y, m, p, e = base
    st = e.get_model_stats()
    assert st["n_obs"] == int((e.ts_data["Valid"].to_numpy(bool) & m).sum())


def test_no_valid_rows_handled():
    X, y, m, p, _ = predictive_xy(make_levels(1, n_rows=140))   # 120 rows < MN + purge
    for fwd in (True, False):
        e = fit_engine(X, y if fwd else p, m, p, MN, MX, purge=H if fwd else 0, fwd=fwd)
        assert len(e.ts_data) == len(X) and not e.ts_data["Valid"].any()
        assert e.wf_meta["n_oos"] == 0 and e.wf_meta["first_forecast_row"] is None
        sig = e.get_current_signal()
        assert sig["regime"] == "N/A" and np.isnan(sig["conviction_score"])
        assert sig["edge_source"] == ("conclusion" if fwd else "engine_levels_test")
        assert e.get_model_stats()["n_obs"] == 0


def test_levels_mode_ddm_warmup_and_regime(base):
    X, y, m, p, _ = base
    e = fit_engine(X, p, None, None, MN, MX, purge=0, fwd=False)
    ts = e.ts_data
    invalid = ~ts["Valid"].to_numpy(bool)
    assert invalid.any()
    assert ts.loc[invalid, "ConvictionScore"].isna().all()       # DDM warm-up rows NaN (A1-05)
    assert (ts.loc[invalid, "Regime"] == "N/A").all()
    assert e.get_current_signal()["edge_source"] == "engine_levels_test"


# ── primitives ───────────────────────────────────────────────────────────────

def test_conformal_z_scale_long_lookback():
    from analytics.conformal import compute_conformal_zscores
    x = np.random.default_rng(42).standard_normal(20000)
    z = compute_conformal_zscores(x, 250, 125)[0]
    z = z[np.isfinite(z)]
    assert 0.95 < z.std() < 1.10                 # A1-04: was mis-scaled
    assert 0.29 < np.mean(np.abs(z) > 1) < 0.37
    assert 0.03 < np.mean(np.abs(z) > 2) < 0.07


def _ar1(rng, n, a, m=0.0):
    e = rng.standard_normal(n)
    v = np.empty(n)
    v[0] = m
    for i in range(1, n):
        v[i] = m + a * (v[i - 1] - m) + e[i]
    return v


def test_ou_mu_unbiased():
    from analytics.ou_process import ornstein_uhlenbeck_estimate
    rng = np.random.default_rng(42)
    est = np.array([ornstein_uhlenbeck_estimate(_ar1(rng, 300, 0.9, 5.0)) for _ in range(200)])
    assert abs(np.median(est[:, 1]) - 5.0) < 0.25
    assert abs(est[:, 0].mean() - (-np.log(0.9))) < 0.02


def test_hurst_dfa_on_increments():
    from analytics.hurst import hurst_dfa
    rng = np.random.default_rng(7)
    rw = np.cumsum(rng.standard_normal(3000))
    assert abs(hurst_dfa(np.diff(rw)) - 0.5) < 0.08               # RW increments → H ≈ 0.5
    assert hurst_dfa(np.diff(rng.standard_normal(3000))) < 0.15    # over-differenced noise → anti-persistent
    assert np.isnan(hurst_dfa(rng.standard_normal(20)))            # too short → NaN, not 0.5


def test_ddm_warmup_nan():
    from analytics.ddm_filter import drift_diffusion_filter
    f, _, _ = drift_diffusion_filter(np.r_[np.full(30, np.nan), np.ones(10) * 20])
    assert np.isnan(f[:30]).all() and np.isfinite(f[30:]).all()
    assert int(np.argmax(np.isfinite(f))) == 30
