"""The Nishkarsh Oscillator (single inference series) and NKO-state precedent."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.analogs import find_state_analogs, nko_state_frame, summarize_forward
from convergence.nishkarsh import COMPONENTS, build_nishkarsh_oscillator, nko_readout
from core.config import NIRNAY_SIGN

N = 1400
IDX = pd.bdate_range("2018-01-01", periods=N)


def _inputs(seed=0, bull=None, nirnay=True):
    """Synthetic engine outputs. ``bull`` (array, ±1) pushes every input to that side."""
    rng = np.random.default_rng(seed)
    b = np.zeros(N) if bull is None else np.asarray(bull, float)
    noise = lambda: rng.standard_normal(N)                       # noqa: E731
    frame = pd.DataFrame({"zA": 0.5 * noise() + b}, index=IDX)
    a = pd.DataFrame({"AvgZ": noise() - b, "OversoldBreadth": 50 + 10 * noise() + 10 * b,
                      "OverboughtBreadth": 50 + 10 * noise() - 10 * b,
                      "BuySignalBreadth": 5 + noise() + b, "SellSignalBreadth": 5 + noise() - b}, index=IDX)
    nd = None
    if nirnay:
        # Nirnay columns written so that "bullish" (b > 0) means bullish UNDER THE CONFIGURED
        # polarity: trend → strength rises with b; contrarian → it falls (S = NIRNAY_SIGN).
        S = NIRNAY_SIGN
        nd = pd.DataFrame({"Avg_Signal": noise() + S * b, "Oversold_Pct": 30 + 5 * noise() - 5 * S * b,
                           "Overbought_Pct": 30 + 5 * noise() + 5 * S * b, "Regime_Bull_Pct": 40 + 5 * noise() + 5 * S * b,
                           "Regime_Bear_Pct": 30 + 5 * noise() - 5 * S * b, "avg_hmm_bull": .4 + .05 * noise() + .05 * S * b,
                           "avg_hmm_bear": .3 + .05 * noise() - .05 * S * b, "Buy_Signals": 5 + noise() - S * b,
                           "Sell_Signals": 5 + noise() + S * b, "n_live": 50.0}, index=IDX)
        frame["SAO"] = np.clip(0.3 * noise() + 0.6 * b, -1, 1)
    return frame, a, nd


def test_every_input_is_used_and_oriented_bullish_positive():
    regime = np.where(np.arange(N) % 200 < 100, 1.0, -1.0)
    f, a, n = _inputs(seed=1, bull=regime)
    k = build_nishkarsh_oscillator(f, a, n)
    assert len(COMPONENTS) == 10
    tail = k.iloc[400:]
    for _, cid, _, _ in COMPONENTS:                               # each oriented input tracks the regime
        assert np.corrcoef(tail[f"in_{cid}"], regime[400:])[0, 1] > 0.3, cid
    assert np.corrcoef(tail["NKO"], regime[400:])[0, 1] > 0.8
    assert k["NKO"].dropna().between(-100, 100).all()


def test_disagreement_mutes_and_agreement_amplifies():
    f, a, n = _inputs(seed=2)
    k = build_nishkarsh_oscillator(f, a, n).iloc[500:]
    agree = (np.sign(k["P_top_down"]) == np.sign(k["P_bottom_up"])) & (np.sign(k["P_bottom_up"]) == np.sign(k["P_agreement"]))
    split = np.sign(k["P_top_down"]) != np.sign(k["P_bottom_up"])
    assert k.loc[agree, "NKO"].abs().mean() > 1.5 * k.loc[split, "NKO"].abs().mean()


def test_causal_prefix_invariant():
    f, a, n = _inputs(seed=3)
    full = build_nishkarsh_oscillator(f, a, n)
    cut = build_nishkarsh_oscillator(f.iloc[:1000], a.iloc[:1000], n.iloc[:1000])
    cols = ["NKO", "P_top_down", "P_bottom_up", "P_agreement"]
    pd.testing.assert_frame_equal(full[cols].iloc[:1000], cut[cols])


def test_single_system_dates_are_flagged():
    f, a, _ = _inputs(seed=4, nirnay=False)
    k = build_nishkarsh_oscillator(f, a, None)
    assert not k["two_system"].any() and k["NKO"].notna().any()
    r = nko_readout(k, pd.Series(np.exp(np.cumsum(np.random.default_rng(0).standard_normal(N) * .01)), index=IDX), 10)
    assert r["status"] == "INSUFFICIENT_DATA" and r["two_system"] is False


def test_null_world_is_not_validated():
    f, a, n = _inputs(seed=5)
    price = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(9).standard_normal(N) * .01)), index=IDX)
    r = nko_readout(build_nishkarsh_oscillator(f, a, n), price, 10)
    assert r["status"] in {"UNPROVEN", "INSUFFICIENT_DATA"}
    assert set(r["pillars"]) == {"top_down", "bottom_up", "agreement"} and len(r["drivers"]) == 3


def test_precedent_matches_on_nko_state_causally():
    f, a, n = _inputs(seed=6)
    k = build_nishkarsh_oscillator(f, a, n)
    price = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(7).standard_normal(N) * .01)), index=IDX)
    holds = (5, 10, 20)
    st = nko_state_frame(k, price, holds)
    analogs, uncond = find_state_analogs(st, holds, gap=20)
    assert 5 <= len(analogs) <= 10 and set(uncond) == set(holds)
    today = st[st["two_system"]].index[-1]
    pos = st.loc[[pd.Timestamp(x["date"]) for x in analogs], "pos"].to_numpy()
    assert (pos <= st.loc[today, "pos"] - max(holds)).all()          # every outcome realized before today
    assert (np.diff(np.sort(pos)) >= 20).all()                         # distinct episodes (Theiler)
    s = summarize_forward(analogs, holds, uncond)
    assert {"median", "positive_pct", "n", "unconditional_positive_pct", "binomial_p"} <= set(s[10])


def test_polarity_switch_flips_every_bottom_up_input_and_nothing_else(monkeypatch):
    """ONE switch (NIRNAY_POLARITY): flipping it negates every Nirnay input and the
    bottom-up pillar exactly, and leaves the top-down inputs untouched."""
    import convergence.nishkarsh as NK
    f, a, n = _inputs(seed=8)
    base = build_nishkarsh_oscillator(f, a, n)
    monkeypatch.setattr(NK, "NIRNAY_SIGN", -NK.NIRNAY_SIGN)
    flip = build_nishkarsh_oscillator(f, a, n)
    for pil, cid, _, _ in COMPONENTS:
        if pil == "bottom_up":
            np.testing.assert_allclose(flip[f"in_{cid}"], -base[f"in_{cid}"], equal_nan=True)
        elif pil == "top_down":
            np.testing.assert_allclose(flip[f"in_{cid}"], base[f"in_{cid}"], equal_nan=True)
    np.testing.assert_allclose(flip["P_bottom_up"], -base["P_bottom_up"], equal_nan=True)


def test_readout_context_is_descriptive_and_correct():
    from convergence.nishkarsh import _context
    idx = pd.bdate_range("2024-01-01", periods=30)
    s = pd.Series([0.0] * 20 + [20.0, 25, 30, 35, 40, 45, 50, 55, 60, 70], index=idx)
    c = _context(s)
    assert c["change_5d"] == 70 - 40 and c["change_20d"] == 70 - 0     # 5 and 20 sessions back
    assert c["streak"] == {"side": "BULLISH", "sessions": 10, "since": str(idx[20].date())}
    assert c["strength_percentile"] == 100.0
