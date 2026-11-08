"""convergence/conclusion.py (contract C-V2). Ports council/b3/{t_det,t_mc}.py — A3 v2-8 tests
1, 2, 3 (slow), 4, 5, 8, 9, 10 + JSON round-trip + exact to_dict() schema."""
import functools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.causal_stats import causal_z
from convergence.conclusion import (CONFIRM, CONTRADICT, NEUTRAL, NO_DATA, LiftResult, agreement_state,
                                    divergence_events, nirnay_readout, run_conclusion, tier_of)
from synth_helpers import conclusion_world

H = 10
FORBID = re.compile(r"calibrated|confirms|convergence adds value", re.I)

# ── exact C-V2 schema (literal, from DECISION_REGISTER §2 — not imported from the module) ──
TOP_KEYS = ["schema", "as_of", "horizon", "label_src", "forward_mode", "kappa_enabled", "single_system", "latest",
            "edge", "edge_C", "lift", "oos", "backtest", "chain", "trust", "nirnay", "agreement", "divergence_recent"]
LATEST_KEYS = {"date", "C", "C_aarambh", "zA", "zN", "state", "kappa", "tier", "direction"}
EDGE_KEYS = {"signal", "ic", "ic_phase0", "mde", "n", "t", "p", "hac_t", "boot_ci", "significant", "phase_min", "phase_max", "yearly",
             "baseline_momentum_ic", "available"}
LIFT_KEYS = {"available", "ic_confirm", "ic_contradict", "n_confirm", "n_contradict", "lift", "z", "p_one_sided",
             "lift_phase_min", "lift_phase_max", "kappa", "passed", "mde"}
OOS_KEYS = {"ic_C", "ic_A", "n", "adds_value"}
BT_KEYS = {"gross_sharpe", "net_sharpe", "t", "ann_return", "turnover_per_year", "breakeven_bps", "n"}
TRUST_TIERS = {"INVERTED", "NO_PROVEN_EDGE", "DIRECTIONAL_EDGE", "TRADEABLE_EDGE", "INSUFFICIENT_DATA"}
FRAME_COLS = ["zA", "zN", "state", "kappa", "lift_passed", "C", "C_aarambh", "tier"]


def jrt(d: dict) -> None:
    s = json.dumps(d, allow_nan=False)
    assert json.loads(s) == d


@pytest.fixture(scope="module")
def kw():
    """Seed 0 of the t_det search: a κ refit passes (κ < 1 on CONTRADICT days), Nirnay has a
    2-day tail gap and 15 interior holes."""
    ats, nd, px = conclusion_world(seed=0, k_lift=0.12, n_nir=3000, nirnay_tail_gap=2, nirnay_holes=15)
    full = run_conclusion(ats, nd, px, forward_mode=True, h=H)
    return ats, nd, px, full


@pytest.fixture(scope="module")
def variants(kw):
    ats, nd, px, full = kw
    return {
        "full": full,
        "single": run_conclusion(ats, pd.DataFrame(), px, forward_mode=True, h=H),
        "kappa_off": run_conclusion(ats, nd, px, forward_mode=True, h=H, kappa_enabled=False),
        "levels": run_conclusion(ats, nd, px, forward_mode=False, h=H),
        "no_price": run_conclusion(ats, nd, pd.Series(dtype=float), forward_mode=True, h=H),
        "empty": run_conclusion(ats.iloc[:0], nd, px, forward_mode=True, h=H),
    }


# ── [1] prefix invariance ────────────────────────────────────────────────────

def test_world_has_shrinking_refit(kw):
    fr = kw[3].frame
    assert ((fr.kappa < 1) & (fr.state == CONTRADICT)).sum() > 50


def test_run_conclusion_prefix_invariant(kw):
    ats, nd, px, full = kw
    fr = full.frame
    for T in (3000, 3700, 4300, 4700, len(ats) - 1):
        a = ats.iloc[:T]
        end = a.index[-1]
        pre = run_conclusion(a, nd[nd.index <= end], px[px.index <= end], forward_mode=True, h=H)
        f2 = fr.loc[pre.frame.index]
        for c in ("zA", "zN", "kappa", "C", "C_aarambh"):
            np.testing.assert_array_equal(f2[c].to_numpy(float), pre.frame[c].to_numpy(float), err_msg=f"{c}@{T}")
        assert (f2.state.fillna("-").astype(str) == pre.frame.state.fillna("-").astype(str)).all()
        assert (f2.lift_passed == pre.frame.lift_passed).all()
        assert full.events.loc[full.events.index <= end].equals(pre.events)


# ── frame semantics ──────────────────────────────────────────────────────────

def test_frame_formula_and_conventions(kw):
    ats, nd, px, full = kw
    fr = full.frame
    assert list(fr.columns) == FRAME_COLS and isinstance(fr.index, pd.DatetimeIndex)
    head = ats.loc[ats.Valid, "Forecast"]
    np.testing.assert_array_equal(fr.zA.to_numpy(), causal_z(head, 126, 4.0).to_numpy())
    np.testing.assert_allclose(fr.C_aarambh, 100 * np.tanh(fr.zA / 2), equal_nan=True)
    m = np.where(fr.state == CONTRADICT, fr.kappa, 1.0)
    np.testing.assert_allclose(fr.C, fr.C_aarambh * m, equal_nan=True)
    assert (fr.kappa <= 1).all() and (fr.kappa >= 0).all()
    fin = fr.C.notna()
    assert (fr.C[fin].abs() <= fr.C_aarambh[fin].abs() + 1e-12).all()       # Nirnay can only SHRINK
    assert (np.sign(fr.C[fin]) == np.sign(fr.C_aarambh[fin])).all() or (fr.kappa == 0).any()
    assert fr.state.iloc[:126].isna().all() and (fr.tier.iloc[:126] == "NO LEAN").all()
    assert (fr.tier == [tier_of(c) for c in fr.C]).all()
    d = full.to_dict()["latest"]
    assert d["direction"] == ("BULLISH" if d["C"] > 10 else "BEARISH" if d["C"] < -10 else "NEUTRAL")


def test_zN_not_forward_filled(kw):
    ats, nd, px, full = kw
    fr = full.frame
    missing = ~fr.index.normalize().isin(pd.DatetimeIndex(nd.index).normalize())
    assert missing.sum() > 15 and fr.zN[missing].isna().all()


def test_forward_mode_requires_forecast(kw):
    ats, nd, px, _ = kw
    with pytest.raises(KeyError):
        run_conclusion(ats.drop(columns="Forecast"), nd, px, forward_mode=True, h=H)
    with pytest.raises(ValueError):
        run_conclusion(ats, nd, px, forward_mode=True, h=0)


# ── [2] NO_DATA / single-system ──────────────────────────────────────────────

def test_no_data_rows(kw):
    ats, nd, px, full = kw
    fr = full.frame
    miss = fr.index[fr.zN.isna() & fr.zA.notna()]
    assert len(miss) > 1000
    assert (fr.loc[miss, "state"] == NO_DATA).all()
    assert (fr.loc[miss, "C"] == fr.loc[miss, "C_aarambh"]).all()
    assert not full.events.index.isin(miss).any()
    d = full.to_dict()
    assert list(fr.state.iloc[-2:]) == [NO_DATA, NO_DATA]           # 2-day tail gap
    assert d["nirnay"] == {"state": NO_DATA, "text": "Nirnay: no data for this date — conclusion is Aarambh-only."}


def test_single_system_mode(variants):
    r = variants["single"]
    d = r.to_dict()
    assert d["single_system"] is True and d["lift"]["available"] is False and d["lift"]["passed"] is False
    assert d["lift"]["n_confirm"] == 0 and d["lift"]["n_contradict"] == 0
    assert d["latest"]["kappa"] == 1.0 and len(r.events) == 0
    assert set(r.frame.state.dropna()) == {NO_DATA}
    assert d["edge_C"]["ic"] == d["edge"]["ic"]
    assert d["nirnay"]["text"].startswith("Single-system mode (Nirnay unavailable)")
    assert (r.frame.C.dropna() == r.frame.C_aarambh.dropna()).all()


def test_kappa_disabled_is_pure_aarambh(variants):
    r = variants["kappa_off"]
    assert (r.frame.kappa == 1).all()
    assert ((r.frame.C - r.frame.C_aarambh).abs().fillna(0) == 0).all()
    assert r.to_dict()["oos"]["adds_value"] is False


def test_degenerate_inputs(variants):
    assert variants["levels"].to_dict()["trust"]["tier"] in TRUST_TIERS
    npd = variants["no_price"].to_dict()
    assert npd["trust"]["tier"] == "INSUFFICIENT_DATA" and npd["backtest"]["by_cost"]["5.0"]["n"] == 0
    ed = variants["empty"].to_dict()
    assert ed["as_of"] is None and ed["trust"]["tier"] == "INSUFFICIENT_DATA"
    assert ed["latest"]["tier"] == "NO LEAN" and ed["latest"]["direction"] == "NEUTRAL"


# ── JSON + exact schema ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["full", "single", "kappa_off", "levels", "no_price", "empty"])
def test_to_dict_json_round_trip_and_schema(variants, name):
    d = variants[name].to_dict()
    jrt(d)
    assert list(d.keys()) == TOP_KEYS
    assert d["schema"] == "nishkarsh.conclusion/1" and d["horizon"] == H
    assert set(d["latest"]) == LATEST_KEYS
    assert d["latest"]["tier"] in {"STRONG", "MODERATE", "WEAK", "NO LEAN"}
    assert d["latest"]["direction"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert set(d["edge"]) == EDGE_KEYS and set(d["edge_C"]) == EDGE_KEYS
    assert d["edge"]["signal"] == "aarambh_head"
    assert all(isinstance(k, str) for k in d["edge"]["yearly"])
    assert d["edge"]["boot_ci"] is None or len(d["edge"]["boot_ci"]) == 2
    assert set(d["lift"]) == LIFT_KEYS and set(d["oos"]) == OOS_KEYS
    bt = d["backtest"]
    assert set(bt) == {"mode", "reference_bps", "by_cost", "nonoverlap_reference"}
    assert bt["mode"] == "tranche" and bt["reference_bps"] == 5.0
    assert set(bt["by_cost"]) == {"0.0", "2.0", "5.0", "10.0"}
    assert all(set(v) == BT_KEYS for v in bt["by_cost"].values()) and set(bt["nonoverlap_reference"]) == BT_KEYS
    assert [c["stage"] for c in d["chain"]][1:] == ["engine_conviction", "C"]
    assert all(set(c) == {"stage", "ic", "n"} for c in d["chain"])
    assert set(d["trust"]) == {"tier", "text"} and d["trust"]["tier"] in TRUST_TIERS
    assert set(d["nirnay"]) == {"state", "text"}
    assert set(d["agreement"]) == {"today_state", "same_sign_rate_60d"}
    assert set(d["divergence_recent"]) == {"AARAMBH_LEADS", "NIRNAY_LEADS", "CONTRADICTION", "window_days"}
    assert d["divergence_recent"]["window_days"] == 20


# ── [8] divergence reachability + day-indexed persistence ────────────────────

def test_divergence_types_and_persistence():
    ix = pd.bdate_range("2020-01-01", periods=60)
    zA = pd.Series(0.0, index=ix)
    zN = pd.Series(0.0, index=ix)
    zA.iloc[0:6], zN.iloc[0:6] = 2.0, 0.1             # AARAMBH_LEADS ×6 → persistent from the 5th
    zA.iloc[10], zN.iloc[10] = 0.2, 2.0               # NIRNAY_LEADS once → not persistent
    zA.iloc[20:26], zN.iloc[20:26] = 1.0, -1.0        # CONTRADICT ×6 → CONTRADICTION from the 5th
    zN.iloc[22] = np.nan                              # NO_DATA inside the run: no event, day counts
    zA.iloc[40], zN.iloc[40] = 1.0, -1.0              # 5th CONTRADICT within the last 20 → CONTRADICTION
    zA.iloc[55], zN.iloc[55] = 1.0, -1.0              # isolated (window 36..55 has 2) → none
    st = agreement_state(zA, zN)
    assert sorted(set(st.dropna())) == [CONFIRM, CONTRADICT, NEUTRAL, NO_DATA]
    ev = divergence_events(pd.DataFrame({"zA": zA, "zN": zN, "state": st}))
    got = [(d.strftime("%m-%d"), t, bool(p)) for d, t, p in zip(ev.index, ev.type, ev.persistent)]
    assert got == [("01-01", "AARAMBH_LEADS", False), ("01-02", "AARAMBH_LEADS", False),
                   ("01-03", "AARAMBH_LEADS", False), ("01-06", "AARAMBH_LEADS", False),
                   ("01-07", "AARAMBH_LEADS", True), ("01-08", "AARAMBH_LEADS", True),
                   ("01-15", "NIRNAY_LEADS", False), ("02-05", "CONTRADICTION", True),
                   ("02-26", "CONTRADICTION", True)]
    assert set(ev.type) == {"AARAMBH_LEADS", "NIRNAY_LEADS", "CONTRADICTION"}


def test_agreement_state_edge_cases():
    ix = pd.bdate_range("2020-01-01", periods=5)
    zA = pd.Series([1.0, -1.0, 0.0, np.nan, 1.0], index=ix)
    zN = pd.Series([0.3, 0.3, 1.0, 1.0, 0.2], index=ix)
    assert agreement_state(zA, zN).tolist() == [CONFIRM, CONTRADICT, NEUTRAL, None, NEUTRAL]


# ── [9] Nirnay texts (A3 v2-6 table) + forbidden words ───────────────────────

def _L(av, passed, ic_c=.15, ic_k=.05):
    return LiftResult(av, ic_c, ic_k, 60, 60, ic_c - ic_k, 2.0 if passed else .5, .02 if passed else .3, .01, .2,
                      max(ic_k, 0) / ic_c if passed else 1.0, passed, .47)


OOS_T = {"ic_C": .13, "ic_A": .12, "adds_value": True}
OOS_F = {"ic_C": .11, "ic_A": .12, "adds_value": False}


def test_nirnay_text_forbidden_words():
    n = 0
    for single in (True, False):
        for stt in (CONFIRM, CONTRADICT, NEUTRAL, NO_DATA, None):
            for lf in (_L(False, False), _L(True, False), _L(True, True)):
                for ke in (True, False):
                    for oo in (OOS_T, OOS_F):
                        t = nirnay_readout(single, stt, lf, lf.kappa, ke, oo)["text"]
                        n += 1
                        allowed = lf.passed and oo["adds_value"] and ke and not single and stt == CONFIRM
                        assert allowed or not FORBID.search(t), t
    assert n == 120


@pytest.mark.parametrize("args, text", [
    ((True, None, _L(False, False), 1, True, OOS_F),
     "Single-system mode (Nirnay unavailable) — conclusion is Aarambh-only; Convergence Lift N/A."),
    ((False, NO_DATA, _L(True, True), 1, True, OOS_T),
     "Nirnay: no data for this date — conclusion is Aarambh-only."),
    ((False, CONFIRM, _L(False, False), 1, True, OOS_F),
     "Nirnay agrees — context only: no measured lift (only 60/60 agree/disagree windows; need ≥ 30 each). "
     "Conviction is Aarambh's."),
    ((False, CONTRADICT, _L(True, False), 1, True, OOS_F),
     "Nirnay disagrees — context only: agreement has not been shown to improve Aarambh's accuracy "
     "(lift +0.100, p 0.30; detectable lift ≥ 0.47). Conviction is Aarambh's."),
    ((False, CONFIRM, _L(True, True), 1, True, OOS_T),
     "Nirnay CONFIRMS — IC +0.150 when confirming vs +0.050 when contradicting (lift +0.100, p 0.020); "
     "out-of-sample IC +0.130 vs Aarambh-only +0.120."),
    ((False, CONFIRM, _L(True, True), 1, True, OOS_F),
     "Nirnay agrees — lift +0.100 (p 0.020) passed in-sample, but out-of-sample IC has not improved "
     "(C +0.110 vs Aarambh +0.120). Conviction is Aarambh's."),
    ((False, CONTRADICT, _L(True, True), .33, True, OOS_T),
     "Nirnay CONTRADICTS — conviction ×0.33 (measured lift +0.100, p 0.020)."),
    ((False, CONTRADICT, _L(True, True), 1, False, OOS_T),
     "Nirnay disagrees — lift +0.100 (p 0.020) passed, but Intelligence Mode is OFF (κ ≡ 1). Conviction is Aarambh's."),
])
def test_nirnay_text_table(args, text):
    assert nirnay_readout(*args)["text"] == text


# ── [10] one agreement definition ────────────────────────────────────────────

def test_single_agreement_state_definition(repo_root):
    hits = []
    for p in Path(repo_root).rglob("*.py"):
        rel = p.relative_to(repo_root)
        if rel.parts[0] in ("tests", ".git", "vault") or "__pycache__" in rel.parts:
            continue
        if re.search(r"^\s*def agreement_state\b", p.read_text(encoding="utf-8", errors="ignore"), re.M):
            hits.append(str(rel))
    assert hits == ["convergence/conclusion.py"]


# ── Monte-Carlo: [3] null size, [4] κ under no lift, [5] power ───────────────

@functools.lru_cache(maxsize=None)
def _mc(seed: int, k_a: float, k_l: float) -> dict:
    ats, nd, px = conclusion_world(seed=seed, k_a=k_a, k_lift=k_l)
    r = run_conclusion(ats, nd, px, forward_mode=True, h=H)
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    f = r.frame
    oos = f.iloc[int(np.argmax(np.isfinite(f.zN.to_numpy()))) + 500:]
    txt = d["nirnay"]["text"] + " " + d["trust"]["text"]
    allowed = d["lift"]["passed"] and d["oos"]["adds_value"]
    return dict(lift_pass=d["lift"]["passed"], edge_sig=d["edge"]["significant"],
                bt_t=d["backtest"]["by_cost"]["5.0"]["t"],
                kappa_med=float(oos.kappa.median()), frac_k_lt1=float((oos.kappa < 1).mean()),
                viol=bool(FORBID.search(txt) and not allowed),
                power_str=("detectable lift" in d["nirnay"]["text"]) or d["lift"]["passed"])


def _batch(seeds, k_a, k_l) -> pd.DataFrame:
    return pd.DataFrame([_mc(s, k_a, k_l) for s in seeds])


def _null_size_checks(g: pd.DataFrame, bound: float):
    assert g.lift_pass.mean() <= bound
    assert g.edge_sig.mean() <= bound
    assert (g.bt_t.abs() > 1.96).mean() <= bound
    assert not g.viol.any()


def test_null_size_smoke():
    """20-rep smoke of A3 test 3 (loose bound; the real ≤ 0.08 size test is @slow)."""
    _null_size_checks(_batch(range(20), 0.0, 0.0), 0.25)


def test_kappa_no_lift_smoke():
    """A3 test 4 smoke: Aarambh edge but no lift → κ stays 1 on (almost) every OOS date."""
    g = _batch(range(1000, 1008), 0.045, 0.0)
    assert g.kappa_med.median() == 1.0 and g.frac_k_lt1.mean() <= 0.2 and not g.viol.any()


@pytest.mark.slow
def test_null_size_100_reps():
    """A3 test 3: null world (no edge, no lift), 100 seeded reps (seeds 0..99 as council/b3/t_mc.py):
    P(lift passes), P(edge significant), P(|backtest t @5bps| > 1.96) all ≤ 0.08."""
    _null_size_checks(_batch(range(100), 0.0, 0.0), 0.08)


@pytest.mark.slow
def test_kappa_no_lift_100_reps():
    """A3 test 4: k_lift = 0 → median κ = 1 and κ < 1 on ≤ 10% of OOS dates."""
    g = _batch(range(1000, 1100), 0.045, 0.0)
    assert g.kappa_med.median() == 1.0
    assert g.frac_k_lt1.mean() <= 0.10
    assert not g.viol.any()


@pytest.mark.slow
def test_lift_power_100_reps():
    """A3 test 5: k_lift = 0.06 → P(lift passes) ≥ 0.12; the UI power string is always present."""
    g = _batch(range(3000, 3100), 0.045, 0.06)
    assert g.lift_pass.mean() >= 0.12
    assert g.power_str.all()
    assert not g.viol.any()
