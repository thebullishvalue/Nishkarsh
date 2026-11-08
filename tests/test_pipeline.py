"""Headless pipeline: config fingerprint, label-alignment guard, end-to-end contract.

The end-to-end run uses the deterministic synthetic world (offline). It is the
integration test that every layer composes: data → Aarambh → Nirnay → label →
conclusion → anatomy → precedent → report, with the exact UI session contract.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from convergence.labels import label_alignment_check
from data.synthetic import synthetic_world
from pipeline.config import PipelineConfig
from pipeline.run import SESSION_KEYS, run_pipeline
from pipeline.report import strip_volatile


class _NullLog:
    def __getattr__(self, name):
        if name == "begin_run":
            return lambda *a, **k: "run"
        return lambda *a, **k: None


def _world(seed=7, n_rows=2300, planted_ic=0.0):
    sheet, src = synthetic_world(seed, n_rows, planted_ic=planted_ic)
    return sheet, src, pd.Timestamp(sheet["DATE"].iloc[-1])


def _run(sheet, src, today, **over):
    cfg = PipelineConfig.build(today=today, target="NIFTY50_PE", date_col="DATE", **over)
    return run_pipeline(sheet, cfg, src, today=today, log=_NullLog())


# ── config fingerprint ───────────────────────────────────────────────────────

def test_feature_order_is_canonical_and_fingerprint_invariant():
    base = PipelineConfig.build(today="2026-01-01")
    rev = PipelineConfig.build(today="2026-01-01", features=tuple(reversed(base.features)))
    # Results must not depend on tick order (red-team T2: ElasticNet is order-sensitive),
    # so the canonical order — not just the hash — must coincide.
    assert rev.features == base.features == tuple(sorted(base.features))
    assert rev.fingerprint() == base.fingerprint()


def test_fingerprint_changes_with_result_changing_inputs(monkeypatch):
    base = PipelineConfig.build(today="2026-01-01")
    fp = base.fingerprint()
    assert base.replace(kappa_enabled=not base.kappa_enabled).fingerprint() != fp
    assert base.replace(target="NIFTY50_PB").fingerprint() != fp
    import core.config as C
    monkeypatch.setattr(C, "CONV2_TAU", C.CONV2_TAU + 0.05)   # a tuned core constant must change the key
    assert base.fingerprint() != fp


# ── label timestamp alignment guard ──────────────────────────────────────────

def test_label_alignment_detects_one_session_misstamp():
    idx = pd.bdate_range("2015-01-01", periods=900)
    rng = np.random.default_rng(3)
    r = rng.standard_normal(900) * 0.01
    price = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    pe = price / 5.0 * np.exp(rng.standard_normal(900) * 0.001)      # shares the price leg
    ok = label_alignment_check(pe, price)
    assert ok["available"] and ok["aligned"] and ok["best_lag"] == 0
    late = pd.Series(price.to_numpy()[:-1], index=idx[1:])           # close of D−1 stamped D
    bad = label_alignment_check(pe, late)
    assert bad["available"] and not bad["aligned"] and bad["best_lag"] != 0


# ── end-to-end ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e2e():
    sheet, src, today = _world()
    return _run(sheet, src, today)


def test_end_to_end_contract(e2e):
    r = e2e
    ss = r.to_session_state()
    assert set(ss) == set(SESSION_KEYS)
    assert int(r.aarambh_ts["Valid"].sum()) > 0, "no genuine out-of-sample forecasts"
    assert r.engine.wf_meta["purge"] == 10 and r.engine.wf_meta["chunks_skipped_starved"] >= 0
    d = r.conclusion.to_dict()
    json.dumps(d, allow_nan=False)                                     # JSON-safe, no NaN anywhere
    json.dumps(r.run_report, allow_nan=False, default=str)
    assert r.run_report["schema"] == "nishkarsh.run_report/1"
    assert d["trust"]["tier"] in {"INVERTED", "NO_PROVEN_EDGE", "DIRECTIONAL_EDGE", "TRADEABLE_EDGE",
                                  "INSUFFICIENT_DATA"}
    assert d["edge"]["mde"] is None or 0 < d["edge"]["mde"] < 1
    # The run must label itself synthetic everywhere it reports provenance.
    assert "SYNTHETIC" in json.dumps(ss["run_config"], default=str).upper()


def test_null_world_claims_no_edge(e2e):
    """planted_ic = 0 → the conclusion must not claim a directional or tradeable edge,
    and κ must stay 1 (Nirnay has no measured lift)."""
    d = e2e.conclusion.to_dict()
    assert d["trust"]["tier"] not in {"DIRECTIONAL_EDGE", "TRADEABLE_EDGE"}
    assert not d["lift"]["passed"] and not d["oos"]["adds_value"]
    assert float((e2e.conclusion.frame["kappa"] < 1).mean()) == 0.0


@pytest.mark.slow
def test_end_to_end_deterministic_and_order_invariant():
    sheet, src, today = _world(seed=11)
    a = _run(sheet, src, today)
    feats = PipelineConfig.build(today=today).features
    b = _run(sheet, src, today, features=tuple(reversed(feats)))
    assert json.dumps(strip_volatile(a.run_report), sort_keys=True, default=str) == \
        json.dumps(strip_volatile(b.run_report), sort_keys=True, default=str)


@pytest.mark.slow
def test_streamlit_app_synthetic_demo_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    radios = [r for r in at.radio if any("Synthetic" in str(o) for o in r.options)]
    assert radios, "Synthetic demo data source missing"
    radios[0].set_value(next(o for o in radios[0].options if "Synthetic" in str(o))).run()
    run_btn = [b for b in at.button if "Run" in str(b.label)]
    assert run_btn, "Run Analysis button missing"
    run_btn[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) >= 6


def test_real_sheet_date_typo_runs_and_is_reported():
    """Regression for the first real-data run: a day-first sheet with an out-of-order
    date cell used to abort the whole run ('not monotonic … after row 4271'). It must
    now run, exclude that one row, and name it by sheet row and raw text."""
    from data.synthetic import synthetic_sheet
    from pipeline.inputs import prepare_sheet
    sheet = synthetic_sheet(seed=5, n_rows=4400, date_fmt="%d/%m/%Y")
    bad = 4271
    d = pd.to_datetime(sheet.loc[bad, "DATE"], format="%d/%m/%Y")
    sheet.loc[bad, "DATE"] = d.replace(year=d.year - 1).strftime("%d/%m/%Y")
    cfg = PipelineConfig.build(today=pd.Timestamp("2027-01-01"), target="NIFTY50_PE", date_col="DATE")
    prep = prepare_sheet(sheet, cfg)
    msg = " ".join(prep.warnings)
    assert f"row {bad + 2}" in msg and "out of chronological sequence" in msg
    assert prep.dates.is_monotonic_increasing and prep.dates.is_unique


def test_late_predictor_history_cost_is_disclosed():
    """A predictor that starts late truncates every row before it; the run must say
    which one and how much history it costs (first real-sheet run: INIRYY, 5.4 years)."""
    from data.synthetic import synthetic_sheet
    from pipeline.inputs import prepare_sheet
    sheet = synthetic_sheet(seed=6, n_rows=4400)
    sheet.loc[:899, "INIRYY"] = np.nan                       # starts 900 rows late
    cfg = PipelineConfig.build(today=pd.Timestamp("2027-01-01"), target="NIFTY50_PE", date_col="DATE")
    prep = prepare_sheet(sheet, cfg)
    msg = " ".join(prep.warnings)
    assert prep.bounding_predictor == "INIRYY"
    assert "Predictor INIRYY" in msg and "truncates the usable history" in msg
