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
    assert [t.label for t in at.tabs] == ["Today", "Drivers", "Precedent", "Reliability", "Data"]
    for t in at.tabs:                         # every page renders without an error box or NaN / None text
        md = " ".join(m.value for m in t.markdown)
        assert not t.error and "Could not draw" not in md, (t.label, [e.value for e in t.error])
        low = md.lower()
        assert "+nan" not in low and ">nan<" not in low and ">none<" not in low and " nan " not in low, t.label
    # no buy/sell/signal/call wording on ANY page unless the read itself is VALIDATED
    import re
    from ui.components import BUY_SELL_RE
    status = ((at.session_state["conclusion"] or {}).get("nko") or {}).get("status")
    if status != "VALIDATED":
        for t in at.tabs:
            text = re.sub(r"<style.*?</style>|<[^>]+>", " ", " ".join(m.value for m in t.markdown), flags=re.S)
            hit = BUY_SELL_RE.search(text)
            assert not hit, (t.label, text[max(0, hit.start() - 60):hit.end() + 60])
    today = " ".join(m.value for m in at.tabs[0].markdown)
    assert "Today’s Nishkarsh read" in today or "Today&#x27;s Nishkarsh read" in today
    assert "Track record:" in today and "What drives today" in today and "moments like today" in today
    assert "moments like today" in " ".join(m.value for m in at.tabs[2].markdown)
    rel = " ".join(m.value for m in at.tabs[3].markdown)
    assert "The ledger" in rel and "Method and limits" in rel and "Glossary" in rel
    # disclosures live in their sections: the only expander left is the sidebar's Model settings
    assert [e.label for e in at.expander] == ["Model settings"]
    assert "Synthetic demo data" in " ".join(m.value for m in at.markdown)


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


def test_agreement_oscillator_is_measured_and_never_claims_without_evidence(e2e):
    """The oscillator is computed every run; on a no-signal world it must not be VALIDATED,
    and its text must carry the numbers (pre-registered rule, docs/research)."""
    o = e2e.conclusion.to_dict().get("oscillator")
    assert o is not None and o["status"] in {"VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA"}
    assert o["status"] != "VALIDATED"
    assert {"agreement_ratio", "consensus", "SAO"} <= set(e2e.conclusion.frame.columns)
    sao = e2e.conclusion.frame["SAO"].dropna()
    assert sao.between(-1, 1).all()
    if o["status"] == "UNPROVEN":
        assert "Not yet a validated buy/sell signal" in o["text"] and "IC" in o["text"]


def test_auto_sources_fall_back_to_nse_when_yahoo_is_empty():
    from data.nse_archive import AutoSources
    from data.sources import FetchResult

    class Empty:
        def constituents(self, force=False):
            return FetchResult([], "empty", note="429")
        def ohlcv(self, symbols, start, end, force=False):
            return FetchResult({}, "empty", note="429")

    class Fallback:
        def constituents(self):
            return FetchResult(["AAA.NS", "BBB.NS"], "live", note="NSE archives")
        def ohlcv(self, symbols, start, end, force=False):
            df = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
            return FetchResult({s: df for s in symbols}, "live", note="NSE bhavcopy")

    a = AutoSources(primary=Empty(), fallback=Fallback())
    c = a.constituents()
    assert c.value == ["AAA.NS", "BBB.NS"] and "NSE" in c.note
    o = a.ohlcv(["AAA.NS", "BBB.NS"], "2024-01-01", "2024-01-02")
    assert set(o.value) == {"AAA.NS", "BBB.NS"} and "Yahoo returned 0/2" in o.note
