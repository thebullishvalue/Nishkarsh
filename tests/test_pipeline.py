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
    """The app on the synthetic demo: cold start (masthead + landing), one run, then EVERY registered
    page rendered through st.navigation without an exception — the page shell on each (tape →
    command bar → notice rail), no NaN/None text, no buy/sell/signal/call wording unless the read
    is VALIDATED, and none of the retired disclosure widgets (rule 133)."""
    import re

    from streamlit.testing.v1 import AppTest
    from streamlit.testing.v1 import element_tree as et
    from streamlit.util import calc_md5

    import app as app_module
    from ui.components import BUY_SELL_RE

    # AppTest 1.52.2 reads every button group's value as a LIST; a single-select
    # st.segmented_control (Appearance, chart window) holds a scalar, which AppTest iterates
    # character by character on the next run. Test-harness shim only — the app is unchanged.
    def _indices(self):
        v = self.value
        vals = v if isinstance(v, (list, tuple)) else ([] if v is None else [v])
        return [self.options.index(self.format_func(x)) for x in vals]

    _orig = et.ButtonGroup.indices
    et.ButtonGroup.indices = property(_indices)
    try:
        at = AppTest.from_file("../app.py", default_timeout=300)
        at.run()
        assert not at.exception, [e.message for e in at.exception]
        cold = " ".join(m.value for m in at.markdown)
        assert "premium-header" in cold and "lede-claim" in cold and "outcome-grid" in cold
        radios = [r for r in at.radio if any("Synthetic" in str(o) for o in r.options)]
        assert radios and radios[0].label == "Data source", "Synthetic demo data source missing"
        radios[0].set_value(next(o for o in radios[0].options if "Synthetic" in str(o))).run()
        run_btn = [b for b in at.button if "Run" in str(b.label)]
        assert run_btn, "Run Analysis button missing"
        run_btn[0].click().run()
        assert not at.exception, [e.message for e in at.exception]

        # navigation: exactly the seven pages, Overview default, in the inference's order
        assert [p[1] for p in app_module.PAGES] == ["Overview", "Aarambh", "Nirnay", "Convergence", "Precedent",
                                                     "System", "Data"]
        assert [p[0] for p in app_module.PAGES] == ["", "Engines", "Engines", "Engines", "Engines", "System",
                                                     "System"]
        status = ((at.session_state["conclusion"] or {}).get("nko") or {}).get("status")
        firsts = {"overview": "Inference Chain", "aarambh": "Aarambh — The Top-Down Forecast",
                  "nirnay": "Nirnay — What the Fifty Stocks Are Doing",
                  "convergence": "Convergence — Do the Systems Agree?",
                  "precedent": "Precedent — What Followed Reads Like Today", "system": "Walk-Forward Edge",
                  "data": "The Record"}
        for section, title, icon, url, _fn, windowed in app_module.PAGES:
            if url != "overview":
                at._page_hash = calc_md5(url)
            at.run()
            assert not at.exception, (title, [e.message for e in at.exception])
            md = " ".join(m.value for m in at.markdown)
            assert "could not be drawn" not in md, title
            assert "command-bar" in md and "ticker" in md and "notice-rail" in md, title
            assert md.index("ticker") < md.index("command-bar") < md.index("notice-rail"), title
            assert firsts[url] in html_unescape(md), title
            assert ('<span class="k">Window</span>' in md) is windowed, title
            text = re.sub(r"<style.*?</style>|<[^>]+>", " ", md, flags=re.S)
            low = f" {text.lower()} "
            assert "+nan" not in low and " nan " not in low and " none " not in low, title
            if status != "VALIDATED":
                hit = BUY_SELL_RE.search(text)
                assert not hit, (title, text[max(0, hit.start() - 60):hit.end() + 60])
            # rule 133: no expanders, in-page tabs, captions, info/warning boxes, metrics or dataframes
            assert not (at.expander or at.tabs or at.caption or at.info or at.warning or at.metric
                        or at.dataframe), title
            assert "Synthetic demo data" in md, title
        at._page_hash = ""
        at.run()
        overview_md = " ".join(m.value for m in at.markdown)
        assert "hero-card" in overview_md and "Limited by" in overview_md
        for label in ("Read", "Nishkarsh Oscillator", "Walk-Forward Edge", "Nirnay Net Breadth", "Top-Down Pillar",
                      "Engine Agreement"):
            assert f">{label}<" in overview_md, label
    finally:
        et.ButtonGroup.indices = _orig


def html_unescape(s: str) -> str:
    import html
    return html.unescape(s)


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
