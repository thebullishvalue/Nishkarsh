"""UI page contracts (pure helpers behind Overview / Convergence / System / Data and the shell).

  P1  the Ledger's status words follow the payload's own flags (never a UI threshold), one row per claim
  P2  no "nan"/"None" text and no buy/sell/signal/call wording in any ledger row, for any payload
  P3  data notes = run warnings (bar SYNTHETIC) + uncovered stale/fallback/empty sources
  P4  the drivers chart lists the three pillars, each followed by exactly its own inputs, top-3 marked
  P5  no page or notice ever shows a link (a sheet URL is a secret)
  P6  the Overview KPI strip prints six cards, no "nan"/"None", the lean word title-cased
  P7  the shell's data (tape, quote, freshness, notices) is built in the pipeline, in the agreed shapes
  P8  the edge contract: {ic, se, n, horizon, blocks[{start,end,ic}], positive_blocks, scored_blocks}
  P9  the Data record joins the run's frames by date; on-screen view is the NEWEST rows
"""
from __future__ import annotations

import os
import random
import re
import sys

_ROOT = os.environ.get("NISHKARSH_ROOT", "/home/user/Nishkarsh")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from convergence.nishkarsh import COMPONENTS, edge_contract  # noqa: E402
from convergence.verdict import build_read_verdict  # noqa: E402
from pipeline.shell import build_freshness, build_notices, build_quote, build_tape, redact  # noqa: E402
from ui.components import BUY_SELL_RE, data_notes, edge_periods  # noqa: E402
from ui.tabs.page_convergence import drivers_rows  # noqa: E402
from ui.tabs.page_data import build_record  # noqa: E402
from ui.tabs.page_overview import kpi_items  # noqa: E402
from ui.tabs.page_system import build_ledger_rows  # noqa: E402

NAN = float("nan")
_BAD = re.compile(r"(?<![a-z])(nan|inf|none)(?![a-z])", re.I)
EV = {"ic": 0.034, "n": 138, "p": 0.35, "mde": 0.22, "boot_ci": [-0.07, 0.14], "yearly": {"2024": 0.1},
      "yearly_positive_share": 1.0}


def payload(nko="UNPROVEN", osc="UNPROVEN", sig=False, lift="fail", tier="NO_PROVEN_EDGE", prec_p=0.4) -> tuple:
    con = {
        "horizon": 10,
        "nko": {"status": nko, "evidence": EV, "pillar_ic": {"top_down": {"ic": 0.03}, "bottom_up": {"ic": -0.02},
                                                               "agreement": {"ic": NAN}}},
        "oscillator": {"status": osc, "evidence": EV},
        "edge": {"available": True, "significant": sig, "ic": 0.12 if sig else 0.04, "p": 0.01 if sig else 0.3,
                 "n": 234},
        "lift": {"available": lift != "na", "passed": lift == "pass", "lift": 0.2, "p_one_sided": 0.02},
        "oos": {"ic_C": 0.05, "ic_A": 0.04},
        "backtest": {"by_cost": {"5.0": {"net_sharpe": 0.4, "t": 2.4}}},
        "trust": {"tier": tier},
    }
    prec = {"hero": {"n": 10, "positive_pct": 90.0, "unconditional_positive_pct": 55.0, "binomial_p": prec_p}}
    return con, prec


def _status(rows, rid):
    return next(r for r in rows if r["id"] == rid)["status"][0]


def test_ledger_status_follows_flags():
    rows = build_ledger_rows(*payload())
    assert [r["id"] for r in rows] == ["L-13", "", "L-12", "L-03", "L-05", "L-04", "L-10"]
    assert _status(rows, "L-13") == "Unproven" and _status(rows, "L-03") == "Not proven"
    assert _status(rows, "L-05") == "Not proven" and _status(rows, "L-04") == "Not tradeable"
    assert _status(rows, "L-10") == "Descriptive"
    rows = build_ledger_rows(*payload(nko="VALIDATED", osc="VALIDATED", sig=True, lift="pass", tier="TRADEABLE_EDGE",
                                      prec_p=0.01))
    assert [_status(rows, i) for i in ("L-13", "L-12", "L-03", "L-05", "L-04", "L-10")] == \
        ["Validated", "Validated", "Proven", "Passed", "Tradeable", "Significant"]
    rows = build_ledger_rows(*payload(nko="INSUFFICIENT_DATA", lift="na"))
    assert _status(rows, "L-13") == "Not enough data" and _status(rows, "L-05") == "Not enough data"
    # a small p on a NEGATIVE IC is "wrong way" only via the payload's own p (1 − p < α)
    con, prec = payload()
    con["edge"] = {"available": True, "significant": False, "ic": -0.15, "p": 0.995, "n": 300}
    assert _status(build_ledger_rows(con, prec), "L-03") == "Wrong way"


def test_ledger_never_prints_nan_or_trade_words():
    rnd = random.Random(7)
    poison = (None, NAN, float("inf"), "x", {}, [], True)
    for _ in range(600):
        con, prec = payload(rnd.choice(("VALIDATED", "UNPROVEN", "BOGUS")), rnd.choice(("UNPROVEN", None)))
        for _ in range(rnd.randint(0, 6)):
            k = rnd.choice(list(con))
            if isinstance(con[k], dict) and con[k] and rnd.random() < 0.7:
                con[k][rnd.choice(list(con[k]))] = rnd.choice(poison)
            else:
                con[k] = rnd.choice(poison)
        rows = build_ledger_rows(rnd.choice((con, None, "x")), rnd.choice((prec, None, {"hero": "x"})))
        assert len(rows) == 7
        for r in rows:
            for t in (r["claim"], r["status"][0], r["number"], r["meaning"]):
                assert not _BAD.search(str(t)), t
                assert not BUY_SELL_RE.search(str(t)), t


def test_data_notes_feed_the_pill():
    rc = {"warnings": ["SYNTHETIC DEMO DATA — x", "Date column 'DATE': row 4274 duplicates 28/08/2023",
                       "index: provenance STALE (as of 2026-11-06)"],
          "provenance": {"index": {"provenance": "stale"}, "macro": {"provenance": "fallback"},
                         "sheet": {"provenance": "live"}, "ohlcv": None}}
    notes = data_notes(rc)
    assert notes == ["Date column 'DATE': row 4274 duplicates 28/08/2023", "index: provenance STALE (as of 2026-11-06)",
                     "macro: provenance FALLBACK"]
    assert data_notes(None) == [] and data_notes({"warnings": "x", "provenance": "y"}) == []


def test_drivers_rows_follow_the_pillars():
    comps = [{"id": cid, "pillar": pil, "label": lab, "z": 0.1 * i} for i, (pil, cid, lab, _) in enumerate(COMPONENTS)]
    k = {"components": comps, "drivers": comps[-3:], "pillars": {"top_down": 0.5, "bottom_up": None,
                                                                   "agreement": -1.0}}
    rows = drivers_rows(k)
    assert len(rows) == 3 + len(COMPONENTS)
    assert [r["label"] for r in rows if r["kind"] == "pillar"] == ["Top-down", "Bottom-up", "Agreement"]
    assert sum(r["driver"] for r in rows) == 3
    pillar_of, cur = {}, None
    for r in rows:
        if r["kind"] == "pillar":
            cur = r["label"]
        else:
            pillar_of[r["label"]] = cur
    assert pillar_of["Signed agreement"] == "Agreement" and pillar_of["Aarambh's forecast"] == "Top-down"
    assert drivers_rows({"components": "x", "drivers": 3, "pillars": None})[0]["z"] is None


def test_pages_hide_links():
    t = redact("sheet: live fetch failed (HTTPError: 404 for https://docs.google.com/spreadsheets/d/abc/edit); x")
    assert "docs.google" not in t and "https" not in t and "[link hidden]" in t


def test_notices_hide_links_and_name_the_facts():
    nko = {"value": 30.0, "two_system": False, "as_of": "2026-11-06"}
    fr = build_freshness("2026-11-06", "2026-11-11")                  # Fri → Wed: 3 trading days
    assert fr["behind"] == 3 and fr["label"] == "STALE" and fr["tone"] == "danger"
    assert build_freshness("2026-11-06", "2026-11-08")["label"] == "CURRENT"     # Friday read on Sunday
    assert build_freshness("2026-11-06", "2026-11-09")["label"] == "1D BEHIND"
    rows = build_notices(warnings=["SYNTHETIC DEMO DATA — x", "sheet: provenance STALE — https://docs.google.com/x",
                                   "Predictor INIRYY starts 2021-01-01 and truncates the usable history: x"],
                         synthetic=True, forward_mode=False, nko=nko, freshness=fr)
    titles = [r["title"] for r in rows]
    assert titles == ["Synthetic demo data", "Valuation context", "Top-down only", "Latest data unavailable",
                      "Your sheet stale", "History bounded"]
    assert all("http" not in r["body"] and "docs.google" not in r["body"] for r in rows)
    assert {r["kind"] for r in rows} <= {"info", "warning"}
    assert build_freshness(None)["label"] == ""


def test_tape_and_quote_shapes():
    idx = pd.bdate_range("2026-11-02", periods=3)
    ohlcv = {"RELIANCE.NS": pd.DataFrame({"Close": [100.0, 101.0, 99.99]}, index=idx),
             "BAD.NS": pd.DataFrame({"Close": [np.nan, 5.0, np.nan]}, index=idx)}
    tape = build_tape(ohlcv)
    assert tape == [{"sym": "RELIANCE", "last": 99.99, "chg_pct": (99.99 / 101.0 - 1) * 100}]
    macro = pd.DataFrame({"US10Y": [4.0, 4.1]}, index=idx[:2])
    assert build_tape({}, macro)[0]["sym"] == "US10Y" and build_tape(None, None) == []
    q = build_quote(pd.Series([24000.0, 24240.0], index=idx[:2]))
    assert q["last"] == 24240.0 and abs(q["chg_pct"] - 1.0) < 1e-9 and q["as_of"] == "2026-11-03"
    assert build_quote(None)["last"] is None


def test_edge_contract_shape():
    idx = pd.bdate_range("2021-03-01", "2023-06-30")
    rng = np.random.default_rng(1)
    sig = pd.Series(rng.normal(size=len(idx)), index=idx)
    fwd = pd.Series(rng.normal(size=len(idx)), index=idx)
    e = {"ic": 0.04, "n": 60, "p": 0.3, "yearly": {"2021": 0.1, "2022": -0.05, "2023": 0.02}}
    c = edge_contract(e, sig, fwd, 10)
    assert set(c) == {"ic", "se", "n", "p", "horizon", "blocks", "positive_blocks", "scored_blocks"}
    assert [b["start"][:4] for b in c["blocks"]] == ["2021", "2022", "2023"]
    assert c["blocks"][0]["start"] == "2021-03-01" and c["blocks"][-1]["end"] == "2023-06-30"
    assert c["positive_blocks"] == 2 and c["scored_blocks"] == 3 and edge_periods(c) == (2, 3)
    assert 0 < c["se"] < 0.2 and c["horizon"] == 10
    empty = edge_contract({"ic": NAN, "n": 2, "yearly": {}}, sig, fwd, 10)
    assert empty["ic"] is None and empty["se"] is None and empty["blocks"] == []


def test_overview_kpis_print_no_nan():
    con = {"horizon": 10, "nko": {"value": -50.7, "status": "UNPROVEN", "tier": "MODERATE", "two_system": True,
                                  "pillars": {"top_down": -0.71, "bottom_up": -0.75, "agreement": -1.32},
                                  "strength_percentile": 76.1, "evidence": EV,
                                  "edge": {"ic": 0.006, "se": 0.085, "n": 148, "positive_blocks": 5,
                                           "scored_blocks": 7}}}
    v = build_read_verdict(con, None, breadth={"overbought_pct": 0.0, "oversold_pct": 16.7}, agreement_ratio=0.79)
    items = kpi_items(v)
    assert [i["label"] for i in items] == ["Read", "Nishkarsh Oscillator", "Walk-Forward Edge",
                                            "Nirnay Net Breadth", "Top-Down Pillar", "Engine Agreement"]
    assert [i["value"] for i in items] == ["Bearish", "−51", "+0.006", "−17 pts", "−0.71σ", "79%"]
    assert items[0]["color_class"] == "danger" and items[3]["color_class"] == "danger"
    assert items[3]["subtext"] == "0% overbought · 17% oversold"
    for v2 in (build_read_verdict(None), build_read_verdict({"nko": {"value": NAN}})):
        for it in kpi_items(v2):
            for t in (it["value"], it["subtext"]):
                assert not _BAD.search(str(t)), t
                assert not BUY_SELL_RE.search(str(t)), t


def test_data_record_joins_by_date_newest_first():
    idx = pd.bdate_range("2026-01-01", periods=5)
    ss = {"aarambh_ts": pd.DataFrame({"Date": idx, "Forecast": [0.1, 0.2, np.nan, 0.4, 0.5],
                                      "AvgZ": [1.0] * 5, "Valid": [True] * 5}, index=idx),
          "conclusion_frame": pd.DataFrame({"NKO": [np.nan, 10.0, 20.0, -5.0, 40.0],
                                            "two_system": [False, True, True, True, True]}, index=idx),
          "label_levels": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx),
          "nirnay_daily": pd.DataFrame({"Overbought_Pct": [50.0] * 3}, index=idx[2:]),
          "conclusion": {"label_src": "^NSEI"}, "run_config": {"forward_mode": True}}
    r = build_record(ss)
    assert list(r.columns[:3]) == ["Nifty 50", "NKO", "Two systems"] and len(r) == 5
    assert r.loc[idx[1], "NKO"] == 10.0 and r.loc[idx[0], "Two systems"] == "no"
    assert np.isnan(r.loc[idx[0], "Overbought %"]) and r.loc[idx[4], "Overbought %"] == 50.0
    assert r.loc[idx[3], "Valid"] == "yes" and "Forecast" in r.columns and r.index.name == "Date"
    assert build_record({}).empty
