"""UI page contracts (pure helpers behind Today / Reliability / Data).

  P1  the Ledger's status words follow the payload's own flags (never a UI threshold), one row per claim
  P2  no "nan"/"None" text and no buy/sell/signal/call wording in any ledger row, for any payload
  P3  the header pill's data notes = run warnings (bar SYNTHETIC) + uncovered stale/fallback/empty sources
  P4  Today's driver chart lists the three pillars, each followed by exactly its own inputs, top-3 marked
  P5  the Data page never shows a link (a sheet URL is a secret)
"""
from __future__ import annotations

import os
import random
import re
import sys

_ROOT = os.environ.get("NISHKARSH_ROOT", "/home/user/Nishkarsh")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from convergence.nishkarsh import COMPONENTS  # noqa: E402
from ui.components import BUY_SELL_RE, data_notes  # noqa: E402
from ui.tabs.tab_data import redact  # noqa: E402
from ui.tabs.tab_reliability import build_ledger_rows  # noqa: E402
from ui.tabs.tab_today import drivers_rows  # noqa: E402

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


def test_data_page_hides_links():
    t = redact("sheet: live fetch failed (HTTPError: 404 for https://docs.google.com/spreadsheets/d/abc/edit); x")
    assert "docs.google" not in t and "https" not in t and "[link hidden]" in t
