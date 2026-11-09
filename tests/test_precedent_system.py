"""Precedent and System page contracts (pure helpers; the pages only format).

  S1  system_edges: all six graded series in the ONE edge contract; nothing re-estimated (IC, n, p
      are the payload's own); a series the payload calls ungraded gets an empty contract
  S2  the edge table and the Ledger markup print no "nan"/"None" and no buy/sell/signal/call words,
      for any payload; every string is escaped
  S3  base-rate cards are NEUTRAL unless the payload's binomial p < EDGE_ALPHA (Override 3)
  S4  analog cards: Tattva's .analog-card anatomy, the lean word of the read then, escaped, no blank
      line (CommonMark would close the HTML block)
  S5  data-layer cards: one per source, provenance word toned, links hidden, SYNTHETIC named
"""
from __future__ import annotations

import os
import random
import re
import sys

_ROOT = os.environ.get("NISHKARSH_ROOT", "/home/user/Nishkarsh")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from convergence.system_views import SYSTEM_EDGE_KEYS, system_edges  # noqa: E402
from core.config import EDGE_ALPHA  # noqa: E402
from pipeline.shell import SOURCE_NAMES  # noqa: E402
from ui.components import BUY_SELL_RE  # noqa: E402
from ui.tabs.page_precedent import analog_card_html, base_rate_items  # noqa: E402
from ui.tabs.page_system import build_ledger_rows, edge_table_rows, ledger_html, provenance_items  # noqa: E402

NAN = float("nan")
_BAD = re.compile(r"(?<![a-z])(nan|inf|none)(?![a-z])", re.I)
_CONTRACT = {"ic", "se", "n", "p", "horizon", "blocks", "positive_blocks", "scored_blocks"}


def _text(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


def _con(osc_status="UNPROVEN", aarambh_available=True) -> dict:
    blocks = [{"start": "2017-01-03", "end": "2017-12-29", "ic": 0.1}, {"start": "2018-01-02", "end": "2018-10-25",
                                                                         "ic": -0.05}]
    pe = {"ic": 0.03, "se": 0.17, "n": 39, "p": 0.42, "horizon": 10, "blocks": blocks, "positive_blocks": 1,
          "scored_blocks": 2}
    return {
        "horizon": 10,
        "nko": {"status": "UNPROVEN", "edge": dict(pe, ic=0.025),
                "pillar_edge": {"top_down": pe, "bottom_up": dict(pe, ic=-0.04), "agreement": dict(pe, ic=0.07)}},
        "oscillator": {"status": osc_status, "evidence": {"ic": -0.003, "n": 64, "p": 0.51,
                                                          "yearly": {"2016": 0.09, "2017": -0.21, "2018": 0.19}}},
        "edge": {"available": aarambh_available, "ic": 0.02, "n": 64, "p": 0.44, "t": 0.15,
                 "yearly": {"2016": 0.15, "2017": -0.11, "2018": 0.1}},
    }


def test_system_edges_contract_and_no_reestimation():
    e = system_edges(_con())
    assert tuple(e) == SYSTEM_EDGE_KEYS
    for k, c in e.items():
        assert set(c) == _CONTRACT, k
    assert e["read"]["ic"] == 0.025 and e["bottom_up"]["ic"] == -0.04
    osc = e["oscillator"]
    assert osc["ic"] == -0.003 and osc["n"] == 64 and osc["p"] == 0.51            # the payload's own numbers
    assert [b["start"][:4] for b in osc["blocks"]] == ["2016", "2017", "2018"]
    assert osc["positive_blocks"] == 2 and osc["scored_blocks"] == 3 and 0 < osc["se"] < 0.2
    assert e["aarambh"]["ic"] == 0.02 and e["aarambh"]["horizon"] == 10


def test_system_edges_ungraded_series_are_empty():
    e = system_edges(_con(osc_status="INSUFFICIENT_DATA", aarambh_available=False))
    assert e["oscillator"]["ic"] is None and e["oscillator"]["blocks"] == []
    assert e["aarambh"]["ic"] is None and e["aarambh"]["se"] is None
    blank = system_edges(None)
    assert tuple(blank) == SYSTEM_EDGE_KEYS and all(c["ic"] is None for c in blank.values())


def test_edge_table_and_ledger_never_print_nan_or_trade_words():
    rnd = random.Random(11)
    poison = (None, NAN, float("inf"), "x", {}, [], True, "<b>")
    for _ in range(300):
        con = _con(rnd.choice(("UNPROVEN", "INSUFFICIENT_DATA", None)), rnd.choice((True, False, None)))
        for _ in range(rnd.randint(0, 6)):
            k = rnd.choice(list(con))
            if isinstance(con[k], dict) and con[k] and rnd.random() < 0.7:
                con[k][rnd.choice(list(con[k]))] = rnd.choice(poison)
            else:
                con[k] = rnd.choice(poison)
        rows = edge_table_rows(system_edges(rnd.choice((con, None, "x"))))
        assert len(rows) == 6
        for r in rows:
            for v in r.values():
                assert not _BAD.search(str(v)), v
                assert not BUY_SELL_RE.search(str(v)), v
        html = ledger_html(build_ledger_rows(con, rnd.choice(({"hero": {"n": 10, "positive_pct": 70.0}}, None))))
        assert "<b>" not in html
        text = _text(html)
        assert not _BAD.search(text) and not BUY_SELL_RE.search(text)
    assert edge_table_rows(system_edges(_con()))[0]["IC"] == "+0.025"


def test_base_rate_cards_neutral_unless_significant():
    summary = {3: {"median": 0.84, "positive_pct": 70.0, "n": 10, "unconditional_positive_pct": 57.3,
                   "binomial_p": 0.53},
               10: {"median": 2.1, "positive_pct": 100.0, "n": 10, "unconditional_positive_pct": 56.0,
                    "binomial_p": 0.003},
               20: {"median": -1.2, "positive_pct": 10.0, "n": 10, "unconditional_positive_pct": 60.0,
                    "binomial_p": 0.001},
               5: {"median": NAN, "positive_pct": None, "n": 0, "unconditional_positive_pct": None,
                   "binomial_p": NAN}}
    items = base_rate_items(summary)
    assert [i["label"] for i in items] == ["+3D Median Return", "+5D Median Return", "+10D Median Return",
                                           "+20D Median Return"]
    tones = {i["label"][:4]: i["color_class"] for i in items}
    assert tones["+3D "] == "neutral" and tones["+5D "] == "neutral"
    assert tones["+10D"] == "success" and tones["+20D"] == "danger"
    assert items[0]["value"] == "+0.8%" and items[0]["subtext"] == "7 of 10 rose · usually 57% · p 0.53"
    for i in items:
        assert not _BAD.search(i["value"] + i["subtext"]), i
    assert 0.003 < EDGE_ALPHA < 0.53


def test_analog_card_anatomy_and_escaping():
    a = {"date": "2017-08-10", "similarity": 0.8956, "nko": -45.8,
         "pillars": {"top_down": -0.95, "bottom_up": NAN, "agreement": None},
         "fwd": {3: -0.66, 5: 0.0, "10": 1.81, 20: None}}
    html = analog_card_html(1, a, (3, 5, 10, 20), "Nifty <50>")
    assert "\n\n" not in html and 'class="analog-card tier-caution"' in html
    assert "Analog 1 · Historical Match" in html and "Nifty &lt;50&gt;" in html and "<50>" not in html
    assert "Bearish" in html and "chip-outline" in html                 # a moderate lean → outline chip
    assert "89.6%" in html and "−46" in html and "+1.8%" in html and 'analog-fwd-tile flat' in html
    text = _text(html)
    assert not _BAD.search(text) and not BUY_SELL_RE.search(text)
    strong = analog_card_html(2, dict(a, nko=72.0), (3,), "Nifty 50")
    assert "tier-strong-buy" in strong and "Strongly Bullish" in strong and "chip-outline" not in strong
    flat = analog_card_html(3, {"date": "x", "nko": "x"}, (3,), "Nifty 50")
    assert "tier-hold" in flat and "No Read" in flat and not _BAD.search(_text(flat))


def test_data_layer_cards_hide_links_and_name_synthetic():
    rc = {"provenance": {
        "sheet": {"provenance": "stale", "as_of": "2026-11-06", "missing": [],
                  "note": "live fetch failed (HTTPError for https://docs.google.com/spreadsheets/d/abc/edit); served cache"},
        "constituents": {"provenance": "live", "as_of": "2026-11-09", "missing": [], "note": "SYNTHETIC seed=7"},
        "ohlcv": {"provenance": "cache", "as_of": "2026-11-09", "missing": ["X.NS", "Y.NS"], "note": "yfinance"},
        "index": None}}
    items = provenance_items(rc)
    assert [i["label"] for i in items] == list(SOURCE_NAMES.values())
    by = {i["label"]: i for i in items}
    assert by["Your sheet"]["value"] == "STALE" and by["Your sheet"]["color_class"] == "warning"
    assert all("docs.google" not in (i["subtext"] + i.get("tooltip", "")) for i in items)
    assert by["Constituents"]["value"] == "SYNTHETIC" and "seeded world" in by["Constituents"]["subtext"]
    assert by["Stock prices"]["value"] == "CACHED" and "2 series missing" in by["Stock prices"]["subtext"]
    assert by["Index"]["value"] == "—" and by["Market macro"]["value"] == "—"
    for i in items:
        assert not _BAD.search(i["value"] + i["subtext"]), i
