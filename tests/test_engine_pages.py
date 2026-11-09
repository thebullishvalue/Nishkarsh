"""The three engine pages (Aarambh · Nirnay · Convergence): their display views and row builders.

  E1  breadth is oriented by the ONE polarity switch: net = bullish-side share − bearish-side share
  E2  a stock is read on a date only when the engine counts it live; its side follows its own zone
  E3  the average oscillator is judged against that date's median cut (causal per date, no repainting)
  E4  the pillar lean words use the read verdict's own band
  E5  divergences: recent ones counted on the conclusion frame's sessions; none recent → most recent shown
  E6  model-quality cards, stage and backtest tables print no "nan"/"None" and no trade words, for any payload
  E7  no module-level colour in the page modules (colours resolve per render, so Paper repaints)
"""
from __future__ import annotations

import ast
import os
import random
import re
import sys
from pathlib import Path

_ROOT = os.environ.get("NISHKARSH_ROOT", "/home/user/Nishkarsh")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from convergence.verdict import _Z_LEAN  # noqa: E402
from convergence.views import (  # noqa: E402
    PILLAR_NO_LEAN, ZONE_WORDS, avg_oscillator_view, breadth_view, cut_bands, lean_phrase, stock_snapshot, zone_side,
)
from core.config import NIRNAY_POLARITY  # noqa: E402
from ui.components import BUY_SELL_RE  # noqa: E402
from ui.tabs.page_aarambh import backtest_table, quality_items, stage_table  # noqa: E402
from ui.tabs.page_convergence import divergence_rows  # noqa: E402
from ui.tabs.page_nirnay import drill_table  # noqa: E402

NAN = float("nan")
_BAD = re.compile(r"(?<![a-z])(nan|inf|none)(?![a-z])", re.I)
IDX = pd.bdate_range("2025-01-01", periods=6)


def _stock(osc, cond, warm=(), close=None):
    n = len(osc)
    return pd.DataFrame({"Close": close if close is not None else [100.0] * n, "Unified_Osc": osc,
                         "Condition": cond, "Warmup": [i in warm for i in range(n)],
                         "OS_Cut": [-2.0 - 0.1 * i for i in range(n)], "OB_Cut": [2.0 + 0.1 * i for i in range(n)]},
                        index=IDX[:n])


def test_breadth_is_oriented_by_the_polarity():
    nd = pd.DataFrame({"Overbought_Pct": [50.0, 10.0], "Oversold_Pct": [10.0, 40.0], "n_live": [50, 49]}, index=IDX[:2])
    b = breadth_view(nd)
    trend = NIRNAY_POLARITY == "trend"
    assert list(b["net"]) == ([40.0, -30.0] if trend else [-40.0, 30.0])
    assert list(b["bull_pct"]) == ([50.0, 10.0] if trend else [10.0, 40.0])
    assert ZONE_WORDS == (("Overbought", "Oversold") if trend else ("Oversold", "Overbought"))
    assert breadth_view(None).empty and breadth_view(pd.DataFrame()).empty


def test_stock_snapshot_reads_live_stocks_on_the_date():
    dfs = {"AAA.NS": _stock([1.0, 3.0, -3.0], ["Neutral", "Overbought", "Oversold"]),
           "BBB.NS": _stock([2.5, 0.1, 0.2], ["Overbought", "Neutral", "Neutral"], warm=(2,)),   # warm-up on day 3
           "CCC.NS": _stock([0.5, np.nan, -0.5], ["Neutral", "Neutral", "Neutral"])}             # no osc on day 2
    s = stock_snapshot(dfs, IDX[1])
    assert set(s.index) == {"AAA", "BBB"}                                # CCC not live on day 2
    assert s.loc["AAA", "zone"] == "Overbought" and s.loc["AAA", "side"] == zone_side("Overbought")
    s3 = stock_snapshot(dfs, IDX[2])
    assert set(s3.index) == {"AAA", "CCC"}                               # BBB is in warm-up on day 3
    assert zone_side("Oversold") == -zone_side("Overbought") != 0 and zone_side("Neutral") == 0 == zone_side(None)
    assert stock_snapshot({}).empty and stock_snapshot(None).empty
    assert set(stock_snapshot(dfs).index) == {"AAA", "CCC"}              # default: the latest live date


def test_average_oscillator_uses_each_dates_median_cut():
    dfs = {"AAA.NS": _stock([1.0, 3.0, -3.0], ["Neutral"] * 3), "BBB.NS": _stock([0.0, 0.0, 0.0], ["Neutral"] * 3)}
    cuts = cut_bands(dfs)
    assert list(cuts["ob_cut"].round(2)) == [2.0, 2.1, 2.2]              # per date, not today's cut painted back
    nd = pd.DataFrame({"Avg_Signal": [2.05, 2.15, -2.5]}, index=IDX[:3])
    av = avg_oscillator_view(nd, dfs)
    assert list(av["zone"]) == [1.0, 1.0, -1.0]                          # 2.15 > 2.1 on day 2; −2.5 < −2.2 on day 3
    sign = 1.0 if NIRNAY_POLARITY == "trend" else -1.0
    assert list(av["side"]) == [sign, sign, -sign]
    av0 = avg_oscillator_view(nd, {})
    assert av0["zone"].isna().all() and avg_oscillator_view(None, dfs).empty


def test_pillar_lean_words_use_the_verdicts_band():
    assert PILLAR_NO_LEAN == _Z_LEAN
    assert lean_phrase(_Z_LEAN) == "leans bullish" and lean_phrase(-_Z_LEAN) == "leans bearish"
    assert lean_phrase(0.0) == "no lean" and lean_phrase(None) == "" and lean_phrase(NAN) == ""


def test_divergences_count_recent_sessions_and_fall_back():
    frame_idx = pd.bdate_range("2025-01-01", periods=40)
    ev = pd.DataFrame({"type": ["CONTRADICTION", "NIRNAY_LEADS", "AARAMBH_LEADS"], "zA": [0.6, -0.1, 1.7],
                       "zN": [-0.9, 2.1, 0.0], "persistent": [True, False, False]},
                      index=[frame_idx[2], frame_idx[30], frame_idx[39]])
    tbl, n = divergence_rows(ev, frame_idx, 20)
    assert n == 2 and len(tbl) == 2 and list(tbl["Type"]) == ["Aarambh leads", "Nirnay leads"]   # newest first
    assert tbl.index.name == "Date"
    old, n0 = divergence_rows(ev.iloc[:1], frame_idx, 20)
    assert n0 == 0 and len(old) == 1 and old["Persistent"].iloc[0] == "yes"                        # most recent shown
    for t in tbl["What happened"]:
        assert not _BAD.search(t) and not BUY_SELL_RE.search(t)
    assert divergence_rows(None, frame_idx)[0].empty and divergence_rows(pd.DataFrame(), [])[1] == 0


def test_quality_cards_and_tables_never_print_nan_or_trade_words():
    good = {"rows": 3979, "valid": 2469, "first_valid": "2015-11-16",
            "wf_meta": {"min_train": 1500, "max_train": 2000, "purge": 10, "refit": 5, "chunks_total": 494,
                        "chunks_skipped_starved": 0, "chunks_fallback": 0},
            "model_stats": {"r2_vs_rw": -0.04, "avg_model_spread": 0.0035}}
    items = quality_items(good, 0.0028, True, 10)
    assert [i["label"] for i in items] == ["Walk-Forward Windows", "Out-of-Sample Rows", "Skill vs No Change",
                                           "Ensemble Spread", "Purge"]
    assert [i["value"] for i in items] == ["494", "2,469", "−0.040", "28 bps", "10 rows"]
    assert items[2]["color_class"] == "warning" and items[4]["color_class"] == "success"
    assert quality_items(good, 0.11, False, 10)[2]["label"] == "Skill vs Random Walk"
    rnd = random.Random(3)
    poison = (None, NAN, float("inf"), "x", {}, [], True)
    for _ in range(300):
        a = {k: (dict(v) if isinstance(v, dict) else v) for k, v in good.items()}
        for _ in range(rnd.randint(0, 5)):
            key = rnd.choice(list(a))
            if isinstance(a[key], dict) and a[key] and rnd.random() < 0.7:
                a[key][rnd.choice(list(a[key]))] = rnd.choice(poison)
            else:
                a[key] = rnd.choice(poison)
        for it in quality_items(rnd.choice((a, None, "x")), rnd.choice(poison + (0.003,)), rnd.random() < 0.5, 10):
            for t in (it["value"], it["subtext"], it["label"]):
                assert not _BAD.search(str(t)), t
                assert not BUY_SELL_RE.search(str(t)), t
    st_ = stage_table([{"stage": "forecast", "ic": 0.02, "n": 234}, {"stage": "C", "ic": NAN, "n": None}, "x"])
    assert list(st_["Stage"]) == ["Aarambh forecast", "Conclusion C"] and st_["IC"].isna().iloc[1]
    assert stage_table(None).empty
    bt = backtest_table({"reference_bps": 5.0, "by_cost": {"10.0": {"net_sharpe": 0.1, "ann_return": 0.01},
                                                            "0.0": {"net_sharpe": 0.2}, "x": {}},
                         "nonoverlap_reference": {"net_sharpe": 0.0, "n": 234}})
    assert list(bt["Cost"]) == ["0 bps", "10 bps", "5 bps · separate periods"]
    assert bt.loc[1, "Return / yr %"] == 1.0 and backtest_table(None).empty
    for t in " ".join(bt.columns) + " ".join(st_.columns):
        assert not BUY_SELL_RE.search(t)


def test_drill_table_is_newest_first_and_flags_read_as_words():
    df = _stock([1.0, 2.0, 3.0], ["Neutral", "Overbought", "Overbought"])
    df["Change_Point"] = [False, True, False]
    df["Buy_Signal"] = [False, True, False]
    t = drill_table(df)
    assert list(t.index) == list(IDX[:3][::-1]) and t.index.name == "Date"
    assert list(t["Change_Point"]) == ["no", "yes", "no"] and "Buy_Signal" not in t.columns
    assert drill_table(None).empty


def test_page_modules_bind_no_colour_at_import():
    """Rule 95: colours resolve per render — no module-level chart_color/chart_rgba/panel_bg/grid_rgba
    call and no hex or rgba literal in the engine page modules."""
    root = Path(_ROOT) / "ui" / "tabs"
    for name in ("page_aarambh.py", "page_nirnay.py", "page_convergence.py", "engine_common.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") in (
                            "chart_color", "chart_rgba", "panel_bg", "grid_rgba"):
                        raise AssertionError(f"{name}: module-level colour call")
        src = (root / name).read_text(encoding="utf-8")
        assert not re.search(r"#[0-9A-Fa-f]{6}\b|(?<![_A-Za-z])rgba\(\s*\d", src), name
