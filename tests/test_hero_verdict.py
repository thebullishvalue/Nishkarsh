"""Hero verdict contract (C-UI / C-V2) — grid + property tests for ``ui.components.build_hero_verdict``.

Successor of council/a4/hero_harness.py (rules R0–R11 of the v1 hero), re-stated for the v2 contract:
  H0  never raises; always returns the documented keys (any missing key / NaN / garbage input)
  H1  no "nan" / "inf" token in any rendered text; missing numbers render "—"
  H2  a coloured directional call (signal_class buy/sell) ONLY when trust ∈ {DIRECTIONAL_EDGE, TRADEABLE_EDGE}
  H3  INVERTED → never a call; NO LEAN / missing C → never a call
  H4  forward mode: no levels-model words (fair value, residual, cheap/expensive, primary trade signal, …)
  H5  levels mode: "valuation context — falsified as a forecaster" in the mode label AND the headline
  H6  the NIRNAY row is conclusion["nirnay"]["text"] verbatim; the trust chip text is conclusion["trust"]["text"]
  H7  PRECEDENT votes only when binomial p < EDGE_ALPHA, and its text carries k/n + unconditional + p
  H8  SYNTHETIC banner iff run_config provenance / warnings say SYNTHETIC
  H9  the hero's own words never claim edge (confirms / calibrated / adds value / validated)
  H10 render_hero_card never raises and escapes injected markup
  H11 market read (Overview): no buy/sell/signal wording unless the oscillator is VALIDATED; the
      oscillator's status is always named in plain words; side/colour follow the oscillator's own
      side; no nan text; levels mode keeps its tag; render_market_hero escapes and never raises
Inputs are produced COHERENTLY with the real conclusion-layer producers (tier_of, direction_of,
trust_readout, nirnay_readout, _json_safe) and then poisoned for robustness. Deterministic.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
from itertools import product

_ROOT = os.environ.get("NISHKARSH_ROOT", "/home/user/Nishkarsh")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest  # noqa: E402

import ui.components as comp  # noqa: E402
from convergence.conclusion import (  # noqa: E402
    LiftResult, _json_safe, direction_of, nirnay_readout, tier_of, trust_readout,
)
from core.config import EDGE_ALPHA  # noqa: E402
from ui.components import (  # noqa: E402
    BUY_SELL_RE, CALL_TIERS, build_hero_verdict, build_market_read, market_texts, render_hero_card, render_market_hero,
)
from ui.copy import EDGE_WORDS, LEVELS_TAG, forbidden_hits  # noqa: E402

NAN = float("nan")
SEED = 20261108
TRUST_TIERS = ("INVERTED", "NO_PROVEN_EDGE", "DIRECTIONAL_EDGE", "TRADEABLE_EDGE", "INSUFFICIENT_DATA")
STATES = ("CONFIRM", "CONTRADICT", "NEUTRAL", "NO_DATA", None)
C_VALUES = (88.0, -72.0, 45.0, -31.0, 12.0, -9.0, 0.0, None, NAN)
_BAD = re.compile(r"(?<![a-z])(nan|inf)(?![a-z])", re.I)
KEYS = ("signal", "signal_class", "call", "direction", "tier", "score", "score_text", "headline",
        "mode_label", "trust", "evidence", "as_of", "synthetic", "banner", "horizon", "forward_mode", "market")
MARKET_KEYS = ("headline", "tag", "side", "validated", "value", "value_text", "gauge_label", "subline",
               "aarambh_text", "nirnay_text", "as_of", "status", "status_label", "status_class", "status_text",
               "numbers", "trust_tier", "levels_note")


# ── coherent producers ───────────────────────────────────────────────────────
def _graded(tier: str) -> tuple[dict, dict, dict]:
    """(graded edge, tranche ref, non-overlap ref) that make trust_readout return ``tier``."""
    if tier == "INSUFFICIENT_DATA":
        return {"ic": NAN, "t": NAN, "n": 12, "p": NAN, "available": False, "significant": False}, {}, {}
    if tier == "INVERTED":
        return {"ic": -0.15, "t": -2.6, "n": 300, "p": 0.995, "available": True, "significant": False}, {}, {}
    if tier == "NO_PROVEN_EDGE":
        return {"ic": 0.04, "t": 0.6, "n": 234, "p": 0.26, "available": True, "significant": False}, \
            {"net_sharpe": 0.13, "t": 0.4}, {"t": 0.02}
    g = {"ic": 0.12, "t": 2.5, "n": 457, "p": 0.006, "available": True, "significant": True}
    if tier == "DIRECTIONAL_EDGE":
        return g, {"net_sharpe": 0.3, "t": 1.1}, {"t": 0.9}
    return g, {"net_sharpe": 0.63, "t": 2.7}, {"t": 2.2}


def _lift(kind: str) -> LiftResult:
    if kind == "na":
        return LiftResult(False, NAN, NAN, 5, 4, NAN, NAN, NAN, NAN, NAN, 1.0, False, NAN)
    if kind == "fail":
        return LiftResult(True, 0.15, 0.03, 71, 62, 0.117, 0.64, 0.26, 0.001, 0.187, 1.0, False, 0.455)
    return LiftResult(True, 0.20, -0.05, 90, 80, 0.25, 2.1, 0.018, -0.02, 0.31, 0.6, True, 0.3)


def make_conclusion(trust_tier, C, state, forward, lift_kind="fail", kappa_enabled=True, single=False,
                    adds_value=False) -> dict:
    g, bt, no = _graded(trust_tier)
    trust = trust_readout(g, "Aarambh head", bt, no, 10)
    assert trust["tier"] == trust_tier
    L = _lift(lift_kind)
    kap = 0.6 if (state == "CONTRADICT" and L.passed and kappa_enabled) else 1.0
    oos = {"ic_C": 0.05, "ic_A": 0.04, "n": 120, "adds_value": adds_value}
    nir = nirnay_readout(single, state, L, kap, kappa_enabled, oos)
    c_ok = C is not None and math.isfinite(C)
    latest = {"date": "2025-05-01", "C": C, "C_aarambh": C, "zA": (C / 50 if c_ok else C), "zN": 0.4,
              "state": state, "kappa": kap,
              "tier": tier_of(C) if c_ok else "NO LEAN", "direction": direction_of(C) if c_ok else "NEUTRAL"}
    payload = {
        "schema": "nishkarsh.conclusion/1", "as_of": "2025-05-01", "horizon": 10, "label_src": "^NSEI",
        "forward_mode": forward, "kappa_enabled": kappa_enabled, "single_system": single,
        "latest": latest, "lift": {"passed": L.passed, "available": L.available, "mde": L.mde},
        "oos": oos, "trust": trust, "nirnay": nir,
        "agreement": {"today_state": state, "same_sign_rate_60d": 0.3},
        "divergence_recent": {"AARAMBH_LEADS": 1, "NIRNAY_LEADS": 0, "CONTRADICTION": 2, "window_days": 20},
    }
    return _json_safe(payload)


PRECEDENTS = (
    None, {},
    {"horizon": 10, "median": 0.93, "positive_pct": 70.0, "n": 10, "dir": 1,
     "unconditional_positive_pct": 50.5, "binomial_p": 0.34},
    {"horizon": 10, "median": -1.8, "positive_pct": 10.0, "n": 10, "dir": -1,
     "unconditional_positive_pct": 55.0, "binomial_p": 0.004},
    {"horizon": 10, "median": 2.1, "positive_pct": 100.0, "n": 10, "dir": 1,
     "unconditional_positive_pct": 52.0, "binomial_p": 0.002},
    {"horizon": None, "median": NAN, "positive_pct": NAN, "n": NAN, "dir": 0,
     "unconditional_positive_pct": None, "binomial_p": NAN},
)
SYN_RC = {"target": "NIFTY50_PE", "forward_mode": True,
          "provenance": {"sheet": {"provenance": "live", "note": "SYNTHETIC seed=7"}}, "warnings": []}
LIVE_RC = {"target": "NIFTY50_PE", "forward_mode": True,
           "provenance": {"sheet": {"provenance": "live", "note": "sheet ok"}, "macro": None}, "warnings": []}


def _texts(v: dict) -> list[str]:
    out = [v["signal"], v["score_text"], v["headline"], v["mode_label"], v["as_of"], v["trust"]["text"],
           v["trust"]["label"], v.get("sign_note", "")]
    out += [r["text"] for r in v["evidence"]] + [f'{r["tag"]} {r["word"]}' for r in v["evidence"]]
    if v["banner"]:
        out.append(v["banner"])
    return [str(t) for t in out]


def _own_texts(v: dict) -> list[str]:
    """Texts the hero itself writes (not the upstream trust / Nirnay strings)."""
    return [v["signal"], v["headline"], v["mode_label"]] + [r["text"] for r in v["evidence"]
                                                            if r["tag"] != "NIRNAY"]


def _grid():
    for tier, C, state, fwd, lk, prec, syn in product(TRUST_TIERS, C_VALUES, STATES, (True, False),
                                                     ("na", "fail", "pass"), range(len(PRECEDENTS)),
                                                     (True, False)):
        yield tier, C, state, fwd, lk, PRECEDENTS[prec], syn


def _check(con, prec, rc, *, forward, nir_text=None, trust_text=None):
    v = build_hero_verdict(con, {"conviction_score": NAN}, prec, rc)
    for k in KEYS:
        assert k in v, k
    for t in _texts(v):
        assert not _BAD.search(t), (t, con)
    t_tier = v["trust"]["tier"]
    if v["signal_class"] in ("buy", "sell") or v["call"]:
        assert t_tier in CALL_TIERS, v
        assert v["tier"] != "NO LEAN" and v["score"] is not None
    if t_tier == "INVERTED":
        assert v["signal_class"] == "hold" and not v["call"]
    if forward:
        for t in _texts(v):
            assert not forbidden_hits(t, True), (forbidden_hits(t, True), t)
    else:
        assert LEVELS_TAG in v["mode_label"] and LEVELS_TAG in v["headline"]
    if nir_text is not None:
        assert [r for r in v["evidence"] if r["tag"] == "NIRNAY"][0]["text"] == nir_text
    if trust_text is not None:
        assert v["trust"]["text"] == trust_text
    pr = [r for r in v["evidence"] if r["tag"] == "PRECEDENT"][0]
    p = comp.num((prec or {}).get("binomial_p"))
    if pr["votes"]:
        assert p is not None and p < EDGE_ALPHA
    if prec and comp.num(prec.get("n")):
        assert "/" in pr["text"] and "unconditional" in pr["text"] and "p=" in pr["text"]
    for t in _own_texts(v):
        for w in EDGE_WORDS:
            assert not re.search(r"\b" + w + r"\b", t, re.I), (w, t)
    assert v["as_of"]
    return v


def test_grid_contract():
    n = 0
    for tier, C, state, fwd, lk, prec, syn in _grid():
        con = make_conclusion(tier, C, state, fwd, lift_kind=lk, single=(state is None and lk == "na"))
        rc = dict(SYN_RC if syn else LIVE_RC, forward_mode=fwd)
        v = _check(con, prec, rc, forward=fwd, nir_text=con["nirnay"]["text"], trust_text=con["trust"]["text"])
        assert v["synthetic"] is syn and (v["banner"] is not None) is syn
        n += 1
    assert n > 5000


def test_call_requires_proven_tier_exhaustive():
    """Every directional C under every trust tier: coloured iff DIRECTIONAL/TRADEABLE."""
    for tier, C, fwd in product(TRUST_TIERS, (88.0, -72.0, 45.0, -31.0, 12.0, -12.0), (True, False)):
        v = build_hero_verdict(make_conclusion(tier, C, "NEUTRAL", fwd), None, None, LIVE_RC)
        assert v["call"] is (tier in CALL_TIERS)
        assert v["signal_class"] == (("buy" if C > 0 else "sell") if tier in CALL_TIERS else "hold")


def test_missing_and_poisoned_inputs():
    rnd = random.Random(SEED)
    poison = (None, NAN, float("inf"), "n/a", "", [], {}, True, -0.0, "<script>x</script>")
    base = make_conclusion("DIRECTIONAL_EDGE", 45.0, "CONTRADICT", True, lift_kind="pass")
    blobs = [None, {}, [], "garbage", 3.0, {"latest": None}, {"latest": {}, "trust": None, "nirnay": None}]
    for _ in range(4000):
        con = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        for _ in range(rnd.randint(1, 6)):
            k = rnd.choice(list(con))
            if isinstance(con[k], dict) and con[k] and rnd.random() < 0.7:
                kk = rnd.choice(list(con[k]))
                con[k][kk] = rnd.choice(poison)
            elif rnd.random() < 0.5:
                con.pop(k, None)
            else:
                con[k] = rnd.choice(poison)
        blobs.append(con)
    precs = list(PRECEDENTS) + [{"n": "ten"}, {"n": 10, "positive_pct": None}, "x", 7]
    rcs = [None, {}, "x", LIVE_RC, SYN_RC, {"provenance": "bad", "warnings": None},
           {"provenance": {"a": "str"}, "warnings": ["SYNTHETIC demo"]}]
    for i, con in enumerate(blobs):
        v = build_hero_verdict(con, rnd.choice([None, {}, {"conviction_score": "x"}, "bad"]),
                               precs[i % len(precs)], rcs[i % len(rcs)])
        for k in KEYS:
            assert k in v
        for t in _texts(v):
            if "script" in t:      # upstream text echoed verbatim; escaped at render (H10)
                continue
            assert not _BAD.search(t), (t, con)
        if v["call"] or v["signal_class"] != "hold":
            assert v["trust"]["tier"] in CALL_TIERS
        fm = con.get("forward_mode") if isinstance(con, dict) else None
        if fm is False:
            assert LEVELS_TAG in v["mode_label"]


def test_render_escapes_and_never_raises(monkeypatch):
    out = []
    monkeypatch.setattr(comp.st, "markdown", lambda body, **kw: out.append(body))
    con = make_conclusion("TRADEABLE_EDGE", 72.0, "CONFIRM", True, lift_kind="pass", adds_value=True)
    con["trust"]["text"] = '<iframe srcdoc="<script>alert(1)</script>"></iframe>'
    con["nirnay"]["text"] = "<script>alert(2)</script>"
    rc = dict(SYN_RC, target="<img src=x onerror=alert(3)>")
    render_hero_card(build_hero_verdict(con, None, PRECEDENTS[3], rc))
    for bad in (None, {}, {"evidence": [None, {"tag": "<b>"}], "trust": None}, {"signal_class": "<x>"}):
        render_hero_card(bad)
    html = "\n".join(out)
    assert "<script" not in html and "<iframe" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "SYNTHETIC" in html


def test_real_pipeline_shape_example():
    """A to_dict()-shaped payload from the synthetic demo run (NO PROVEN EDGE, CONTRADICT)."""
    con = make_conclusion("NO_PROVEN_EDGE", -47.8, "CONTRADICT", True, lift_kind="fail")
    v = build_hero_verdict(con, {"conviction_score": 47.8}, PRECEDENTS[2], SYN_RC)
    assert v["signal"] == "LEAN BEARISH (MODERATE) · unvalidated" and v["signal_class"] == "hold" and not v["call"]
    assert v["score_text"] == "C -47.8" and v["trust"]["tier"] == "NO_PROVEN_EDGE"
    pr = [r for r in v["evidence"] if r["tag"] == "PRECEDENT"][0]
    assert pr["text"].startswith("7/10 analogs rose over 10d vs 50% unconditional (binomial p=0.34)")
    assert not pr["votes"] and v["banner"]


# ── H11: the Overview market read and the Agreement Oscillator ────────────────
OSC_EVIDENCE = {"ic": 0.0914, "n": 164, "p": 0.13, "mde": 0.199, "hac_t": 1.4, "boot_ci": [-0.02, 0.2],
                "yearly": {"2020": 0.1, "2021": 0.2, "2022": -0.05, "2023": 0.1, "2024": 0.2, "2025": 0.05,
                           "2026": -0.1}, "yearly_positive_share": 5 / 7}
OSC_VALUES = (0.72, -0.72, 0.35, -0.12, 0.0, None, NAN, 3.0, "x")
NIRNAY_LATEST = (None, {}, {"n_live": 50, "Oversold_Pct": 10.0, "Overbought_Pct": 52.0},
                 {"n_live": 12, "Oversold_Pct": 16.7, "Overbought_Pct": 0.0},
                 {"n_live": 50, "Oversold_Pct": 0.0, "Overbought_Pct": 0.0},
                 {"n_live": NAN, "Oversold_Pct": NAN, "Overbought_Pct": None})


def _osc(status, value):
    v = comp.num(value)
    side = None if v is None else ("BUY-SIDE" if v > 0 else "SELL-SIDE" if v < 0 else "SPLIT")
    ev = None if status == "INSUFFICIENT_DATA" else OSC_EVIDENCE
    return {"value": value, "side": side, "strength": None if v is None else abs(v), "status": status,
            "evidence": ev, "as_of": "2025-05-01",
            "text": "Not yet a validated buy/sell signal — IC +0.091 over 164 windows (p 0.13)."}


def _check_market(m: dict, status: str, forward: bool) -> None:
    for k in MARKET_KEYS:
        assert k in m, k
    texts = market_texts(m)
    for t in texts:
        assert not _BAD.search(t), t
    if status != "VALIDATED":
        assert not m["validated"]
        for t in texts:
            assert not BUY_SELL_RE.search(t), (t, status)
    assert m["status"] == status and m["status_label"] and m["status_text"]
    assert m["side"] in ("buy", "sell", "split", "none")
    v = comp.num(m["value"])
    if m["side"] == "buy":
        assert v is not None and v > 0 and "buy-side" in m["headline"]
    if m["side"] == "sell":
        assert v is not None and v < 0 and "sell-side" in m["headline"]
    if m["side"] == "none":
        assert v is None and m["value_text"] == "—"
    if not forward:
        assert m["levels_note"] == LEVELS_TAG
    for w in EDGE_WORDS:
        for t in texts:
            if status != "VALIDATED" or w != "validated":
                assert not re.search(r"\b" + w + r"\b", t, re.I), (w, t)


def test_market_read_oscillator_honesty_grid():
    n = 0
    for status, value, tier, C, fwd, nl in product(("VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA"), OSC_VALUES,
                                                   TRUST_TIERS, (72.0, -31.0, 4.0, None), (True, False),
                                                   NIRNAY_LATEST):
        con = make_conclusion(tier, C, "CONTRADICT", fwd)
        con["oscillator"] = _osc(status, value)
        m = build_market_read(con, dict(LIVE_RC, forward_mode=fwd), nl)
        _check_market(m, status, fwd)
        v = build_hero_verdict(con, None, None, dict(LIVE_RC, forward_mode=fwd), nirnay_latest=nl)
        assert v["market"]["status"] == status
        n += 1
    assert n > 3000


def test_market_read_split_and_missing():
    con = make_conclusion("NO_PROVEN_EDGE", 3.9, "CONTRADICT", True)
    con["oscillator"] = _osc("UNPROVEN", 0.0)
    m = build_market_read(con, LIVE_RC, {"n_live": 50, "Oversold_Pct": 10.0, "Overbought_Pct": 52.0})
    assert m["side"] == "split" and m["value_text"] == "0.00" and m["status_label"] == "Unproven"
    assert "52% of the 50 Nifty stocks overbought" in m["subline"] and "forecast flat" in m["subline"]
    for bad in (None, {}, "x", {"oscillator": None}, {"oscillator": {"status": "BOGUS", "value": "nan"}}):
        mm = build_market_read(bad)
        assert mm["status"] == "INSUFFICIENT_DATA" and mm["side"] == "none"
        for t in market_texts(mm):
            assert not _BAD.search(t) and not BUY_SELL_RE.search(t)


def test_market_hero_escapes_and_never_raises(monkeypatch):
    out = []
    monkeypatch.setattr(comp.st, "markdown", lambda body, **kw: out.append(body))
    con = make_conclusion("NO_PROVEN_EDGE", -47.8, "CONTRADICT", True)
    con["oscillator"] = _osc("UNPROVEN", -0.63)
    con["oscillator"]["evidence"] = dict(OSC_EVIDENCE, yearly={"<script>": 0.1})
    render_market_hero(build_market_read(con, SYN_RC, {"n_live": "<img src=x onerror=alert(1)>"}))
    for bad in (None, {}, {"numbers": [None, {"label": "<b>x</b>"}], "side": "<x>"}):
        render_market_hero(bad)
    html = "\n".join(out)
    assert "<script" not in html and "<img" not in html and "<b>x" not in html
    assert "Unproven" in html and "sell-side" in html.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
