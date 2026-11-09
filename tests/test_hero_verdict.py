"""Hero contract (C-UI / NKO) — grid + property tests for the Today hero.

The HERO (``ui.components.build_hero_verdict``) infers from the Nishkarsh Oscillator ONLY
(``conclusion["nko"]`` + ``precedent["hero"]``):
  N0  never raises; always returns the documented keys (missing key / NaN / garbage input)
  N1  no "nan" / "inf" token in any text it writes; missing numbers render "—"
  N2  buy/sell/signal/call wording (BUY_SELL_RE) and a directional ``call`` ONLY when nko status is VALIDATED
  N3  NO LEAN / missing value → never a call; out-of-range values are rejected
  N4  the track record is always named (word + status label); ``nko["text"]`` (which says "buy/sell") is
      never printed by the hero
  N5  a single-system date (two_system False) says "top-down only"
  N6  PRECEDENT votes only when binomial p < EDGE_ALPHA, and its text carries k, n, the usual rate and p
  N7  forward mode: no levels-model words; levels mode: the falsified-as-a-forecaster tag on the title
  N8  SYNTHETIC banner iff run_config provenance / warnings say SYNTHETIC
  N9  its own words never claim edge (confirms / calibrated / adds value / validated) unless VALIDATED
  N10 render_hero never raises and escapes injected markup
Inputs are produced COHERENTLY with the real producers (trust_readout, nirnay_readout, _json_safe,
convergence.nishkarsh.nko_tier) and then poisoned. Deterministic.
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
from convergence.nishkarsh import COMPONENTS, nko_tier  # noqa: E402
from ui.components import BUY_SELL_RE, build_hero_verdict, hero_texts, render_hero  # noqa: E402
from ui.copy import EDGE_WORDS, LEVELS_TAG, forbidden_hits  # noqa: E402

NAN = float("nan")
SEED = 20261108
TRUST_TIERS = ("INVERTED", "NO_PROVEN_EDGE", "DIRECTIONAL_EDGE", "TRADEABLE_EDGE", "INSUFFICIENT_DATA")
STATES = ("CONFIRM", "CONTRADICT", "NEUTRAL", "NO_DATA", None)
C_VALUES = (88.0, -72.0, 45.0, -31.0, 12.0, -9.0, 0.0, None, NAN)
_BAD = re.compile(r"(?<![a-z])(nan|inf)(?![a-z])", re.I)
HERO_KEYS = ("eyebrow", "headline", "side", "value", "value_text", "tier", "direction", "call", "validated",
             "gauge_label", "two_system", "single_system", "explain", "chips", "track", "status", "status_label",
             "precedent", "as_of", "as_of_note", "horizon", "forward_mode", "levels_note", "synthetic", "banner")


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




# ── N: the hero infers from the Nishkarsh Oscillator only ─────────────────────
NKO_EVIDENCE = {"ic": 0.034, "n": 138, "p": 0.35, "mde": 0.22, "hac_t": 0.6, "boot_ci": [-0.078, 0.148],
                "yearly": {"2021": 0.1, "2022": -0.05, "2023": 0.1, "2024": -0.2, "2025": 0.05, "2026": -0.1},
                "yearly_positive_share": 0.5}
NKO_TEXT = ("Not yet a validated buy/sell signal — accuracy (rank IC) +0.034 over 138 non-overlapping 10-day "
            "windows (p 0.35; positive in 3/6 years); this sample can only confirm an IC of 0.22 or more.")
NKO_VALUES = (82.0, -82.0, 48.0, -48.0, 15.0, -12.0, 5.0, 0.0, None, NAN, 300.0, "x")
PILLAR_SETS = ({"top_down": 0.77, "bottom_up": -2.14, "agreement": -0.87},
               {"top_down": 1.2, "bottom_up": 0.9, "agreement": 1.1},
               {"top_down": -0.5, "bottom_up": None, "agreement": None}, {}, {"top_down": NAN, "bottom_up": "x"})
REAL_Z = {"forecast": 0.4, "extension": 0.2, "zone_breadth_a": 0.9, "crossings_a": 0.5, "oscillator_n": -0.6,
          "zone_breadth_n": -0.8, "regime_mix": -1.9, "hmm_state": -1.5, "cut_crossings_n": -2.3,
          "signed_agreement": -0.87}
CONTEXTS = ({"change_5d": -12.4, "change_20d": 30.0, "strength_percentile": 85.2,
             "streak": {"side": "BEARISH", "sessions": 12, "since": "2025-04-14"}},
            {"change_5d": None, "change_20d": None, "strength_percentile": None, "streak": None},
            {"change_5d": NAN, "strength_percentile": "x", "streak": {"side": "NO LEAN", "sessions": 1,
                                                                      "since": "garbage"}})
STATUSES = ("VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA")
_STATUS_LABELS = {"VALIDATED": "Validated", "UNPROVEN": "Unproven", "INSUFFICIENT_DATA": "Not enough data"}
_TRACK_WORDS = {"VALIDATED": "validated", "UNPROVEN": "unproven", "INSUFFICIENT_DATA": "not enough data"}


def make_nko(status, value, two, pillars, z=REAL_Z, context=CONTEXTS[0]) -> dict:
    """An ``nko_readout``-shaped dict (components / drivers built the way the producer builds them)."""
    comps = [{"id": cid, "pillar": pil, "label": lab, "source": src, "z": z.get(cid)} for pil, cid, lab, src in COMPONENTS]
    drivers = sorted([c for c in comps if comp.num(c["z"]) is not None], key=lambda c: -abs(c["z"]))[:3]
    v = comp.num(value)
    return {"value": value, "as_of": "2025-05-01", "tier": nko_tier(v), "two_system": two, "pillars": pillars,
            "components": comps, "drivers": drivers, "status": status,
            "evidence": None if status == "INSUFFICIENT_DATA" else NKO_EVIDENCE, "pillar_ic": {},
            "text": NKO_TEXT, **context}


def _hero(status, value, two=True, pillars=PILLAR_SETS[0], prec=None, rc=LIVE_RC, forward=True, context=CONTEXTS[0]):
    con = make_conclusion("NO_PROVEN_EDGE", -47.8, "CONTRADICT", forward)
    con["nko"] = make_nko(status, value, two, pillars, context=context)
    return build_hero_verdict(con, prec, dict(rc, forward_mode=forward))


def _check_hero(v, status, *, two, prec, forward, synth):
    for k in HERO_KEYS:
        assert k in v, k
    texts = hero_texts(v)
    for t in texts:
        assert not _BAD.search(t), t
        assert "None" not in t, t
    # N2/N3: wording and the call are reserved for a VALIDATED read
    if status != "VALIDATED":
        assert not v["call"] and not v["validated"]
        for t in texts:
            assert not BUY_SELL_RE.search(t), (t, status)
    if v["call"]:
        assert status == "VALIDATED" and v["tier"] in ("STRONG", "MODERATE", "WEAK") and v["value"] is not None
    if v["value"] is None:
        assert v["value_text"] == "—" and not v["call"] and v["tier"] == "NO READ" and v["side"] == "none"
    else:
        assert -100 <= v["value"] <= 100
    if v["tier"] in ("STRONG", "MODERATE", "WEAK"):
        assert v["headline"].startswith("Nishkarsh reads ") and v["direction"].lower() in v["headline"]
    # N4: the track record is always named; the upstream sentence (with its buy/sell words) is never printed
    assert v["status"] == status and v["status_label"] == _STATUS_LABELS[status]
    assert v["track"]["word"] == _TRACK_WORDS[status] and v["track"]["text"].endswith("Details on Reliability.")
    assert all(NKO_TEXT not in t for t in texts)
    # N5: single-system said
    if not two and v["value"] is not None:
        assert v["single_system"] and "top-down only" in v["headline"]
    # N6: precedent votes only when significant
    pr = v["precedent"]
    p = comp.num((prec if isinstance(prec, dict) else {}).get("binomial_p"))
    if pr["votes"]:
        assert p is not None and p < EDGE_ALPHA
    if isinstance(prec, dict) and comp.num(prec.get("n")) and comp.num(prec.get("positive_pct")) is not None:
        assert "rose" in pr["text"] and "usually" in pr["text"] and "(p " in pr["text"]
    # N7: mode words
    if forward:
        for t in texts:
            assert not forbidden_hits(t, True), t
    else:
        assert LEVELS_TAG in v["eyebrow"] and v["levels_note"] == LEVELS_TAG
    # N8 / N9
    assert v["synthetic"] is synth and (v["banner"] is not None) is synth
    for t in texts:
        for w in EDGE_WORDS:
            if status != "VALIDATED" or w != "validated":
                assert not re.search(r"\b" + w + r"\b", t, re.I), (w, t)
    assert v["as_of"]


def test_hero_nko_honesty_grid():
    n = 0
    for status, value, two, pil, prec, fwd, syn in product(STATUSES, NKO_VALUES, (True, False), PILLAR_SETS,
                                                            PRECEDENTS, (True, False), (True, False)):
        rc = SYN_RC if syn else LIVE_RC
        v = _hero(status, value, two, pil, prec, rc, fwd, context=CONTEXTS[n % len(CONTEXTS)])
        _check_hero(v, status, two=two, prec=prec, forward=fwd, synth=syn)
        n += 1
    assert n > 8000


def test_hero_call_only_when_validated_exhaustive():
    for status, value in product(STATUSES, (82.0, -82.0, 48.0, -48.0, 15.0, -15.0, 9.0, 0.0)):
        v = _hero(status, value)
        directional = abs(value) > 10
        assert v["call"] is (status == "VALIDATED" and directional)
        assert v["side"] == (("bull" if value > 0 else "bear") if directional else "flat")
        if status == "VALIDATED" and directional:
            assert "call" in v["track"]["text"]            # the wording is unlocked only here


def test_hero_real_reading_example():
    """A real reading (NKO −48, MODERATE, BEARISH; the systems disagree; UNPROVEN)."""
    prec = {"horizon": 10, "median": -0.8, "positive_pct": 40.0, "n": 10, "dir": -1,
            "unconditional_positive_pct": 56.0, "binomial_p": 0.34, "basis": "Nishkarsh Oscillator state"}
    v = _hero("UNPROVEN", -48.0, prec=prec)
    assert v["headline"] == "Nishkarsh reads moderately bearish" and v["value_text"] == "−48"
    assert v["tier"] == "MODERATE" and v["direction"] == "BEARISH" and not v["call"]
    assert v["explain"] == ("Driven by broad weakness in the Nifty's stocks (bottom-up −2.1), led by stock cut "
                            "crossings (−2.3); Aarambh's top-down view disagrees mildly (+0.8).")
    assert [c["value"] for c in v["chips"]] == ["−12 pts", "12 sessions · since 14 Apr", "85% of past readings"]
    assert v["chips"][1]["label"] == "On this side for"
    assert v["precedent"]["text"] == ("In 10 past moments like today, the Nifty rose 4 times over the next 10 sessions "
                                      "(usually 56%) — median −0.8%; not significant (p 0.34).")
    assert not v["precedent"]["votes"] and v["status_label"] == "Unproven"
    assert v["track"]["text"].startswith("on 138 past 10-day periods this read has not yet shown it anticipates the "
                                         "Nifty (accuracy +0.03).")
    sig = _hero("UNPROVEN", -48.0, prec=PRECEDENTS[3])
    assert sig["precedent"]["votes"] and sig["precedent"]["state"] == "confirm"
    single = _hero("UNPROVEN", 22.0, two=False, pillars=PILLAR_SETS[2])
    assert single["headline"] == "Nishkarsh reads mildly bullish (top-down only today)"
    assert single["explain"].startswith("Top-down only today")
    agree = _hero("UNPROVEN", 70.0, pillars=PILLAR_SETS[1])
    assert agree["explain"].startswith("Driven by Aarambh's top-down view leaning up (top-down +1.2)")
    assert agree["explain"].endswith("the stocks agree mildly (+0.9).")
    flat = _hero("UNPROVEN", 4.0)
    assert flat["headline"] == "Nishkarsh has no lean today" and flat["explain"].startswith("The pillars roughly cancel")


def test_hero_missing_and_poisoned():
    rnd = random.Random(SEED)
    poison = (None, NAN, float("inf"), "n/a", "", [], {}, True, "<script>x</script>", 1e9)
    for _ in range(3000):
        k = make_nko(rnd.choice(STATUSES + ("BOGUS",)), rnd.choice(NKO_VALUES), rnd.choice((True, False, None)),
                     rnd.choice(PILLAR_SETS), context=rnd.choice(CONTEXTS))
        for _ in range(rnd.randint(1, 5)):
            key = rnd.choice(list(k))
            k[key] = rnd.choice(poison)
        con = rnd.choice([{"nko": k}, {"nko": k, "horizon": "x"}, {}, None, "x", {"nko": "bad"}])
        v = build_hero_verdict(con, rnd.choice(list(PRECEDENTS) + [{"n": "ten"}, "x", 7]),
                               rnd.choice([None, {}, "x", LIVE_RC, SYN_RC]))
        for key in HERO_KEYS:
            assert key in v
        st_ = v["status"]
        assert st_ in STATUSES and v["status_label"] == _STATUS_LABELS[st_]
        for t in hero_texts(v):
            if "script" in t:
                continue
            assert not _BAD.search(t), t
            if st_ != "VALIDATED":
                assert not BUY_SELL_RE.search(t), t
        assert not v["call"] or st_ == "VALIDATED"


def test_hero_render_escapes_and_never_raises(monkeypatch):
    out = []
    monkeypatch.setattr(comp.st, "markdown", lambda body, **kw: out.append(body))
    con = make_conclusion("NO_PROVEN_EDGE", -47.8, "CONTRADICT", True)
    con["nko"] = make_nko("UNPROVEN", -48.0, True, PILLAR_SETS[0])
    con["nko"]["text"] = "<script>alert(1)</script>"
    con["nko"]["evidence"] = dict(NKO_EVIDENCE, yearly={"<script>": 0.1})
    con["nko"]["drivers"] = [{"id": "<img src=x onerror=alert(2)>", "label": "<b>x</b>", "z": -2.0}]
    con["nko"]["components"][0]["label"] = "<b>x</b>"
    con["nko"]["components"][0]["id"] = "<img src=x onerror=alert(3)>"
    render_hero(build_hero_verdict(con, PRECEDENTS[3], SYN_RC))
    for bad in (None, {}, {"chips": [None, {"label": "<b>x</b>", "class": "<x>"}], "side": "<x>", "track": "x"}):
        render_hero(bad)
    html = "\n".join(out)
    assert "<script" not in html and "<img" not in html and "<b>x" not in html
    assert "Track record: <b>" not in html and "Track record: unproven" in html


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
