"""The page shell's data — built ONCE per run, outside the UI (the UI formats only).

    tape       the Nifty 50 constituents' last close and session change (the bottom-up
               world, as Tattva's tape is its macro world); the sheet's macro series when
               there is no constituent data; else empty.
    freshness  ONE reading of how current the run's newest session is, in NSE trading days
               (Mon–Fri) between it and today — shared by the command-bar chip and the
               notices so the two can never disagree.
    notices    every disclosure the page shell shows under the command bar, as
               ``{kind: info|warning, title, body}`` rows: synthetic data, levels mode,
               single-system, freshness, and the run's own data notes (date anomalies, the
               late-predictor note, missing / stale / fallback sources). Links are hidden —
               a sheet URL is a secret and must never reach the page.

Pure functions of the run's frames; nothing here estimates anything.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

import core.config as C

__all__ = ["build_tape", "build_quote", "build_freshness", "build_notices", "redact", "SOURCE_NAMES"]

SOURCE_NAMES = {"sheet": "Your sheet", "constituents": "Constituents", "ohlcv": "Stock prices",
                "index": "Index", "macro": "Market macro"}
_URL = re.compile(r"(https?://|docs\.google\.com/)\S+", re.I)


def redact(text) -> str:
    """Hide any link in a note (a sheet URL is a secret and must never reach the page)."""
    return _URL.sub("[link hidden]", str(text))


def _sym(s: str) -> str:
    s = str(s)
    for suf in (".NS", ".BO"):
        if s.upper().endswith(suf):
            return s[: -len(suf)]
    return s.lstrip("^")


def _last_two(series: pd.Series) -> tuple[float, float] | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) < 2:
        return None
    return float(s.iloc[-1]), float(s.iloc[-2])


def build_tape(ohlcv: dict | None, macro_panel: pd.DataFrame | None = None,
               macro_columns=None, limit: int = 60) -> list[dict]:
    """``[{sym, last, chg_pct}]`` — change in PERCENT POINTS (-0.42 == -0.42%)."""
    out: list[dict] = []
    for sym, df in (ohlcv or {}).items():
        if df is None or not len(df) or "Close" not in getattr(df, "columns", ()):
            continue
        lt = _last_two(df["Close"])
        if lt is None:
            continue
        last, prev = lt
        out.append({"sym": _sym(sym), "last": last, "chg_pct": (last / prev - 1.0) * 100.0})
    if out:
        return out[:limit]
    if macro_panel is not None and len(macro_panel):
        cols = list(macro_columns or macro_panel.columns)
        for c in cols:
            if c not in macro_panel.columns:
                continue
            lt = _last_two(macro_panel[c])
            if lt is None:
                continue
            last, prev = lt
            out.append({"sym": str(c).upper()[:12], "last": last, "chg_pct": (last / prev - 1.0) * 100.0})
    return out[:limit]


def build_quote(levels: pd.Series | None, name: str = "NIFTY 50") -> dict:
    """The command bar's instrument quote: the index's last level and its session change in
    PERCENT POINTS, from the run's resolved return-label levels. Empty values when absent."""
    out = {"name": name, "last": None, "chg_pct": None, "as_of": None}
    if levels is None or not len(levels):
        return out
    s = pd.to_numeric(pd.Series(levels), errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s > 0]
    if not len(s):
        return out
    out["last"] = float(s.iloc[-1])
    out["as_of"] = str(pd.Timestamp(s.index[-1]).date())
    if len(s) >= 2:
        out["chg_pct"] = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
    return out


def _today(today) -> date:
    if today is not None:
        return pd.Timestamp(today).date()
    # The EARLIER of UTC and machine-local: a UTC-hosted deploy rolls past midnight ahead of
    # IST, and taking the earlier date means a time-zone skew can never overstate staleness.
    return min(datetime.now(timezone.utc).date(), datetime.now().date())


def build_freshness(as_of, today=None, stale_days: int | None = None) -> dict:
    """Trading days (NSE, Mon–Fri) after ``as_of`` up to and including today: 0 → CURRENT
    ("current", not "live" — this is end-of-day data), 1..stale−1 → "nD BEHIND",
    ≥ stale → STALE. Friday data read on a Sunday is current."""
    stale = int(C.STALENESS_DAYS if stale_days is None else stale_days)
    t = _today(today)
    try:
        a = pd.Timestamp(as_of).date() if as_of is not None and not pd.isna(pd.Timestamp(as_of)) else None
    except (TypeError, ValueError):
        a = None
    if a is None:
        return {"as_of": None, "today": str(t), "behind": None, "label": "", "tone": "neutral"}
    behind = int(np.busday_count(np.datetime64(a) + 1, np.datetime64(t) + 1)) if t > a else 0
    if behind >= stale:
        label, tone = "STALE", "danger"
    elif behind >= 1:
        label, tone = f"{behind}D BEHIND", "warning"
    else:
        label, tone = "CURRENT", "success"
    return {"as_of": str(a), "today": str(t), "behind": behind, "label": label, "tone": tone}


def _note_title(w: str) -> tuple[str, str]:
    """(kind, title) for one run warning."""
    m = re.match(r"^(\w+): (provenance (\w+)|(\d+) missing)", w)
    if m and m.group(1) in SOURCE_NAMES:
        name = SOURCE_NAMES[m.group(1)]
        if m.group(3):
            word = m.group(3).lower()
            return ("warning" if word in ("stale", "empty", "fallback") else "info"), f"{name} {word}"
        return "warning", f"{name} incomplete"
    lw = w.lower()
    if lw.startswith("date column"):
        return "info", "Sheet dates"
    if lw.startswith("predictor ") and "truncates" in lw:
        return "info", "History bounded"
    if lw.startswith("predictors"):
        return "info", "Predictors"
    if lw.startswith("stale sheet"):
        return "warning", "Sheet is stale"
    if lw.startswith("nirnay"):
        return "warning", "Nirnay coverage"
    if lw.startswith("label timestamp"):
        return "warning", "Return label misaligned"
    if "unavailable" in lw:
        return "info", "Return label"
    if "failed" in lw:
        return "warning", "Step failed"
    return "info", "Data note"


def build_notices(*, warnings, synthetic: bool, forward_mode: bool, nko: dict | None,
                  freshness: dict) -> list[dict]:
    """The notice rail's rows, in severity-then-reading order."""
    out: list[dict] = []
    if synthetic:
        out.append({"kind": "warning", "title": "Synthetic demo data",
                    "body": "Not the market — a seeded, simulated world; no reading here is about the real "
                            "Nifty."})
    if not forward_mode:
        out.append({"kind": "info", "title": "Valuation context",
                    "body": "Aarambh runs in levels mode — valuation context, falsified as a forecaster; "
                            "the read is not a forecast."})
    k = nko if isinstance(nko, dict) else {}
    if k.get("value") is not None and k.get("two_system") is not True:
        d = k.get("as_of")
        ds = pd.Timestamp(d).strftime("%d %b %Y") if d else "this date"
        out.append({"kind": "warning", "title": "Top-down only",
                    "body": f"Top-down only — Nirnay has no stock data for {ds}. The read is Aarambh's alone, "
                            "and such dates are left out of its measurement."})
    f = freshness or {}
    if f.get("behind") and f.get("as_of"):
        ds = pd.Timestamp(f["as_of"]).strftime("%d %b %Y")
        b = int(f["behind"])
        if f.get("label") == "STALE":
            out.append({"kind": "warning", "title": "Latest data unavailable",
                        "body": f"Newest data is {ds} — {b} trading days behind. Every reading reflects {ds}, "
                                "not today. Use Refresh data in the rail once the source updates."})
        else:
            out.append({"kind": "info", "title": "Data freshness",
                        "body": f"Readings are as of {ds} ({b} trading day{'s' if b > 1 else ''} behind — "
                                "today's bar may not be published yet)."})
    for w in warnings or []:
        w = str(w).strip()
        # The synthetic banner has its own row above; "Stale sheet" is a calendar-day reading
        # of the same fact the ONE freshness reading states in trading days.
        if not w or w.upper().startswith("SYNTHETIC") or w.lower().startswith("stale sheet"):
            continue
        kind, title = _note_title(w)
        out.append({"kind": kind, "title": title, "body": redact(w)})
    return out
