"""
Nishkarsh v2.0.0 — Nifty 50 constituent list: niftyindices.com → Wikipedia → stale → fallback.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

No Streamlit import (audit A5-21): caching is the pickle-free ``data.cache.constituents_cache``
(TTL 1h), so the module runs headless. Every symbol is upper-cased with an ``.NS`` suffix
(yfinance upper-cases tickers; "EicherMot.NS" silently vanished — audit A5-11).

Provenance (surfaced, never hidden — audit A5-11):
  live      niftyindices.com or Wikipedia answered now        (note names which)
  cache     a live list fetched within the TTL
  stale     both live sources failed; the last live list (any age) is served
  fallback  nothing live ever succeeded; the hard-coded ``core.config.NIFTY50_FALLBACK``
            (membership is dated — verify; see ``FALLBACK_NOTE``)
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from core import config as _cfg
from data import cache as _cache
from data.circuit_breaker import constituents_circuit
from data.sources import FetchResult

log = logging.getLogger(__name__)

NIFTY50_URL = _cfg.NIFTY50_URL
MIN_CONSTITUENTS = 40          # a "Nifty 50" list shorter than this is treated as a bad parse

_INDIA_INDEX_WIKI = {
    "NIFTY 50": "https://en.wikipedia.org/wiki/NIFTY_50",
    "NIFTY NEXT 50": "https://en.wikipedia.org/wiki/NIFTY_Next_50",
    "NIFTY 500": "https://en.wikipedia.org/wiki/NIFTY_500",
}
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

FALLBACK_AS_OF = getattr(_cfg, "NIFTY50_FALLBACK_AS_OF", None)
FALLBACK_NOTE = (
    "hard-coded fallback list"
    + (f" (membership as of {FALLBACK_AS_OF})" if FALLBACK_AS_OF else " (membership date unknown)")
    + " — niftyindices.com and Wikipedia were unreachable; verify against "
      "niftyindices.com before relying on constituent-level results"
)


def to_ns_symbols(raw) -> list[str]:
    """Normalise exchange symbols → upper-case ``SYMBOL.NS`` (dedupe, order kept)."""
    out, seen = [], set()
    for s in raw or []:
        u = str(s).strip().upper()
        if not u or u in ("NAN", "NONE"):
            continue
        if not u.endswith((".NS", ".BO")) and not u.startswith("^"):
            u = u + ".NS"
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fallback_constituents() -> list[str]:
    return to_ns_symbols(_cfg.NIFTY50_FALLBACK)


def _from_niftyindices() -> list[str] | None:
    resp = requests.get(NIFTY50_URL, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig", errors="replace")))
    if "Symbol" not in df.columns:
        return None
    syms = to_ns_symbols(df["Symbol"].dropna().astype(str).tolist())
    return syms if len(syms) >= MIN_CONSTITUENTS else None


def _parse_wikipedia_constituents(url: str, min_count: int = MIN_CONSTITUENTS) -> list[str] | None:
    """Parse a Wikipedia constituent table → bare symbols (no suffix), or None."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    for tbl in pd.read_html(io.StringIO(resp.text)):
        if "Symbol" in tbl.columns:
            syms = [s for s in tbl["Symbol"].dropna().astype(str).str.strip().tolist()
                    if s and len(s) <= 20 and s.lower() != "nan"]
            if len(syms) >= min_count:
                return syms
    return None


def _live_chain() -> tuple[list[str], str]:
    errors = []
    for name, fn in (("niftyindices", _from_niftyindices),
                     ("wikipedia", lambda: _parse_wikipedia_constituents(_INDIA_INDEX_WIKI["NIFTY 50"]))):
        try:
            syms = fn()
            if syms:
                return to_ns_symbols(syms), name
            errors.append(f"{name}: no usable table")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors))


def fetch_constituents(*, force: bool = False) -> FetchResult:
    """Nifty 50 constituents as a :class:`FetchResult` (value = list of ``SYM.NS``)."""
    cache = _cache.constituents_cache
    key = ("NIFTY 50",)
    if not (force or _cache.force_window_active()):
        e = cache.get_entry(*key)
        if e is not None and e.age_seconds < cache.ttl:
            return FetchResult(list(e.value), "cache", e.meta.get("as_of"), [],
                               f"{e.meta.get('source', 'live')} (cached {e.age_seconds / 60:.0f} min ago)")
    try:
        syms, source = constituents_circuit.call(_live_chain)
        today = str(pd.Timestamp.today().date())
        cache.put(*key, value=syms, meta={"source": source, "as_of": today})
        return FetchResult(syms, "live", today, [], source)
    except Exception as ex:  # noqa: BLE001 — incl. CircuitBreakerError (circuit open)
        reason = f"{type(ex).__name__}: {ex}"
        log.warning("Constituents live fetch failed: %s", reason)
    e = cache.get_entry(*key)
    if e is not None and e.value:
        return FetchResult(list(e.value), "stale", e.meta.get("as_of"), [],
                           f"{e.meta.get('source', 'live')} list from {e.meta.get('as_of')} "
                           f"(live sources failed: {reason})")
    syms = fallback_constituents()
    log.warning("Constituents: using %s", FALLBACK_NOTE)
    return FetchResult(syms, "fallback", FALLBACK_AS_OF, [], f"{FALLBACK_NOTE} [{reason}]")


def fetch_nifty50_constituents(*, force: bool = False) -> tuple[list[str], str]:
    """Legacy wrapper → ``(symbols, source)``; ``source`` names the provenance explicitly
    (e.g. "niftyindices", "wikipedia (cached …)", "stale: …", "FALLBACK: …")."""
    r = fetch_constituents(force=force)
    label = {"live": r.note, "cache": r.note, "stale": f"STALE: {r.note}",
             "fallback": f"FALLBACK: {r.note}"}.get(r.provenance, r.note)
    return list(r.value), label
