"""
Nishkarsh v2.0.0 — data fetchers: Google Sheets (requests) + yfinance (OHLCV, macro, index).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Every external call goes through:
  1. a pickle-free TTL cache with tagged last-good snapshots      (data/cache.py)
  2. a per-service circuit breaker (consecutive failures)           (data/circuit_breaker.py)
  3. retry-with-backoff for transient failures only

Core API — each returns a :class:`data.sources.FetchResult` (value + provenance ∈
{live, cache, stale, fallback, empty} + as_of + missing + note); a stale fallback is DATA
WITH PROVENANCE, never an error string (audit A5-10):

    fetch_ohlcv(symbols, start, end, *, force=False)          -> FetchResult[dict[str, DataFrame]]
    fetch_index_levels(symbol, start, end, *, force=False)    -> FetchResult[pd.Series]  (Close)
    fetch_macro(start, end, *, force=False, tickers=None)     -> FetchResult[DataFrame]
    fetch_sheet(url, *, force=False)                          -> FetchResult[DataFrame]

Legacy names (v1 app) are thin wrappers with the old return shapes plus ``force=``:
``fetch_constituent_ohlcv``, ``fetch_macro_live``, ``load_google_sheet``.

``force=True`` bypasses the fresh cache but keeps snapshots as the failure fallback — it is
an explicit argument because worker threads carry no Streamlit session (audit A5-05).

Window rule: the stale fallback is the newest snapshot for the same SYMBOL SET, whatever its
window, trimmed to the requested window (audit A5-06). OHLCV merges per symbol with that
snapshot and never shrinks it; symbols with no data anywhere are reported in ``missing``
(audit A5-07). Tickers are upper-cased (yfinance upper-cases them; audit A5-11).
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from urllib.parse import urlparse

import pandas as pd
import requests

from core.config import GLOBAL_MACRO_MAP, MACRO_SYMBOLS_YF
from data import cache as _cache
from data.circuit_breaker import (
    FetchRejected,
    RetryWithBackoff,
    sheets_circuit,
    yfinance_circuit,
)
from data.sources import FetchResult

log = logging.getLogger(__name__)

# A snapshot that was incomplete when written (stale-filled / missing symbols) is "fresh"
# only this long, so a rate-limited batch is retried soon instead of pinned for the full TTL.
PARTIAL_TTL_SECONDS = 600
# Macro columns missing from a live batch are back-filled only from a snapshot this recent
# (an older column would forward-fill flat across the gap and look current).
_BACKFILL_MAX_AGE_DAYS = 7.0
SHEET_TIMEOUT = (5.0, 30.0)   # (connect, read) seconds
_HEADERS = {"User-Agent": "Mozilla/5.0 (Nishkarsh data fetcher)"}


# ─── helpers ────────────────────────────────────────────────────────────────

def normalize_symbols(symbols) -> list[str]:
    """Strip, upper-case, de-duplicate (order kept)."""
    out, seen = [], set()
    for s in symbols or []:
        u = str(s).strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def symset_tag(symbols) -> str:
    """Stable tag of a symbol SET (order/case-insensitive) — the snapshot lookup key."""
    raw = "|".join(sorted(normalize_symbols(symbols)))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _window(start, end) -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    return s, e, str(s.date()), str((e + pd.Timedelta(days=1)).date())   # yfinance end is exclusive


def _trim(obj, s: pd.Timestamp, e: pd.Timestamp):
    idx = obj.index
    m = (idx >= s) & (idx < e + pd.Timedelta(days=1))
    return obj.loc[m]


def _as_of(frames) -> str | None:
    last = [f.index.max() for f in frames if f is not None and len(f)]
    return str(pd.Timestamp(max(last)).date()) if last else None


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _effective_force(force: bool) -> bool:
    return bool(force) or _cache.force_window_active()


def _fresh(cache: _cache.Cache, *key) -> _cache.CacheEntry | None:
    e = cache.get_entry(*key)
    if e is None:
        cache.misses += 1
        return None
    ttl = cache.ttl if e.meta.get("complete", True) else min(cache.ttl, PARTIAL_TTL_SECONDS)
    if e.age_seconds < ttl:
        cache.hits += 1
        return e
    cache.misses += 1
    return None


# ─── Constituent / index OHLCV (yfinance) ───────────────────────────────────

@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _yfinance_batch_download(symbols_tuple: tuple[str, ...], start_yf: str, end_yf: str) -> pd.DataFrame:
    """Single raw yfinance batch call — wrapped with retry."""
    import yfinance as yf
    raw = yf.download(list(symbols_tuple), start=start_yf, end=end_yf, progress=False,
                      auto_adjust=True, group_by="ticker")
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Empty yfinance batch response")
    return raw


def _parse_ohlcv_batch(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split a yfinance batch into {SYMBOL: OHLCV}; drops rows with NaN Close (pre-listing
    union rows) and symbols with no data. Case-insensitive ticker lookup."""
    out: dict[str, pd.DataFrame] = {}
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return out
    raw = _strip_tz(raw)
    if isinstance(raw.columns, pd.MultiIndex):
        lvl = 0
        for i in range(raw.columns.nlevels):
            if {str(v).upper() for v in raw.columns.get_level_values(i)} & set(symbols):
                lvl = i
                break
        by_upper = {str(v).upper(): v for v in raw.columns.get_level_values(lvl).unique()}
        for sym in symbols:
            if sym not in by_upper:
                continue
            sub = raw.xs(by_upper[sym], level=lvl, axis=1)
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = [c[0] for c in sub.columns]
            out_df = _clean_ohlcv(sub)
            if out_df is not None:
                out[sym] = out_df
    elif len(symbols) == 1:
        out_df = _clean_ohlcv(raw)
        if out_df is not None:
            out[symbols[0]] = out_df
    return out


def _clean_ohlcv(sub: pd.DataFrame) -> pd.DataFrame | None:
    sub = sub.copy()
    sub.columns = [str(c) for c in sub.columns]
    if "Close" not in sub.columns:
        return None
    sub = sub[sub["Close"].notna()]
    sub = sub[~sub.index.duplicated(keep="last")].sort_index()
    sub.columns.name = None
    if sub.index.name is None:
        sub.index.name = "Date"
    return sub if len(sub) else None


def fetch_ohlcv(symbols, start, end, *, force: bool = False) -> FetchResult:
    """OHLCV for ``symbols`` over [start, end] (inclusive). See module docstring."""
    syms = normalize_symbols(symbols)
    if not syms:
        return FetchResult({}, "empty", None, [], "no symbols requested")
    s, e, start_yf, end_yf = _window(start, end)
    tag = symset_tag(syms)
    cache = _cache.ohlcv_cache

    if not _effective_force(force):
        hit = _fresh(cache, tag, start_yf, end_yf)
        if hit is not None:
            val = {k: v for k, v in hit.value.items() if k in syms}
            filled = list(hit.meta.get("stale_filled", []))
            missing = [x for x in syms if x not in val]
            note = "cache hit" + (f"; stale-filled: {', '.join(filled)}" if filled else "")
            return FetchResult(val, "stale" if filled else "cache", _as_of(val.values()), missing, note)

    live: dict[str, pd.DataFrame] = {}
    err: Exception | None = None
    try:
        raw = yfinance_circuit.call(_yfinance_batch_download, tuple(syms), start_yf, end_yf)
        live = _parse_ohlcv_batch(raw, syms)
    except Exception as ex:  # noqa: BLE001 — degrade to snapshot
        err = ex
        log.warning("yfinance OHLCV batch failed (%s): %s", type(ex).__name__, ex)

    snap = cache.latest(tag)
    base = {k: _trim(v, s, e) for k, v in (snap.value.items() if snap else [])}
    base = {k: v for k, v in base.items() if len(v)}

    merged = dict(live)
    filled = [x for x in syms if x not in live and x in base]
    for x in filled:
        merged[x] = base[x]
    missing = [x for x in syms if x not in merged]

    if live:
        complete = not filled and not missing
        # Store the merged set: the snapshot for this symbol set never shrinks.
        cache.put(tag, start_yf, end_yf, value=merged, tag=tag,
                  meta={"complete": complete, "stale_filled": filled, "missing": missing,
                        "start": start_yf, "end": end_yf, "n_live": len(live)})
        parts = [f"live {len(live)}/{len(syms)}"]
        if filled:
            parts.append(f"stale-filled {len(filled)} from snapshot of "
                         f"{pd.Timestamp(snap.ts, unit='s').strftime('%Y-%m-%d %H:%M')}: {', '.join(filled)}")
        if missing:
            parts.append(f"missing {len(missing)}: {', '.join(missing)}")
            log.warning("OHLCV missing symbols: %s", missing)
        return FetchResult(merged, "stale" if filled else "live", _as_of(merged.values()), missing, "; ".join(parts))

    if merged:
        age_h = snap.age_seconds / 3600.0
        note = (f"live fetch failed ({type(err).__name__ if err else 'no data'}: {err}); serving snapshot "
                f"{age_h:.1f}h old, trimmed to window")
        log.warning("OHLCV: %s", note)
        return FetchResult(merged, "stale", _as_of(merged.values()), missing, note)
    note = f"live fetch failed ({type(err).__name__ if err else 'no data'}: {err}); no snapshot"
    log.error("OHLCV: %s", note)
    return FetchResult({}, "empty", None, missing, note)


def fetch_constituent_ohlcv(symbols, start_date, end_date, *, force: bool = False) -> dict[str, pd.DataFrame]:
    """Legacy wrapper: ``fetch_ohlcv(...).value`` (keys are upper-cased tickers)."""
    return fetch_ohlcv(symbols, start_date, end_date, force=force).value


def fetch_index_levels(symbol: str, start, end, *, force: bool = False) -> FetchResult:
    """Close-level Series of an index (e.g. ``^NSEI``) — same cache/stale rules as OHLCV."""
    r = fetch_ohlcv([symbol], start, end, force=force)
    sym = normalize_symbols([symbol])[0] if symbol else ""
    df = r.value.get(sym) if isinstance(r.value, dict) else None
    if df is None or "Close" not in df.columns or df.empty:
        return FetchResult(pd.Series(dtype="float64", name=sym), "empty", None, [sym], r.note)
    close = df["Close"].astype("float64").rename(sym)
    return FetchResult(close, r.provenance, r.as_of, [], r.note)


# ─── Macro panel (Global Macro bond ETFs + FX + commodities) ────────────────

def macro_tickers() -> tuple[str, ...]:
    return tuple(sorted({str(t).upper() for t in list(GLOBAL_MACRO_MAP.values()) + list(MACRO_SYMBOLS_YF.values())}))


@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _yfinance_batch_download_macro(tickers_tuple: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Macro yfinance batch fetch (separate retry budget; yfinance-internal threads)."""
    import yfinance as yf
    raw = yf.download(list(tickers_tuple), start=start, end=end, progress=False,
                      auto_adjust=True, threads=True)
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Empty yfinance macro response")
    return raw


def _parse_macro_close(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = _strip_tz(raw)
    if isinstance(raw.columns, pd.MultiIndex):
        panel = None
        for field in ("Close", "Adj Close"):
            for i in range(raw.columns.nlevels):
                if field in set(raw.columns.get_level_values(i)):
                    panel = raw.xs(field, level=i, axis=1)
                    break
            if panel is not None:
                break
        if panel is None:
            return pd.DataFrame()
    elif "Close" in raw.columns and len(tickers) == 1:
        panel = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        panel = raw
    panel = panel.copy()
    panel.columns = [str(c).upper() for c in panel.columns]
    panel.columns.name = None
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    if panel.index.name is None:
        panel.index.name = "Date"
    return panel.astype("float64")


def fetch_macro(start, end, *, force: bool = False, tickers=None) -> FetchResult:
    """Close panel (DatetimeIndex × ticker), forward-filled on the union calendar.

    Columns a live batch dropped (absent / all-NaN) are back-filled from the newest snapshot
    of the same ticker set if it is ≤ 7 days old; older → the column is reported missing.
    Total failure → newest snapshot (any age) trimmed to the window, provenance "stale".
    NOTE: not lagged here — the pipeline applies ``data.macro_lag.lag_non_ist``.
    """
    tick = tuple(normalize_symbols(tickers)) if tickers else macro_tickers()
    if not tick:
        return FetchResult(pd.DataFrame(), "empty", None, [], "no macro tickers configured")
    s, e, start_s, end_s = _window(start, end)
    tag = symset_tag(tick)
    cache = _cache.macro_cache

    if not _effective_force(force):
        hit = _fresh(cache, tag, start_s, end_s)
        if hit is not None:
            filled = list(hit.meta.get("stale_filled", []))
            return FetchResult(hit.value, "stale" if filled else "cache", _as_of([hit.value]),
                               list(hit.meta.get("missing", [])),
                               "cache hit" + (f"; stale-filled: {', '.join(filled)}" if filled else ""))

    combined, err = pd.DataFrame(), None
    try:
        raw = yfinance_circuit.call(_yfinance_batch_download_macro, tick, start_s, end_s)
        combined = _parse_macro_close(raw, tick)
    except Exception as ex:  # noqa: BLE001
        err = ex
        log.warning("yfinance macro batch failed (%s): %s", type(ex).__name__, ex)

    if not combined.empty:
        combined = combined.ffill()
        absent = [t for t in tick if t not in combined.columns or combined[t].isna().all()]
        filled: list[str] = []
        if absent:
            snap = cache.latest(tag, max_age_seconds=_BACKFILL_MAX_AGE_DAYS * 86400.0)
            if snap is not None and isinstance(snap.value, pd.DataFrame):
                aligned = snap.value.reindex(combined.index).ffill()
                for t in absent:
                    if t in aligned.columns and not aligned[t].isna().all():
                        combined[t] = aligned[t]
                        filled.append(t)
        combined = combined.drop(columns=[t for t in combined.columns if combined[t].isna().all()])
        missing = [t for t in tick if t not in combined.columns]
        cache.put(tag, start_s, end_s, value=combined, tag=tag,
                  meta={"complete": not filled and not missing, "stale_filled": filled,
                        "missing": missing, "start": start_s, "end": end_s})
        note = f"live {len(tick) - len(filled) - len(missing)}/{len(tick)} tickers"
        if filled:
            note += f"; stale-filled (≤{_BACKFILL_MAX_AGE_DAYS:.0f}d snapshot): {', '.join(filled)}"
            log.warning("Macro back-fill from recent snapshot: %s", filled)
        if missing:
            note += f"; missing: {', '.join(missing)}"
            log.warning("Macro tickers unavailable (dropped, not filled stale): %s", missing)
        return FetchResult(combined, "stale" if filled else "live", _as_of([combined]), missing, note)

    snap = cache.latest(tag)
    if snap is not None and isinstance(snap.value, pd.DataFrame):
        panel = _trim(snap.value, s, e)
        if not panel.empty:
            note = (f"live fetch failed ({type(err).__name__ if err else 'empty'}: {err}); serving snapshot "
                    f"{snap.age_seconds / 3600.0:.1f}h old, trimmed to window")
            log.warning("Macro: %s", note)
            return FetchResult(panel, "stale", _as_of([panel]),
                               [t for t in tick if t not in panel.columns], note)
    note = f"live fetch failed ({type(err).__name__ if err else 'empty'}: {err}); no snapshot"
    log.error("Macro: %s", note)
    return FetchResult(pd.DataFrame(), "empty", None, list(tick), note)


def fetch_macro_live(start_date, end_date, *, force: bool = False) -> pd.DataFrame:
    """Legacy wrapper: ``fetch_macro(...).value``."""
    return fetch_macro(start_date, end_date, force=force).value


# ─── Aarambh sheet (Google Sheets, via requests) ────────────────────────────

_ID = r"[A-Za-z0-9_-]+"


def sheet_csv_url(url: str) -> str:
    """CSV-export URL for any common Google Sheets link form.

    Handles ``/d/<id>/edit#gid=N``, ``/d/<id>/edit?usp=sharing``, ``/u/0/d/<id>/htmlview``,
    existing ``/export?format=csv`` links, published ``/d/e/2PACX…/pubhtml?gid=N`` and
    ``/d/e/2PACX…/pub?output=csv`` links, and a bare sheet id. Raises ``FetchRejected``.
    """
    u = (url or "").strip()
    if not u:
        raise FetchRejected("Empty Google Sheets URL")
    if re.fullmatch(r"[A-Za-z0-9_-]{25,}", u):
        return f"https://docs.google.com/spreadsheets/d/{u}/export?format=csv&gid=0"
    p = urlparse(u if "://" in u else "https://" + u)
    if not (p.netloc == "docs.google.com" or p.netloc.endswith(".docs.google.com")):
        raise FetchRejected("Not a Google Sheets URL (expected docs.google.com/spreadsheets/…)")
    gid_m = re.search(r"[?&#]gid=(\d+)", u)
    pub = re.search(rf"/spreadsheets/d/e/({_ID})", p.path)
    if pub:
        out = f"https://docs.google.com/spreadsheets/d/e/{pub.group(1)}/pub?output=csv"
        return out + (f"&gid={gid_m.group(1)}&single=true" if gid_m else "")
    m = re.search(rf"/spreadsheets/(?:u/\d+/)?d/({_ID})", p.path)
    if not m:
        raise FetchRejected("Invalid Google Sheets URL (no sheet id found)")
    gid = gid_m.group(1) if gid_m else "0"
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"


@RetryWithBackoff(max_retries=2, initial_delay=1.0, backoff_factor=2.0,
                  exceptions=(requests.RequestException, OSError))
def _sheets_download(url: str) -> pd.DataFrame:
    """Fetch the sheet as CSV with a timeout and a content check. Transient errors retry;
    client errors / HTML pages raise ``FetchRejected`` (no retry, no circuit trip)."""
    csv_url = sheet_csv_url(url)
    resp = requests.get(csv_url, headers=_HEADERS, timeout=SHEET_TIMEOUT, allow_redirects=True)
    if 400 <= resp.status_code < 500 and resp.status_code != 429:
        raise FetchRejected(f"Google Sheets answered HTTP {resp.status_code} — is the sheet shared "
                            "('Anyone with the link') or published to the web?")
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").lower()
    text = resp.content.decode("utf-8-sig", errors="replace")
    if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        raise FetchRejected("Google Sheets returned an HTML page, not CSV — the sheet is private or "
                            "the link is not a sheet. Share it ('Anyone with the link') or publish it to the web.")
    if ctype and not any(t in ctype for t in ("text/csv", "text/plain", "application/csv",
                                               "application/octet-stream")):
        raise FetchRejected(f"Unexpected content-type from Google Sheets: {ctype}")
    if not text.strip():
        raise FetchRejected("Google Sheet is empty")
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        raise FetchRejected("Google Sheet has a header but no rows")
    return df


def fetch_sheet(url: str, *, force: bool = False) -> FetchResult:
    """The Aarambh sheet as a raw DataFrame (dates still strings — parse with
    ``data.dates.parse_dates``). The URL is a secret: it is hashed into the cache key and
    never written to disk or logs."""
    if not url:
        return FetchResult(None, "empty", None, [], "Empty URL")
    cache = _cache.sheets_cache
    if not _effective_force(force):
        hit = _fresh(cache, url)
        if hit is not None:
            return FetchResult(hit.value, "cache", None, [], "cache hit")
    try:
        df = sheets_circuit.call(_sheets_download, url)
        cache.put(url, value=df, meta={"rows": int(len(df))})
        return FetchResult(df, "live", None, [], f"{len(df)} rows")
    except Exception as ex:  # noqa: BLE001
        reason = f"{type(ex).__name__}: {ex}"
        stale = cache.get_entry(url)
        if stale is not None and stale.value is not None:
            note = (f"live fetch failed ({reason}); serving last-good snapshot "
                    f"{stale.age_seconds / 3600.0:.1f}h old")
            log.warning("Sheets: live fetch failed (%s); serving last-good snapshot", type(ex).__name__)
            return FetchResult(stale.value, "stale", None, [], note)
        log.error("Sheets: fetch failed (%s)", type(ex).__name__)
        return FetchResult(None, "empty", None, [], reason)


def load_google_sheet(url: str, *, force: bool = False) -> tuple[pd.DataFrame | None, str | None]:
    """Legacy wrapper → ``(df, error)``. A stale snapshot is returned as ``(df, None)`` (data,
    not an error — audit A5-10); its provenance is in ``fetch_sheet(url).note`` and the log."""
    r = fetch_sheet(url, force=force)
    if r.value is None:
        return None, r.note or "No data"
    return r.value, None
