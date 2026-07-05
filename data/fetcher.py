"""
Nishkarsh v1.4.0 — Unified data fetcher: Google Sheets + yfinance.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Each external call is wrapped with:
  1. **Two-tier cache** (memory + disk, TTL + versioned keys) — `data/cache.py`
  2. **Circuit breaker** (CLOSED → OPEN → HALF_OPEN per service) — `data/circuit_breaker.py`
  3. **Retry-with-backoff** (1s → 2s → 4s) for transient failures
  4. **Stale-fallback** — if a fetch fails AND the circuit is open, the last-good
     snapshot is returned so the UI keeps working through API outages.

Macro data is sourced from Sanket's Global Macro bond-ETF universe via yfinance
(replaces the broken Stooq direct-yield endpoints, which started returning HTML
error pages instead of CSV in late 2025).
"""

from __future__ import annotations

import logging
import pickle
import re

import pandas as pd
import yfinance as yf

from core.config import (
    GLOBAL_MACRO_MAP,
    MACRO_SYMBOLS_YF,
)
from data.cache import sheets_cache, ohlcv_cache, macro_cache
from data.circuit_breaker import (
    yfinance_circuit,
    sheets_circuit,
    CircuitBreakerError,
    RetryWithBackoff,
)

log = logging.getLogger(__name__)


# ─── Aarambh data (Google Sheets) ────────────────────────────────────────────

@RetryWithBackoff(max_retries=2, initial_delay=1.0, backoff_factor=2.0)
def _sheets_download(url: str) -> pd.DataFrame:
    """Raw Google Sheets CSV fetch — circuit + retry protected internally."""
    sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not sheet_id_match:
        raise ValueError("Invalid Google Sheets URL")
    sheet_id = sheet_id_match.group(1)
    gid_match = re.search(r"gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    )
    df = pd.read_csv(csv_url)
    if df is None or df.empty:
        raise ValueError("Empty response from Google Sheets")
    return df


def load_google_sheet(url: str) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch valuation data from a Google Sheet with cache + circuit + retry."""
    if not url:
        return None, "Empty URL"

    cached = sheets_cache.get(url)
    if cached is not None:
        return cached, None

    try:
        df = sheets_circuit.call(_sheets_download, url)
        sheets_cache.put(url, value=df)
        return df, None
    except CircuitBreakerError as e:
        # Circuit is open — try last-good snapshot before failing.
        stale = sheets_cache.get_stale(url)
        if stale is not None:
            log.warning("Sheets circuit open, serving stale snapshot")
            return stale, f"Serving last-good snapshot (circuit open): {e}"
        return None, str(e)
    except Exception as e:
        log.error("Google Sheets fetch failed: %s", e)
        stale = sheets_cache.get_stale(url)
        if stale is not None:
            return stale, f"Serving last-good snapshot: {e}"
        return None, str(e)


# ─── Constituent OHLCV (yfinance) ────────────────────────────────────────────

@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _yfinance_batch_download(
    symbols_tuple: tuple[str, ...],
    start_yf: str,
    end_yf: str,
) -> pd.DataFrame:
    """Single raw yfinance batch call — wrapped with retry."""
    raw = yf.download(
        list(symbols_tuple),
        start=start_yf,
        end=end_yf,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Empty yfinance batch response")
    return raw


def fetch_constituent_ohlcv(
    symbols: list[str],
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV for Nifty 50 constituents via yfinance."""
    start_yf = str(pd.Timestamp(start_date).date())
    end_yf = str(pd.Timestamp(end_date).date() + pd.Timedelta(days=1))
    # Sorted tuple → deterministic cache key regardless of input order.
    sym_key = tuple(sorted(symbols))

    cached = ohlcv_cache.get(sym_key, start_yf, end_yf)
    if cached is not None:
        return cached

    try:
        raw = yfinance_circuit.call(
            _yfinance_batch_download, sym_key, start_yf, end_yf
        )
    except CircuitBreakerError as e:
        stale = ohlcv_cache.get_stale(sym_key, start_yf, end_yf)
        if stale is not None:
            log.warning("yfinance circuit open, serving stale OHLCV")
            return stale
        log.error("yfinance unavailable, no stale snapshot: %s", e)
        return {}
    except Exception as e:
        log.error("yfinance batch download failed: %s", e)
        stale = ohlcv_cache.get_stale(sym_key, start_yf, end_yf)
        return stale if stale is not None else {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                sub = raw.xs(sym, level=0, axis=1)
                close_col = sub.get(
                    "Close", sub.iloc[:, 0] if len(sub.columns) else pd.Series()
                )
                if not sub.empty and not close_col.isnull().all():
                    if isinstance(sub.columns, pd.MultiIndex):
                        sub.columns = [c[0] for c in sub.columns]
                    result[sym] = sub
            except KeyError:
                pass

    if result:
        ohlcv_cache.put(sym_key, start_yf, end_yf, value=result)
    return result


# ─── Macro data (yfinance Global Macro universe + commodities/FX) ───────────


# Snapshots older than this are NOT used to backfill a missing column — a
# column resurrected from a weeks-old snapshot forward-fills flat across the
# gap and looks like real, current data downstream (silent staleness).
_BACKFILL_MAX_AGE_DAYS = 7.0


def _load_macro_snapshots_newest_first() -> list[pd.DataFrame]:
    """Prior cached macro frames (ticker-columned), newest first, for column
    backfill. Snapshots past the staleness ceiling are excluded."""
    snaps: list[pd.DataFrame] = []
    try:
        import time as _time
        cutoff = _time.time() - _BACKFILL_MAX_AGE_DAYS * 86400.0
        paths = sorted(
            (p for p in macro_cache._disk_dir.glob("*.pkl") if p.stat().st_mtime >= cutoff),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:  # noqa: BLE001
        return snaps
    for p in paths:
        try:
            with open(p, "rb") as f:
                val, _ = pickle.load(f)
            if isinstance(val, pd.DataFrame) and not val.empty:
                snaps.append(val)
        except Exception:  # noqa: BLE001
            continue
    return snaps


def _backfill_missing_columns(combined: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Refill columns yfinance dropped/rate-limited (absent or all-NaN) from the most
    recent prior snapshot that has them.

    yfinance routinely rate-limits a handful of tickers per batch while the rest
    succeed. The partial frame is non-empty, so it bypasses the all-or-nothing stale
    fallback and gets cached — silently dropping a predictor/target column and failing
    the walk-forward with "Need N+ data points". Backfilling the missing columns keeps
    the frame complete (tail is at most ~1 day stale) and re-heals the cache. Scans
    newest→oldest so it recovers even a previously-poisoned snapshot.
    """
    if combined.empty:
        return combined
    missing = [t for t in tickers if t not in combined.columns or combined[t].isna().all()]
    if not missing:
        return combined

    filled: list[str] = []
    for snap in _load_macro_snapshots_newest_first():
        if not missing:
            break
        snap_aligned = snap.reindex(combined.index).ffill()
        for t in list(missing):
            if t in snap_aligned.columns and not snap_aligned[t].isna().all():
                combined[t] = snap_aligned[t]
                filled.append(t)
                missing.remove(t)

    if filled:
        log.warning("Macro backfill from recent snapshot (≤%.0fd old) for rate-limited/missing tickers: %s",
                    _BACKFILL_MAX_AGE_DAYS, filled)
    if missing:
        log.warning("Macro tickers unavailable (no snapshot within the %.0fd staleness ceiling) — "
                    "columns dropped rather than filled stale: %s", _BACKFILL_MAX_AGE_DAYS, missing)
    return combined


def _fetch_macro_live_uncached(start_str: str, end_str: str) -> pd.DataFrame:
    """Single yfinance batch for the full macro universe (Sanket-style).

    Combines Sanket's Global Macro bond ETFs (proxy for global yield dynamics)
    with the existing commodity + FX symbols. One batch call, one circuit hit.
    """
    tickers = tuple(sorted(set(GLOBAL_MACRO_MAP.values()) | set(MACRO_SYMBOLS_YF.values())))
    if not tickers:
        return pd.DataFrame()

    try:
        yf_raw = yfinance_circuit.call(
            _yfinance_batch_download_macro, tickers, start_str, end_str
        )
    except CircuitBreakerError as e:
        log.warning("yfinance macro fetch blocked by circuit: %s", e)
        return pd.DataFrame()
    except Exception as e:
        log.warning("Yahoo Finance macro fetch failed: %s", e)
        return pd.DataFrame()

    if yf_raw is None or yf_raw.empty:
        return pd.DataFrame()

    if isinstance(yf_raw.columns, pd.MultiIndex):
        if "Close" in yf_raw.columns.get_level_values(0):
            combined = yf_raw["Close"]
        elif "Adj Close" in yf_raw.columns.get_level_values(0):
            combined = yf_raw["Adj Close"]
        else:
            combined = yf_raw
    else:
        combined = yf_raw

    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    combined = combined.sort_index()

    if not combined.empty:
        combined = combined.ffill()
        # yfinance rate-limits a few tickers per batch; refill those columns from the
        # last good snapshot so a partial response never drops a predictor/target.
        combined = _backfill_missing_columns(combined, tickers)
    return combined


@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _yfinance_batch_download_macro(
    tickers_tuple: tuple[str, ...], start: str, end: str
) -> pd.DataFrame:
    """Macro yfinance batch fetch — separated to allow distinct retry budget.

    Uses ``auto_adjust=True`` and ``threads=True`` (matching Sanket's
    ``fetch_batch_data``) so the macro universe is pulled in parallel by
    yfinance's internal pool.
    """
    raw = yf.download(
        list(tickers_tuple),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Empty yfinance macro response")
    return raw


def fetch_macro_live(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Fetch macro indicators with cache + rate-limit + circuit + stale fallback."""
    start_str = str(pd.Timestamp(start_date).date())
    end_str = str(pd.Timestamp(end_date).date() + pd.Timedelta(days=1))

    cached = macro_cache.get(start_str, end_str)
    if cached is not None:
        return cached

    try:
        combined = _fetch_macro_live_uncached(start_str, end_str)
    except Exception as e:
        log.error("Macro fetch raised unexpectedly: %s", e)
        combined = pd.DataFrame()

    if not combined.empty:
        macro_cache.put(start_str, end_str, value=combined)
        return combined

    # Nothing came back this run — try a stale snapshot before returning empty.
    stale = macro_cache.get_stale(start_str, end_str)
    if stale is not None:
        log.warning("Macro fetch empty; serving last-good snapshot")
        return stale
    return combined
