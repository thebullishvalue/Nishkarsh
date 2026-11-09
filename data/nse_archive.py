"""
NSE archive source — official end-of-day data straight from the exchange's public
archives (no API key, no third-party aggregator).

* **Constituents** — ``ind_nifty50list.csv`` (symbol + ISIN).
* **Constituent OHLCV** — the daily equity *bhavcopy* (``cmDDMONYYYYbhav.csv.zip``
  until mid-2024, ``BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip`` after). Stocks are
  tracked by **ISIN**, so a symbol rename (e.g. ZOMATO → ETERNAL) keeps one history.
  Bhavcopy prices are unadjusted; splits / bonuses are back-adjusted with NSE's own
  ``PREVCLOSE`` (the exchange publishes the corporate-action-adjusted previous close
  on the ex-date), so the adjustment uses no outside data.
* **Index levels** — ``ind_close_all_DDMMYYYY.csv`` (close + P/E, P/B, dividend yield).

Every downloaded day is cached as a small parquet file and never fetched twice; a day
with no file (an exchange holiday) is remembered too. Used by the app when Yahoo
Finance is unavailable, and by the measurement scripts. Network access happens only
in :func:`_get`; everything else is pure and unit-testable.
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data.sources import FetchResult

__all__ = ["bhavcopy_urls", "parse_bhavcopy", "adjusted_ohlcv", "NSEArchiveSources",
           "load_bhavcopies", "load_index_closes", "fetch_constituents_nse"]

_LOG = logging.getLogger("data.nse_archive")
_UA = {"User-Agent": "Mozilla/5.0 (Nishkarsh research; end-of-day archive reader)"}
_HOSTS = ("https://nsearchives.nseindia.com", "https://archives.nseindia.com")
_NEW_FORMAT_FROM = pd.Timestamp("2024-07-08")
CONSTITUENTS_PATH = "/content/indices/ind_nifty50list.csv"
# A |PREVCLOSE / previous CLOSE − 1| beyond this marks a corporate action (split, bonus,
# rights). Ordinary days differ by rounding only.
ADJUSTMENT_TOLERANCE = 0.02


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "nishkarsh" / "nse_archive"


# ── URLs & parsing (pure) ────────────────────────────────────────────────────

def bhavcopy_urls(day: pd.Timestamp) -> list[str]:
    """Candidate archive URLs for one trading day, most likely first."""
    d = pd.Timestamp(day)
    old = f"/content/historical/EQUITIES/{d:%Y}/{d.strftime('%b').upper()}/cm{d:%d}{d.strftime('%b').upper()}{d:%Y}bhav.csv.zip"
    new = f"/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    paths = (new, old) if d >= _NEW_FORMAT_FROM else (old, new)
    return [h + p for p in paths for h in _HOSTS]


def index_close_urls(day: pd.Timestamp) -> list[str]:
    return [h + f"/content/indices/ind_close_all_{pd.Timestamp(day):%d%m%Y}.csv" for h in _HOSTS]


_OLD = {"SYMBOL": "symbol", "SERIES": "series", "OPEN": "open", "HIGH": "high", "LOW": "low",
        "CLOSE": "close", "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume", "ISIN": "isin"}
_NEW = {"TckrSymb": "symbol", "SctySrs": "series", "OpnPric": "open", "HghPric": "high",
        "LwPric": "low", "ClsPric": "close", "PrvsClsgPric": "prev_close",
        "TtlTradgVol": "volume", "ISIN": "isin"}


def parse_bhavcopy(raw: bytes, day: pd.Timestamp) -> pd.DataFrame:
    """Equity (EQ/BE/BZ series) rows of one bhavcopy, in a normalized schema:
    date, isin, symbol, open, high, low, close, prev_close, volume."""
    data = raw
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            data = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(data), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    m = _NEW if "TckrSymb" in df.columns else _OLD
    missing = [k for k in m if k not in df.columns]
    if missing:
        raise ValueError(f"bhavcopy {pd.Timestamp(day).date()}: unexpected schema (missing {missing})")
    df = df[list(m)].rename(columns=m)
    df["series"] = df["series"].str.strip()
    df = df[df["series"].isin(["EQ", "BE", "BZ"])].copy()
    for c in ("open", "high", "low", "close", "prev_close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].str.strip()
    df["isin"] = df["isin"].str.strip()
    df.insert(0, "date", pd.Timestamp(day).normalize())
    cols = ["date", "isin", "symbol", "open", "high", "low", "close", "prev_close", "volume"]
    return df[cols].reset_index(drop=True)


def adjusted_ohlcv(long: pd.DataFrame, isin: str, tol: float = ADJUSTMENT_TOLERANCE) -> pd.DataFrame:
    """Split/bonus-adjusted daily OHLCV for one ISIN from long bhavcopy rows.

    On an ex-date NSE publishes PREVCLOSE adjusted for the action, so
    ``f_t = PREVCLOSE_t / CLOSE_{t−1}`` (≈1 on ordinary days) is the factor by which all
    earlier prices must be scaled; factors are compounded backwards from the latest day.
    Volume is scaled inversely. Returns Open/High/Low/Close/Volume on a DatetimeIndex."""
    g = long[long["isin"] == isin].sort_values("date").drop_duplicates("date", keep="last")
    if g.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    close = g["close"].to_numpy(dtype=float)
    prev = g["prev_close"].to_numpy(dtype=float)
    f = np.ones(len(g))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = prev[1:] / close[:-1]
    act = np.isfinite(r) & (np.abs(r - 1.0) > tol)
    f[1:][act] = r[act]
    # price at row i is multiplied by the product of factors of all LATER rows
    cum = np.ones(len(g))
    cum[:-1] = np.cumprod(f[::-1])[::-1][1:]
    out = pd.DataFrame({
        "Open": g["open"].to_numpy(float) * cum, "High": g["high"].to_numpy(float) * cum,
        "Low": g["low"].to_numpy(float) * cum, "Close": close * cum,
        "Volume": g["volume"].to_numpy(float) / cum,
    }, index=pd.DatetimeIndex(g["date"].to_numpy(), name="Date"))
    out.attrs["corporate_actions"] = int(act.sum())
    return out


# ── network + cache ──────────────────────────────────────────────────────────

def _get(urls: list[str], timeout: float = 20.0) -> bytes | None:
    """First successful body among ``urls``; None when every URL answers 404 (no file)."""
    import requests
    last_err = None
    for u in urls:
        for attempt in range(3):
            try:
                r = requests.get(u, headers=_UA, timeout=timeout)
                if r.status_code == 200 and r.content:
                    return r.content
                if r.status_code == 404:
                    break
                last_err = f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001 — network faults are retried
                last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    if last_err:
        _LOG.debug("NSE archive: %s (%s)", urls[0], last_err)
    return None


def _day_cached(kind: str, day: pd.Timestamp, cache_dir: Path, fetch) -> pd.DataFrame | None:
    d = cache_dir / kind
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{pd.Timestamp(day):%Y%m%d}.parquet"
    miss = d / f"{pd.Timestamp(day):%Y%m%d}.none"
    if f.exists():
        return pd.read_parquet(f)
    if miss.exists() and pd.Timestamp(day) < pd.Timestamp.today().normalize() - pd.Timedelta(days=3):
        return None
    raw = fetch(day)
    if raw is None:
        miss.touch()
        return None
    df = raw
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(f)
    return df


def _weekdays(start, end) -> list[pd.Timestamp]:
    return list(pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()))


def load_bhavcopies(start, end, *, cache_dir: Path | None = None, max_workers: int = 6,
                    progress=None) -> pd.DataFrame:
    """Long equity rows for every trading day in [start, end] (holidays have no file)."""
    cache_dir = Path(cache_dir or default_cache_dir())
    days = _weekdays(start, end)

    def one(day):
        def fetch(dd):
            raw = _get(bhavcopy_urls(dd))
            return None if raw is None else parse_bhavcopy(raw, dd)
        try:
            return _day_cached("bhav", day, cache_dir, fetch)
        except Exception as e:  # noqa: BLE001 — a corrupt day must not abort the history
            _LOG.warning("NSE bhavcopy %s skipped: %s", pd.Timestamp(day).date(), e)
            return None

    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, df in enumerate(ex.map(one, days)):
            if df is not None and len(df):
                frames.append(df)
            if progress and (i % 50 == 0 or i == len(days) - 1):
                progress(i + 1, len(days))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_index_closes(start, end, index_name: str = "Nifty 50", *, cache_dir: Path | None = None,
                      max_workers: int = 6) -> pd.DataFrame:
    """Daily close, P/E, P/B and dividend yield of one NSE index (official files)."""
    cache_dir = Path(cache_dir or default_cache_dir())
    days = _weekdays(start, end)

    def one(day):
        def fetch(dd):
            raw = _get(index_close_urls(dd))
            if raw is None:
                return None
            t = pd.read_csv(io.BytesIO(raw), dtype=str)
            t.columns = [c.strip() for c in t.columns]
            return t
        try:
            t = _day_cached("index", day, cache_dir, fetch)
        except Exception:  # noqa: BLE001
            return None
        if t is None or "Index Name" not in t.columns:
            return None
        row = t[t["Index Name"].str.strip().str.lower() == index_name.lower()]
        if row.empty:
            return None
        r = row.iloc[0]
        num = lambda k: pd.to_numeric(r.get(k), errors="coerce")  # noqa: E731
        return {"Date": pd.Timestamp(day), "Close": num("Closing Index Value"), "PE": num("P/E"),
                "PB": num("P/B"), "DY": num("Div Yield")}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = [r for r in ex.map(one, days) if r]
    if not rows:
        return pd.DataFrame(columns=["Close", "PE", "PB", "DY"])
    return pd.DataFrame(rows).set_index("Date").sort_index()


def fetch_constituents_nse(cache_dir: Path | None = None) -> tuple[list[tuple[str, str]], str]:
    """[(symbol, ISIN)] of the current Nifty 50 from NSE's archive, cached for a day."""
    cache_dir = Path(cache_dir or default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / "nifty50_constituents.json"
    if f.exists() and time.time() - f.stat().st_mtime < 86400:
        d = json.loads(f.read_text())
        return [tuple(x) for x in d["rows"]], d["as_of"]
    raw = _get([h + CONSTITUENTS_PATH for h in _HOSTS])
    if raw is None:
        if f.exists():
            d = json.loads(f.read_text())
            return [tuple(x) for x in d["rows"]], d["as_of"] + " (stale)"
        return [], ""
    t = pd.read_csv(io.BytesIO(raw), dtype=str)
    t.columns = [c.strip() for c in t.columns]
    rows = [(s.strip(), i.strip()) for s, i in zip(t["Symbol"], t["ISIN Code"])]
    as_of = str(pd.Timestamp.today().date())
    f.write_text(json.dumps({"rows": rows, "as_of": as_of}))
    return rows, as_of


# ── DataSources implementation ───────────────────────────────────────────────

class NSEArchiveSources:
    """``DataSources`` backed by NSE's public archives (see module docstring).

    Constituents and their OHLCV come from the exchange; the index label is the
    official NIFTY 50 close. NSE publishes no global macro panel, so ``macro()`` is
    empty — the pipeline then uses the sheet's own rate columns for Nirnay's MMR."""

    def __init__(self, cache_dir: Path | None = None, max_workers: int = 6, progress=None):
        self.cache_dir = Path(cache_dir or default_cache_dir())
        self.max_workers = int(max_workers)
        self.progress = progress
        self._isin: dict[str, str] = {}

    def constituents(self) -> FetchResult:
        rows, as_of = fetch_constituents_nse(self.cache_dir)
        if not rows:
            return FetchResult([], "empty", note="NSE archive constituent list unavailable")
        self._isin = {f"{s}.NS": i for s, i in rows}
        return FetchResult([f"{s}.NS" for s, _ in rows], "stale" if "stale" in as_of else "live",
                           as_of=as_of.replace(" (stale)", ""), note="NSE archives (ind_nifty50list.csv)")

    def macro(self, start, end, force: bool = False) -> FetchResult:
        return FetchResult(pd.DataFrame(), "empty", note="NSE archives publish no global macro panel")

    def ohlcv(self, symbols, start, end, force: bool = False) -> FetchResult:
        if not self._isin:
            self.constituents()
        long = load_bhavcopies(start, end, cache_dir=self.cache_dir, max_workers=self.max_workers,
                               progress=self.progress)
        if long.empty:
            return FetchResult({}, "empty", note="no NSE bhavcopy available for the window")
        out, missing, actions = {}, [], 0
        for sym in symbols:
            isin = self._isin.get(sym)
            if isin is None:
                hit = long.loc[long["symbol"] == sym.replace(".NS", ""), "isin"]
                isin = hit.iloc[-1] if len(hit) else None
            df = adjusted_ohlcv(long, isin) if isin else pd.DataFrame()
            if len(df):
                out[sym] = df
                actions += int(df.attrs.get("corporate_actions", 0))
            else:
                missing.append(sym)
        last = str(pd.Timestamp(long["date"].max()).date())
        return FetchResult(out, "live", as_of=last, missing=missing,
                           note=f"NSE bhavcopy · {len(out)} ISIN-linked histories · {actions} split/bonus adjustments")

    def index_levels(self, symbol, start, end, force: bool = False) -> FetchResult:
        idx = load_index_closes(start, end, "Nifty 50", cache_dir=self.cache_dir, max_workers=self.max_workers)
        s = idx["Close"].dropna() if len(idx) else pd.Series(dtype="float64")
        if s.empty:
            return FetchResult(s, "empty", note="NSE index closes unavailable")
        return FetchResult(s, "live", as_of=str(s.index[-1].date()), note="NSE archives (ind_close_all) · NIFTY 50 close")
