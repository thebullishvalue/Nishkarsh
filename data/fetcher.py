"""
Unified data fetcher — Google Sheets, yfinance, and Stooq sources.

Fetches Aarambh valuation data from Google Sheets, Nifty 50 constituent
OHLCV from yfinance, and macro indicators from Stooq / Yahoo Finance.
All data is merged into a ``UnifiedDataset`` with aligned date indices.
"""

from __future__ import annotations

import io
import logging
import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from core.config import (
    MACRO_SYMBOLS_STOOQ,
    MACRO_SYMBOLS_YF,
    DEFAULT_SHEET_URL,
)
from data.cache import Cache
from data.schema import UnifiedDataset
from data.constituents import fetch_nifty50_constituents

cache = Cache(ttl=3600)


# ─── Aarambh data (Google Sheets) ────────────────────────────────────────────


def fetch_aarambh_data(
    source: str = DEFAULT_SHEET_URL,
) -> pd.DataFrame | None:
    """Fetch valuation data from a Google Sheet.

    Parameters
    ----------
    source : str
        Google Sheets URL.

    Returns
    -------
    pd.DataFrame | None
        Raw sheet data, or ``None`` on failure.
    """
    cache_key = f"aarambh_sheet_{source}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", source)
        if not sheet_id_match:
            return None
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"gid=(\d+)", source)
        gid = gid_match.group(1) if gid_match else "0"
        csv_url = (
            f"https://docs.google.com/spreadsheets/d/"
            f"{sheet_id}/export?format=csv&gid={gid}"
        )
        df = pd.read_csv(csv_url)
        cache.put(cache_key, value=df)
        return df
    except Exception as e:
        logging.error("Failed to load Google Sheets: %s", e)
        return None


# ─── Constituent OHLCV (yfinance) ────────────────────────────────────────────


def fetch_constituent_ohlcv(
    symbols: list[str],
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV for Nifty 50 constituents via yfinance.

    Parameters
    ----------
    symbols : list[str]
        Constituent symbols with ``.NS`` suffix.
    start_date : pd.Timestamp | str
        Start of the date range.
    end_date : pd.Timestamp | str
        End of the date range.

    Returns
    -------
    dict[str, pd.DataFrame]
        Per-symbol OHLCV DataFrames, keyed by symbol.
    """
    cache_key = f"ohlcv_{','.join(sorted(symbols))}_{start_date}_{end_date}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    start_yf = str(pd.Timestamp(start_date).date())
    end_yf = str(pd.Timestamp(end_date).date())

    try:
        raw = yf.download(
            symbols,
            start=start_yf,
            end=end_yf,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        logging.error("yfinance batch download failed: %s", e)
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                sub = raw.xs(sym, level=0, axis=1)
                close_col = sub.get("Close", sub.iloc[:, 0] if len(sub.columns) else pd.Series())
                if not sub.empty and not close_col.isnull().all():
                    if isinstance(sub.columns, pd.MultiIndex):
                        sub.columns = [c[0] for c in sub.columns]
                    result[sym] = sub
            except KeyError:
                pass

    cache.put(cache_key, value=result)
    return result


# ─── Macro data (Stooq + Yahoo Finance) ──────────────────────────────────────


def _fetch_stooq_symbol(
    symbol: str, start_date: str, end_date: str
) -> pd.Series | None:
    """Fetch a single yield series from Stooq via HTTP."""
    try:
        url = (
            f"https://stooq.com/q/d/l/?s={symbol}"
            f"&d1={start_date}&d2={end_date}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and len(resp.text) > 50:
            df = pd.read_csv(io.StringIO(resp.text))
            if "Date" in df.columns and "Close" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                return df.set_index("Date").sort_index()["Close"]
    except Exception:
        pass
    return None


def fetch_macro_live(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Fetch macro indicators from Stooq (yields) and Yahoo Finance (FX, commodities).

    Parameters
    ----------
    start_date : pd.Timestamp | str
        Start of the date range.
    end_date : pd.Timestamp | str
        End of the date range.

    Returns
    -------
    pd.DataFrame
        Combined macro DataFrame, forward-filled.
    """
    cache_key = f"macro_{start_date}_{end_date}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    start_str = str(pd.Timestamp(start_date).date())
    end_str = str(pd.Timestamp(end_date).date())

    # Stooq yields
    stooq_parts: dict[str, pd.Series] = {}
    for _name, symbol in MACRO_SYMBOLS_STOOQ.items():
        series = _fetch_stooq_symbol(symbol, start_str, end_str)
        if series is not None and len(series) > 0:
            stooq_parts[symbol] = series
    stooq_df = pd.DataFrame(stooq_parts).sort_index() if stooq_parts else pd.DataFrame()

    # Yahoo Finance macro
    yf_df = pd.DataFrame()
    try:
        yf_tickers = list(MACRO_SYMBOLS_YF.values())
        yf_raw = yf.download(
            yf_tickers, start=start_str, end=end_str, progress=False
        )
        if not yf_raw.empty:
            if isinstance(yf_raw.columns, pd.MultiIndex):
                if "Close" in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw["Close"]
                elif "Adj Close" in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw["Adj Close"]
            else:
                yf_df = yf_raw
            if yf_df.index.tz is not None:
                yf_df.index = yf_df.index.tz_localize(None)
            yf_df = yf_df.sort_index()
    except Exception as e:
        logging.warning("Yahoo Finance macro fetch failed: %s", e)

    # Combine
    if not stooq_df.empty and not yf_df.empty:
        combined = pd.concat([stooq_df, yf_df], axis=1).sort_index()
    elif not stooq_df.empty:
        combined = stooq_df
    elif not yf_df.empty:
        combined = yf_df
    else:
        combined = pd.DataFrame()

    if not combined.empty:
        combined = combined.ffill()

    cache.put(cache_key, value=combined)
    return combined


# ─── Unified dataset builder ─────────────────────────────────────────────────


def build_unified_dataset(
    aarambh_df: pd.DataFrame,
    target_col: str = "NIFTY50_PE",
    feature_cols: list[str] | None = None,
    date_col: str | None = None,
    constituents_ohlcv: dict[str, pd.DataFrame] | None = None,
    macro_df: pd.DataFrame | None = None,
) -> UnifiedDataset:
    """Build a ``UnifiedDataset`` by merging all data sources.

    Parameters
    ----------
    aarambh_df : pd.DataFrame
        Raw Google Sheet data.
    target_col : str
        Column name for the Nifty 50 PE target variable.
    feature_cols : list[str] | None
        Predictor column names.
    date_col : str | None
        Date column name in the sheet.
    constituents_ohlcv : dict[str, pd.DataFrame] | None
        Per-constituent OHLCV data.
    macro_df : pd.DataFrame | None
        Combined macro indicators.

    Returns
    -------
    UnifiedDataset
        Unified dataset with aligned date indices.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in aarambh_df.columns
            if c != target_col and c != date_col
        ]

    # Clean Aarambh data
    cols = [target_col] + feature_cols
    if date_col and date_col in aarambh_df.columns:
        cols.append(date_col)
    valid_cols = [c for c in cols if c in aarambh_df.columns]
    data = aarambh_df[valid_cols].copy()

    if date_col and date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data.dropna(subset=[date_col]).sort_values(date_col)

    for col in [target_col] + feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    numeric_cols = [target_col] + feature_cols
    data[numeric_cols] = data[numeric_cols].ffill().bfill()
    data = data.dropna(subset=numeric_cols).reset_index(drop=True)

    if date_col and date_col in data.columns:
        date_index = pd.DatetimeIndex(data[date_col])
    else:
        date_index = pd.DatetimeIndex(
            pd.date_range(end=pd.Timestamp.today(), periods=len(data), freq="B")
        )

    nifty50_pe = data[target_col].values
    predictors = data[feature_cols].copy()

    macro_aligned = macro_df.reindex(date_index).ffill() if macro_df is not None and not macro_df.empty else pd.DataFrame()

    if constituents_ohlcv is None:
        constituents_ohlcv = {}

    return UnifiedDataset(
        date_index=date_index,
        nifty50_pe=nifty50_pe,
        aarambh_predictors=predictors,
        constituent_ohlcv=constituents_ohlcv,
        macro_df=macro_aligned,
        trading_days=list(date_index),
    )
