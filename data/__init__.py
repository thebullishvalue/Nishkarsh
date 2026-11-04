"""Nishkarsh v2.0.0 — DATA: the only IO boundary.

fetcher (yfinance / Google Sheets), constituents (niftyindices / Wikipedia / fallback),
cache (pickle-free Parquet/JSON, atomic), circuit_breaker, sources (FetchResult + the
DataSources protocol: Live / Snapshot / Synthetic), synthetic (deterministic worlds),
dates (parse_dates), macro_lag (lag_non_ist). No Streamlit import anywhere in data/.
Submodules are imported explicitly; importing ``data`` itself does nothing.
"""
