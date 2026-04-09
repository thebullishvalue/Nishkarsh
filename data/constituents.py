"""
Nifty 50 constituent list fetching.

Retrieves constituent symbols from niftyindices.com with Wikipedia
fallback and a hardcoded fallback for when both are unavailable.
"""

from __future__ import annotations

import io
import logging
import warnings

import pandas as pd
import requests
import streamlit as st
import urllib3

from core.config import NIFTY50_URL, NIFTY50_FALLBACK

# Suppress insecure request warnings just for these external fetches
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Wikipedia fallback URLs for major indices
_INDIA_INDEX_WIKI = {
    "NIFTY 50": "https://en.wikipedia.org/wiki/NIFTY_50",
    "NIFTY NEXT 50": "https://en.wikipedia.org/wiki/NIFTY_Next_50",
    "NIFTY 500": "https://en.wikipedia.org/wiki/NIFTY_500",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36"
    )
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty50_constituents() -> tuple[list[str], str]:
    """Fetch Nifty 50 constituent symbols with fallback chain.

    Attempts sources in order:
    1. niftyindices.com (official CSV)
    2. Wikipedia (HTML table parse)
    3. Hardcoded fallback list

    Returns
    -------
    symbols : list[str]
        Constituent symbols with ``.NS`` suffix.
    source : str
        Which source provided the data (``"niftyindices"``,
        ``"wikipedia"``, or ``"fallback"``).
    """
    # Primary: niftyindices.com
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = requests.get(NIFTY50_URL, headers=_HEADERS, verify=False, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "Symbol" in df.columns:
            symbols = df["Symbol"].tolist()
            symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
            if symbols_ns:
                return symbols_ns, "niftyindices"
    except Exception as e:
        logging.warning("niftyindices.com fetch failed: %s", e)

    # Fallback: Wikipedia
    try:
        symbols = _parse_wikipedia_constituents(_INDIA_INDEX_WIKI["NIFTY 50"])
        if symbols:
            return [s + ".NS" for s in symbols], "wikipedia"
    except Exception as e:
        logging.warning("Wikipedia fallback failed: %s", e)

    # Hardcoded fallback
    return list(NIFTY50_FALLBACK), "fallback"


def _parse_wikipedia_constituents(url: str, min_count: int = 40) -> list[str] | None:
    """Parse a Wikipedia index constituent page.

    Parameters
    ----------
    url : str
        Wikipedia page URL.
    min_count : int
        Minimum number of symbols expected for a valid table.

    Returns
    -------
    list[str] | None
        Constituent symbols without suffix, or ``None`` if no valid table found.
    """
    tmp_resp = requests.get(url, headers=_HEADERS, timeout=15)
    tmp_resp.raise_for_status()
    tables = pd.read_html(io.StringIO(tmp_resp.text))
    for tbl in tables:
        if "Symbol" in tbl.columns:
            symbols = tbl["Symbol"].dropna().astype(str).str.strip().tolist()
            symbols = [s for s in symbols if s and len(s) <= 20 and s != "nan"]
            if len(symbols) >= min_count:
                return symbols
    return None
