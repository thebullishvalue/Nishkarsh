"""NSE archive source — parsing and corporate-action adjustment (offline)."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from data.nse_archive import adjusted_ohlcv, bhavcopy_urls, parse_bhavcopy

OLD = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
       "RELIANCE,EQ,2343.5,2371.8,2338.5,2368.9,2371,2342.5,5861389,1.38e10,27-SEP-2023,233089,INE002A01018,\n"
       "SGBJUN28,GB,1,1,1,1,1,1,1,1,27-SEP-2023,1,IN0020200104,\n")
NEW = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,"
       "OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
       "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
       "2025-09-30,2025-09-30,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE IND,1377.10,1377.10,1362.80,"
       "1364.00,1364.0,1372.80,,,,,14604684,1.99e10,1,F1,1,,,,,\n")


def _zip(text: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.csv", text)
    return b.getvalue()


def test_parse_both_bhavcopy_formats_keep_equity_only():
    old = parse_bhavcopy(_zip(OLD), pd.Timestamp("2023-09-27"))
    new = parse_bhavcopy(_zip(NEW), pd.Timestamp("2025-09-30"))
    for df, close, prev in ((old, 2368.9, 2342.5), (new, 1364.0, 1372.8)):
        assert list(df.columns) == ["date", "isin", "symbol", "open", "high", "low", "close", "prev_close", "volume"]
        assert df["symbol"].tolist() == ["RELIANCE"]                 # the GB (gold bond) row is dropped
        assert df["close"].iloc[0] == close and df["prev_close"].iloc[0] == prev


def test_urls_choose_format_by_date():
    assert "cm27SEP2023bhav.csv.zip" in bhavcopy_urls(pd.Timestamp("2023-09-27"))[0]
    assert "BhavCopy_NSE_CM_0_0_0_20250930" in bhavcopy_urls(pd.Timestamp("2025-09-30"))[0]


def _long(closes, prevs, isin="INE1", symbols=None):
    d = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"date": d, "isin": isin, "symbol": symbols or ["OLDNAME"] * len(closes),
                         "open": closes, "high": closes, "low": closes, "close": closes,
                         "prev_close": prevs, "volume": [100.0] * len(closes)})


def test_split_is_back_adjusted_with_nse_prevclose():
    # 1:2 split on day 3: price 200 → 100; NSE prints PREVCLOSE 100 (adjusted) that day.
    closes = [200.0, 202.0, 101.0, 102.0]
    prevs = [199.0, 200.0, 101.0, 101.0]
    adj = adjusted_ohlcv(_long(closes, prevs), "INE1")
    assert adj.attrs["corporate_actions"] == 1
    np.testing.assert_allclose(adj["Close"].to_numpy(), [100.0, 101.0, 101.0, 102.0])
    np.testing.assert_allclose(adj["Volume"].to_numpy()[:2], [200.0, 200.0])   # volume scaled inversely
    r = np.diff(np.log(adj["Close"].to_numpy()))
    assert np.abs(r).max() < 0.02                                                 # no fake −50% day


def test_isin_links_history_across_a_symbol_rename():
    df = _long([10.0, 10.5, 11.0], [9.9, 10.0, 10.5], symbols=["ZOMATO", "ZOMATO", "ETERNAL"])
    adj = adjusted_ohlcv(df, "INE1")
    assert len(adj) == 3 and adj.attrs["corporate_actions"] == 0
