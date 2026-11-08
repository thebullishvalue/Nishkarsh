"""data/ layer (contract C-DATA). Ports council/b4a/{verify_data_layer,verify_sources,verify_dates}.py.
Offline: yfinance is replaced by emulators reproducing yfinance 1.x's output shape."""
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data.cache as DC
import data.circuit_breaker as CB
import data.fetcher as F
from data.dates import parse_dates
from data.macro_lag import lag_non_ist


def _bad():
    raise RuntimeError("429")


# ── circuit breaker ──────────────────────────────────────────────────────────

def test_breaker_counts_consecutive_failures():
    cb = CB.CircuitBreaker(failure_threshold=5, recovery_timeout=60, name="t-consec")
    for _ in range(5):                       # 5 isolated failures among 100 successes
        for _ in range(20):
            cb.call(lambda: 1)
        with pytest.raises(RuntimeError):
            cb.call(_bad)
    assert cb.state is CB.CircuitState.CLOSED and cb.failure_count == 1 and cb.total_failures == 5
    assert cb.call(lambda: 7) == 7
    for _ in range(5):
        with pytest.raises(RuntimeError):
            cb.call(_bad)
    assert cb.state is CB.CircuitState.OPEN
    with pytest.raises(CB.CircuitBreakerError):
        cb.call(lambda: 1)


def _probe_threads(cb, n, hold):
    out = []

    def slow():
        hold.wait(5)
        return 1
    th = [threading.Thread(target=lambda: out.append(cb.call(slow))) for _ in range(n)]
    for t in th:
        t.start()
    deadline = time.time() + 5
    while cb.half_open_calls < n and time.time() < deadline:
        time.sleep(0.005)
    return th, out


def test_breaker_half_open_admits_two_probes():
    cb = CB.CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, name="t-half")
    with pytest.raises(RuntimeError):
        cb.call(_bad)
    assert cb.state is CB.CircuitState.OPEN
    hold = threading.Event()
    th, out = _probe_threads(cb, 2, hold)          # macro + OHLCV probes in flight together
    assert cb.state is CB.CircuitState.HALF_OPEN and cb.half_open_calls == 2
    with pytest.raises(CB.CircuitBreakerError):    # a 3rd concurrent probe is refused
        cb.call(lambda: 1)
    hold.set()
    for t in th:
        t.join()
    assert out == [1, 1] and cb.state is CB.CircuitState.CLOSED


def test_breaker_failed_probe_reopens():
    cb = CB.CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, name="t-reopen")
    with pytest.raises(RuntimeError):
        cb.call(_bad)
    with pytest.raises(RuntimeError):
        cb.call(_bad)                              # the HALF_OPEN probe fails
    assert cb.state is CB.CircuitState.OPEN


# ── cache: round-trip, no pickle ─────────────────────────────────────────────

def _same(a, b) -> bool:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_freq=False)
        return True
    if isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, check_freq=False)
        return True
    if isinstance(a, dict) and a and all(isinstance(x, pd.DataFrame) for x in a.values()):
        return a.keys() == b.keys() and all(_same(a[x], b[x]) for x in a)
    return a == b


def test_cache_round_trip_every_type_no_pickle(tmp_path):
    c = DC.Cache(ttl=3600, disk_dir=tmp_path, namespace="rt")
    idx = pd.bdate_range("2024-01-01", periods=30, name="Date")
    frame = pd.DataFrame({"f": np.linspace(0, 1, 30), "i": np.arange(30, dtype="int64"), "b": np.arange(30) % 2 == 0,
                          "s": [f"x{i}" for i in range(30)], "t": idx}, index=idx)
    values = {"frame": frame, "series": frame["f"].rename("close"),
              "frames": {"AAA.NS": frame[["f", "i"]], "BBB.NS": frame[["f", "b", "s"]].iloc[5:],
                         "EMPTY.NS": frame[["f"]].iloc[:0]},
              "list": ["A.NS", "B.NS"], "dict": {"a": 1, "b": [1.5, None], "c": "x"}, "str": "hi", "int": 7,
              "float": 2.5, "none": None, "bool": True}
    for k, v in values.items():
        c.put(k, value=v, tag="t-" + k)
    fresh = DC.Cache(ttl=3600, disk_dir=tmp_path, namespace="rt")       # new process view: disk only
    for k, v in values.items():
        assert _same(v, fresh.get(k)), k
    try:                                                              # unsupported → never pickled
        c.put("arr", value=np.arange(3))
        assert fresh.get("arr") is None
    except TypeError:
        pass
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files and {p.suffix for p in files} <= {".parquet", ".json"}
    assert not list(tmp_path.rglob("*.pkl")) and not list(tmp_path.rglob(".tmp-*"))
    v3 = DC.Cache(ttl=3600, disk_dir=tmp_path, namespace="rt", version="v3")
    assert v3.get("frame") is None and fresh.get("frame") is not None   # version bump invalidates


class _Payload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (Path.write_text, (self.marker, "code executed"))


def test_planted_pickle_never_executed(cache_root, monkeypatch, tmp_path):
    marker = tmp_path / "PWNED_MARKER"
    blob = pickle.dumps((_Payload(marker), time.time()))
    for d in (DC.macro_cache.disk_dir.parent, DC.macro_cache.disk_dir, DC.ohlcv_cache.disk_dir):
        (d / "planted.pkl").write_bytes(blob)
    tick = F.macro_tickers()
    key = DC.macro_cache._key(F.symset_tag(tick), "2026-01-01", "2026-06-02")
    (DC.macro_cache.disk_dir / f"{key}.pkl").write_bytes(blob)       # a key-named .pkl
    (DC.macro_cache.disk_dir / f"{key}.json").write_bytes(blob)      # pickle bytes behind a .json name

    def yf_macro(tt, s, e):
        ix = pd.bdate_range(s, pd.Timestamp(e) - pd.Timedelta(days=1))
        cols = pd.MultiIndex.from_product([["Close", "Open"], list(tt)])
        raw = pd.DataFrame(np.random.default_rng(1).random((len(ix), len(cols))) + 50, index=ix, columns=cols)
        raw.loc[:, ("Close", tt[0])] = np.nan
        return raw
    monkeypatch.setattr(F, "_yfinance_batch_download_macro", yf_macro)
    F.yfinance_circuit.reset()
    try:
        r = F.fetch_macro("2026-01-01", "2026-06-01")
        assert r.provenance in ("live", "stale")
        DC.macro_cache.clear()
        monkeypatch.setattr(F, "_yfinance_batch_download_macro", lambda *a, **k: _bad())
        F.fetch_macro("2026-01-01", "2026-06-01", force=True)        # outage → stale scan over planted files
    finally:
        F.yfinance_circuit.reset()
    assert not marker.exists()
    assert DC.purge_legacy_pickles(cache_root) == 4 and not list(cache_root.rglob("*.pkl"))
    assert not marker.exists()


# ── fetcher: stale lookup, partial batch ─────────────────────────────────────

def _yf_ok(symbols_tuple, start_yf, end_yf):
    ix = pd.bdate_range(start_yf, pd.Timestamp(end_yf) - pd.Timedelta(days=1))
    cols = pd.MultiIndex.from_product([[s.upper() for s in symbols_tuple], ["Open", "High", "Low", "Close", "Volume"]])
    return pd.DataFrame(np.random.default_rng(0).random((len(ix), len(cols))) + 1, index=ix, columns=cols)


@pytest.fixture
def yf(monkeypatch, cache_root):
    def use(fn):
        monkeypatch.setattr(F, "_yfinance_batch_download", fn)
        F.yfinance_circuit.reset()
    yield use
    F.yfinance_circuit.reset()


def test_stale_lookup_across_date_change(yf):
    syms = ["AAA.NS", "BBB.NS"]
    yf(_yf_ok)
    d1_end = pd.Timestamp("2026-11-02")
    r1 = F.fetch_ohlcv(syms, d1_end - pd.Timedelta(days=365 * 7), d1_end)
    assert r1.provenance == "live" and sorted(r1.value) == syms
    yf(lambda *a: _bad())                                            # next day: rate-limited
    DC.ohlcv_cache.clear()
    DC.ohlcv_cache.ttl = 0
    d2_end = d1_end + pd.Timedelta(days=1)
    d2_start = d2_end - pd.Timedelta(days=365 * 7)
    r2 = F.fetch_ohlcv(syms, d2_start, d2_end)
    assert r2.provenance == "stale" and sorted(r2.value) == syms and r2.as_of is not None
    assert all(v.index.min() >= d2_start and v.index.max() <= d2_end for v in r2.value.values())
    assert sorted(F.fetch_constituent_ohlcv(syms, d2_start, d2_end)) == syms   # legacy wrapper


def test_partial_batch_never_shrinks_snapshot(yf):
    syms = [f"S{i}.NS" for i in range(10)]
    win = ("2026-01-01", "2026-06-01")
    yf(_yf_ok)
    assert len(F.fetch_ohlcv(syms, *win).value) == 10

    def partial(symbols_tuple, s, e):
        raw = _yf_ok(symbols_tuple, s, e)
        for x in list(symbols_tuple)[4:]:
            raw.loc[:, x.upper()] = np.nan
        return raw
    yf(partial)
    DC.ohlcv_cache.clear()
    DC.ohlcv_cache.ttl = 0
    part = F.fetch_ohlcv(syms, *win)
    assert len(part.value) == 10 and part.provenance == "stale" and part.missing == []
    DC.ohlcv_cache.ttl = 3600
    DC.ohlcv_cache.clear()
    snap = DC.ohlcv_cache.latest(F.symset_tag(syms))
    assert len(snap.value) == 10 and snap.meta["complete"] is False and len(snap.meta["stale_filled"]) == 6

    def with_new(symbols_tuple, s, e):
        raw = _yf_ok(symbols_tuple, s, e)
        raw.loc[:, "NEW9.NS"] = np.nan
        return raw
    yf(with_new)
    r = F.fetch_ohlcv(syms[:3] + ["NEW9.NS"], *win, force=True)
    assert r.missing == ["NEW9.NS"] and r.provenance == "live"


def test_force_bypasses_fresh_cache(yf):
    calls = []
    yf(lambda *a: (calls.append(1), _yf_ok(*a))[1])
    syms = ["X1.NS", "X2.NS"]
    F.fetch_ohlcv(syms, "2026-01-01", "2026-03-01")
    hit = F.fetch_ohlcv(syms, "2026-01-01", "2026-03-01")
    assert hit.provenance == "cache" and len(calls) == 1
    F.fetch_ohlcv(syms, "2026-01-01", "2026-03-01", force=True)
    assert len(calls) == 2


def test_fallback_tickers_upper_cased(yf):
    from core.config import NIFTY50_FALLBACK
    yf(_yf_ok)
    fb = list(NIFTY50_FALLBACK)
    r = F.fetch_ohlcv(fb, "2026-01-01", "2026-03-01")
    assert len(r.value) == len({s.upper() for s in fb}) and "EICHERMOT.NS" in r.value


def test_sheet_url_forms_and_host_check():
    forms = {
        "https://docs.google.com/spreadsheets/d/1AbC_d-9/edit#gid=123": "/d/1AbC_d-9/export?format=csv&gid=123",
        "https://docs.google.com/spreadsheets/d/1AbC_d-9/edit?usp=sharing": "/d/1AbC_d-9/export?format=csv&gid=0",
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQxyz/pubhtml?gid=55&single=true":
            "/d/e/2PACX-1vQxyz/pub?output=csv&gid=55&single=true",
    }
    for u, tail in forms.items():
        assert F.sheet_csv_url(u).endswith(tail)
    with pytest.raises(CB.FetchRejected):
        F.sheet_csv_url("https://evil.example.com/spreadsheets/d/abc/edit")


# ── parse_dates ──────────────────────────────────────────────────────────────

DATES = pd.bdate_range("2016-01-01", periods=2500)


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"])
def test_parse_dates_accepts(fmt):
    p = parse_dates(pd.Series(DATES.strftime(fmt), name="DATE"))
    assert (pd.DatetimeIndex(p) == DATES).all()


@pytest.mark.parametrize("label, vals", [
    ("US mm/dd", list(DATES.strftime("%m/%d/%Y"))),
    ("2% garbage", list(DATES.strftime("%Y-%m-%d")[:-50]) + ["garbage"] * 50),
    ("shuffled", list(pd.Series(DATES.strftime("%Y-%m-%d")).sample(frac=1, random_state=0))),
    ("numeric", list(range(2500))),
])
def test_parse_dates_rejects(label, vals):
    with pytest.raises(ValueError):
        parse_dates(pd.Series(vals, name="DATE"))


def test_parse_dates_descending_blanks_datetime():
    assert parse_dates(pd.Series(list(DATES.strftime("%Y-%m-%d")[::-1]))).is_monotonic_decreasing
    blanks = pd.Series(list(DATES.strftime("%d/%m/%Y")[:-3]) + ["", "n/a", None])
    assert int(parse_dates(blanks).isna().sum()) == 3
    assert pd.api.types.is_datetime64_any_dtype(parse_dates(pd.Series(DATES)).dtype)


# ── lag_non_ist ──────────────────────────────────────────────────────────────

def test_lag_non_ist_row_mode():
    idx = pd.bdate_range("2024-01-01", periods=5)
    toy = pd.DataFrame({"IN10Y": [1., 2, 3, 4, 5], "LIQUIDBEES.NS": [10., 20, 30, 40, 50],
                        "US10Y": [.1, .2, .3, .4, .5], "HYG": [100., 101, 102, 103, 104],
                        "INR=X": [80., 81, 82, 83, 84]}, index=idx)
    lt = lag_non_ist(toy)
    assert lt["IN10Y"].equals(toy["IN10Y"]) and lt["LIQUIDBEES.NS"].equals(toy["LIQUIDBEES.NS"])
    assert lt["US10Y"].iloc[1:].tolist() == [.1, .2, .3, .4] and np.isnan(lt["HYG"].iloc[0])
    assert {"US10Y", "HYG"} <= set(lt.attrs["lagged_columns"])
    assert toy["US10Y"].tolist() == [.1, .2, .3, .4, .5]               # pure


def test_lag_non_ist_calendar_mode():
    us = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05"])   # US trades Fri
    nse = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08"])  # NSE trades Thu, Mon
    panel = pd.DataFrame({"TLT": [1., 2, 3, 5], "GILT5YBEES.NS": [7., 8, 9, 11]}, index=us)
    la = lag_non_ist(panel, nse_index=nse)
    assert la.index.equals(nse)
    assert la["TLT"].tolist() == [1.0, 2.0, 3.0, 5.0]               # last close strictly before t
    assert la["GILT5YBEES.NS"].tolist() == [8.0, 9.0, 9.0, 11.0]    # IST: as-of t


# ── synthetic sources ────────────────────────────────────────────────────────

def test_synthetic_sheet_contract(small_sheet):
    from core.config import DEFAULT_PREDICTORS
    assert list(small_sheet.columns) == ["DATE", "NIFTY50_PE", *DEFAULT_PREDICTORS]
    assert small_sheet["DATE"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert parse_dates(small_sheet["DATE"]).is_monotonic_increasing
    assert (small_sheet["NIFTY50_PE"] > 0).all()


def test_synthetic_sources_deterministic(small_world):
    from data.sources import DataSources
    from data.synthetic import synthetic_sheet, synthetic_world
    sh1, src1 = small_world
    sh2, src2 = synthetic_world(7, n_rows=900)
    S, E = "2000-01-01", "2030-12-31"

    def frames(src):
        cons = src.constituents().value
        return [src.macro(S, E).value, *src.ohlcv(cons, S, E).value.values(),
                src.index_levels("^NSEI", S, E).value.to_frame()]
    assert sh1.equals(sh2)
    fa, fb = frames(src1), frames(src2)
    assert len(fa) > 10 and all(a.equals(b) for a, b in zip(fa, fb))
    assert synthetic_sheet(7, n_rows=900).equals(sh1)
    assert not synthetic_sheet(8, n_rows=900).equals(sh1)
    assert isinstance(src1, DataSources)
    assert len(src1.constituents().value) >= 10


def test_synthetic_failure_injection():
    from data.synthetic import synthetic_world
    _, bad = synthetic_world(7, n_rows=900, empty=("macro",), drop_symbols=("SYN03.NS",))
    cons = bad.constituents().value
    assert bad.macro("2000-01-01", "2030-12-31").provenance == "empty"
    assert bad.ohlcv(cons, "2000-01-01", "2030-12-31").missing == ["SYN03.NS"]


# ── no streamlit in data/ ────────────────────────────────────────────────────

def test_data_package_does_not_import_streamlit(repo_root):
    code = ("import sys, pkgutil, importlib, data\n"
            "for m in pkgutil.iter_modules(data.__path__):\n"
            "    importlib.import_module('data.' + m.name)\n"
            "bad = sorted(k for k in sys.modules if k == 'streamlit' or k.startswith('streamlit.'))\n"
            "print('STREAMLIT' if bad else 'CLEAN')\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(repo_root), capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "CLEAN"
