"""
Nishkarsh v2.0.0 — deterministic synthetic data for offline runs, tests and power studies.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

    synthetic_sheet(seed, n_rows=4000, start="2010-01-01", planted_ic=0.0) -> DataFrame
        DATE (ISO strings, like a CSV export) + NIFTY50_PE + core.config.DEFAULT_PREDICTORS.
    synthetic_world(seed, ...) -> (sheet_df, SyntheticSources)
        + ≥ 10 constituents (one LATE LISTER), OHLCV over the last HISTORY_YEARS of the sheet
          (window anchored to 1 Jan, as in production), a macro panel of real ticker names
          (US ETFs / ^TNX / FX / futures + three .NS ETFs) and ^NSEI levels = PE × EPS.

Everything is a pure function of the arguments: independent ``SeedSequence`` streams per
component (sheet, EPS, constituents, macro), no dependence on today's date.

DGP. logPE is a slowly mean-reverting random walk (daily σ 0.8%). With ``planted_ic = ρ ≠ 0``
a GENUINE, CAUSAL forward relation is planted: let
    s(t)  = (z[Δ20 REL_BREADTH](t) − z[Δ20 IN10Y](t)) / √2        (20-day predictor momentum)
    D(t)  = Σ_{k=t−9..t} s(k)                                     (known at the close of t)
and the daily drift of logPE on day u is c·s(u−10), so logPE(t+10) − logPE(t) = c·D(t) + noise
with c chosen so corr(D(t), fwd 10d ΔlogPE) = ρ. Only IST-synchronous predictors carry the
signal, so the one-session lag of US columns cannot destroy it. A forecaster built on
per-column 20-day momentum can recover most (not all) of ρ; the realised driver IC is stored
in ``sheet.attrs["synthetic"]``. ρ = 0 is the null world (no edge anywhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import DEFAULT_PREDICTORS, HISTORY_YEARS

__all__ = ["synthetic_sheet", "synthetic_world", "build_world", "SyntheticWorld",
           "MACRO_TICKERS", "INDEX_SYMBOL"]

INDEX_SYMBOL = "^NSEI"
_H = 10            # forward horizon of the planted relation (= AARAMBH_FWD_HORIZON default)
_K = 20            # momentum window (= AARAMBH_FWD_MOM_K default)
_PE_SIGMA = 0.008

# (ticker, start level, daily idio σ, loadings scale) — names consumed by analytics/custom_features.
MACRO_TICKERS: tuple[tuple[str, float, float, float], ...] = (
    ("HYG", 80.0, 0.003, 1.0), ("LQD", 110.0, 0.003, 1.0), ("TLT", 100.0, 0.006, 1.5),
    ("IEF", 95.0, 0.003, 1.0), ("SHY", 82.0, 0.001, 0.3), ("TIP", 110.0, 0.003, 0.8),
    ("BND", 75.0, 0.002, 0.8), ("BNDX", 50.0, 0.002, 0.6), ("EMB", 90.0, 0.004, 1.0),
    ("EMHY", 40.0, 0.004, 1.0), ("FALN", 27.0, 0.004, 1.0), ("PFF", 33.0, 0.004, 1.0),
    ("DX-Y.NYB", 100.0, 0.004, 0.8), ("INR=X", 75.0, 0.003, 0.5), ("EURINR=X", 85.0, 0.005, 0.6),
    ("GC=F", 1500.0, 0.009, 1.2), ("CL=F", 60.0, 0.02, 2.0),
    ("LIQUIDBEES.NS", 1000.0, 0.0002, 0.0), ("GILT5YBEES.NS", 50.0, 0.002, 0.5),
    ("LTGILTBEES.NS", 25.0, 0.003, 0.8),
)


@dataclass
class SyntheticWorld:
    seed: int
    sheet: pd.DataFrame                       # DATE strings + target + predictors
    dates: pd.DatetimeIndex                   # sheet calendar (business days)
    constituents: list[str]
    ohlcv: dict[str, pd.DataFrame]
    macro: pd.DataFrame
    index_ohlcv: dict[str, pd.DataFrame]      # {"^NSEI": OHLCV}
    late_lister: str | None
    params: dict = field(default_factory=dict)


def _ar1(rng: np.random.Generator, n: int, mu: float, phi: float, sd: float, x0: float | None = None) -> np.ndarray:
    e = rng.normal(0.0, sd, n)
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + e[t]
    return x


def _streams(seed: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(4)]


def _sheet_core(seed: int, n_rows: int, start: str, planted_ic: float, date_fmt: str):
    if n_rows < 3 * _K:
        raise ValueError("n_rows too small")
    if not -0.95 <= planted_ic <= 0.95:
        raise ValueError("planted_ic must be in [-0.95, 0.95]")
    rng = _streams(seed)[0]
    n = int(n_rows)
    dates = pd.bdate_range(pd.Timestamp(start), periods=n, name="DATE")

    rel_breadth = _ar1(rng, n, 0.0, 0.97, 0.08)
    ad_ratio = np.abs(_ar1(rng, n, 1.0, 0.6, 0.25)) + 0.05
    count = np.clip(np.round(_ar1(rng, n, 25.0, 0.8, 4.0)), 0, 50)
    rel_ad = _ar1(rng, n, 0.0, 0.7, 0.2)
    in10 = _ar1(rng, n, 7.0, 0.999, 0.02, 7.2)
    in02 = in10 - 0.6 + _ar1(rng, n, 0.0, 0.99, 0.01)
    in30 = in10 + 0.3 + _ar1(rng, n, 0.0, 0.99, 0.01)
    iniryy = _ar1(rng, n, 5.0, 0.995, 0.05)
    repo = np.round(_ar1(rng, n, 6.0, 0.9995, 0.01, 6.0) * 4.0) / 4.0
    us10 = _ar1(rng, n, 3.5, 0.999, 0.03, 2.0)
    us02 = us10 - 0.3 + _ar1(rng, n, 0.0, 0.995, 0.02)
    us30 = us10 + 0.4 + _ar1(rng, n, 0.0, 0.995, 0.01)
    eps = rng.normal(0.0, _PE_SIGMA, n)
    dy_noise = rng.normal(0.0, 0.01, n)
    pb_noise = rng.normal(0.0, 0.01, n)

    # planted driver (causal): 20d momentum composite → 10-day trailing sum
    m1 = pd.Series(rel_breadth).diff(_K)
    m2 = pd.Series(in10).diff(_K)
    s = ((m1 / m1.std()) - (m2 / m2.std())) / np.sqrt(2.0)
    driver = s.rolling(_H).sum()
    c = 0.0
    if planted_ic != 0.0:
        c = planted_ic * _PE_SIGMA * np.sqrt(_H) / np.sqrt(1.0 - planted_ic ** 2) / float(driver.std())
    s_np = s.fillna(0.0).to_numpy()
    log22 = np.log(22.0)
    lpe = np.empty(n)
    lpe[0] = log22
    for t in range(1, n):
        drift = c * s_np[t - _H] if t >= _H else 0.0
        lpe[t] = lpe[t - 1] + 0.002 * (log22 - lpe[t - 1]) + drift + eps[t]
    pe = np.exp(lpe)

    cols = {
        "NIFTY50_PE": pe, "AD_RATIO": ad_ratio, "COUNT": count, "REL_AD_RATIO": rel_ad,
        "REL_BREADTH": rel_breadth, "IN10Y": in10, "IN02Y": in02, "IN30Y": in30,
        "INIRYY": iniryy, "REPO": repo, "US02Y": us02, "US10Y": us10, "US30Y": us30,
        "NIFTY50_DY": 1.3 * 22.0 / pe * np.exp(dy_noise),
        "NIFTY50_PB": pe / 6.3 * np.exp(pb_noise),
    }
    missing = [p for p in DEFAULT_PREDICTORS if p not in cols]
    if missing:
        raise RuntimeError(f"synthetic sheet has no generator for predictors {missing}")
    sheet = pd.DataFrame({"DATE": dates.strftime(date_fmt)})
    sheet["NIFTY50_PE"] = pe
    for p in DEFAULT_PREDICTORS:
        sheet[p] = cols[p]

    fwd = pd.Series(lpe).shift(-_H) - pd.Series(lpe)
    ok = driver.notna() & fwd.notna()
    realized = float(driver[ok].rank().corr(fwd[ok].rank())) if ok.sum() > 10 else float("nan")
    sheet.attrs["synthetic"] = {"seed": int(seed), "planted_ic": float(planted_ic),
                                "realized_driver_ic": realized, "horizon": _H, "mom_k": _K,
                                "driver": "10d sum of z(Δ20 REL_BREADTH) − z(Δ20 IN10Y)"}
    return sheet, pd.DatetimeIndex(dates, name="Date"), lpe, us10


def synthetic_sheet(seed: int = 0, n_rows: int = 4000, start: str = "2010-01-01",
                    planted_ic: float = 0.0, *, date_fmt: str = "%Y-%m-%d") -> pd.DataFrame:
    """The Aarambh sheet: DATE (strings in ``date_fmt``, ISO by default) + NIFTY50_PE + the
    14 DEFAULT_PREDICTORS. Deterministic per (seed, n_rows, start, planted_ic)."""
    return _sheet_core(seed, n_rows, start, planted_ic, date_fmt)[0]


def _ohlc_from_close(rng: np.random.Generator, close: np.ndarray, idx: pd.DatetimeIndex,
                     vol_mu: float = 13.0) -> pd.DataFrame:
    n = len(close)
    prev = np.concatenate([[close[0]], close[:-1]])
    op = prev * np.exp(rng.normal(0.0, 0.004, n))
    hi = np.maximum(op, close) * np.exp(np.abs(rng.normal(0.0, 0.006, n)))
    lo = np.minimum(op, close) * np.exp(-np.abs(rng.normal(0.0, 0.006, n)))
    vol = np.round(rng.lognormal(vol_mu, 0.4, n))
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": vol},
                        index=pd.DatetimeIndex(idx, name="Date"))


def build_world(seed: int = 0, n_rows: int = 4000, start: str = "2010-01-01", planted_ic: float = 0.0, *,
                n_constituents: int = 12, late_lister: bool = True, history_years: int = HISTORY_YEARS,
                date_fmt: str = "%Y-%m-%d", pad_prelisting: bool = False) -> SyntheticWorld:
    """Build the full deterministic world (see module docstring)."""
    if n_constituents < 10:
        raise ValueError("n_constituents must be ≥ 10")
    sheet, dates, lpe, us10 = _sheet_core(seed, n_rows, start, planted_ic, date_fmt)
    _, r_eps, r_con, r_mac = _streams(seed)
    n = len(dates)

    # ^NSEI = PE × EPS (EPS: 10%/yr trend + slow AR noise) over the full sheet span.
    leps = np.log(6000.0 / 22.0) + 0.10 * np.arange(n) / 252.0 + _ar1(r_eps, n, 0.0, 0.999, 0.0008)
    lidx = lpe + leps
    index_close = np.exp(lidx)
    index_ohlcv = {INDEX_SYMBOL: _ohlc_from_close(r_eps, index_close, dates, vol_mu=19.0)}

    # Production window rule: 1 January of (last year − history_years), clipped to the sheet.
    w0 = max(pd.Timestamp(year=dates[-1].year - int(history_years), month=1, day=1), dates[0])
    cal = dates[dates >= w0]
    pos0 = int(np.searchsorted(dates.asi8, cal.asi8[0]))
    r_mkt = np.diff(lidx)[pos0 - 1:] if pos0 > 0 else np.concatenate([[0.0], np.diff(lidx)])
    r_mkt = r_mkt[:len(cal)]

    symbols = [f"SYN{i:02d}.NS" for i in range(n_constituents)]
    ohlcv: dict[str, pd.DataFrame] = {}
    late = symbols[-1] if late_lister else None
    for i, sym in enumerate(symbols):
        beta = r_con.uniform(0.6, 1.4)
        idio = r_con.normal(0.0001, r_con.uniform(0.008, 0.018), len(cal))
        close = r_con.uniform(80.0, 3000.0) * np.exp(np.cumsum(beta * r_mkt + idio))
        df = _ohlc_from_close(r_con, close, cal)
        if sym == late:
            list_at = int(0.6 * len(cal))
            listed = df.iloc[list_at:]
            df = listed.reindex(cal) if pad_prelisting else listed
        ohlcv[sym] = df

    # Macro panel: 3 latent factor random walks + idiosyncratic walks; ^TNX = sheet US10Y.
    nm = len(cal)
    f = np.cumsum(r_mac.normal(0.0, 0.004, (nm, 3)), axis=0)
    macro = {}
    for tkr, lvl, sd, scale in MACRO_TICKERS:
        load = r_mac.normal(0.0, 1.0, 3) * scale
        macro[tkr] = lvl * np.exp(f @ load * 0.5 + np.cumsum(r_mac.normal(0.0, sd, nm)))
    macro["^TNX"] = us10[pos0:pos0 + nm] + r_mac.normal(0.0, 0.005, nm)
    macro_df = pd.DataFrame(macro, index=pd.DatetimeIndex(cal, name="Date"))
    macro_df = macro_df[sorted(macro_df.columns)]

    return SyntheticWorld(seed=int(seed), sheet=sheet, dates=dates, constituents=symbols, ohlcv=ohlcv,
                          macro=macro_df, index_ohlcv=index_ohlcv, late_lister=late,
                          params={"n_rows": n_rows, "start": start, "planted_ic": planted_ic,
                                  "n_constituents": n_constituents, "history_years": history_years,
                                  "window_start": str(cal[0].date()), "date_fmt": date_fmt})


def synthetic_world(seed: int = 0, n_rows: int = 4000, start: str = "2010-01-01", planted_ic: float = 0.0,
                    **kwargs):
    """``(sheet_df, SyntheticSources)`` for the same deterministic world. ``kwargs`` go to
    :func:`build_world` (n_constituents, late_lister, history_years, date_fmt, pad_prelisting)
    and to :class:`data.sources.SyntheticSources` (empty, drop_symbols)."""
    from data.sources import SyntheticSources
    src_kw = {k: kwargs.pop(k) for k in ("empty", "drop_symbols") if k in kwargs}
    world = build_world(seed, n_rows, start, planted_ic, **kwargs)
    return world.sheet.copy(), SyntheticSources(world=world, **src_kw)
