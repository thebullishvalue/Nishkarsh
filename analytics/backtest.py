"""
Cost-aware long/short backtest instrument — pure functions, no IO, no state.

Closes FINDINGS open item (b): "IC → net Sharpe". A position ``p_t ∈ [−1, +1]`` is
decided at the close of day ``t`` (from information ≤ t) and earns the log return
from ``t`` to ``t+1`` onwards. Costs are ``cost_bps`` per unit of one-way notional
traded (``|Δposition|``).

Two accounting modes:

* ``"tranche"`` — ``h`` staggered tranches, each 1/h of capital, each held ``h``
  days: the book on day t is the mean of the last ``h`` decisions, and each day one
  tranche is rolled (trade ``|p_t − p_{t−h}| / h``, which equals the book's own
  ``|Δbook|``). Uses every day's signal; the daily P&L is autocorrelated by
  construction, so the t-statistic is Newey–West with ``h`` lags. Annualization √252.
* ``"nonoverlap"`` — rebalance every ``h`` rows (phase 0); one P&L per h-window,
  ``P&L_k = p_k·R_h − c·|p_k − p_{k−1}|``; plain t-statistic (windows do not overlap).
  Annualization √(252/h).

Sanity: for a forecast with rank IC ρ against h-day returns, gross Sharpe ≈ ρ·√(252/h).
Every float returned is finite or ``None`` (never NaN) so it can go straight into a
JSON report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.causal_stats import newey_west_t

TRADING_DAYS_PER_YEAR = 252.0
BACKTEST_MODES = ("tranche", "nonoverlap")


def _finite_or_none(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _sharpe(x: np.ndarray, periods_per_year: float) -> float | None:
    if len(x) < 2:
        return None
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return _finite_or_none(float(np.mean(x)) / sd * np.sqrt(periods_per_year))


def _empty(mode: str, h: int, cost_bps: float) -> dict:
    return {"mode": mode, "h": int(h), "cost_bps": float(cost_bps),
            "gross_sharpe": None, "net_sharpe": None, "t": None,
            "ann_return": None, "gross_ann_return": None,
            "turnover_per_year": None, "breakeven_bps": None, "n": 0}


def backtest(position: pd.Series, logp: pd.Series, h: int, cost_bps: float,
             mode: str = "tranche") -> dict:
    """Backtest ``position`` (index = decision dates) against log price ``logp``.

    Parameters
    ----------
    position : pd.Series
        Target exposure in [−1, +1] decided at each row's close (clipped; NaN → flat).
    logp : pd.Series
        Log price of the traded label, aligned to ``position.index`` (reindexed;
        rows without a price are excluded from P&L, never forward-filled here).
    h : int
        Holding horizon (rows).
    cost_bps : float
        One-way cost in basis points per unit of notional traded.
    mode : {"tranche", "nonoverlap"}

    Returns
    -------
    dict with keys ``mode, h, cost_bps, gross_sharpe, net_sharpe, t`` (net-of-cost
    t-statistic: Newey–West(h) for tranche, plain for nonoverlap), ``ann_return``
    (net, log-return units per year), ``gross_ann_return``, ``turnover_per_year``
    (one-way notional / yr), ``breakeven_bps`` (one-way cost at which the net
    annual return is zero = gross annual return / turnover; ≤ 0 when the gross
    return is ≤ 0; None when nothing was traded), ``n`` (P&L periods).
    """
    if mode not in BACKTEST_MODES:
        raise ValueError(f"backtest: mode must be one of {BACKTEST_MODES}, got {mode!r}")
    h = int(h)
    if h < 1:
        raise ValueError("backtest: h must be >= 1")
    if position is None or len(position) == 0:
        return _empty(mode, h, cost_bps)
    pos = pd.Series(position, dtype="float64")
    lp = pd.Series(logp, dtype="float64").reindex(pos.index)
    p = pos.clip(-1.0, 1.0).fillna(0.0)
    c = float(cost_bps) / 1e4

    if mode == "nonoverlap":
        k = np.arange(0, len(p), h)
        P = p.to_numpy()[k]
        lpa = lp.to_numpy()
        Rh = np.full(len(k), np.nan)
        ok = k + h < len(lpa)
        Rh[ok] = lpa[k[ok] + h] - lpa[k[ok]]
        dP = np.abs(np.diff(np.concatenate([[0.0], P])))
        gross = P * Rh
        net = gross - c * dP
        valid = np.isfinite(net)
        g, nt, tr = gross[valid], net[valid], dP[valid]
        ppy = TRADING_DAYS_PER_YEAR / h
        if len(nt) < 3:
            return _empty(mode, h, cost_bps)
        sd = float(np.std(nt, ddof=1))
        t_net = float(np.mean(nt) / (sd / np.sqrt(len(nt)))) if sd > 0 else float("nan")
        years = len(tr) / ppy
    else:
        book = p.rolling(h, min_periods=1).sum() / h          # mean of the last h tranches (missing = flat)
        trade = (p - p.shift(h).fillna(0.0)).abs() / h         # one tranche rolled per day
        r1 = (lp.shift(-1) - lp).to_numpy()                    # t → t+1
        gross = book.to_numpy() * r1
        net = gross - c * trade.to_numpy()
        valid = np.isfinite(net)
        g, nt, tr = gross[valid], net[valid], trade.to_numpy()[valid]
        ppy = TRADING_DAYS_PER_YEAR
        if len(nt) < h + 5:
            return _empty(mode, h, cost_bps)
        t_net = newey_west_t(nt, lags=h)
        years = len(tr) / ppy

    turnover = float(tr.sum() / years) if years > 0 else float("nan")
    gross_ann = float(np.mean(g) * ppy)
    net_ann = float(np.mean(nt) * ppy)
    breakeven = gross_ann / turnover * 1e4 if (np.isfinite(turnover) and turnover > 0) else float("nan")
    return {
        "mode": mode, "h": h, "cost_bps": float(cost_bps),
        "gross_sharpe": _sharpe(g, ppy),
        "net_sharpe": _sharpe(nt, ppy),
        "t": _finite_or_none(t_net),
        "ann_return": _finite_or_none(net_ann),
        "gross_ann_return": _finite_or_none(gross_ann),
        "turnover_per_year": _finite_or_none(turnover),
        "breakeven_bps": _finite_or_none(breakeven),
        "n": int(len(nt)),
    }
