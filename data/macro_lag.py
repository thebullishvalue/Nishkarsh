"""
Nishkarsh v2.0.0 — one-NSE-session lag for non-IST-synchronous market data (audit A2-04).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

The look-ahead it removes: a US / EU / FX / commodity close dated D prints AFTER the NSE
close of D (US 16:00 ET ≈ 01:30–02:30 IST on D+1; London 16:30 ≈ 21:00–22:00 IST on D).
Joining it onto NSE date D hands the engines information that did not exist at the NSE
close. The fix is to lag every such column by one NSE session before it enters either engine.

Classification (``core.config``, the single source of truth):
  * IST-synchronous, never lagged: a column whose name ends with one of
    ``IST_SYNCHRONOUS_SUFFIXES`` (".NS", ".BO" — NSE/BSE-listed instruments) or is listed in
    ``IST_SYNCHRONOUS_COLUMNS`` (the sheet's Indian rates / breadth / valuation columns).
  * Everything else is lagged — US ETFs and yields, ``^TNX``, ``.L``/``.AX``/``.TO`` listings,
    FX (``INR=X`` …), futures (``=F``), and the sheet's ``US02Y/US10Y/US30Y``. Unknown names
    are lagged: being one session late is safe, being one session early is a leak.
  * Datetime-typed columns are labels, not market data, and are passed through untouched.

Apply it to the RAW panel, before deriving features (spreads, PCA factors): a derived
column's name says nothing about which closes it embeds.

Pure: no IO, does not read ``MACRO_LAG_NON_IST`` (the caller decides whether to lag).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.config import IST_SYNCHRONOUS_COLUMNS, IST_SYNCHRONOUS_SUFFIXES

__all__ = ["lag_non_ist", "is_ist_synchronous", "non_ist_columns"]

_IST_COLS = frozenset(str(c).upper() for c in IST_SYNCHRONOUS_COLUMNS)
_IST_SUFFIXES = tuple(str(s).upper() for s in IST_SYNCHRONOUS_SUFFIXES)


def is_ist_synchronous(name: object) -> bool:
    """True when column ``name`` is synchronous with the NSE close (never lagged)."""
    u = str(name).strip().upper()
    return u in _IST_COLS or u.endswith(_IST_SUFFIXES)


def non_ist_columns(columns: Iterable[object]) -> list:
    """The columns :func:`lag_non_ist` would lag (in input order)."""
    return [c for c in columns if not is_ist_synchronous(c)]


def lag_non_ist(panel: pd.DataFrame, *, nse_index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Return a copy of ``panel`` with every non-IST-synchronous column lagged one NSE session.

    ``nse_index=None``: the panel's own rows ARE the NSE session calendar (e.g. the sheet,
        or a panel already aligned to NSE dates). Non-IST columns are shifted down one row;
        IST columns are unchanged. Index and shape are preserved.
    ``nse_index`` given (a DatetimeIndex of NSE sessions; the panel must have a sorted
        DatetimeIndex on any calendar, e.g. the US one): the result is indexed by
        ``nse_index``. IST columns take their as-of value (last observation dated ≤ t);
        non-IST columns take the last observation dated STRICTLY BEFORE t — i.e. the close
        that had printed before the NSE session t, including a close made on an NSE holiday.
        Rows before a column's first eligible observation are NaN.

    The lagged column list is recorded in ``result.attrs["lagged_columns"]``.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("lag_non_ist expects a DataFrame")
    if isinstance(panel.columns, pd.MultiIndex):
        raise TypeError("lag_non_ist expects flat (single-level) columns")
    lag_cols = [c for c in non_ist_columns(panel.columns)
                if not pd.api.types.is_datetime64_any_dtype(panel[c].dtype)]

    if nse_index is None:
        out = panel.copy()
        if lag_cols:
            out[lag_cols] = out[lag_cols].shift(1)
        out.attrs["lagged_columns"] = list(lag_cols)
        return out

    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("with nse_index, the panel must have a DatetimeIndex")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    target = pd.DatetimeIndex(nse_index)
    if not target.is_monotonic_increasing:
        raise ValueError("nse_index must be sorted ascending")

    src_ns = panel.index.as_unit("ns").asi8
    tgt_ns = target.as_unit("ns").asi8
    out = pd.DataFrame(index=target)
    for c in panel.columns:
        col = panel[c]
        ok = col.notna().to_numpy()
        vals, times = col.to_numpy()[ok], src_ns[ok]
        side = "left" if c in lag_cols else "right"     # left → strictly before t
        pos = np.searchsorted(times, tgt_ns, side=side) - 1
        res = pd.Series(np.nan if col.dtype.kind in "fiub" else None, index=target,
                        dtype="float64" if col.dtype.kind in "fiub" else col.dtype)
        has = pos >= 0
        if has.any():
            res.iloc[np.flatnonzero(has)] = vals[pos[has]]
        out[c] = res
    out.attrs["lagged_columns"] = list(lag_cols)
    return out
