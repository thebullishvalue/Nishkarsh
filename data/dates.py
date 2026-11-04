"""
Nishkarsh v2.0.0 — the single date parser for every date column that crosses the IO boundary.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Why this exists (audit A5-01): ``pd.to_datetime(s, dayfirst=True)`` under pandas 3
infers ONE format from the first value. On an ISO column ("2016-01-12") that inferred
format becomes ``%Y-%d-%m``: 2 500 rows → 984 kept, 900 silently mis-dated, row order
lost. Every call site now goes through :func:`parse_dates`, which

1. parses ISO-8601 strings as ISO (never dayfirst),
2. parses everything else day-first (the sheet's native ``dd/mm/yyyy`` / ``dd-Mon-yyyy``),
3. FAILS LOUDLY — ``ValueError`` — when more than 1% of the non-blank values do not
   parse, or when the parsed dates are not monotonic in FILE ORDER (after dropping
   NaT). A mis-parsed day/month swap almost always breaks monotonicity, so this is
   the tripwire for an ambiguous US-format export. Increasing and decreasing files
   are both accepted; the caller decides how to sort.

Pure: no IO, no global state.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = ["parse_dates", "is_iso_like", "MAX_NAT_FRACTION"]

# Above this share of unparseable non-blank values the column is rejected.
MAX_NAT_FRACTION = 0.01

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"                       # date
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"  # optional time
    r"(?:Z|[+-]\d{2}:?\d{2})?$"                  # optional offset
)
_BLANKS = {"", "nan", "nat", "none", "null", "n/a", "na", "-"}


def is_iso_like(values: pd.Series) -> bool:
    """True when ≥ 99% of the non-blank string values look like ISO-8601 dates."""
    s = values.dropna().astype(str).str.strip()
    s = s[~s.str.lower().isin(_BLANKS)]
    if s.empty:
        return False
    return bool(s.str.match(_ISO_RE).mean() >= 1.0 - MAX_NAT_FRACTION)


def _check(parsed: pd.Series, n_values: int, name: str, how: str,
           max_nat_fraction: float, require_monotonic: bool) -> pd.Series:
    n_nat = int(parsed.isna().sum())
    if n_values and n_nat / n_values > max_nat_fraction:
        raise ValueError(
            f"Date column {name!r}: {n_nat}/{n_values} values unparseable as {how} "
            f"(> {max_nat_fraction:.0%}). Use one consistent format (ISO yyyy-mm-dd recommended)."
        )
    if require_monotonic:
        vals = parsed.dropna().to_numpy()
        if len(vals) > 1:
            d = np.diff(vals.astype("datetime64[ns]").astype(np.int64))
            if not ((d >= 0).all() or (d <= 0).all()):
                bad = int(np.argmax(d < 0) if (d > 0).sum() >= (d < 0).sum() else np.argmax(d > 0))
                raise ValueError(
                    f"Date column {name!r} parsed as {how} is not monotonic in file order "
                    f"(first break after row {bad}) — a day/month swap or an unsorted export. "
                    "Use ISO yyyy-mm-dd dates in a chronologically ordered file."
                )
    return parsed


def parse_dates(series: pd.Series, *, max_nat_fraction: float = MAX_NAT_FRACTION,
                require_monotonic: bool = True) -> pd.Series:
    """Parse a date column. Returns a ``datetime64`` Series aligned to ``series.index``.

    ISO-8601 when the values match ISO, else day-first. Blank cells ('', 'NaN', 'n/a',
    None) become NaT and do not count against the NaT budget. Raises ``ValueError`` if
    more than ``max_nat_fraction`` of the non-blank values fail to parse, or (when
    ``require_monotonic``) if the parsed dates are not monotonic in file order after
    dropping NaT. Timezone-aware input is converted to naive wall time. Never sorts,
    never drops rows: the caller decides.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    name = str(series.name) if series.name is not None else "<dates>"

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        out = series.copy()
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
        return _check(out, int(out.notna().sum()), name, "datetime", max_nat_fraction, require_monotonic)

    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
        # Integers would silently become nanoseconds since 1970 (audit A5-13).
        raise ValueError(f"Date column {name!r} is numeric ({series.dtype}); convert it explicitly.")

    # Object column of Timestamp/datetime objects (e.g. pd.read_excel).
    nonnull = series.dropna()
    if len(nonnull) and all(isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year")
                            for v in nonnull.iloc[:50]) and not isinstance(nonnull.iloc[0], str):
        out = pd.to_datetime(series, errors="coerce")
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
        return _check(out, len(nonnull), name, "datetime", max_nat_fraction, require_monotonic)

    s = series.astype("object").where(series.notna(), None)
    s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    blank = s.map(lambda v: v is None or (isinstance(v, str) and v.lower() in _BLANKS))
    s = s.where(~blank, None)
    n_values = int((~blank).sum())

    if is_iso_like(s[~blank]):
        how = "ISO-8601"
        out = pd.to_datetime(s, format="ISO8601", errors="coerce", utc=False)
    else:
        how = "day-first"
        out = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    out.name = series.name
    return _check(out, n_values, name, how, max_nat_fraction, require_monotonic)
