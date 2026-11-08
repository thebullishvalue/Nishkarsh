"""
The single date parser for every date column that crosses the IO boundary.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Why this exists (audit A5-01): ``pd.to_datetime(s, dayfirst=True)`` under pandas 3
infers ONE format from the first value. On an ISO column ("2016-01-12") that inferred
format becomes ``%Y-%d-%m``: 2 500 rows → 984 kept, 900 silently mis-dated, row order
lost. Every call site goes through :func:`parse_dates`, which never guesses silently:

1. **ISO-8601** when most values are ISO strings (never day-first).
2. **Numeric d/m/y-style dates** (``15/01/2024``, ``15-01-2024``, ``15.01.24``) are read
   with a convention PROVEN BY THE DATA: a value whose first number exceeds 12 can only
   be day-first, one whose second number exceeds 12 only month-first. If the column
   carries both kinds of evidence (a sheet whose entry locale changed part-way), each
   stretch of rows takes the convention of its own unambiguous dates, and ambiguous
   values are resolved by that local evidence and file order. The regimes are reported.
3. Anything else (``15-Jan-2024`` …) is parsed per element, day-first.
4. **File-order repair, never silent:**
   - a handful of isolated out-of-sequence rows (typos: a wrong year, a swapped entry)
     — the rows outside the longest chronological run of the column — are set to NaT
     and listed with their sheet row number and raw text;
   - a column that is merely in a different order (an unsorted export: many rows out of
     sequence but every date distinct) is accepted and reported — the caller sorts;
   - anything else (out-of-sequence rows whose dates COLLIDE with others — the
     signature of an undetected day/month swap) raises ``ValueError`` naming the rows.
5. More than ``MAX_NAT_FRACTION`` unparseable values → ``ValueError``.

Pass ``report={}`` to receive what was inferred and repaired (the pipeline turns it
into run warnings). Pure: no IO, no global state.
"""

from __future__ import annotations

import re
from bisect import bisect_right

import numpy as np
import pandas as pd

__all__ = ["parse_dates", "is_iso_like", "MAX_NAT_FRACTION", "MAX_OUT_OF_SEQUENCE_FRACTION"]

# Above this share of unparseable non-blank values the column is rejected.
MAX_NAT_FRACTION = 0.01
# Isolated out-of-sequence rows up to this share (min 3 rows) are dropped and reported;
# beyond it the column must be either a clean re-ordering (accepted, sorted by the
# caller) or it is rejected.
MAX_OUT_OF_SEQUENCE_FRACTION = 0.005

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"                       # date
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"  # optional time
    r"(?:Z|[+-]\d{2}:?\d{2})?$"                  # optional offset
)
# a/b/yyyy, a-b-yy, a.b.yyyy (optional trailing time)
_DMY_RE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})(?:[ T].*)?$")
_BLANKS = {"", "nan", "nat", "none", "null", "n/a", "na", "-"}


def is_iso_like(values: pd.Series, threshold: float = 0.5) -> bool:
    """True when more than ``threshold`` of the non-blank values look like ISO-8601 dates
    (the format choice; the NaT budget then decides whether the column is acceptable)."""
    s = values.dropna().astype(str).str.strip()
    s = s[~s.str.lower().isin(_BLANKS)]
    if s.empty:
        return False
    return bool(s.str.match(_ISO_RE).mean() > threshold)


def _sheet_row(pos: int) -> int:
    """Spreadsheet row number of data position ``pos`` (row 1 = header)."""
    return int(pos) + 2


def _mk(y: int, m: int, d: int):
    try:
        return pd.Timestamp(year=y, month=m, day=d)
    except (ValueError, OverflowError):
        return pd.NaT


def _parse_numeric_dmy(s: pd.Series, report: dict) -> pd.Series:
    """Resolve numeric a/b/y dates with a data-proven convention (see module doc)."""
    n = len(s)
    out = [pd.NaT] * n
    cand_d: list = [pd.NaT] * n   # day-first reading
    cand_m: list = [pd.NaT] * n   # month-first reading
    ev = np.zeros(n, dtype=np.int8)   # +1 day-first only, −1 month-first only, 0 ambiguous/none
    for i, v in enumerate(s.tolist()):
        if not isinstance(v, str):
            continue
        mt = _DMY_RE.match(v)
        if not mt:
            continue
        a, b, y = int(mt.group(1)), int(mt.group(2)), int(mt.group(3))
        if y < 100:
            y += 2000 if y < 70 else 1900
        cd, cm = _mk(y, b, a), _mk(y, a, b)
        cand_d[i], cand_m[i] = cd, cm
        if pd.notna(cd) and pd.isna(cm):
            ev[i] = 1
        elif pd.isna(cd) and pd.notna(cm):
            ev[i] = -1
    has_d, has_m = bool((ev == 1).any()), bool((ev == -1).any())

    if not has_m:            # the sheet's native dd/mm/yyyy (or all-ambiguous): day-first
        report["how"] = "day-first"
        return pd.Series(cand_d, index=s.index, dtype="datetime64[ns]")
    if not has_d:            # only month-first evidence: a US-format export
        report["how"] = "month-first (proven: no value fits day-first)"
        return pd.Series(cand_m, index=s.index, dtype="datetime64[ns]")

    # Mixed evidence: each row follows the convention of the nearest unambiguous rows.
    evs = pd.Series(np.where(ev != 0, ev, np.nan), dtype="float64")
    prev_ev = evs.ffill().to_numpy()
    next_ev = evs.bfill().to_numpy()
    last = pd.NaT
    for i in range(n):
        cd, cm = cand_d[i], cand_m[i]
        if ev[i] == 1:
            out[i] = cd
        elif ev[i] == -1:
            out[i] = cm
        elif pd.isna(cd) and pd.isna(cm):
            out[i] = pd.NaT
        elif cd == cm or pd.isna(cm):
            out[i] = cd
        elif pd.isna(cd):
            out[i] = cm
        else:
            p, q = prev_ev[i], next_ev[i]
            regime = p if (np.isfinite(p) and (not np.isfinite(q) or p == q)) else (q if np.isfinite(q) and not np.isfinite(p) else np.nan)
            if np.isfinite(regime):
                out[i] = cd if regime > 0 else cm
            else:   # regime boundary: the reading that keeps file order, nearest to the last date
                ok = [c for c in (cd, cm) if pd.isna(last) or c >= last]
                out[i] = min(ok, key=lambda c: abs((c - last).days)) if (ok and pd.notna(last)) else (ok[0] if ok else cd)
        if pd.notna(out[i]):
            last = out[i]
    # regimes = maximal runs of the resolved convention over the unambiguous rows
    runs, cur, start = [], None, None
    for i in range(n):
        if ev[i] == 0:
            continue
        if cur is None:
            cur, start = int(ev[i]), i
        elif int(ev[i]) != cur:
            runs.append((cur, start, i - 1))
            cur, start = int(ev[i]), i
    if cur is not None:
        runs.append((cur, start, n - 1))
    report["how"] = "mixed day-first / month-first (proven by unambiguous dates)"
    report["convention_regimes"] = [
        {"convention": "day-first" if c > 0 else "month-first",
         "from_sheet_row": _sheet_row(a), "to_sheet_row": _sheet_row(b)} for c, a, b in runs]
    return pd.Series(out, index=s.index, dtype="datetime64[ns]")


def _longest_nondecreasing(vals: np.ndarray) -> np.ndarray:
    """Positions (into ``vals``) of one longest non-decreasing subsequence — O(n log n)."""
    tails: list[int] = []         # tails[k] = index of the smallest tail of a run of length k+1
    tail_vals: list = []
    prev = np.full(len(vals), -1, dtype=np.int64)
    for i, v in enumerate(vals):
        k = bisect_right(tail_vals, v)
        if k > 0:
            prev[i] = tails[k - 1]
        if k == len(tails):
            tails.append(i)
            tail_vals.append(v)
        else:
            tails[k] = i
            tail_vals[k] = v
    keep = []
    j = tails[-1] if tails else -1
    while j >= 0:
        keep.append(j)
        j = int(prev[j])
    return np.array(keep[::-1], dtype=np.int64)


def _check(parsed: pd.Series, raw: pd.Series | None, n_values: int, name: str, how: str,
           max_nat_fraction: float, require_monotonic: bool, report: dict) -> pd.Series:
    n_nat = int(parsed.isna().sum()) - (int(raw.isna().sum()) if raw is not None else 0)
    n_nat = max(n_nat, 0)
    report.setdefault("how", how)
    report["unparseable"] = n_nat
    if n_values and n_nat / n_values > max_nat_fraction:
        raise ValueError(
            f"Date column {name!r}: {n_nat}/{n_values} values unparseable as {how} "
            f"(> {max_nat_fraction:.0%}). Use one consistent format (ISO yyyy-mm-dd recommended)."
        )
    if not require_monotonic:
        return parsed
    pos = np.flatnonzero(parsed.notna().to_numpy())
    if len(pos) < 2:
        return parsed
    ints = parsed.to_numpy()[pos].astype("datetime64[ns]").astype(np.int64)
    d = np.diff(ints)
    if (d >= 0).all() or (d <= 0).all():
        return parsed
    descending = int((d < 0).sum()) > int((d > 0).sum())
    seq = -ints if descending else ints
    keep = _longest_nondecreasing(seq)
    off = np.setdiff1d(np.arange(len(pos)), keep)          # positions (into pos) out of sequence
    rawv = raw.to_numpy() if raw is not None else parsed.astype(str).to_numpy()

    def _row(j: int) -> dict:
        i = int(pos[j])
        return {"sheet_row": _sheet_row(i), "raw": str(rawv[i]),
                "parsed": str(pd.Timestamp(parsed.iloc[i]).date())}

    limit = max(3, int(np.floor(MAX_OUT_OF_SEQUENCE_FRACTION * len(pos))))
    if len(off) <= limit:
        out = parsed.copy()
        out.iloc[pos[off]] = pd.NaT
        report["dropped_out_of_sequence"] = [_row(j) for j in off]
        return out
    vals = parsed.to_numpy()[pos]
    if len(np.unique(vals)) == len(vals):
        report["reordered"] = {"rows_out_of_sequence": int(len(off)),
                               "examples": [_row(j) for j in off[:5]]}
        return parsed       # a clean re-ordering: every date distinct — the caller sorts
    ex = "; ".join(f"sheet row {r['sheet_row']}: {r['raw']!r} → {r['parsed']}" for r in (_row(j) for j in off[:6]))
    raise ValueError(
        f"Date column {name!r} parsed as {how}: {len(off)} rows are out of chronological "
        f"sequence and their dates collide with other rows — the signature of a day/month "
        f"swap, not a re-ordering. First offending rows: {ex}. Fix those cells (ISO "
        "yyyy-mm-dd avoids the ambiguity)."
    )


def parse_dates(series: pd.Series, *, max_nat_fraction: float = MAX_NAT_FRACTION,
                require_monotonic: bool = True, report: dict | None = None) -> pd.Series:
    """Parse a date column. Returns a ``datetime64`` Series aligned to ``series.index``.

    See the module docstring for the rules. Blank cells ('', 'NaN', 'n/a', None) become
    NaT and do not count against the NaT budget. Isolated out-of-sequence rows are
    returned as NaT (listed in ``report["dropped_out_of_sequence"]``); a clean
    re-ordering is returned as-is (``report["reordered"]``); never sorts, never drops
    rows itself — the caller decides. Raises ``ValueError`` for an unparseable or
    swap-corrupted column. ``report`` (optional dict) receives ``how``,
    ``convention_regimes``, ``unparseable`` and the repairs.
    """
    rep = report if report is not None else {}
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    name = str(series.name) if series.name is not None else "<dates>"

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        out = series.copy()
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
        return _check(out, series, int(out.notna().sum()), name, "datetime", max_nat_fraction,
                      require_monotonic, rep)

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
        return _check(out, series.where(series.notna(), None), len(nonnull), name, "datetime",
                      max_nat_fraction, require_monotonic, rep)

    s = series.astype("object").where(series.notna(), None)
    s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    blank = s.map(lambda v: v is None or (isinstance(v, str) and v.lower() in _BLANKS))
    s = s.where(~blank, None)
    n_values = int((~blank).sum())
    raw_for_check = s.where(~blank, None)

    if is_iso_like(s[~blank]):
        how = "ISO-8601"
        out = pd.to_datetime(s, format="ISO8601", errors="coerce", utc=False)
    else:
        sv = s[~blank].astype(str)
        if len(sv) and sv.str.match(_DMY_RE).mean() >= 0.9:
            out = _parse_numeric_dmy(s, rep)
            how = rep.get("how", "day-first")
            # stragglers in another notation (e.g. '15-Jan-2024'): per element, day-first
            miss = out.isna() & ~blank
            if miss.any():
                out = out.copy()
                out[miss] = pd.to_datetime(s[miss], format="mixed", dayfirst=True, errors="coerce")
        else:
            how = "day-first (per element)"
            out = pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    out = out.astype("datetime64[ns]")
    out.name = series.name
    return _check(out, raw_for_check, n_values, name, how, max_nat_fraction, require_monotonic, rep)
