"""prepare_sheet — the Aarambh sheet → engine inputs (pure; no I/O, no Streamlit).

Order (each step is an audit item):
  1. ``parse_dates`` on the selected date column — a date is REQUIRED (A5-01/13: no
     dayfirst guessing, no "None" date path that mapped every row to 1970-01-01).
  2. drop unparseable-date rows, stable sort by date, de-duplicate dates (keep last).
  3. numeric coercion; forward-fill ONLY (no bfill look-ahead).
  4. non-IST sheet columns (US yields, ``NON_IST_SHEET_COLUMNS``) lagged one row — their
     close for date D prints after the NSE close of D (A2-04/R5).
  5. predictor history guard: a predictor still NaN in the last ``min_data_points`` rows
     is dropped; the effective start and the predictor bounding it are reported (A1-18).
  6. predictive mode: X = k-day predictor momentum, y = forward h-day log change of the
     target, ``label_mask`` False on the unrealized tail, ``price`` = target level.
  7. row requirement ``min_train + h + min_oos_rows`` (with a clear error) and a content
     hash of the prepared inputs (the data half of the app's cache key, A5-02).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.dates import parse_dates
from pipeline.config import PipelineConfig


class InputError(ValueError):
    """The sheet cannot support a run (message is user-facing)."""


@dataclass
class PreparedInputs:
    data: pd.DataFrame                 # RangeIndex; "Date" + target + used features (levels, post-lag)
    dates: pd.DatetimeIndex            # == data["Date"], the Aarambh calendar
    features: tuple[str, ...]          # predictors actually used
    dropped_features: tuple[str, ...]  # removed by the history guard / absent from the sheet
    X: np.ndarray
    y: np.ndarray
    label_mask: np.ndarray | None
    price: np.ndarray | None
    purge: int
    sheet_panel: pd.DataFrame          # ALL numeric sheet columns on the parsed date index (unlagged, ffilled)
    sheet_span: tuple[str, str]        # first / last parsed date of the whole sheet
    effective_start: str
    bounding_predictor: str | None
    lagged_columns: tuple[str, ...]
    rows_raw: int
    rows_used: int
    content_hash: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"rows_raw": self.rows_raw, "rows_used": self.rows_used,
                "sheet_span": list(self.sheet_span),
                "aarambh_span": [str(self.dates[0].date()), str(self.dates[-1].date())],
                "effective_start": self.effective_start, "bounding_predictor": self.bounding_predictor,
                "features": list(self.features), "dropped_features": list(self.dropped_features),
                "lagged_columns": list(self.lagged_columns),
                "label_complete_rows": (int(self.label_mask.sum()) if self.label_mask is not None
                                        else self.rows_used),
                "content_hash": self.content_hash}


def frame_hash(df: pd.DataFrame) -> str:
    """Deterministic sha256 of a frame's values + index + column names."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def required_rows(cfg: PipelineConfig) -> int:
    return int(cfg.min_train) + (int(cfg.horizon) if cfg.forward_mode else 0) + int(cfg.min_oos_rows)


def _date_warnings(col: str, rep: dict) -> list[str]:
    """Human-readable run warnings for what ``parse_dates`` inferred or repaired."""
    out: list[str] = []
    regimes = rep.get("convention_regimes") or []
    if len(regimes) > 1:
        spans = "; ".join(f"sheet rows {r['from_sheet_row']}–{r['to_sheet_row']} {r['convention']}"
                          for r in regimes[:6])
        out.append(f"Date column {col!r} switches day/month convention ({spans}) — proven by "
                   "unambiguous dates; ambiguous dates were read with their stretch's convention.")
    elif rep.get("how", "").startswith("month-first"):
        out.append(f"Date column {col!r} is month-first (US) format — proven: no value fits day-first.")
    drop = rep.get("dropped_out_of_sequence") or []
    if drop:
        ex = "; ".join(f"row {d['sheet_row']} {d['raw']!r}→{d['parsed']}" for d in drop[:10])
        out.append(f"{len(drop)} date cell(s) out of chronological sequence excluded (likely typos — "
                   f"fix them in the sheet): {ex}" + (" …" if len(drop) > 10 else ""))
    reo = rep.get("reordered")
    if reo:
        out.append(f"Date column {col!r} is not in chronological order ({reo['rows_out_of_sequence']} "
                   "rows out of sequence, every date distinct) — rows sorted by date.")
    return out


def prepare_sheet(df: pd.DataFrame, cfg: PipelineConfig) -> PreparedInputs:
    if df is None or len(df) == 0:
        raise InputError("The sheet is empty.")
    if cfg.date_col not in df.columns:
        raise InputError(f"Date column {cfg.date_col!r} not found in the sheet.")
    if cfg.target not in df.columns:
        raise InputError(f"Target column {cfg.target!r} not found in the sheet.")
    warnings: list[str] = []
    absent = [f for f in cfg.features if f not in df.columns]
    if absent:
        warnings.append(f"Predictors not in the sheet (ignored): {', '.join(absent)}")
    features = [f for f in cfg.features if f in df.columns and f != cfg.target and f != cfg.date_col]
    if not features:
        raise InputError("None of the selected predictors is in the sheet.")

    # 1–2. dates
    date_report: dict = {}
    try:
        parsed = parse_dates(df[cfg.date_col], report=date_report)
    except ValueError as e:
        raise InputError(str(e)) from e
    warnings += _date_warnings(cfg.date_col, date_report)
    raw = df.copy()
    raw["__date__"] = parsed.to_numpy()
    n_seq = len(date_report.get("dropped_out_of_sequence") or [])
    n_nat = int(raw["__date__"].isna().sum()) - n_seq
    if n_nat > 0:
        warnings.append(f"{n_nat} row(s) with a blank or unparseable date dropped.")
    raw = raw.dropna(subset=["__date__"]).sort_values("__date__", kind="mergesort")
    n_dup = int(raw["__date__"].duplicated(keep="last").sum())
    if n_dup:
        warnings.append(f"{n_dup} duplicated date row(s) dropped (kept the last).")
        raw = raw[~raw["__date__"].duplicated(keep="last")]
    rows_raw = int(len(raw))
    if rows_raw == 0:
        raise InputError("No row has a parseable date.")
    dates_all = pd.DatetimeIndex(raw["__date__"], name="Date")

    # 3. numeric coercion (all non-date columns, for the macro panel too) + ffill only
    num_cols = [c for c in raw.columns if c not in (cfg.date_col, "__date__")]
    numeric = raw[num_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any()]
    numeric.index = dates_all
    numeric = numeric.ffill()
    sheet_panel = numeric.copy()
    if cfg.target not in numeric.columns:
        raise InputError(f"Target column {cfg.target!r} has no numeric values.")
    features = [f for f in features if f in numeric.columns]

    # 4. lag non-IST predictors one NSE session (the sheet rows ARE the NSE calendar)
    data = numeric[[cfg.target] + features].copy()
    lagged = tuple(c for c in cfg.non_ist_sheet_columns if c in features) if cfg.macro_lag_non_ist else ()
    for c in lagged:
        data[c] = data[c].shift(1)

    # 5. predictor history guard
    tail = data[features].tail(int(cfg.min_data_points))
    too_young = [f for f in features if tail[f].isna().any()]
    if too_young:
        warnings.append(f"Predictors without {cfg.min_data_points} rows of history dropped: {', '.join(too_young)}")
        features = [f for f in features if f not in too_young]
        data = data.drop(columns=too_young)
    if not features:
        raise InputError("No predictor has enough history after cleaning.")
    first_valid = {c: data[c].first_valid_index() for c in [cfg.target] + features}
    data = data.dropna(subset=[cfg.target] + features)
    if data.empty:
        raise InputError("No row has the target and every predictor populated.")
    eff_start = data.index[0]
    # the column whose first real value is latest bounds the effective start (None = the sheet's own start)
    late = {c: pd.Timestamp(v) for c, v in first_valid.items() if v is not None}
    bounding = max(late, key=late.get) if late else None
    if bounding is not None and late[bounding] <= dates_all[0]:
        bounding = None
    # Power disclosure: a single late-starting predictor truncates EVERY row before its
    # first value (the walk-forward needs all predictors). Say what it costs, so the user
    # can weigh it against the statistical power it removes (first real-sheet run: INIRYY
    # cost 5.4 years — the edge test fell from ~300 to ~160 windows). The model is not
    # user-configurable, so the note states the fact only — it names no control.
    if bounding is not None:
        others = [v for c, v in late.items() if c != bounding]
        alt_start = max(others) if others else dates_all[0]
        gained = int(((dates_all >= alt_start) & (dates_all < late[bounding])).sum())
        if gained >= 252:
            warnings.append(
                f"Predictor {bounding} starts {late[bounding].date()} and truncates the usable history: "
                f"without it the data would start {pd.Timestamp(alt_start).date()} (+{gained} rows, "
                f"≈{gained / 252:.1f} years), which would give the edge tests more non-overlapping "
                "windows.")
    if len(data) < int(cfg.min_data_points):
        raise InputError(f"Need {cfg.min_data_points}+ complete rows for the walk-forward; have {len(data)} "
                         f"(effective start {eff_start.date()}"
                         + (f", bounded by {bounding}" if bounding else "") + ").")

    # 6. engine representation
    data = data.reset_index()  # "Date" column, RangeIndex
    if cfg.forward_mode:
        lvl = data[[cfg.target] + features].astype(float)
        tlog = np.log(lvl[cfg.target].where(lvl[cfg.target] > 0))
        fwd = tlog.shift(-int(cfg.horizon)) - tlog
        mom = lvl[features].diff(int(cfg.mom_k)).replace([np.inf, -np.inf], np.nan)
        valid = mom.notna().all(axis=1).to_numpy()
        data = data.loc[valid].reset_index(drop=True)
        X = mom.loc[valid].to_numpy(dtype=np.float64)
        fwd_v = fwd.loc[valid].to_numpy(dtype=np.float64)
        y = np.nan_to_num(fwd_v, nan=0.0)
        label_mask = np.isfinite(fwd_v)
        price = data[cfg.target].to_numpy(dtype=np.float64)
        purge = int(cfg.horizon)
        n_label = int(label_mask.sum())
    else:
        X = data[features].to_numpy(dtype=np.float64)
        y = data[cfg.target].to_numpy(dtype=np.float64)
        label_mask, price, purge = None, None, 0
        n_label = len(data)

    # 7. row requirement + content hash
    need = required_rows(cfg)
    if len(data) < need:
        raise InputError(
            f"Need ≥ {need} usable rows (MIN_TRAIN {cfg.min_train} + horizon {cfg.horizon if cfg.forward_mode else 0} "
            f"+ MIN_OOS {cfg.min_oos_rows}) for a genuine out-of-sample read; have {len(data)} "
            f"({n_label} label-complete) from {data['Date'].iloc[0].date()}.")
    dates = pd.DatetimeIndex(data["Date"], name="Date")
    hashed = data[["Date", cfg.target] + features].copy()
    content_hash = frame_hash(hashed)

    return PreparedInputs(
        data=data, dates=dates, features=tuple(features),
        dropped_features=tuple(absent) + tuple(too_young), X=X, y=y, label_mask=label_mask,
        price=price, purge=purge, sheet_panel=sheet_panel,
        sheet_span=(str(dates_all[0].date()), str(dates_all[-1].date())),
        effective_start=str(pd.Timestamp(eff_start).date()), bounding_predictor=bounding,
        lagged_columns=lagged, rows_raw=rows_raw, rows_used=int(len(data)),
        content_hash=content_hash, warnings=warnings)


def sheet_content_hash(df: pd.DataFrame) -> str:
    """Hash of the RAW sheet (before any selection) — cheap identity for the app's cache key."""
    return frame_hash(df.astype(str))
