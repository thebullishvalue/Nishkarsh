"""PipelineConfig — the ONE run configuration (audit A5-19).

Every knob that changes a run's results is a field here, read from ``core.config``
(the single source of truth) and overridden only by the UI/CLI selections. The
``fingerprint()`` is a sha256 over all fields + the product version: it is the
config half of the app's result-cache key (the other half is the sheet content
hash), so the Intelligence toggle, the target, the predictor set, the date column,
the history window … all invalidate cached results (audit A5-02/03).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

import core.config as C


def history_start_for(today=None, years: int = C.HISTORY_YEARS) -> str:
    """1 January of (year(today) − years) as ISO date — the window slides once a year (A2-06)."""
    t = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    return pd.Timestamp(year=int(t.year) - int(years), month=1, day=1).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class PipelineConfig:
    # ── selections (UI / CLI) ───────────────────────────────────────────────
    target: str = "NIFTY50_PE"
    features: tuple[str, ...] = tuple(C.DEFAULT_PREDICTORS)
    date_col: str = "DATE"
    kappa_enabled: bool = True                  # Intelligence toggle: ON = walk-forward κ (only if lift passes)
    # ── modelling (core.config) ─────────────────────────────────────────────
    forward_mode: bool = bool(C.AARAMBH_FORWARD_SIGNAL)
    horizon: int = int(C.AARAMBH_FWD_HORIZON)
    mom_k: int = int(C.AARAMBH_FWD_MOM_K)
    return_label: str = str(C.CALIBRATION_RETURN_LABEL)
    min_train: int = int(C.MIN_TRAIN_SIZE)
    max_train: int = int(C.MAX_TRAIN_SIZE)
    min_data_points: int = int(C.MIN_DATA_POINTS)
    min_oos_rows: int = int(C.MIN_OOS_ROWS)
    # ── data window / alignment ─────────────────────────────────────────────
    history_years: int = int(C.HISTORY_YEARS)
    history_start: str = field(default_factory=history_start_for)
    index_symbol: str = "^NSEI"
    universe: str = "NIFTY 50"
    macro_lag_non_ist: bool = bool(C.MACRO_LAG_NON_IST)
    non_ist_sheet_columns: tuple[str, ...] = tuple(C.NON_IST_SHEET_COLUMNS)
    # ── macro panel / factors ───────────────────────────────────────────────
    custom_predictors: bool = bool(C.CUSTOM_PREDICTORS_ENABLED)
    pca_passthrough: tuple[str, ...] = tuple(C.PCA_PASSTHROUGH) if C.PCA_PASSTHROUGH_ENABLED else ()
    macro_factor_components: int = int(C.MACRO_FACTOR_COMPONENTS)
    macro_factor_min_columns: int = int(C.MACRO_FACTOR_MIN_COLUMNS)
    aarambh_pca_predictors: bool = bool(C.AARAMBH_PCA_PREDICTORS)
    aarambh_pca_n_components: int = int(C.AARAMBH_PCA_N_COMPONENTS)
    # ── Nirnay ──────────────────────────────────────────────────────────────
    nirnay_msf_length: int = int(C.NIRNAY_MSF_LENGTH)
    nirnay_roc_len: int = int(C.NIRNAY_ROC_LEN)
    nirnay_regime_sensitivity: float = float(C.NIRNAY_REGIME_SENSITIVITY)
    nirnay_base_weight: float = float(C.NIRNAY_BASE_WEIGHT)
    nirnay_mmr_num_vars: int = int(C.NIRNAY_MMR_NUM_VARS)
    # ── precedent ───────────────────────────────────────────────────────────
    precedent_horizons: tuple[int, ...] = tuple(int(h) for h in C.PRECEDENT_HOLD_HORIZONS)

    def __post_init__(self) -> None:
        # normalise sequences to tuples so the dataclass stays hashable / frozen
        for f in ("features", "non_ist_sheet_columns", "pca_passthrough", "precedent_horizons"):
            v = getattr(self, f)
            if not isinstance(v, tuple):
                object.__setattr__(self, f, tuple(v))
        if not self.target:
            raise ValueError("PipelineConfig: target is required")
        if not self.date_col or str(self.date_col) == "None":
            raise ValueError("PipelineConfig: a date column is required (no 'None' date path)")
        if self.target in self.features:
            object.__setattr__(self, "features", tuple(f for f in self.features if f != self.target))
        # Canonical predictor ORDER: results must not depend on the order a user ticked
        # the boxes (ElasticNet's coordinate descent is order-sensitive at its tolerance —
        # red-team T2 measured ΔC up to 10.9 for a reversed list under one cache key).
        object.__setattr__(self, "features", tuple(sorted(dict.fromkeys(self.features))))
        if not self.features:
            raise ValueError("PipelineConfig: at least one predictor is required")
        if self.return_label not in ("nsei", "target", "basket"):
            raise ValueError(f"PipelineConfig: return_label {self.return_label!r} invalid")
        pd.Timestamp(self.history_start)  # raises on garbage

    @classmethod
    def build(cls, *, today=None, **overrides) -> "PipelineConfig":
        """Config from core.config + overrides; ``history_start`` anchored to ``today``'s year."""
        if "history_start" not in overrides:
            overrides["history_start"] = history_start_for(
                today, overrides.get("history_years", C.HISTORY_YEARS))
        return cls(**overrides)

    def replace(self, **changes) -> "PipelineConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    def fingerprint(self) -> str:
        """sha256 over every result-changing field (+ product version), order-independent
        for the predictor set."""
        d = self.to_dict()
        d["features"] = sorted(d["features"])
        d["_version"] = C.VERSION
        # Every core.config constant participates: a tuned threshold (CONV2_*, EDGE_*,
        # BACKTEST_*, AARAMBH_WF_REFIT, …) changes results, so it must change the key.
        d["_core_config"] = {k: getattr(C, k) for k in dir(C) if k.isupper()}
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
