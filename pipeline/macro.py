"""assemble_macro_panel — Nirnay's macro panel on the NSE calendar (pure; no I/O).

  macro FetchResult (yfinance, US/union calendar)  ∪  sheet bond + custom-feature columns
  (canonical parsed dates — never a hard-coded "DATE" column, A5-13)
  → de-duplicate rate series the sheet already carries (US10Y vs ^TNX, US30Y vs ^TYX)
  → ``lag_non_ist(..., nse_index=...)`` calendar mode: non-IST closes enter session t only
    if they printed strictly before t (A2-04/R5)
  → custom engineered features (causal) on the full span
  → restricted to the stock window [history_start, end] (anchored 1 Jan, A2-06)
  → causal macro-factor PCA when the panel is wide (as in v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import core.config as C
from data.macro_lag import lag_non_ist
from pipeline.config import PipelineConfig


@dataclass
class MacroPanel:
    panel: pd.DataFrame                    # what Nirnay's MMR joins (factors or raw), NSE calendar, window
    columns: list[str]
    raw: pd.DataFrame                      # pre-PCA panel (window) — Aarambh levels-mode PCA input
    n_raw: int = 0
    factors: bool = False
    passthrough: list[str] = field(default_factory=list)
    custom_features: list[str] = field(default_factory=list)
    leakage_excluded: frozenset = frozenset()
    dropped_duplicates: list[str] = field(default_factory=list)
    lagged_columns: list[str] = field(default_factory=list)
    yf_columns: int = 0
    sheet_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"yf_columns": self.yf_columns, "sheet_columns": list(self.sheet_columns),
                "raw_columns": self.n_raw, "custom_features": len(self.custom_features),
                "custom_leakage_excluded_from_aarambh": sorted(self.leakage_excluded),
                "dropped_duplicate_rates": list(self.dropped_duplicates),
                "lagged_non_ist": len(self.lagged_columns), "factors": self.factors,
                "mmr_columns": len(self.columns), "passthrough": list(self.passthrough),
                "rows": int(len(self.panel)),
                "span": ([str(self.panel.index[0].date()), str(self.panel.index[-1].date())]
                         if len(self.panel) else None)}


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None) \
        if getattr(out.index, "tz", None) is not None else pd.DatetimeIndex(pd.to_datetime(out.index))
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def assemble_macro_panel(macro_df: pd.DataFrame | None, sheet_panel: pd.DataFrame,
                         nse_index: pd.DatetimeIndex, cfg: PipelineConfig, *,
                         end=None, log=None) -> MacroPanel:
    warn: list[str] = []
    macro = _clean_index(macro_df)
    yf_cols = int(macro.shape[1])
    sheet = _clean_index(sheet_panel)
    bond_cols = [c for c in C.MACRO_SHEET_BOND_COLUMNS if c in sheet.columns]
    extra_cols = [c for c in C.CUSTOM_FEATURE_SHEET_COLUMNS
                  if c in sheet.columns and c not in bond_cols] if cfg.custom_predictors else []

    dropped = []
    for sheet_col, tkr in C.MACRO_DUPLICATE_RATES.items():
        if sheet_col in bond_cols and tkr in macro.columns:
            macro = macro.drop(columns=[tkr])
            dropped.append(f"{tkr} (= sheet {sheet_col})")
    macro = macro.drop(columns=[c for c in macro.columns if c in bond_cols or c in extra_cols])

    combined = macro.join(sheet[bond_cols + extra_cols], how="outer") if not macro.empty \
        else sheet[bond_cols + extra_cols].copy()
    combined = combined.sort_index()
    nse = pd.DatetimeIndex(nse_index).sort_values().unique()
    if combined.empty or len(nse) == 0:
        empty = pd.DataFrame(index=nse)
        return MacroPanel(panel=empty, columns=[], raw=empty, yf_columns=yf_cols,
                          sheet_columns=bond_cols, dropped_duplicates=dropped,
                          warnings=["Macro panel empty — Nirnay runs on price only."])

    if cfg.macro_lag_non_ist:
        lagged = lag_non_ist(combined, nse_index=nse)
        lag_cols = list(lagged.attrs.get("lagged_columns", []))
    else:  # as-of alignment without the lag
        lagged = combined.reindex(combined.index.union(nse)).ffill().reindex(nse)
        lag_cols = []
    lagged.index.name = "Date"

    custom_new: list[str] = []
    leakage: frozenset = frozenset()
    if cfg.custom_predictors:
        try:
            from analytics.custom_features import build_custom_features, AARAMBH_LEAKAGE_EXCLUDE
            leakage = frozenset(AARAMBH_LEAKAGE_EXCLUDE)
            cf = build_custom_features(lagged)
            custom_new = [c for c in cf.columns if c not in lagged.columns]
            if custom_new:
                lagged = lagged.join(cf[custom_new], how="left")
        except Exception as e:  # noqa: BLE001 — degrade, but say so
            warn.append(f"Custom features failed: {e}")
    panel = lagged.drop(columns=extra_cols)  # sheet extras feed the custom features only (v1 behaviour)

    start = pd.Timestamp(cfg.history_start)
    stop = pd.Timestamp(end) if end is not None else panel.index.max()
    panel = panel.loc[(panel.index >= start) & (panel.index <= stop)]
    panel = panel.loc[:, panel.notna().any()]
    raw = panel.copy()
    cols = list(panel.columns)
    n_raw = len(cols)
    factors = False
    passthrough: list[str] = []
    if n_raw > int(cfg.macro_factor_min_columns):
        try:
            from analytics.factors import build_causal_macro_factors
            fac = build_causal_macro_factors(panel, n_components=int(cfg.macro_factor_components),
                                             passthrough=list(cfg.pca_passthrough) or None,
                                             stationarize=False, return_loadings=False)
            if isinstance(fac, pd.DataFrame) and not fac.empty and fac.shape[1] >= 2:
                panel, cols, factors = fac, list(fac.columns), True
                passthrough = [c for c in cfg.pca_passthrough if c in cols]
            else:
                warn.append(f"Macro-factor PCA unavailable — MMR uses {n_raw} raw macros.")
        except Exception as e:  # noqa: BLE001
            warn.append(f"Macro-factor PCA failed ({e}) — MMR uses {n_raw} raw macros.")
    return MacroPanel(panel=panel, columns=cols, raw=raw, n_raw=n_raw, factors=factors,
                      passthrough=passthrough, custom_features=custom_new, leakage_excluded=leakage,
                      dropped_duplicates=dropped, lagged_columns=lag_cols, yf_columns=yf_cols,
                      sheet_columns=bond_cols, warnings=warn)


def aarambh_pca_inputs(prepared, macro: MacroPanel, cfg: PipelineConfig):
    """LEVELS mode + ``AARAMBH_PCA_PREDICTORS`` only: causal PCA of (sheet predictors + raw macro
    panel) as Aarambh's design matrix. Macro columns enter the expanding fit only once they have
    real history (``build_causal_macro_factors`` column-entry rule) — they are NOT flat-filled.
    Returns (X, feature_names, loadings) or None."""
    if cfg.forward_mode or not cfg.aarambh_pca_predictors or macro.raw.empty:
        return None
    from analytics.factors import build_causal_macro_factors
    sheet = prepared.data[list(prepared.features)].reset_index(drop=True)
    mac = macro.raw.drop(columns=[c for c in macro.raw.columns if c in sheet.columns or c in macro.leakage_excluded])
    mac = mac.reindex(mac.index.union(prepared.dates)).ffill().reindex(prepared.dates).reset_index(drop=True)
    comb = pd.concat([sheet, mac], axis=1)
    comb.index = pd.RangeIndex(len(comb))
    ncomp = int(min(cfg.aarambh_pca_n_components, comb.shape[1]))
    fac, load = build_causal_macro_factors(comb, n_components=ncomp, passthrough=list(cfg.pca_passthrough) or None,
                                           stationarize=True, return_loadings=True)
    if fac is None or fac.empty or fac.shape[1] < 2:
        return None
    return fac.fillna(0.0).to_numpy(dtype=np.float64), list(fac.columns), load
