"""run_pipeline — the whole Nishkarsh analysis, headless (contract C-PIPE, ruling R6).

    data (constituents / macro / OHLCV via ``sources``, force explicit)
    → Aarambh walk-forward (Price column set)
    → Nirnay per stock + daily aggregate (failures COUNTED)
    → return label resolved ONCE (^NSEI over the FULL sheet span → target → basket)
    → run_conclusion (κ only when the Intelligence toggle is on)
    → agreement anatomy (descriptive) → precedent (computed ONCE)
    → run report (JSON) → console SYSTEM DIAGNOSTICS rendered from the report.

Everything runs under ``threadpoolctl.threadpool_limits(1)`` (A5-18): the walk-forward is
hundreds of small solves for which one BLAS thread is fastest on every host.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

import core.config as C
from core.logger_config import console as _default_console
from data.sources import FetchResult
from pipeline.config import PipelineConfig
from pipeline.inputs import PreparedInputs, prepare_sheet
from pipeline.macro import MacroPanel, aarambh_pca_inputs, assemble_macro_panel

ProgressFn = Callable[..., None]
PROVENANCE_KEYS = ("constituents", "macro", "ohlcv", "index", "sheet")
SESSION_KEYS = ("engine", "aarambh_ts", "nirnay_daily", "nirnay_constituent_dfs", "anatomy_df",
                "conclusion", "conclusion_frame", "conclusion_events", "precedent", "run_report",
                "run_config", "label_levels")


@dataclass
class PipelineResult:
    config: PipelineConfig
    prepared: PreparedInputs
    engine: Any
    aarambh_ts: pd.DataFrame
    nirnay_daily: pd.DataFrame
    nirnay_constituent_dfs: dict
    nirnay_failures: dict
    anatomy_df: pd.DataFrame
    conclusion: Any                       # convergence.conclusion.ConclusionResult
    precedent: dict
    macro: MacroPanel
    provenance: dict                      # {key: FetchResult | None}
    label_src: str
    run_id: str
    today: str
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    run_report: dict = field(default_factory=dict)
    label_alignment: dict | None = None    # convergence.labels.label_alignment_check
    label_levels: pd.Series | None = None  # the resolved return-label LEVELS (index price) — for display

    @property
    def synthetic(self) -> bool:
        return any(isinstance(r, FetchResult) and "SYNTHETIC" in str(r.note)
                   for r in self.provenance.values())

    def run_config(self) -> dict:
        cfg = self.config
        latest = (self.conclusion.to_dict().get("latest") or {}) if self.conclusion is not None else {}
        prov = {k: (v.to_dict() if isinstance(v, FetchResult) else None) for k, v in self.provenance.items()}
        return {"target": cfg.target, "features": list(self.prepared.features), "date_col": cfg.date_col,
                "forward_mode": cfg.forward_mode, "kappa_enabled": cfg.kappa_enabled,
                "label_src": self.label_src, "as_of": latest.get("date"),
                "history_start": cfg.history_start, "fingerprint": cfg.fingerprint(),
                "provenance": {k: prov.get(k) for k in PROVENANCE_KEYS},
                "warnings": list(self.warnings)}

    def to_session_state(self) -> dict:
        """EXACTLY the C-PIPE keys (UI contract C-UI)."""
        out = {"engine": self.engine, "aarambh_ts": self.aarambh_ts, "nirnay_daily": self.nirnay_daily,
               "nirnay_constituent_dfs": self.nirnay_constituent_dfs, "anatomy_df": self.anatomy_df,
               "conclusion": self.conclusion.to_dict(), "conclusion_frame": self.conclusion.frame,
               "conclusion_events": self.conclusion.events, "precedent": self.precedent,
               "run_report": self.run_report, "run_config": self.run_config(),
               "label_levels": self.label_levels}
        assert tuple(out) == SESSION_KEYS
        return out


# ── helpers ──────────────────────────────────────────────────────────────────
def _call(fn, *args, force=False):
    """Call a DataSources method with an explicit ``force`` (protocol: constituents() may lack it)."""
    try:
        return fn(*args, force=force)
    except TypeError:
        return fn(*args)


def _as_fetch(x, empty_value) -> FetchResult:
    if isinstance(x, FetchResult):
        return x
    return FetchResult(x if x is not None else empty_value, "live" if x is not None else "empty")


def _provenance_warnings(prov: dict) -> list[str]:
    out = []
    for k, r in prov.items():
        if not isinstance(r, FetchResult):
            continue
        if r.provenance in ("stale", "fallback", "empty"):
            out.append(f"{k}: provenance {r.provenance.upper()}" + (f" (as of {r.as_of})" if r.as_of else "")
                       + (f" — {r.note}" if r.note else ""))
        if r.missing:
            m = ", ".join(map(str, r.missing[:8])) + (" …" if len(r.missing) > 8 else "")
            out.append(f"{k}: {len(r.missing)} missing ({m})")
    return out


def _nirnay_one(sym, ohlcv_df, macro: MacroPanel, cfg: PipelineConfig):
    from engines.nirnay import run_full_analysis
    merged = ohlcv_df.copy()
    if macro.columns:
        merged = merged.join(macro.panel[macro.columns], how="left")
        merged[macro.columns] = merged[macro.columns].ffill()
    res, _ = run_full_analysis(merged, length=cfg.nirnay_msf_length, roc_len=cfg.nirnay_roc_len,
                               regime_sensitivity=cfg.nirnay_regime_sensitivity,
                               base_weight=cfg.nirnay_base_weight, macro_columns=list(macro.columns),
                               num_vars=cfg.nirnay_mmr_num_vars)
    return res


def compute_precedent(ts_rows: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Analogs + base rates computed ONCE per run (A4-16). ``ts_rows``: engine ts with RangeIndex,
    ``Date`` and ``Price`` columns. Returns {"analogs", "summary", "hero"}."""
    from analytics.analogs import find_similar_periods, summarize_forward
    holds = tuple(int(h) for h in cfg.precedent_horizons)
    analogs = find_similar_periods(ts_rows, cfg.target, hold_horizons=holds, mom_window=int(cfg.mom_k))
    summary = summarize_forward(analogs, holds, ts=ts_rows) if analogs else {}
    # The precedent read is graded at the SAME horizon as the conclusion (both modes).
    hp = int(cfg.horizon)
    if not summary.get(hp):
        hp = int(max(holds))
    row = summary.get(hp)
    hero = None
    if row and row.get("n"):
        med = float(row["median"])
        hero = {"horizon": hp, "median": med, "positive_pct": float(row["positive_pct"]), "n": int(row["n"]),
                "dir": 1 if med > 0 else -1 if med < 0 else 0,
                "unconditional_positive_pct": row.get("unconditional_positive_pct"),
                "binomial_p": row.get("binomial_p")}
    return {"analogs": list(analogs or []), "summary": {int(k): v for k, v in summary.items()}, "hero": hero}


# ── the pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(sheet_df: pd.DataFrame, cfg: PipelineConfig, sources, *, progress: ProgressFn | None = None,
                 log=None, force: bool = False, sheet_info: FetchResult | None = None,
                 today=None) -> PipelineResult:
    """Run everything. ``force`` re-fetches market data (Refresh); ``sheet_info`` is the sheet's
    FetchResult (provenance only — the frame is ``sheet_df``); ``today`` anchors the fetch end
    and staleness (default: now)."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _run(sheet_df, cfg, sources, progress=progress, log=log or _default_console,
                    force=bool(force), sheet_info=sheet_info, today=today)


def _run(sheet_df, cfg, sources, *, progress, log, force, sheet_info, today) -> PipelineResult:
    from engines.aarambh import FairValueEngine
    from engines.nirnay import aggregate_constituent_timeseries
    from convergence.conclusion import run_conclusion
    from convergence.cross_validator import compute_agreement_anatomy
    from convergence.labels import build_index_return_levels, label_alignment_check, resolve_return_levels

    prog = progress or (lambda pct, title, sub="": None)
    T: dict[str, float] = {}
    t_all = time.perf_counter()
    today_ts = (pd.Timestamp.today() if today is None else pd.Timestamp(today)).normalize()
    run_id = log.begin_run()
    warnings: list[str] = []

    # ── 0. inputs ────────────────────────────────────────────────────────────
    t = time.perf_counter()
    prepared = prepare_sheet(sheet_df, cfg)          # raises InputError with a user-facing message
    warnings += prepared.warnings
    T["inputs"] = time.perf_counter() - t
    log.header("NISHKARSH — Unified Analysis", f"v{C.VERSION}")
    log.main_header("ANALYSIS CONFIGURATION", {
        "Run ID": run_id, "Target": cfg.target, "Predictors": f"{len(prepared.features)} columns",
        "Rows": f"{prepared.rows_used} used / {prepared.rows_raw} in sheet ({prepared.sheet_span[0]} → {prepared.sheet_span[1]})",
        "Effective start": prepared.effective_start + (f" (bounded by {prepared.bounding_predictor})"
                                                       if prepared.bounding_predictor else ""),
        "Intelligence (κ)": "ON" if cfg.kappa_enabled else "OFF (κ ≡ 1, pure Aarambh)",
        "Fingerprint": cfg.fingerprint()[:12]})
    sheet_last = pd.Timestamp(prepared.sheet_span[1])
    age = int((today_ts - sheet_last.normalize()).days)
    if age > int(C.STALENESS_DAYS):
        warnings.append(f"Stale sheet: last row {sheet_last.date()} is {age} days old.")

    # ── 1. data acquisition ──────────────────────────────────────────────────
    t = time.perf_counter()
    log.start_phase("DATA ACQUISITION", 1, 5)
    prog(2, "Fetching Constituents", "index membership")
    start, end = pd.Timestamp(cfg.history_start), today_ts
    r_cons = _as_fetch(_call(sources.constituents, force=force), [])
    constituents = list(r_cons.value or [])
    log.item("Constituents", f"{len(constituents)} · {r_cons.provenance}" + (f" · {r_cons.note}" if r_cons.note else ""))
    if constituents:
        log.item("Symbols", ", ".join(constituents[:3]) + (" …" if len(constituents) > 3 else ""))
    prog(5, "Fetching Market Data", "macro panel · constituent OHLCV")
    with ThreadPoolExecutor(max_workers=2) as pool:   # independent batches; force passed explicitly
        f_mac = pool.submit(_call, sources.macro, start, end, force=force)
        f_ohl = pool.submit(_call, sources.ohlcv, constituents, start, end, force=force) if constituents else None
        r_macro = _as_fetch(f_mac.result(), pd.DataFrame())
        r_ohlcv = _as_fetch(f_ohl.result(), {}) if f_ohl is not None else FetchResult({}, "empty", None, [], "no constituents")
    ohlcv = dict(r_ohlcv.value or {})
    log.item("History window", f"{start.date()} → {end.date()} (anchored 1 Jan; survivorship-biased to current members)")
    log.item("Macro", f"{getattr(r_macro.value, 'shape', (0, 0))[1]} series · {r_macro.provenance}")
    log.item("OHLCV", f"{len(ohlcv)}/{len(constituents)} constituents · {r_ohlcv.provenance}")
    prog(15, "Assembling Macro Panel", "lag non-IST closes · custom features · factors")
    nse_index = prepared.sheet_panel.index
    for df_ in ohlcv.values():
        nse_index = nse_index.union(pd.DatetimeIndex(df_.index))
    macro = assemble_macro_panel(r_macro.value, prepared.sheet_panel, nse_index, cfg, end=end, log=log)
    warnings += macro.warnings
    ms = macro.summary()
    log.item("Macro panel", f"{ms['raw_columns']} raw ({ms['yf_columns']} yfinance + {len(ms['sheet_columns'])} sheet"
             f" + {ms['custom_features']} custom) · {ms['lagged_non_ist']} non-IST lagged one session")
    if macro.dropped_duplicates:
        log.item("Duplicate rates dropped", ", ".join(macro.dropped_duplicates))
    log.item("MMR inputs", f"{len(macro.columns)} " + ("causal PCA factors" if macro.factors else "raw macros")
             + (f" (passthrough {', '.join(macro.passthrough)})" if macro.passthrough else ""))
    log.end_phase("DATA ACQUISITION")
    T["data"] = time.perf_counter() - t
    prog(20, "Data Acquisition Complete", f"{len(ohlcv)} constituents · {len(macro.columns)} macro inputs")

    # ── 2. Aarambh ───────────────────────────────────────────────────────────
    t = time.perf_counter()
    log.start_phase("AARAMBH ENGINE", 2, 5)
    X, names = prepared.X, list(prepared.features)
    pca = aarambh_pca_inputs(prepared, macro, cfg)
    if pca is not None:
        X, names, _load = pca
        log.item("Predictor PCA", f"{len(names)} causal factors (levels mode; macros enter once they have real history)")
    log.item("Mode", (f"PREDICTIVE · {cfg.horizon}d forward Δlog({cfg.target}) from {cfg.mom_k}d momentum"
                      if cfg.forward_mode else "LEVELS · valuation context — falsified as a forecaster"))
    log.item("Train window", f"{cfg.min_train}→{cfg.max_train} label-complete rows · purge {prepared.purge}")
    last = [0.0]

    def _aar_progress(p, msg):
        prog(int(20 + float(p) * 20), "Running Aarambh Engine", str(msg))
        now = time.time()
        if p >= 1.0 or now - last[0] >= 4.0:
            last[0] = now
            log.detail(f"Walk-forward {float(p) * 100:3.0f}% · {msg}")

    engine = FairValueEngine()
    engine.fit(X, prepared.y, feature_names=names, progress_callback=_aar_progress,
               forward_signal=cfg.forward_mode, purge=prepared.purge, price=prepared.price,
               label_mask=prepared.label_mask, min_train=cfg.min_train, max_train=cfg.max_train)
    if len(engine.ts_data) != len(prepared.data):
        raise RuntimeError(f"Aarambh output rows {len(engine.ts_data)} != inputs {len(prepared.data)}")
    engine.ts_data["Price"] = prepared.data[cfg.target].to_numpy(dtype=np.float64)
    aarambh_ts = engine.ts_data.copy()
    aarambh_ts.index = pd.DatetimeIndex(prepared.dates.to_numpy())
    aarambh_ts["Date"] = aarambh_ts.index
    n_valid = int(aarambh_ts["Valid"].sum()) if "Valid" in aarambh_ts.columns else 0
    log.item("Valid forecasts", f"{n_valid}/{len(aarambh_ts)} · wf {engine.wf_meta}")
    if n_valid == 0:
        warnings.append("Aarambh produced no valid out-of-sample forecast.")
    log.end_phase("AARAMBH ENGINE")
    T["aarambh"] = time.perf_counter() - t
    prog(40, "Aarambh Engine Complete", f"{n_valid} valid forecasts")

    # ── 3. Nirnay ────────────────────────────────────────────────────────────
    t = time.perf_counter()
    log.start_phase("NIRNAY ENGINE", 3, 5)
    dfs: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    total = len(ohlcv)
    for i, (sym, odf) in enumerate(ohlcv.items()):
        try:
            res = _nirnay_one(sym, odf, macro, cfg)
            if res is None or res.empty:
                failures[sym] = "no genuine bars"
            else:
                dfs[sym] = res
                lr = res.iloc[-1]
                log.detail(f"[{i + 1}/{total}] {sym}: osc={float(lr.get('Unified_Osc', np.nan)):+.1f} "
                           f"[{lr.get('Condition', 'N/A')}] regime={lr.get('Regime', 'N/A')} rows={len(res)}")
        except Exception as e:  # noqa: BLE001 — counted and reported, never silently swallowed
            failures[sym] = f"{type(e).__name__}: {e}"
            log.failure(sym, str(e))
        prog(int(45 + (i + 1) / max(total, 1) * 30), f"Analyzing {sym}", f"{i + 1}/{total}")
    nirnay_daily = aggregate_constituent_timeseries(dfs) if dfs else pd.DataFrame()
    if failures:
        warnings.append(f"Nirnay: {len(failures)}/{total} constituents failed ({', '.join(list(failures)[:5])}).")
    log.item("Analyzed", f"{len(dfs)}/{total} ({len(failures)} failed) · {len(nirnay_daily)} trading days")
    log.end_phase("NIRNAY ENGINE")
    T["nirnay"] = time.perf_counter() - t
    prog(75, "Nirnay Engine Complete", f"{len(dfs)} stocks · {len(nirnay_daily)} days")

    # ── 4. conclusion ────────────────────────────────────────────────────────
    t = time.perf_counter()
    log.start_phase("CONCLUSION", 4, 5)
    prog(78, "Resolving Return Label", f"{cfg.index_symbol} over the full sheet span")
    r_index = _as_fetch(_call(sources.index_levels, cfg.index_symbol, pd.Timestamp(prepared.sheet_span[0]), end,
                              force=force), pd.Series(dtype="float64"))
    target_levels = pd.Series(prepared.data[cfg.target].to_numpy(dtype=np.float64), index=prepared.dates)
    from core.config import INDEX_SHEET_COLUMNS
    idx_col = next((c for c in INDEX_SHEET_COLUMNS if c in prepared.sheet_panel.columns), None)
    if (r_index.value is None or len(r_index.value) == 0) and idx_col is not None:
        sheet_idx = prepared.sheet_panel[idx_col].dropna()
        sheet_idx = sheet_idx[sheet_idx > 0]
        if len(sheet_idx):
            r_index = FetchResult(sheet_idx, "live", as_of=str(sheet_idx.index[-1].date()),
                                  note=f"sheet column {idx_col} (IST index level) — {cfg.index_symbol} unavailable")
            warnings.append(f"{cfg.index_symbol} unavailable — the return label is the sheet's {idx_col} column.")
    basket = build_index_return_levels(ohlcv) if ohlcv else None
    levels, label_src, lw = resolve_return_levels(cfg.return_label, r_index.value, target_levels, basket,
                                                  target_name=cfg.target)
    warnings += lw
    if str(getattr(r_index, "note", "")).startswith("sheet column") and str(label_src).startswith("NIFTY 50 index"):
        label_src = f"NIFTY 50 index (sheet {idx_col} column) fwd return"
    # Timestamp-alignment guard: an external label must co-move with the sheet target
    # at lag 0 (a one-session misstamp can fabricate an edge — red-team T2).
    label_alignment = None
    if len(levels) and not str(label_src).startswith("Target"):
        label_alignment = label_alignment_check(target_levels, levels)
        if label_alignment.get("available") and not label_alignment.get("aligned"):
            warnings.append(
                f"LABEL TIMESTAMP MISALIGNMENT suspected: daily returns of the return label "
                f"({label_src}) co-move with {cfg.target} most strongly at lag "
                f"{label_alignment['best_lag']:+d} sessions, not 0 — edge / backtest read-outs "
                f"may be invalid until the label's dates are corrected.")
    log.item("Return label", f"{label_src or 'NONE'} · {len(levels)} levels"
             + (f" ({levels.index[0].date()} → {levels.index[-1].date()})" if len(levels) else ""))
    prog(82, "Computing Conclusion", "causal z · walk-forward κ · edge ledger")
    conclusion = run_conclusion(aarambh_ts, nirnay_daily, levels, forward_mode=cfg.forward_mode,
                                h=int(cfg.horizon), kappa_enabled=cfg.kappa_enabled, label_src=label_src)
    try:
        anatomy = compute_agreement_anatomy(aarambh_ts, nirnay_daily) if not nirnay_daily.empty else pd.DataFrame()
    except Exception as e:  # noqa: BLE001 — descriptive only
        anatomy = pd.DataFrame()
        warnings.append(f"Agreement anatomy failed: {e}")
    try:   # the Agreement Oscillator — measured every run (docs/research/AGREEMENT_OSCILLATOR_*.md)
        from convergence.conclusion import attach_oscillator
        attach_oscillator(conclusion, anatomy, levels, int(cfg.horizon))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Agreement oscillator failed: {e}")
    log.end_phase("CONCLUSION")
    T["conclusion"] = time.perf_counter() - t

    # ── 5. precedent + report ────────────────────────────────────────────────
    t = time.perf_counter()
    log.start_phase("PRECEDENT & REPORT", 5, 5)
    prog(92, "Finding Precedents", "Mahalanobis analogs")
    try:
        precedent = compute_precedent(aarambh_ts.reset_index(drop=True), cfg)
    except Exception as e:  # noqa: BLE001
        precedent = {"analogs": [], "summary": {}, "hero": None}
        warnings.append(f"Precedent failed: {e}")
    T["precedent"] = time.perf_counter() - t
    prov = {"constituents": r_cons, "macro": r_macro, "ohlcv": r_ohlcv, "index": r_index, "sheet": sheet_info}
    warnings += _provenance_warnings(prov)
    result = PipelineResult(config=cfg, prepared=prepared, engine=engine, aarambh_ts=aarambh_ts,
                            nirnay_daily=nirnay_daily, nirnay_constituent_dfs=dfs, nirnay_failures=failures,
                            anatomy_df=anatomy, conclusion=conclusion, precedent=precedent, macro=macro,
                            label_alignment=label_alignment, label_levels=levels, provenance=prov, label_src=label_src, run_id=run_id, today=str(today_ts.date()),
                            warnings=warnings)
    if result.synthetic:
        result.warnings.insert(0, "SYNTHETIC DEMO DATA — not market data; no conclusion here is about the real market.")
    log.end_phase("PRECEDENT & REPORT")
    T["total"] = time.perf_counter() - t_all
    result.timings = {k: round(float(v), 3) for k, v in T.items()}
    from pipeline.report import build_run_report, render_diagnostics
    result.run_report = build_run_report(result)
    render_diagnostics(result.run_report, log)
    log.end_run()
    prog(100, "Analysis Complete", (result.run_report.get("conclusion") or {}).get("latest", {}).get("direction", ""))
    return result
