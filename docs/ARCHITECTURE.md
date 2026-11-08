# Nishkarsh — Architecture (2.0)

## 1. What the system computes

```
            Google Sheet (NIFTY50_PE + 14 predictors)          yfinance (constituents · macro · ^NSEI)
                          │                                                   │
                          ▼                                                   ▼
   AARAMBH  purged walk-forward ensemble          NIRNAY  per-stock MSF + MMR oscillator,
   forecast of forward 10-day Δlog(PE)            online regime ensemble → daily breadth
   from 20-day predictor momentum                 over n_live warm constituents
                          │  Forecast → zA = causal z           │  −Avg_Signal → zN = causal z (NaN = NO_DATA)
                          └──────────────┬──────────────────────┘
                                         ▼
                      CONCLUSION (convergence/conclusion.py)
     state  = agreement_state(zA, zN)  ∈ CONFIRM · CONTRADICT · NEUTRAL · NO_DATA
     κ      = walk-forward shrink, < 1 only if the Convergence-Lift test passes
     C      = 100 · tanh(zA / 2) · (κ on CONTRADICT days, else 1)      (+ = bullish)
     evidence on C at the displayed horizon: Fisher / HAC / block-bootstrap IC,
     yearly ICs, momentum baseline, cost-aware backtest, signal-chain ICs, trust tier
```

**Why this shape.** The 2.0 council measured that (a) Aarambh's forecast is the only
component with a directional lead, (b) a second system that carries no information costs
the lead 29–42% of its IC when it is allowed to gate or average with it, and (c) the v1
eight-parameter calibration was not statistically identified. So Aarambh is the
directional head; Nirnay may only **shrink** conviction, on days it contradicts Aarambh,
and only after the data show that Aarambh is measurably more accurate when the two
agree. "Two systems. One conclusion." is therefore a claim the run *tests* and reports —
with its statistical power — rather than an assumption baked into the number.

## 2. Layering

```
analytics/  →  engines/  →  convergence/  →  pipeline/  →  app.py + ui/
   pure math     stateful      pure, no IO      orchestration     rendering only
                 orchestrators                   (headless)
data/  — the only IO boundary (fetchers, cache, sources, synthetic world)
core/  — config (single source of truth, validated at import) + logging
```

Never upward, never sideways; `tests/test_layering.py` enforces it with an AST scan.
`ui/` reads only the session keys published by `PipelineResult.to_session_state()`.

| Package | Role |
|---|---|
| `analytics/` | `causal_stats` (causal_z, Spearman, Fisher-z, Newey-West, block bootstrap), `backtest`, `conformal`, `ddm_filter`, `ou_process`, `hurst`, `structural_breaks` (heuristic, diagnostic only), `regime` (numba kernel + class oracle), `factors` (causal PCA), `custom_features`, `analogs`, `utils`. |
| `engines/` | `aarambh.FairValueEngine` (purged walk-forward; `Valid`, `Forecast`, `Label`, `ForecastZ`, `wf_meta`), `nirnay` (`run_full_analysis`, `aggregate_constituent_timeseries`). |
| `convergence/` | `conclusion.run_conclusion → ConclusionResult` (the product's conclusion), `cross_validator.compute_agreement_anatomy` (descriptive only), `labels` (return-label resolution). |
| `data/` | `fetcher`, `cache` (parquet/JSON, atomic, no pickle), `circuit_breaker`, `constituents`, `dates.parse_dates`, `macro_lag.lag_non_ist`, `sources` (`FetchResult` provenance; Live / Snapshot / Synthetic), `synthetic`. |
| `pipeline/` | `config.PipelineConfig` (+ `fingerprint()`), `inputs.prepare_sheet`, `macro.assemble_macro_panel`, `run.run_pipeline → PipelineResult`, `report.build_run_report` / `render_diagnostics`, `__main__` (CLI). |

## 3. Running headlessly

```bash
python -m pipeline synth --seed 7 --json run.json          # deterministic synthetic world, offline
python -m pipeline run --csv sheet.csv --date-col DATE --json run.json   # your sheet + live market data
python -m pipeline run --csv sheet.csv --snapshot snap/                  # replay saved market data offline
```

```python
from pipeline import PipelineConfig, run_pipeline
from data.sources import LiveSources            # or SnapshotSources(dir) / SyntheticSources(seed)
result = run_pipeline(sheet_df, PipelineConfig.build(target="NIFTY50_PE"), LiveSources())
result.conclusion        # ConclusionResult; .to_dict() is JSON-safe
result.run_report        # nishkarsh.run_report/1 (JSON-safe)
```

## 4. Integrity guarantees (each is a test in `tests/`)

| Guarantee | Mechanism |
|---|---|
| No look-ahead in the forecast | Training rows end `h` rows before every forecast point; a window too short to keep that gap yields NaN (`wf_meta.chunks_skipped_starved`), never a dropped purge. Perturbing any label that is not realized by row *t* cannot change `Forecast[t]`. |
| Prefix invariance | Appending rows never changes an earlier causal output (engine, Nirnay, conclusion). Fixed refit cadence; history start anchored to 1 January. |
| Cross-timezone causality | US / EU / FX / commodity closes are lagged one NSE session; the sheet's US yields one row. |
| Honest significance | Every reported IC is h-spaced at its own horizon with Fisher, HAC and stationary-block-bootstrap inference; null false-positive rates of the lift, edge and backtest tests are ≤ 0.08 (slow suite, 100 seeded reps). |
| Idempotent runs | No persisted calibration profiles; κ is re-estimated walk-forward on every run. Cache key = config fingerprint + data content hash. |
| Safe IO | No pickle; atomic writes; HTML-escaped rendering; secrets never sent to the browser. |
| Single source of truth | `core/config.validate_config()` runs at import and refuses incoherent settings. |
