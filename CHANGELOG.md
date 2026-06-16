# Changelog

All notable changes to **Nishkarsh** are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Sections used: **Added · Changed · Deprecated · Removed · Fixed · Security · Performance · Docs**.

---

## [Unreleased]

### Added
- `CALIBRATION_RETURN_LABEL` config toggle (`"target"` | `"nsei"` | `"basket"`)
  controlling what forward series the Intelligence layer — both the
  `ConvergenceTuner` calibration AND the diagnostics directional test — is
  optimized to predict. Default `"target"` makes the calibration follow the
  selected engine target (e.g. forward Δlog NIFTY50_PE) instead of hard-wiring
  `^NSEI`, so the whole pipeline stays coherent with the Target dropdown. Over
  the 3–20d horizons earnings are ~flat so ΔlogPE ≈ price return; `"nsei"`
  restores the true survivorship-correct price-return label, `"basket"` uses the
  EW constituent basket. Run-log "Return Label" / "Label Source" and the
  directional-test header now name the active label.
- **Re-calibrating walk-forward IC** (`walk_forward_ic` + `_optimize_frame` in
  `convergence/intelligence.py`) — a durability grade that *re-optimizes* the
  convergence weights/thresholds on each expanding train block and scores the
  next **purged** OOS block. Unlike `cross_validated_ic` (one fixed param set
  across folds), it distinguishes edge that survives recalibration from a single
  lucky parameterization. Renders in SYSTEM DIAGNOSTICS as "Walk-Forward IC
  (re-calibrated per window · purged OOS)". *(Adapted from Tattva.)*
- **Experimental predictive Aarambh mode** — `AARAMBH_FORWARD_SIGNAL` (default
  **off**) plus `AARAMBH_FWD_HORIZON` (10) and `AARAMBH_FWD_MOM_K` (20). When on,
  Aarambh forecasts the forward H-day log-change of the target from trailing
  K-day predictor momentum (ex-ante) instead of regressing the PE level;
  conviction becomes the forecast itself (`−prediction` → bullish pole) and
  R²-vs-RW measures real forecast skill. Bypasses the combined-PCA gate (raw
  momentum features). A research head to test whether a returns forecast carries
  IC the near-tautological levels model structurally cannot. Predictor momentum
  uses a sign-agnostic K-day difference (Nishkarsh predictors include breadth
  ratios/spreads that can be ≤0, so a log return would be invalid). *(Adapted
  from Tattva's returns-based engine.)*
  - Run-log made **mode-aware**: Engine Results / model-stats relabel R² for the
    forecast (magnitude R²≈0 is normal, edge is in IC); the Signal-Performance
    and OU/Hurst sections are flagged NOT-meaningful in predictive mode (the
    target is already a forward return, so "forward Δ of the target" is a
    double-forward artifact and the OU/half-life describe the forecast series).

### Changed
- **Direction-carrying convergence score.** `convergence_score` now equals
  `−consensus_direction · agreement_strength · 100`, where `consensus_direction`
  is the agreed market direction of the two engines (+1 bull / −1 bear / 0
  disagree). Previously the score encoded *agreement magnitude* with no
  direction, so both a bottom and a top scored alike and the hero card could
  contradict the DDM in sign (the documented agreement-vs-direction flaw). The
  calibration's `_composite_signal` is now orientated the same way, so the tuner
  optimizes a genuinely directional signal. `consensus_direction` is persisted in
  `ConvergenceSignal` and the convergence series. *(Adapted from Tattva.)*
- **Purged k-fold calibration objective.** `ConvergenceTuner._objective` now
  scores params across K contiguous time folds (`_score_cv`, mean − 0.5·std
  consistency penalty) instead of a single 70/30 slice, and the train/holdout
  split inserts a **purge gap of max(horizon)** so the overlapping forward-return
  label can't leak across the boundary. Rewards params robust across regimes;
  attacks the single-slice IC volatility seen in prior runs. *(Adapted from
  Tattva.)*
- **Data-backed defaults from the 2026-06-15 study.** `AARAMBH_PCA_PREDICTORS`
  default → **False** (inert in predictive mode, harmful in levels — neutral when
  forward-on, correct when forward-off). `CALIBRATION_RETURN_LABEL` default →
  **"nsei"**: the survivorship-biased basket overfits survivor drift (val IC
  +0.038, 67% durable) and was the silent `target` fallback that produced every
  prior "walk-forward NOT durable" verdict; ^NSEI/PE give barely-fit,
  fully-durable calibrations (val IC +0.13–0.15, 100% walk-forward). Predictive
  `target` now uses the real target LEVEL, not the basket fallback.

### Fixed
- **Predictive-mode messaging coherence.** In predictive mode the engine's
  `tradeable`/`edge_assessment`/`inverted`/`display_strength` come from the
  level/magnitude forward-edge test, which is NOT the verdict — yet the run-log
  headline and the Aarambh-tab banner surfaced "NO FORECAST EDGE — valuation
  context only", directly contradicting the "durable directional edge" in the
  diagnostics. Now: the engine signal exposes a `forward_signal` flag; the Final
  Verdict headline is built from the Directional + Walk-Forward IC (the real
  predictive verdict); Engine Results / Model-&-Edge relabel the magnitude test
  as "NOT the predictive verdict"; the Aarambh-tab banner shows a predictive-mode
  card pointing to the Convergence tab; and `display_strength` shows the
  conviction strength (the directional forecast) instead of "NO EDGE". Levels
  mode is unchanged.
- **Diagnostics tab refreshed to current features.** (1) The HMM regime chart
  read a dead session key (`nirnay_results`) and never rendered — fixed to
  `nirnay_daily`. (2) `REGIME PERSISTENCE` was a hardcoded `0.98` — now the
  empirical day-over-day state stickiness. (3) Intelligence Center: "IC vs
  forward PE" → the actual calibration label (^NSEI / PE / basket); "chronological
  70/30 split" → "purged 70/30 · K-fold CV objective"; the `STABILITY = val/train`
  card (which mislabeled a healthy barely-fit profile as a warning) → `FOLD
  STABILITY` (% CV folds positive, the gate metric). (4) New **Out-of-Sample Edge
  Durability** row surfacing Directional IC, re-calibrating Walk-Forward
  durability, and CV fold IC — the predictive verdict, previously absent. (5)
  Predictive-mode aware throughout: OU/stationarity, "fair value" feature impact,
  and Signal Performance are relabeled as not-the-verdict when forecasting
  returns. App stashes `intelligence_diag` (IC/durability) into session for the tab.
  (6) Feature Impact now states the exact predictors feeding the run (roster +
  count), labels them as `N`-day momentum in predictive mode, and — when the
  combined-PCA gate is on — flags that the names are orthogonal `MACRO_PC*`
  factors (not raw predictors) so the section is never silently cryptic.

---

## [1.4.22] — 2026-06-15 — *Data-backed train-window default*

### Changed
- `MIN_TRAIN_SIZE` / `MAX_TRAIN_SIZE` set to **252 / 756** after a real-data
  sweep (cached 4,940-row PE sheet, 5 windows). Findings: no forward edge at any
  window; smaller windows (126–504) post higher OOS R² and less-negative R² vs
  RW — the overfit-flattery trap, still edgeless; larger min-windows (1000+)
  degrade R² vs RW. 252–756 is the sound sweet spot (healthy p/n, adaptive,
  best honest R² vs RW among non-overfit options). PB/DY confirmed as ~7pts of
  tautological R² (0.91 → 0.84 when dropped).

---

## [1.4.21] — 2026-06-15 — *Passthrough master toggle*

### Added
- **`PCA_PASSTHROUGH_ENABLED`** boolean (default `True`) — master on/off for the
  reserved passthrough, mirroring `AARAMBH_PCA_PREDICTORS`/`CUSTOM_PREDICTORS_
  ENABLED`. Off → `PCA_PASSTHROUGH` is ignored and those columns flow through the
  PCA like everything else. Resolved once into an effective list used by both
  engines, the run-log lines, and the cache key (so flipping it re-runs).

---

## [1.4.20] — 2026-06-15 — *Reserved PCA passthrough*

### Added
- **`PCA_PASSTHROUGH`** config (`IN10Y, IN02Y, IN30Y, INIRYY`) — these
  predictors now **bypass the PCA gate** and ride alongside the `MACRO_PC#`
  factors as their own raw, named columns, in **both** engines (Aarambh combined
  PCA and MMR macro PCA). Their direct India rate/inflation signal is no longer
  diluted across the orthogonal components. `build_causal_macro_factors` gained
  a `passthrough` arg; verified the passthroughs are kept raw, excluded from the
  PCA/loadings, and still causal (prefix-invariant). Folded into the cache key.

---

## [1.4.19] — 2026-06-15 — *Sidebar predictor sync*

### Fixed
- The Predictor Columns multiselect now always reflects the predictors the
  system **actually ran with**. Streamlit persists widget state and ignores
  `default=` across reruns, so when `active_features` changed programmatically
  (e.g. the target-change auto-apply dropping a predictor) the list showed a
  stale selection. The widget is now keyed to the active predictor set, so it
  re-syncs on any external change while still supporting local staging edits.

---

## [1.4.18] — 2026-06-15 — *Custom Predictors toggle*

### Added
- **`CUSTOM_PREDICTORS_ENABLED`** config flag (default `True`) — mirrors
  `AARAMBH_PCA_PREDICTORS`. Off → the 44 engineered features are skipped and
  both engines run on the base macro panel, for clean A/B. Folded into the
  engine cache key, so flipping it re-runs automatically; logs
  `Custom Predictors: disabled` when off.

---

## [1.4.17] — 2026-06-15 — *Custom Predictors — overlap fix*

### Fixed
- Custom-feature build was aborting ("columns overlap but no suffix specified")
  because the bond yields (IN10Y…) exist in *both* the macro panel and the sheet
  extras. Now only sheet columns absent from the macro panel are joined; the
  builder reads the yields from the macro panel and `REPO`/`INIRYY`/breadth/
  valuation from the sheet. Verified the 44 features build across both sources.

---

## [1.4.16] — 2026-06-15 — *Custom Engineered Predictors*

### Added
- **44 custom predictors** (`analytics/custom_features.py`, spec in
  `CUSTOM_PREDICTORS.md`): yield-curve spreads, real rates, cross-country
  differentials, inflation/breadth momentum, credit (HY/IG, EM, fallen-angels),
  duration/inflation-expectation ETF ratios, FX momentum, commodity ratios
  (gold/copper, oil, agri), EM/China, and cross-asset composites (RISK_ON_OFF,
  FIN_CONDITIONS). Every feature is **stationary** (native spread / causal 252d
  z-score / momentum) and **causal** (verified prefix-invariant) — so they
  won't make a level regression extrapolate.
- Wired into **both engines via the causal PCA gate**: appended to the macro
  panel (→ MMR factors) and to Aarambh's combined-predictor PCA. The 3
  PE/PB/DY-embedding features (`EQ_RISK_PREMIUM`, `IMPLIED_ROE`, `DY_SPREAD`)
  are fed to **MMR only** — including them in Aarambh would be *target leakage*
  (predicting PE from 1/PE), so they're auto-excluded there.

---

## [1.4.15] — 2026-06-15 — *Sidebar Reactivity Fix*

### Fixed
- **Target / Date changes now apply immediately.** Previously, changing the
  Target Variable (or Date Column) dropdown only *staged* the change — and the
  "Apply" button plus the "Pending" hint were both buried inside the collapsed
  Predictor Columns expander, so a target change looked like a no-op (the
  analysis kept running on the old target). The two dropdowns now recompute on
  change. Predictor edits still stage behind an **Apply Predictors** button
  (you typically toggle several at once), which is the only case where staging
  helps.

---

## [1.4.14] — 2026-06-14 — *Directional Test & Conclusions*

### Added
- **Directional Convergence Test** (`directional_convergence_ic`, in the run's
  SYSTEM DIAGNOSTICS). The calibrated `convergence_score` is an *agreement*
  metric (both-bullish and both-bearish score the same), so it cannot be a
  directional predictor — the root cause of the calibration's thin, unstable IC.
  This measures the *signed* normalized convergence against forward index
  returns across folds, settling whether any directional timing edge exists.
  Validated: detects a planted edge (100% folds positive) and reports noise as
  no-edge.
- **`FINDINGS.md`** — consolidated research conclusions (no robust tradeable edge;
  Aarambh tautological/inverted; MMR ≈0; convergence = agreement-not-direction).

### Changed
- **`AARAMBH_PCA_PREDICTORS` defaulted OFF** — feeding macros into Aarambh was
  proven harmful in every form (R² −4 / −47 / −0.78-inverted).
- De-duplicate columns in the combined Aarambh predictor panel (the bond yields
  appeared in both the sheet predictors and the macro panel).

---

## [1.4.12] — 2026-06-14 — *Stationarize & Name the Combined Factors*

### Fixed
- **Combined-predictor PCA no longer blows up the level regression.** Feeding
  the raw macro panel (non-stationary ETF/commodity *price levels*) into
  Aarambh's level regression caused catastrophic extrapolation (OOS R² −47,
  R² vs RW −9302, RMSE 30). `build_causal_macro_factors(stationarize=True)` now
  replaces each input with its causal rolling z-score (clipped ±5) before the
  PCA, so every feature is bounded and stationary. Verified prefix-invariant.

### Added
- **Factor interpretability.** `build_causal_macro_factors(return_loadings=True)`
  returns the loadings; the run log now names each Aarambh factor by its
  top-weighted inputs (`MACRO_PC1 ≈ US10Y, Gold, DXY`), resolving the
  "MACRO_PC# is opaque" caveat.

---

## [1.4.11] — 2026-06-14 — *Aarambh Combined-Predictor PCA*

### Added
- **Aarambh can train on causal-PCA factors of (sheet predictors + full macro
  panel)** (`AARAMBH_PCA_PREDICTORS`, default on; `AARAMBH_PCA_N_COMPONENTS=12`).
  The sheet predictors and the ~89-series macro panel are concatenated, aligned
  on the Aarambh dates, and pushed through the same expanding-window causal PCA
  gate as MMR; the orthogonal factors become the walk-forward's predictors.
  Verified prefix-invariant (no look-ahead) end-to-end. NOTE: the macro panel
  has only ~5y of real history, so for older dates the macro columns are
  flat-filled and the factors are effectively sheet-driven — logged in the run.
  Toggle the flag to A/B against the raw-predictor model.

---

## [1.4.10] — 2026-06-14 — *Honest Signal-Performance Significance*

### Fixed
- **Forward-change significance no longer inflated by overlapping windows.**
  `get_signal_performance` collected an h-day forward return at *every*
  qualifying signal bar, so signals only a few days apart shared almost all of
  their forward window — heavily autocorrelated observations that the t-test
  treated as independent, inflating significance by ~√h (a 20d BUY t≈4.8 / p≈0
  corresponds to a real t≈1.1 / p≈0.4 once the ~20-fold redundancy is removed).
  Consecutive included signals must now be ≥ `period` bars apart (non-overlapping
  forward windows), so the reported t-stats and p-values are honest.

---

## [1.4.9] — 2026-06-14 — *Apples-to-Apples Calibration Gate*

### Fixed
- **Incumbent comparison is now on the same validation data.** The gate
  rejected a fresh candidate (val IC +0.098 on 365 rows) for "regressing"
  against a stored incumbent (+0.135 on 142 rows from an older 2y window) —
  but those ICs were measured on different samples and aren't comparable. The
  incumbent's params are now re-scored on the *current* run's validation frame
  (`ConvergenceTuner.score_on_validation`) before the regression check, so a
  larger/more-recent candidate is no longer rejected in favour of a stale,
  noisier number. The re-scored incumbent IC is logged next to the stored one.

---

## [1.4.8] — 2026-06-14 — *Duplicate-Date Root Cause*

### Fixed
- **Root cause of the "cannot convert the series to float" crash.** A
  constituent whose OHLCV index contained a duplicate date (yfinance
  occasionally returns one on longer pulls; the 5y window exposed it) made
  `df.loc[date]` return a *DataFrame* in `aggregate_constituent_timeseries`, so
  `row.get("Unified_Osc")` yielded a Series → `Signal_Sum`/`Avg_Signal` became a
  Series → the convergence loop's `float(row_n.get("Avg_Signal"))` aborted the
  run. Fixed at the source on three layers: (1) `run_full_analysis` now
  guarantees a unique, sorted index per constituent; (2) the aggregation
  collapses any DataFrame row to its last row and coerces the accumulator to
  float; (3) the macro/factor panel is de-duplicated before it is joined into
  each constituent (a duplicated macro index would multiply rows on join).
  Verified end-to-end against a constituent with an injected duplicate date.

---

## [1.4.7] — 2026-06-14 — *Crash Hardening*

### Fixed
- **`UnifiedConvictionModel.fit` no longer aborts the run** on a non-scalar
  score element. If a length-1 `pd.Series`/array slipped into the
  `convergence_score` list (e.g. via a `.get()` on a frame with a duplicated
  column/index, or a pandas-version difference where `float(Series)` raises
  instead of extracting), `np.array(..., dtype=float64)` raised
  *"cannot convert the series to float"* and killed the whole pipeline. Inputs
  are now coerced element-wise (`_coerce_scores`): a length-1 Series/array is
  flattened to its scalar (value preserved), genuine non-numerics → 0.0. Both
  conviction-model call sites (first-pass and post-calibration) are protected.

---

## [1.4.6] — 2026-06-14 — *Longer History & Orthogonal Macro Factors*

### Changed
- **History window 2y → 5y** ([app.py]). ~Triples the Aarambh↔Nirnay overlap
  (was ~10% coverage), stabilizing the calibration sample. Logged with an
  explicit survivorship-bias caveat (the constituent set is current membership);
  5y is the deliberate ceiling — full history would be badly distorted.

### Added
- **Causal macro-factor reduction** (`analytics.factors.build_causal_macro_factors`).
  The ~90-series macro panel (dominated by near-duplicate bond ETFs / co-moving
  FX) is compressed to 8 orthogonal principal factors via an EXPANDING-window
  PCA — refit periodically, sign-aligned, `svd_solver="full"` for determinism.
  MMR now selects drivers from de-duplicated, orthogonal factors instead of
  90 collinear raw series, cutting the "spurious top-correlated macro" selection
  variance. Verified deterministic and prefix-invariant (no look-ahead) at
  real scale (90×4967).

---

## [1.4.5] — 2026-06-14 — *Performance-Driven Edge Verdict*

Driven by a real run whose diagnostics exposed (a) my own edge gate was brittle
at the Hurst boundary (0.552 slipped past a ≤0.55 cutoff → "tradeable=True"),
and (b) the Aarambh signal is actually *inverted* on history (BUY preceded PE
declines at 5–10d, p<0.05) — yet the system reported "edge present".

### Fixed
- **Edge verdict is now driven by realized forward performance, not Hurst.**
  `FairValueEngine._compute_signal_edge` checks whether BUY/SELL signals
  actually preceded the correct direction with significance (t/p from the
  forward-change analysis). New signal fields `has_forward_edge`, `inverted`.
  A signal that is anti-predictive (significant wrong-direction moves) is
  flagged **INVERTED** and forced non-tradeable; Hurst/R²-vs-RW are demoted to
  supporting context. Verified: the real run's signature (BUY −0.86% p=0.018,
  SELL −0.82% p=0.041) now classifies as INVERTED / not tradeable.
- **OU dynamic-θ instability fallback.** When the rolling θ dispersion exceeds
  the base θ (it produced a 1.7d half-life from a θ 10× the base), the engine
  falls back to the median rolling θ, then the base θ — no more implausible
  sub-day half-lives driving the projection bands.
- **Stability gate made scale-free.** The previous catastrophic-fold rule used
  an arbitrary −0.04 cutoff and rejected a genuinely stable profile (80% folds
  positive, mean +0.105) over one −0.067 fold. Now: ≥60% folds positive, mean
  fold ≥ margin, and worst fold ≥ −mean (no regime loses more than the average
  gains). No magic IC numbers.

### Changed
- Half-life "not meaningful" messaging corrected ("residuals do not mean-revert"
  rather than "random walk", which contradicted `random_walk=False`).
- Convergence diagnostics report dimension means over BOTH full history and the
  overlap window, exposing the ~90% Nirnay-neutral dilution instead of hiding it.

---

## [1.4.4] — 2026-06-14 — *Edge Honesty & Full Telemetry*

Fourth-pass remediation, driven by a real run whose `OOS R² 0.913` masked an
`R² vs RW −15.8` (no forecasting edge) — plus a comprehensive diagnostics dump
so a single run log is sufficient to evaluate and tune the system.

### Added
- **Forecast-edge assessment on every Aarambh signal.** New fields `tradeable`,
  `edge_assessment`, `forecast_skill_vs_rw`, `mean_reverting`,
  `random_walk_regime`, `half_life_meaningful`, `display_strength`. When the
  series tests as a random walk (Hurst≈0.5 / ADF non-significant) *and* the
  model loses to a naive forecast, the signal is marked **NO EDGE** and the
  displayed strength collapses — the system no longer presents a confident
  directional call on a series it has just measured as unpredictable. The run
  log now **leads with R² vs RW** (skill) instead of the persistence-inflated
  OOS R².
- **Multi-fold calibration stability.** `ConvergenceTuner.cross_validated_ic`
  scores the chosen params across 5 chronological folds; the profile stores
  `cv_fold_ics`, `cv_fraction_positive`, `cv_ic_mean/min/std`. `is_profile_
  acceptable` now rejects profiles that aren't stable across regimes (<60% of
  folds positive, or a catastrophic worst fold) — a single 70/30 IC (which
  swung +0.011→+0.200 between runs) is no longer trusted on its own.
- **SYSTEM DIAGNOSTICS run-log section** — full per-run telemetry: data quality
  (NaN%, coverage, constituent row spread), Aarambh model/edge/OU/Hurst/ADF/
  KPSS/adaptive-bands/feature-impacts, forward-change significance (t/p, hit
  rate) at 5/10/20d, Nirnay aggregate (MMR √R², regime & vol distributions,
  oscillator spread), convergence dimensions/zones/lead-lag, and the full
  calibration profile (label source, fold ICs, weights, thresholds, param
  importances). Each subsection is independently guarded.

### Changed
- **OU half-life is flagged not-meaningful** when the residuals test as a random
  walk (resolves the 2d-half-life vs Hurst-0.55 contradiction).
- **Convergence coverage is reported** (overlap/total %) so the cross-system
  claim isn't oversold when most history is Aarambh-only.
- `Min Train Size` log line already corrected in 1.4.3.

### Removed
- Delisted `BUNL.L` macro ticker (404 on every fetch).

---

## [1.4.3] — 2026-06-14 — *True Benchmark & Honest Reporting*

### Changed
- **Calibration label prefers the actual NIFTY 50 index (`^NSEI`)** forward
  return — the true, survivorship-correct benchmark — over the equal-weighted
  constituent basket (which is built from *today's* members and is therefore
  survivorship-biased). Fallback order: `^NSEI` → EW basket → PE proxy, with the
  active source named in the run log.
- **`Min Train Size` run-log line fixed.** It printed `MIN_DATA_POINTS` (1500,
  the data *requirement*) under a label implying the training window. Now shows
  `Min Data Required` and a separate `Train Window` (`MIN_TRAIN_SIZE→
  MAX_TRAIN_SIZE`, expanding/capped) so the two are not conflated.

### Removed
- Dead in-sample `gram_schmidt_orthogonalize` (superseded by the causal,
  unique-magnitude variant, which is the only one on the signal path).

---

## [1.4.2] — 2026-06-14 — *Causality & Reproducibility*

Second-pass audit remediation. The v1.4.1 "market-led" thresholds were made
data-driven but *in-sample*; this release makes them **causal** (backtest-safe)
and fixes a non-determinism bug that made the signal flicker run-to-run.

### Fixed
- **Look-ahead in adaptive thresholds.** Aarambh conviction bands and Nirnay
  oscillator/agreement bands were full-sample percentiles, so a historical
  bar's label depended on future bars (and leaked into the calibration IC).
  Both are now **expanding-window quantiles, shifted one bar** — verified
  prefix-invariant (truncating the series leaves historical labels unchanged).
- **Look-ahead in Gram-Schmidt.** The MSF orthogonalization basis was computed
  in-sample. Replaced with `causal_gram_schmidt_orthogonalize` (expanding
  cumulative projections, shifted) — verified prefix-invariant.
- **Gram-Schmidt noise amplification.** The in-sample variant renormalized each
  orthogonal column to unit RMS, promoting a redundant component's noise to full
  weight. The causal variant keeps each residual at its *unique* magnitude, so a
  redundant component (e.g. trend vs momentum, ~98% shared) contributes ~0.1×
  rather than 1× — measured.
- **Look-ahead in structural-break detection.** The walk-forward consulted a
  globally-detected break list, so whether a break existed before `t_start` was
  decided using data after it. Breaks are now detected causally per chunk on
  `y[:t_start]`. With this + the determinism fix, interior FairValue/Regime are
  prefix-invariant.
- **Non-deterministic engine.** `ElasticNetCV(selection="random")` had no
  `random_state`, so identical data produced different signals each run (≈0.025
  FairValue drift). Pinned `random_state=42`; identical fits now match exactly.
- **Conviction scale mismatch.** The divergence detector hardcoded ±60/±20 while
  the engine's adaptive "STRONG" could fire near |conv|≈17, so EXTREME stances
  almost never triggered. The detector now reads the engine's adaptive
  `conviction_levels`, so "extreme/strong" is consistent system-wide.

### Changed
- **MMR R² benchmarked against a random walk** instead of price-level variance
  (the latter is trend-dominated and inflates R² toward 1 for any drift-tracker).
- **Calibration gate now requires a significance margin** (`val_ic ≥ 0.02`, not
  merely `> 0`) so a near-zero, noise-level IC can no longer be persisted.
- **Calibration bin floors scale with the frame** (≥5% of rows, ≥5) so the
  threshold-spread objective can't be gamed by isolating a few lucky bars.

- **Calibration label is now a true price return.** Intelligence Mode
  calibrated weights/thresholds against forward Δlog(PE), which conflates price
  moves with earnings revisions. It now builds an equal-weighted index-return
  series from the constituent OHLCV already in memory and calibrates against
  *that* — the IC measures predicted return, not predicted valuation drift. The
  PE proxy remains an automatic fallback when no usable closes align (the active
  label is recorded on the profile as `label_kind` and shown in the run log).

### Added
- `analytics.utils.causal_gram_schmidt_orthogonalize` — causal, unique-magnitude
  orthogonalization (the in-sample `gram_schmidt_orthogonalize` is retained but
  no longer used by the signal path).
- `convergence.intelligence.build_index_return_levels` — equal-weighted index
  price level from constituent closes, used as the calibration return label.

---

## [1.4.1] — 2026-06-14 — *Audit Hardening*

First-principles audit remediation. Closes the gap between the system's
"market-led, no fixed thresholds" philosophy and the implementation, and fixes
several correctness issues in the modeling core.

### Fixed
- **Ensemble member weighting was computed on the wrong data.** The walk-forward
  ensemble weighted each member by an MAE that compared validation targets
  against *forward-chunk* predictions (an unrelated time slice), making every
  blend weight meaningless. Each member is now scored by predicting the actual
  held-out validation rows and weighted by inverse out-of-sample MAE.
- **Degenerate walk-forward training window.** `MIN_TRAIN_SIZE`/`MAX_TRAIN_SIZE`
  were `15`/`30` against a ~14-feature ensemble (near-singular p/n). Raised to
  `252`/`756` (1y bootstrap → 3y cap) so the expanding window is well-conditioned.
- **Ungated calibration persistence.** Intelligence Mode saved and applied every
  calibrated profile unconditionally — an overfit profile (strong train IC,
  negative val IC) could be locked in as the live signal. Calibration now passes
  through `is_profile_acceptable()`: a profile is persisted/applied only if its
  validation IC is finite, positive, and does not regress against the incumbent.
- **MMR "quality" overstated fit.** Replaced the `Σr²²/Σr²` statistic (an
  r²-weighted mean of r², not a model R²) with a true rolling coefficient of
  determination of the composite fair-value model.

### Added
- **Real Gram-Schmidt orthogonalization** (`analytics.utils.gram_schmidt_orthogonalize`)
  applied to the four MSF components before they are combined, so the composite
  is a sum of independent contributions rather than a collinear blend. The
  docstring claim of "orthogonal components" is now backed by code.
- **Threshold identifiability.** The calibration objective now includes a
  normalized long/short spread term, so the four classification thresholds are
  driven by realized economic separation instead of a weak 5-point monotonicity
  score alone.
- **Model coverage** surfaced in `model_stats` (`model_coverage`,
  `n_fallback_chunks`) — the fraction of walk-forward chunks where ≥1 ensemble
  member fit rather than silently falling back to the train mean.

### Changed
- **Market-led thresholds replace fixed cut-offs.** Aarambh conviction bands
  (was 60/40/20) are now empirical percentiles of the realized |conviction|
  distribution; Nirnay oscillator bands (was ±5) and the agreement gate
  (was 0.3) are per-name empirical quantiles; the significant-bar cut-off
  (was a fixed 0.33%) is now a fraction of each name's recent realized vol;
  HMM regime cuts are tied to the 3-state uniform baseline. Config constants
  remain as cold-start priors / fallbacks only.

### Removed
- **Dead duplicate `analytics/signals.py`** — an unused second MSF/MMR
  implementation whose MMR driver selection used full-sample (look-ahead)
  correlation. The live, causal path in `engines/nirnay.py` is the single source.
- **Numba dependency** — the `@njit` 3×3 HMM forward step's JIT warm-up cost
  exceeded its per-call saving; reverted to plain NumPy and dropped `numba`.

### Performance
- Removed a 250 ms blocking `time.sleep` on Streamlit's script thread after
  analysis completion.

### Docs
- Streamlit `use_container_width` → `width="stretch"` across remaining buttons;
  transient confirmations consolidated to `st.toast`.

---

## [1.4.0] — 2026-05-27 — *Self-Calibrating Convergence*

Headline release: **Intelligence Mode** is now the default pipeline path —
auto-calibrated profiles applied on every Run Analysis in a single flow, with
the progress bar and sidebar rewritten to expose the new behaviour clearly.

### Added
- **Intelligence-aware progress bar.** The CONVERGENCE phase now surfaces its
  sub-stages explicitly so users can see what the calibration loop is doing:
  - `83%` First-Pass Conviction Model (DDM filter · prior weights)
  - `84%` Intelligence Mode · Setup (tuner build · 70/30 split · N trials)
  - `84 → 90%` Intelligence Mode · Calibrating (live Optuna trial counter
    and best score)
  - `90%` Intelligence Mode · Profile Saved (Train IC · Val IC)
  - `91%` Applying Calibrated Profile (vectorized re-weight)
  - `92%` Re-Fitting Conviction Model (post-calibration DDM pass)
  - `93%` Detecting Divergences
  - `94%` Convergence Phase Complete (with `calibrated profile applied`
    or `factory defaults` suffix so the user can see which path ran)
  - `95%` Storing Results · `100%` Analysis Complete
  When Intelligence Mode is OFF, the `84–92%` band is skipped end-to-end.

### Changed
- **Progress bar typography** — every progress label and subtitle migrated
  to Title Case for consistency across the pipeline.
- **Sidebar rhythm tightened.** The vertical gap between consecutive setting
  groups (Data Source ↔ Model Configuration ↔ Model Passport ↔ System Spec)
  reduced from `1.5rem` / `3rem` to `0.5–0.75rem` so the right rail reads
  as a compact toolset rather than three scattered panels:
  - `.section-divider` margin: `var(--sp-6)` → `var(--sp-3)` (globally);
    `var(--sp-2)` inside `[data-testid="stSidebar"]`.
  - `.sidebar-title` margin: `var(--sp-6) 0 var(--sp-3) 0` →
    `var(--sp-3) 0 var(--sp-2) 0`.
  - Pre-`system-spec` `<hr>` margin: `3.00rem 0` → `1rem 0 0.75rem 0`.
- **Reset Analysis** button uses `use_container_width=True` — full-width like
  Export Profile and Reset to Defaults below it in the Model Passport.

### Docs
- README now reflects the Intelligence Mode pipeline as the default flow:
  new `Intelligence Mode` section explaining what gets calibrated, the
  Optuna-TPE objective, and the per-universe profile persistence path.
- Pipeline-flow diagram updated to include the **Phase 4 calibration loop**
  (first-pass → tuner → apply → re-fit).
- `What You See` table now lists the Intelligence Center and Model Passport.
- Configuration table notes the Intelligence Mode toggle and trial count.
- Module headers and version constants unified at **`v1.4.0`** across all
  Python files, `requirements.txt`, `README.md`, `LICENSE.md`, and CHANGELOG.

---

## [1.3.0] — 2026-05-26 — *Resilient Convergence*

Production-grade data layer, refactored convergence wiring, self-calibrating
**Intelligence Mode**, and full UI parity with the sibling **Pragyam** terminal.

### Added (Intelligence Mode — new in this release)
- **Self-calibrating convergence profile.** New module `convergence/intelligence.py`
  ports Sanket's Bayesian-TPE calibration pattern to Nishkarsh's convergence
  layer. An Optuna search finds the per-universe optimum for:
  - Four dimension weights (`w_direction`, `w_breadth`, `w_magnitude`, `w_regime`)
    used inside `CrossValidator.compute_convergence`, replacing the static
    `0.30 / 0.25 / 0.25 / 0.20` defaults and the ±10% adaptive shift heuristic.
  - Four asymmetric classification thresholds (`buy_strong`, `buy_moderate`,
    `sell_moderate`, `sell_strong`) used inside `convergence/normalization.py`
    `classify_normalized_signal`, replacing the symmetric ±0.3 / ±0.5 defaults.
- **Calibration objective:** maximize the Spearman Information Ratio of the
  composite convergence signal against forward NIFTY-50-PE returns at
  horizons `[3, 5, 10, 20]` trading days, with L2 regularization toward
  uniform weights to discourage overfit. Validates via a chronological
  70/30 train/val split (no shuffling — preserves causality).
- **Disk persistence** at `~/.cache/nishkarsh/intelligence/profiles.json`,
  one profile per `(universe · selected_index)` key, with versioning
  (`PROFILE_VERSION = "v1-nishkarsh-convergence"`).
- **Sidebar "Model Passport" card** ported faithfully from Sanket
  (`_render_model_passport_sidebar` in `app.py`). Shows Default / Calibrated
  / Calibrated · ⚠ profile state, Trained-on label, Train IC, Val IC, last
  updated timestamp, plus universe-mismatch warnings, an Import / Export /
  Reset control group, and an **Intelligence Mode toggle** (default ON).
- **Intelligence Center** section in the Diagnostics tab — read-only
  diagnostic dashboard surfacing Train IC, Val IC, Stability %, Trials,
  learned-weights bar chart (calibrated vs default), threshold values,
  Optuna fANOVA factor sensitivity, and a list of all saved profiles
  on disk. No calibrate button — calibration is auto-triggered by
  Run Analysis (see below).
- **Single-flow auto-calibration.** The `CONVERGENCE` phase of every
  Run Analysis now runs the full calibration loop end-to-end with no
  manual user input:
  1. First-pass `CrossValidator` with the **prior profile** (loaded from
     disk for this universe, or factory defaults if none exists).
  2. Initial `UnifiedConvictionModel` fit to populate the convergence
     time-series (which the calibrator needs as its input).
  3. **Auto-calibration** — `ConvergenceTuner` runs Optuna TPE on the
     fresh `convergence_df` + `aarambh_ts`, with live progress on the
     pipeline progress bar (85% → 90%) and per-trial console output.
  4. **Apply in same run** — `intelligence.apply_calibrated_weights()`
     does vectorized recomputation of `convergence_score` and
     `convergence_zone` from the existing `dim_*` columns and the
     newly-learned weights (no need to re-loop CrossValidator).
  5. **Conviction model re-fit** on the recomputed scores, so the
     final DDM-filtered conviction reflects the calibrated state.
  6. **Normalized convergence** classified with the calibrated
     asymmetric thresholds.
  7. **Profile persisted** to disk so the next run starts from this
     calibration (warm path), and the **Passport sidebar updates** on
     the post-analysis rerun to show the freshly-saved profile.

  When the Intelligence Mode toggle is OFF, steps 3–5 are skipped —
  the system runs on factory defaults end-to-end (this is also the
  fall-back if calibration raises an exception).

### Dependencies
- Added `optuna>=3.5.0` to `requirements.txt` for the Bayesian TPE search.

### Added
- **Smart data layer.** Three production-grade primitives now sit between the
  app and every external API:
  - **Circuit breaker** (`data/circuit_breaker.py`) — `CLOSED → OPEN → HALF_OPEN`
    state machine per service, thread-safe, with two module-level breakers
    (`yfinance_circuit`, `sheets_circuit`) and a shared `all_circuits()` helper.
  - **Retry-with-backoff** — exponential decorator (1s → 2s → 4s, capped at 60s,
    max 3 retries), `@yfinance_circuit.protect` / `@RetryWithBackoff` stack.
  - **Two-tier cache** (`data/cache.py`, revived from dead code) — memory + disk,
    TTL expiry, **versioned keys** (`version="v1"` bump invalidates a namespace
    atomically), `get_stale()` last-good-snapshot fallback used automatically
    when a fetch fails *and* the circuit is open, full `stats()` snapshot
    (hits / misses / stale_hits / writes / hit_rate / namespace / TTL).
- **Data Layer Health diagnostics** — new section in the Diagnostics tab
  showing per-namespace cache hit rate, disk entry count, last-fetch
  timestamp, and per-service circuit-breaker state (CLOSED / HALF-OPEN / OPEN).
- **Global Macro bond-ETF universe.** Ported from the sibling **Sanket**
  project — 66 yfinance-available bond ETF tickers covering US Treasuries
  (full curve + raw yields ^IRX / ^FVX / ^TNX / ^TYX), TIPS, aggregate
  bonds, corporate IG / HY, mortgage-backed, municipals, developed-markets
  sovereign (Europe + Asia-Pacific), India fixed income, and emerging
  markets. Replaces the broken Stooq endpoints.
- **Shared normalisation module** (`convergence/normalization.py`) — single
  source of truth for the math behind the Convergence Analysis cards *and*
  the Unified Signal plot. Five small pure functions:
  `align_aarambh_nirnay`, `compute_norm_params`, `zscore_clip`,
  `classify_normalized_signal`, `compute_normalized_convergence`.
- **Pragyam-style UI uplift.** Full port of the *Obsidian Quant Terminal*
  design system from Pragyam:
  - `ui/theme.css` expanded from 47KB to 114KB — adds backdrop-filter glass,
    SVG noise + grid underlays, 11+ entrance / shimmer / gradient-shift
    keyframes, premium springy easing `cubic-bezier(0.16, 1, 0.3, 1)`,
    expanded palette (`--orange`, `--slate-warm`, `--card-base` DRY tokens).
  - `ui/components.py` expanded from 18KB to 27KB — new helpers
    `get_icon(name, size, stroke_width)` (dynamic-sized SVG), `get_signal_badge`
    (5-tier conviction badge), `render_conviction_signal`, `render_system_card`,
    `render_kv_table`. Icon library grew from 18 → 34.
  - Signal card design language unified with metric cards: corner accent dot
    + bottom gradient sweep + tinted background gradients per variant.
  - **Equal-height metric cards** — replaced the brittle `height: 100%` cascade
    with a `flex: 1 1 auto` propagation that's robust against Streamlit DOM
    changes (e.g. inserted `stVerticalBlock` wrappers).
- **System info card** in the sidebar — adopts Pragyam's `system-spec` markup
  with `.spec-row` / `.spec-label` / `.spec-value` flex layout (replaces the
  earlier `info-box` paragraph version).

### Changed
- **Convergence Analysis cards re-wired to the Unified Signal plot.** The four
  metric cards now show *exactly* what the Unified Signal plot rows display:
  - **NISHKARSH CONVICTION** ← normalized convergence (`norm_avg[-1]` in
    `[−1, +1]`), formatted `+0.42`. Signal classification re-thresholded for
    the new scale (`±0.3` moderate, `±0.5` strong).
  - **AARAMBH CONVICTION** ← `aarambh_ts["ConvictionRaw"]` (was
    `ConvictionBounded`), formatted `+0.42`.
  - **NIRNAY AVG SIGNAL** ← unchanged source, format upgraded from 1 to
    2 decimal places.
  - **AGREEMENT** ← unchanged.
- **Hero signal card** (`_render_primary_signal` in `app.py`) now reads the
  normalized value too. Interpretation paragraph rewritten to surface
  Aarambh and Nirnay contributions independently.
- **`render_nishkarsh_signal_card`** conviction format changed from `:+.0f`
  to `:+.2f` to render the new `[−1, +1]` scale meaningfully.
- **Sidebar masthead** typography tightened to match Pragyam exactly
  (`1.35rem` brand size, `0.04em` letter-spacing, `<hr>` divider instead of
  `.section-divider`).

### Removed
- **Stooq direct-yield endpoints.** Stooq started returning HTML error pages
  instead of CSV in late 2025, causing a cascade of `ParserError` retry
  loops. Replaced wholesale with the Global Macro yfinance universe.
- **`MACRO_SYMBOLS_STOOQ`** and dead **`MACRO_COLUMN_MAP`** removed from
  `core/config.py`.
- **`stooq_circuit`** breaker removed — no longer reachable.

### Fixed
- **DataFrame fragmentation `PerformanceWarning`** in `engines/nirnay.py` —
  two batches of column-by-column assignments (12 + 6 columns) collapsed into
  single `df.assign(...)` calls. `Series.shift(1)` replaced with
  `np.concatenate(([nan], arr[:-1]))` to keep behaviour byte-identical
  (verified on a 5-element test sequence).
- **Metric tooltip ring/dot misalignment** — the `::before` glow dot and the
  `.metric-tooltip` help-circle were both anchored at `top: var(--sp-3); right:
  var(--sp-3)`, but their centres differed by 3px each axis. Tooltip now has
  explicit `width/height: 12px` and adjusted position so the ring is
  concentric with the dot.
- **Header text alignment** inside metric cards — the equal-height flex rule
  was inadvertently forcing `flex-direction: column` on `<h4>`, which combined
  with the inherited `align-items: center` to horizontally centre the label.
  Narrowed the flex propagation selector to `div`-only descendants.
- **Sanskrit name on the landing page** — `.premium-header h1::after` content
  was still "प्रज्ञम" (left over from the Pragyam CSS port); corrected to
  "निष्कर्ष".

### Performance
- **Macro fetch concurrency** is now driven by yfinance's internal `threads=True`
  pool — one batch call for all 84 unique macro tickers (66 Global Macro +
  18 commodities/FX), one circuit hit, no manual `ThreadPoolExecutor` loop.

### Docs
- Module headers and version constants unified at **`v1.3.0`** across all
  35 Python files, `requirements.txt`, `README.md`, `LICENSE.md`, and the
  CHANGELOG.
- `data/cache.py`, `data/circuit_breaker.py`, and `convergence/normalization.py`
  carry full module-level docstrings explaining the lifecycle, state machine,
  and pipeline respectively.

---

## [1.2.0] — 2026-04-13 — *Obsidian Quant Terminal*

Full standardisation pass: module headers, documentation, and system integrity fixes.

### Added
- **Standardised module headers** across all 33 Python files. Every module now carries
  the `Nishkarsh v1.2.0` header with the निष्कर्ष tagline and a system-level description
  (AARAMBH, NIRNAY, CONVERGENCE, DATA, ANALYTICS, UI, CORE).
- **Structural break fallback** — `BaiPerronTest` is only in unreleased statsmodels 0.15+.
  A rolling-mean change-point heuristic is now used as a fallback when the native
  implementation is unavailable.

### Changed
- **UI aesthetic** rewritten as *"Obsidian Quant Terminal"* design language:
  - **Typography:** Syne (geometric, authoritative) for display, JetBrains Mono for data.
  - **Palette:** Obsidian (#0A0E17 → #050810), Amber Gold (#D4A853),
    Cyan (#22D3EE), Emerald (#34D399), Rose (#FB7185).
  - **Surfaces:** Frameless glass panels with thin border strokes.
  - **Plotly defaults** centralised in `ui/theme.py` — single source of truth for
    font, grid, hover, legend, and margin config across all tab renderers.
- **`VERSION` unified** — `ui/theme.py` VERSION corrected from `3.1.0` to `1.2.0`
  to match `core/config.py`. All module headers now reference `v1.2.0`.
- **`README.md`** rewritten from scratch with Obsidian Quant Terminal branding,
  full architecture diagram, pipeline flow, and troubleshooting section.
- **`CHANGELOG.md`** restructured — removed legacy Aarambh-only version entries
  that predate the Nishkarsh unification.

### Fixed
- **statsmodels compatibility** — `BaiPerronTest` import no longer crashes on
  statsmodels 0.14.x. Graceful fallback to heuristic detection.
- **VERSION consistency** — `core/config.py` (`1.2.0`) and `ui/theme.py` (`3.1.0`)
  now both report `1.2.0`.

---

## [1.1.0] — 2026-04-07 — *Nishkarsh Production Release*

The first release under the **Nishkarsh** name. Replaces the prior
"Samyoga" branding and ships the unified two-system convergence engine
end-to-end.

### Added
- **Unified Convergence Engine.** Two orthogonal systems (Aarambh top-down +
  Nirnay bottom-up) merged into a single convergence pipeline with adaptive
  4-dimension scoring (Direction 30% · Breadth 25% · Magnitude 25% · Regime 20%).
- **Cross-system divergence detection.** Three event types:
  - `AARAMBH_LEADS` — valuation extreme, constituents lagging (early warning).
  - `NIRNAY_LEADS` — momentum-first move (breadth turning before valuation).
  - `CONTRADICTION` — persistent disagreement (uncertain regime).
- **Terminal logging system** — direct console output with timed phases,
  per-constituent analysis logs, and formatted run summaries.
- **Progressive UI** — animated pulse-dot progress cards with gradient bar
  and real-time status during pipeline execution.
- **Nifty 50 constituent analysis.** All 50 symbols processed through
  MSF + MMR + the four-method regime ensemble (Adaptive Kalman / 3-state HMM /
  GARCH-like / CUSUM), aggregated to daily breadth statistics.
- **Macro data integration.** 18 yfinance commodities/FX symbols + 6 bond
  yields from Google Sheets = 24 macro indicators feeding the MMR regression.
- **Timeframe filtering.** 3M / 6M / 1Y / 2Y / ALL buttons with synchronised
  x-axis zoom across all convergence charts.
- **Hover templates.** All Plotly charts show date + value tooltips.

### Changed
- **System renamed** from *Samyoga* to **Nishkarsh** (निष्कर्ष — *"Conclusion"*).
  Complete branding update across every file.
- **Default timeframe** is now 6M (was 1Y); 3M added (replacing 1M).
- **Sigmoid formula** corrected to the original Nirnay form
  `2 / (1 + exp(−x / scale)) − 1` (was `2 / (1 + exp(−scale·x)) − 1`).
- **Market markers** on the unified signal plot now use conviction-plot-style
  markers (size 7/8/10 by signal strength, colour-coded green/red/grey).
- **Nirnay tab** completely rewritten to match the original Nirnay charts:
  oversold/overbought distribution, raw counts, buy/sell signal counts,
  HMM regime probabilities.

### Fixed
- **Stooq bond yields fallback.** Integrated Google Sheets bond yields
  (IN10Y / IN02Y / IN30Y / US10Y / US30Y / US02Y) when Stooq blocks
  automated access.
- **Column-name normalisation.** All Nirnay aggregation outputs use exact
  original column names (`Oversold_Pct`, `Overbought_Pct`, `Buy_Signals`, …).
- **Convergence date alignment.** Proper inner-join between Aarambh
  calendar dates and Nirnay trading dates.
- **Warning suppression.** Silenced yfinance, urllib3, pandas, and numpy
  warnings for clean terminal output.

### Removed
- All `Samyoga` branding references (sidebar, headers, signal cards,
  console output, metric cards).
- Progress-bar jargon — replaced with clean progress cards.
- Dead code paths from multi-phase development iterations.

---

## [1.0.0] — 2026-04-05 — *Initial Nishkarsh Release*

The first unified Nishkarsh release, inheriting the Aarambh-only lineage.

### Added
- **Aarambh FairValueEngine** — walk-forward ensemble regression (Ridge / Huber /
  OLS / ElasticNet / PCA-WLS) on Nifty 50 PE ratio with conformal prediction
  intervals and DDM smoothing.
- **Nirnay constituent engine** — per-stock MSF + MMR with four-method regime
  ensemble (Adaptive Kalman / 3-state HMM / GARCH-like / CUSUM).
- **Convergence cross-validator** — 4-dimension adaptive scoring with
  Drift-Diffusion filtering and divergence classification.
- **Streamlit UI** — five-tab interface with Obsidian Quant Terminal styling,
  timeframe filtering, and CSV export.
- **Google Sheets + yfinance data pipeline** — unified fetcher with TTL caching.
- **Nifty 50 live constituent fetching** from niftyindices.com with Wikipedia fallback.

### Inherited from Aarambh lineage (pre-unification)

The following mathematical foundations were hardened during the Aarambh-only
development cycle (versions 2.0.0–3.2.2) and carried forward:

- **True conformal quantiles** — empirical `compute_conformal_zscores` replacing
  pseudo-Gaussian z-scores.
- **Bai-Perron regime binding** — expanding windows bound to the most recent
  structural break.
- **DDM variance capping** — geometric variance scaling to prevent ballooning
  standard errors during prolonged regimes.
- **Andrews (1993) median-unbiased AR(1)** — jackknife correction for near-unit-root.
- **DFA Hurst exponent** — Peng et al. (1994) replacing biased R/S estimator.
- **Look-ahead bias elimination** — rolling mean/std with `shift(1)`.
- **Conviction soft bounds** — `tanh` transformation to `[−100, +100]`.
- **Thread-safe walk-forward** — lock-protected sequential execution.

---

© 2026 Nishkarsh · [@thebullishvalue](https://twitter.com/thebullishvalue)
