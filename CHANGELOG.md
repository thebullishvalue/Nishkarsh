# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-04-07 — Nishkarsh Production Release

### Added
- **Unified Convergence Engine**: Two orthogonal systems (Aarambh top-down + Nirnay bottom-up) merged into a single convergence pipeline with adaptive-weighted 4-dimension scoring (Direction 30%, Breadth 25%, Magnitude 25%, Regime 20%)
- **Divergence Detection**: Cross-system early warnings — AARAMBH_LEADS (valuation extreme, constituents lagging), NIRNAY_LEADS (momentum-first), CONTRADICTION (uncertain environment)
- **Terminal Logging System**: Arthagati-style direct console output with timed phases, detailed per-constituent analysis logs, and formatted run summaries
- **Progressive UI**: Arthagati-style progress cards with animated pulse-dot, gradient progress bar, and real-time status updates during pipeline execution
- **Nifty 50 Constituent Analysis**: All 50 constituents processed with MSF + MMR + HMM/GARCH/CUSUM regime intelligence, aggregated into daily statistics
- **Macro Data Integration**: 18 Yahoo Finance commodities/FX symbols + 6 bond yields from Google Sheets = 24 macro indicators for MMR regression
- **Timeframe Filtering**: 3M, 6M, 1Y, 2Y, ALL buttons with synchronized x-axis zoom across all convergence charts
- **Hover Templates**: All Plotly charts now show Date + Value in tooltips

### Changed
- **System Renamed**: Samyoga → **Nishkarsh (निष्कर्ष)** — "Conclusion / Inference" — with complete branding update across all files
- **Default Timeframe**: 6M (was 1Y) with 3M added (was 1M)
- **Sigmoid Formula**: Corrected to original Nirnay formula `2/(1 + exp(-x/scale)) - 1` (was `2/(1 + exp(-scale*x)) - 1`)
- **Market Markers**: Unified signal plot now uses conviction-plot-style markers (size 7/8/10 based on signal strength, color-coded green/red/gray)
- **Nirnay Tab**: Complete rewrite matching original Nirnay charts — Oversold/Overbought distribution, raw counts, buy/sell signal counts, HMM regime probabilities

### Fixed
- **Stooq Bond Yields**: Integrated Google Sheets bond yields (IN10Y, IN02Y, IN30Y, US10Y, US30Y, US02Y) as fallback when Stooq blocks automated access
- **Column Name Normalization**: All Nirnay aggregation outputs use exact original column names (Oversold_Pct, Overbought_Pct, Buy_Signals, etc.)
- **Convergence Date Alignment**: Proper inner-join between Aarambh calendar dates and Nirnay trading dates
- **Warning Suppression**: Suppressed yfinance, urllib3, pandas, and numpy warnings for clean terminal output

### Removed
- All Samyoga branding references (sidebar, headers, signal cards, console output, metric cards)
- Progress bar jargon — replaced with clean Arthagati-style progress cards
- Dead code paths from multi-phase development iterations

---

## [3.2.2] - 2026-04-05 — Production Deployment Hardening

### Fixed
- **Devcontainer Configuration**: Corrected container name from `v2.0` to `v3.2.2` and fixed entry point script from `aarambh.py` to `correl.py`, ensuring consistent development environment setup
- **Dynamic Version Logging**: Replaced hardcoded `v3.2.1` in engine startup log message with dynamic `VERSION` constant, preventing version display inconsistencies across releases

### Changed
- **Production Metadata**: Updated version references across `requirements.txt`, `LICENSE.md`, and documentation to reflect current release state

---

## [3.2.1] - 2026-04-01 — ADAM Phase II Logic Guard Hardening + Phase III Production Release

### Critical — Mathematical Consistency
- **Dynamic Ensemble Inverse-MAE Weights**: Eradicated static simple averages across walk-forward predictions. Ensemble constituents are now dynamically weighted using an inverse-MAE schema built on OOS trailing validation sets, featuring a 5% baseline preservation floor.
- **Double-Bounding Singularity**: `ConvictionScore` math natively applies and exports `ConvictionBounded`, strictly enforcing identical numerical states between the underlying logic loops and the display UI.

### Major — Temporal Stability
- **DDM State Initialization Bias**: `drift_diffusion_filter` now initializes transient state arrays utilizing rolling forward sums of the initial 20 observations, eliminating early systemic drag.
- **Temporal Record Alignment**: Feature impact matrix updates structurally map directly to block end times `t_index`, rather than global arbitrary limits.
- **OU Dynamics**: Projection bands override full sample estimations dynamically with trailing `theta_history[-1]` values, reflecting localized confidence bounds exactly.
- **Hurst Stationarity Guard**: Implemented ADF tests prior to DFA Hurst invocations. If signals diverge violently from base stationarity, deterministic bounds return `H=0.5` instead of hallucinating fractional trend structures.

### UI/UX Reorganization

#### Architecture
- **Tab Overhaul**: Collapsed fragmented 5-tab structure into **4 logically flowing tabs**:
  - **Dashboard**: Primary signal, Base Conviction, DDM Confidence, Market State, Model Quality, Fair Value plot
  - **Breadth Topology**: Zone breadth, Signal Frequency, Average Z-Score, Current Lookback States
  - **ML Diagnostics**: OU diagnostics, Feature Impact, Signal Performance
  - **Data Table**: Full time-series with CSV export
- **Primary Signal Relocation**: Moved from tab content to above all tabs (always visible, full-width card)
- **Timeframe Filter**: Replaced radio buttons with Swing-style button row (1M, 6M, 1Y, 2Y, ALL)

#### Visualization Standards
- **Unified Chart Margins**: `dict(t=10, l=60, r=20, b=10)` across all Plotly figures
- **Consistent Line Widths**: 2.5px main data lines, 1px reference lines
- **Standardized Legends**: Horizontal orientation, top-right position, font size 9
- **Base Conviction Plot**: Threshold markers at ±20 (small) and ±40 (large), color-coded
- **DDM Interpretation Card**: Dynamic 5-state analysis matching Regime Context style

#### Code Quality
- **Dead Code Removal**: ~500 lines of old tab rendering functions eliminated
- **DRY Implementation**: `_render_metric_card()` helper for consistent card rendering
- **Threading Removed**: Sequential walk-forward execution (no ThreadPoolExecutor warnings)
- **Streamlit API Updated**: `use_container_width=True` → `width="stretch"` (15+ instances)

---

## [3.2.0] - 2026 — ADAM Phase I Architectural Hardening

### Critical — Mathematical Correctness
- **True Conformal Quantiles**: Replaced pseudo-Gaussian z-score implementations with the rigorous empirical `compute_conformal_zscores` primitive. Signal zones now mathematically align with fat-tailed reality.
- **Bai-Perron Regime Binding**: Expanding windows now mathematically bind to the most recent structural break (`max(max_lookback, last_break)`), severing legacy coefficients from polluting out-of-sample regressions across shifting regimes.
- **DDM Variance Capping**: `drift_diffusion_filter` now caps scaling variance geometrically (`min(abs(drift) * 0.5, long_run_var * 0.5)`) to prevent ballooning standard errors during prolonged regimes.

### Added
- **True Thread-Safety**: Re-established `ThreadPoolExecutor` async processing safely locked behind state mutation conditions.

### Fixed
- **ElasticNetCV Silent Exception**: Repaired the `ElasticNetCV(n_alphas=10)` silent exception bug, restoring the ElasticNet sub-model to the Walk-Forward ensemble.

---

## [3.1.0] - 2026 — Initial Refactor

### Fixed
- **Look-Ahead Bias in Z-Score Computation**: Rolling mean/std now use `shift(1)` to ensure only past data is used in standardization. Previously, current residual was included in its own z-score calculation, inflating OOS R² by 10–20%.
- **Jackknife Correction for Near-Unit-Root AR(1)**: Replaced biased jackknife estimator with Andrews (1993) median-unbiased correction. Half-life estimates now accurate for persistent series (θ ≈ 0.95).
- **DDM Variance Unbounded Growth**: Added mean-reverting variance term: σ²_t = (1-λ)σ²_{t-1} + λσ²_LR. Confidence bands now stable during extended regimes.

### Added
- **Structural Break Detection**: Bai-Perron multiple breakpoint testing before walk-forward. Expanding window resets if regime shift detected within trailing period.
- **DFA Hurst Exponent**: Replaced biased R/S estimator with Detrended Fluctuation Analysis (Peng et al., 1994). Proper lag range: max(4, n/10) to n/4.
- **Significance Testing**: t-statistics and p-values for all signal performance metrics.
- **Rolling θ Estimation**: Time-series of OU mean-reversion speed computed over trailing 60-day windows.
- **Feature Impact History**: Time-varying feature weights stored at each refit interval.
- **Conformal Quantile Z-Scores**: Empirical quantile-based z-scores preserve fat tails.
- **θ Stability Diagnostic**: New metric card showing whether mean-reversion speed is stable (CV < 50%).

### Changed
- **Thread-Safe Walk-Forward Execution**: Replaced parallel ThreadPoolExecutor with lock-protected sequential execution.
- **Conviction Score Soft Bounds**: Applied tanh transformation: Conviction_bounded = 100 × tanh(Conviction_raw / 100).
- **Winsorized Forward Changes**: Forward % changes capped at ±100% to prevent spurious outliers.
- **DRY Helper Functions**: Extracted `_safe_array_operation`, `_classify_zones`, `_detect_crossover_signals`, `_compute_significance`, `_apply_conviction_bounds`.

### Removed
- Dead code: `kalman_filter_1d`, `_compute_kalman_conviction` (fully replaced by DDM).

### Performance Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| OOS R² | Inflated | Honest | -10–20% |
| Half-Life Estimate | Biased low | Unbiased | +20–40% |
| Confidence Band Stability | Diverging | Stable | Fixed |
| Thread Safety | Race condition | Lock-protected | Fixed |

---

## [3.0.0] - 2026 — ADAM Refactor

### Added
- **Adaptive Conformal Unbounded Bounds**: Native continuous-scale expanding empirical quantiles natively mapping limits in Out-of-Sample residuals.
- **Drift-Diffusion Accumulator (DDM)**: A Sequential Probability Ratio Test integrating the evidence signals into an orthogonal bounding space.
- **Strict Decoupled OU Regression**: Stationary variance projections derived accurately exclusively via ln(2)/θ.

### Changed
- Replaced the constrained rank approximation (`stats.norm.ppf()`) bounding with conformal unbounded standard deviations, permanently repairing extreme "black swan" tail-clipping bugs.
- Removed `kalman_filter_1d` and `_compute_kalman_conviction` in favor of physically valid Drift-Diffusion (DDM), directly solving the boundary-break covariance mis-specification.
- Decoupled `dynamic_theta` from the arbitrary `vol_multiplier` to prevent mathematically illegal OU diffusion contraction.
- Complete UI refactor across Streamlit cards mapping UI strings to accurately reflect new DDM and Conformal realities.

---

## [2.2.1] - Prior Release

### Fixed
- **Dataset Transition Crash**: Resolved state desynchronization when switching between datasets with non-overlapping predictor columns.
- **Predictor Mutation Cache Bypass**: Fixed inline edits to predictor variables being ignored by Streamlit session state cache if the target sum remained identical.

---

## [2.2.0] - Prior Release

### Changed
- **Vectorized Walk-Forward Engine**: Massively optimized performance by vectorizing prediction loops across multi-row execution chunks.
- **sklearn LinearRegression Swap**: Replaced heavy `statsmodels` WLS implementation with `scikit-learn` `LinearRegression`.
- **ElasticNetCV Tuning**: Repaired parameter grid and relaxed `HuberRegressor` convergence tolerance to dramatically reduce ensemble fit times.
- Updated devcontainer environments to reflect v2.2.0.

---

## [2.1.0] - Prior Release

### Added
- `CHANGELOG.md` to track version history.
- Dynamic Feature Impact tracking using WLS coefficients and PCA back-projection.

### Changed
- Reorganized dashboard diagnostic cards with concise sub-metrics for uniform heights.

### Fixed
- **ConvergenceWarning**: Resolved `sklearn.linear_model._coordinate_descent` warning by increasing `max_iter` to `10000`, relaxing `tol`, and filtering non-consequential warnings.

---

## [2.0.0] - Prior Release — Major Update

### Added
- Walk-forward expanding-window ensemble regression (Ridge + Huber + OLS) for Fair Value estimation.
- Ornstein-Uhlenbeck (OU) mean-reversion parameter estimation on OOS residuals.
- Kalman-filtered breadth conviction scoring with 95% confidence bands.
- Residual Hurst Exponent to validate mean-reversion empirical properties.
- Swing-based divergences detection using local extrema logic.
- "Apply & Compute" staging system to prevent heavy engine recalculations during feature selection.
- Data staleness warnings.

### Changed
- Transitioned from in-sample baseline metrics to strictly out-of-sample (OOS) validation framework.
- Enhanced application theme with the @thebullishvalue Design System styling.

---

## [1.0.0] - Initial Release

### Added
- Static multi-lookback Z-score banding and fair-value linear models.
