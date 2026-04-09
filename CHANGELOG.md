# Changelog

All notable changes to **Nishkarsh** are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Sections used: **Added · Changed · Deprecated · Removed · Fixed · Security · Performance · Docs**.

---

## [1.2.0] — 2026-04-09 — *Editorial Quant Terminal*

A total UI/UX redesign and a full documentation rebuild. No engine math
changed in this release; the system was reviewed end-to-end via the
`multi-agent-brainstorming` protocol and the findings are tracked in
`docs/DESIGN_REVIEW.md`. Three high-severity items are scheduled as a
1.2.1 hotfix.

### Added
- **`docs/ARCHITECTURE.md`** — full system architecture: layered diagram,
  end-to-end pipeline walkthrough, mathematical primitive inventory with
  formulas and citations, configuration surface, performance profile,
  data-flow contracts.
- **`docs/DESIGN_REVIEW.md`** — multi-agent peer review of system logic,
  math, and performance. 22 objections raised across Skeptic / Constraint
  Guardian / User Advocate roles; all logged with disposition and target
  version. Arbiter ruling: **REVISE** (acceptable for 1.2.0, hotfix slate
  scheduled for 1.2.1).
- **Differentiation anchor (UI):** 3px vermillion vertical rail fixed to
  the left edge of the viewport — ledger margin mark visible across all tabs.
- **Numbered Plotly colourway** in `apply_chart_theme()` so chart series
  inherit the editorial palette by default.
- **`prefers-reduced-motion` honoured** — pulse animation and all
  transitions disabled when the OS flag is set.

### Changed
- **UI aesthetic** rewritten end-to-end as *"Editorial Quant Terminal"*
  via `ui-ux-pro-max` + `frontend-design` skills (DFII = 15/15):
  - **Typography:** Fraunces (variable serif) for display numerals,
    IBM Plex Sans for body, JetBrains Mono retained for tabular figures.
    Replaces Space Grotesk + JetBrains Mono.
  - **Palette:** warm charcoal stack `#16161A → #1E1E22 → #28282E` with
    cream ink `#F5F2EC` (14.9:1 contrast). Vermillion accent `#E0533D`
    used for chrome only — never P&L semantics.
  - **Signal cards:** oversized 4rem Fraunces serif numerals; coloured
    4px left rule preserved.
  - **Metric cards:** top-rule animates from `--rule` to `--accent` on
    hover with no layout shift.
  - **Tabs:** monospaced small-caps with vermillion underline on the
    active tab.
  - **Status badges:** ledger-style with `[ … ]` bracket marks.
  - **Section divider:** vermillion segment on the leading 32px, hairline rule beyond.
- **`README.md`** rewritten from scratch — narrative-led, no emojis as
  icons, full project layout, math citations, honest known-limitations
  section that cross-references the design review.
- **`CHANGELOG.md`** restructured to Keep-a-Changelog 1.1.0 with consistent
  section ordering.
- **Plotly hover labels** restyled with vermillion border on warm charcoal
  background.

### Preserved (deliberate non-changes — multi-agent review rejected reskinning)
- **Trader signal semantics.** `--buy: #10B981`, `--sell: #EF4444`,
  `--neutral: #F59E0B` are sacred and untouched. Skeptic + User Advocate
  rejected any change to these.
- **All math primitives** in `analytics/`. The DFII review and the system
  Skeptic both confirmed the *individual* math is sound; risk is in
  composition and calibration, not in any single function.

### Docs
- New `docs/` directory with two long-form documents (architecture and
  design review).
- Streamlit Cloud secrets setup documented inline in the README with
  exact TOML example.
- Math primitives now have inline citations: Andrews (1993), Bai & Perron
  (2003), Peng et al. (1994), Ratcliff & McKoon (2008).

### Known issues (all tracked in `docs/DESIGN_REVIEW.md`)
- **CG-1** No engine output caching → wasted compute on every Streamlit
  interaction. *Hotfix in 1.2.1.*
- **CG-3** yfinance has no retry/backoff → reliability cliff. *Hotfix in 1.2.1.*
- **CG-4** `requirements.txt` missing upper bounds → numpy 2.0 ABI risk.
  *Hotfix in 1.2.1.*
- **S-1** Walk-forward ensemble may overfit to trailing fold. *Target 1.3.0.*
- **S-2** DDM-on-DDM compounds lag (true half-life ~110–140 d, not 80).
  *Target 1.3.0.*
- **S-3** HMM 0.98 self-transition is structurally late. *Target 1.3.0.*
- **S-9** No regression test suite. *Target 1.3.0.*
- **UA-1** Headline conviction not visually privileged above tabs.
  *Target 1.3.0.*

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

## [3.2.2] — 2026-04-05 — *Production Deployment Hardening*

Last release under the legacy "Aarambh-only" version line before the
unification into Nishkarsh 1.x.

### Fixed
- **Devcontainer config.** Container name corrected from `v2.0` to `v3.2.2`;
  entry point fixed from `aarambh.py` to `correl.py`.
- **Dynamic version logging.** Replaced hardcoded `v3.2.1` in engine
  startup logs with the dynamic `VERSION` constant.

### Changed
- **Production metadata.** Version references updated across
  `requirements.txt`, `LICENSE.md`, and documentation.

---

## [3.2.1] — 2026-04-01 — *Phase II Logic Guards + Phase III Production*

A heavy mathematical-correctness pass plus a major UI reorganisation.

### Fixed
- **Dynamic ensemble inverse-MAE weights.** Replaced static simple averages
  across walk-forward predictions. Ensemble constituents are now weighted
  by inverse-MAE on a trailing OOS validation set, with a 5% baseline floor.
- **Double-bounding singularity.** `ConvictionScore` now natively applies
  and exports `ConvictionBounded`, enforcing identical numerical state
  between the underlying logic loops and the display UI.
- **DDM state initialisation bias.** `drift_diffusion_filter` now initialises
  transient state arrays via rolling forward sums of the first 20 observations,
  eliminating early systemic drag.
- **Temporal record alignment.** Feature impact matrix updates structurally
  map to block end times `t_index`, not global limits.
- **OU dynamics.** Projection bands now override full-sample estimations
  with trailing `theta_history[-1]` values, reflecting localised confidence.
- **Hurst stationarity guard.** ADF tests run before DFA-Hurst invocations.
  If signals diverge from base stationarity, deterministic bounds return
  `H = 0.5` instead of fabricating fractional trend structure.

### Changed
- **Tab overhaul.** Collapsed five fragmented tabs into four logically
  flowing tabs: **Dashboard · Breadth Topology · ML Diagnostics · Data Table**.
- **Primary signal relocation.** Moved from inside a tab to above all tabs
  as an always-visible, full-width card.
- **Timeframe filter** replaced radio buttons with a Swing-style button row
  (1M / 6M / 1Y / 2Y / ALL).
- **Unified chart margins** — `dict(t=10, l=60, r=20, b=10)` across all Plotly figures.
- **Consistent line widths** — 2.5px for main data lines, 1px for reference lines.
- **Standardised legends** — horizontal orientation, top-right, font size 9.
- **Base conviction plot** — threshold markers at ±20 (small) and ±40 (large), colour-coded.
- **DDM interpretation card** — dynamic 5-state analysis matching the
  Regime Context style.

### Removed
- **~500 lines of dead code** — old tab rendering functions eliminated.

### Performance
- **Threading removed** from walk-forward execution. Sequential processing
  eliminates the `ThreadPoolExecutor` warnings.
- **Streamlit API updated** — `use_container_width=True` migrated to
  `width="stretch"` (15+ instances).

### Refactored
- **DRY helper extraction** — `_render_metric_card()` for consistent card rendering.

---

## [3.2.0] — 2026 — *Phase I Architectural Hardening*

### Fixed
- **True conformal quantiles.** Replaced pseudo-Gaussian z-score
  implementations with the rigorous empirical `compute_conformal_zscores`
  primitive. Signal zones now mathematically align with fat-tailed reality.
- **Bai-Perron regime binding.** Expanding windows now bind to the most
  recent structural break (`max(max_lookback, last_break)`), severing
  legacy coefficients from polluting OOS regressions across regime shifts.
- **DDM variance capping.** `drift_diffusion_filter` now caps scaling
  variance geometrically (`min(abs(drift)·0.5, long_run_var·0.5)`) to
  prevent ballooning standard errors during prolonged regimes.
- **ElasticNetCV silent exception.** Repaired the `ElasticNetCV(n_alphas=10)`
  silent exception bug, restoring ElasticNet to the walk-forward ensemble.

### Added
- **Thread-safe walk-forward.** Re-established `ThreadPoolExecutor` async
  processing safely behind state-mutation locks.

---

## [3.1.0] — 2026 — *Initial Refactor*

### Fixed
- **Look-ahead bias in z-score computation.** Rolling mean/std now use
  `shift(1)` so only past data feeds standardisation. The previous
  implementation included the current residual in its own z-score,
  inflating OOS R² by 10–20%.
- **Jackknife correction for near-unit-root AR(1).** Replaced biased
  jackknife estimator with the **Andrews (1993)** median-unbiased
  correction. Half-life estimates are now accurate for persistent series
  (θ ≈ 0.95).
- **DDM variance unbounded growth.** Added the mean-reverting variance
  term: `σ²_t = (1−λ)·σ²_{t−1} + λ·σ²_LR`. Confidence bands are now stable
  during extended regimes.

### Added
- **Structural break detection** — Bai-Perron multiple breakpoint testing
  before walk-forward. Expanding window resets if a regime shift is
  detected within the trailing period.
- **DFA Hurst exponent** — replaced biased R/S estimator with **Detrended
  Fluctuation Analysis** (Peng et al. 1994). Lag range: `max(4, n/10)` to `n/4`.
- **Significance testing** — t-statistics and p-values for all signal
  performance metrics.
- **Rolling θ estimation** — time-series of OU mean-reversion speed over
  trailing 60-day windows.
- **Feature impact history** — time-varying feature weights stored at each
  refit interval.
- **Conformal quantile z-scores** — empirical quantile-based z-scores
  preserve fat tails.
- **θ stability diagnostic** — metric card showing whether mean-reversion
  speed is stable (CV < 50%).

### Changed
- **Thread-safe walk-forward execution** — `ThreadPoolExecutor` replaced
  with lock-protected sequential execution.
- **Conviction score soft bounds** — `tanh` transformation:
  `Conviction_bounded = 100 · tanh(Conviction_raw / 100)`.
- **Winsorised forward changes** — forward % changes capped at ±100% to
  prevent spurious outliers.

### Removed
- **Dead code** — `kalman_filter_1d`, `_compute_kalman_conviction`
  (fully replaced by DDM).

### Refactored
- **DRY helper extraction** — `_safe_array_operation`, `_classify_zones`,
  `_detect_crossover_signals`, `_compute_significance`,
  `_apply_conviction_bounds`.

### Performance impact

| Metric | Before | After | Change |
|---|---|---|---|
| OOS R² | inflated | honest | −10% to −20% |
| Half-life estimate | biased low | unbiased | +20% to +40% |
| Confidence band stability | diverging | stable | fixed |
| Thread safety | race condition | lock-protected | fixed |

---

## [3.0.0] — 2026 — *ADAM Refactor*

### Added
- **Adaptive conformal unbounded bounds** — native continuous-scale
  expanding empirical quantiles mapping limits in OOS residuals.
- **Drift-Diffusion Accumulator (DDM)** — sequential probability ratio
  test integrating evidence signals into an orthogonal bounding space.
- **Strict decoupled OU regression** — stationary variance projections
  derived exclusively via `ln(2)/θ`.

### Changed
- **Replaced rank approximation bounding** (`stats.norm.ppf()`) with
  conformal unbounded standard deviations, repairing extreme black-swan
  tail-clipping bugs.
- **Decoupled `dynamic_theta`** from the arbitrary `vol_multiplier` to
  prevent illegal OU diffusion contraction.
- **UI refactor** across Streamlit cards to accurately reflect new DDM
  and conformal realities.

### Removed
- `kalman_filter_1d` and `_compute_kalman_conviction` — superseded by DDM.

---

## [2.2.1] — Prior release

### Fixed
- **Dataset transition crash.** Resolved state desynchronisation when
  switching between datasets with non-overlapping predictor columns.
- **Predictor mutation cache bypass.** Inline edits to predictor variables
  were being ignored by the Streamlit session-state cache when the target
  sum remained identical.

---

## [2.2.0] — Prior release

### Performance
- **Vectorised walk-forward engine.** Massively optimised by vectorising
  prediction loops across multi-row execution chunks.
- **sklearn LinearRegression swap.** Replaced heavy `statsmodels` WLS with
  `scikit-learn` `LinearRegression`.
- **ElasticNetCV tuning.** Repaired parameter grid; relaxed `HuberRegressor`
  convergence tolerance to dramatically reduce ensemble fit times.

### Changed
- Devcontainer environments updated to v2.2.0.

---

## [2.1.0] — Prior release

### Added
- `CHANGELOG.md` to track version history.
- **Dynamic feature impact tracking** using WLS coefficients and PCA
  back-projection.

### Changed
- Reorganised dashboard diagnostic cards with concise sub-metrics for
  uniform card heights.

### Fixed
- **`ConvergenceWarning`** — resolved `sklearn.linear_model._coordinate_descent`
  warning by raising `max_iter` to `10000`, relaxing `tol`, and filtering
  non-consequential warnings.

---

## [2.0.0] — Prior release — *Major update*

### Added
- **Walk-forward expanding-window ensemble regression** (Ridge + Huber + OLS)
  for fair-value estimation.
- **Ornstein-Uhlenbeck mean-reversion** parameter estimation on OOS residuals.
- **Kalman-filtered breadth conviction** scoring with 95% confidence bands.
- **Residual Hurst exponent** to validate mean-reversion empirical properties.
- **Swing-based divergence detection** using local extrema logic.
- **"Apply & Compute" staging** to prevent heavy engine recalculations
  during feature selection.
- **Data staleness warnings.**

### Changed
- **OOS-only validation framework.** Transitioned from in-sample baseline
  metrics to strictly out-of-sample validation.
- **Theme update** — `@thebullishvalue` Design System styling applied.

---

## [1.0.0] — Initial release

### Added
- Static multi-lookback z-score banding.
- Fair-value linear models.

---

[1.2.0]: #120--2026-04-09--editorial-quant-terminal
[1.1.0]: #110--2026-04-07--nishkarsh-production-release
[3.2.2]: #322--2026-04-05--production-deployment-hardening
[3.2.1]: #321--2026-04-01--phase-ii-logic-guards--phase-iii-production
[3.2.0]: #320--2026--phase-i-architectural-hardening
[3.1.0]: #310--2026--initial-refactor
[3.0.0]: #300--2026--adam-refactor
[2.2.1]: #221--prior-release
[2.2.0]: #220--prior-release
[2.1.0]: #210--prior-release
[2.0.0]: #200--prior-release--major-update
[1.0.0]: #100--initial-release
