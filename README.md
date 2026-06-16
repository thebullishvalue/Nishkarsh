# Nishkarsh

> **निष्कर्ष** *(Nishkarsha)* — *"Conclusion / Inference."*
> Two systems. One conclusion.

**Nishkarsh** is a unified quantitative convergence engine for the Nifty 50.
It runs two **orthogonal** analytical systems in parallel, treats their
agreement — and their disagreement — as the signal, and **self-calibrates**
its convergence weights and thresholds against forward returns on every run.

| | |
|---|---|
| **Version** | 1.4.14 — *Edge-Honest Convergence* |
| **Stack** | Python 3.12+ · Streamlit · scikit-learn · statsmodels · Optuna · Plotly |
| **Universe** | Nifty 50 (live, fetched from niftyindices.com) |
| **License** | See `LICENSE.md` |

> **A note on rigor.** Nishkarsh is built to be *honest about its own edge*. Every
> signal is evaluated causally (no look-ahead, deterministic), with persistence-aware
> reporting (`R² vs RW`, not just OOS R²), **non-overlapping** significance tests,
> and fold-stability gating on the self-calibration. Under that scrutiny it has **not**
> found a robust, tradeable forecasting edge on the Nifty-50 PE target — and it
> **says so**, in the UI and the run log (edge verdicts, `INVERTED` / `NO-EDGE`
> flags). Read it as a rigorous **valuation / breadth / regime / divergence context**
> instrument, not a return-signal generator. The full, evidence-based conclusions are
> in [`FINDINGS.md`](FINDINGS.md); the iteration history is in [`CHANGELOG.md`](CHANGELOG.md).

---

## Table of Contents

- [Why Two Systems](#why-two-systems)
- [Quick Start](#quick-start)
- [What You See](#what-you-see)
- [The Convergence Score](#the-convergence-score)
- [Intelligence Mode](#intelligence-mode)
- [Resilient Data Layer](#resilient-data-layer)
- [Divergence Types](#divergence-types)
- [Architecture](#architecture)
  - [Directory Structure](#directory-structure)
  - [Layering Rule](#layering-rule)
  - [Pipeline Flow](#pipeline-flow)
- [Configuration](#configuration)
- [Performance Profile](#performance-profile)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Academic References](#academic-references)

---

## Why Two Systems

A single quantitative model is one bet. Two **orthogonal** models that agree
are a stronger bet — and when they disagree, the disagreement is itself
information.

| | **AARAMBH** *(आरंभ — "Beginning")* | **NIRNAY** *(निर्णय — "Judgement")* |
|---|---|---|
| **Direction** | Top-down | Bottom-up |
| **Question** | *Is the index fairly valued?* | *What are the 50 constituents doing?* |
| **Method** | Walk-forward ensemble regression on the Nifty 50 PE ratio with conformal intervals, DDM smoothing, and OU mean-reversion diagnostics | Per-stock MSF (Market Strength Factor) + MMR (Macro-Micro Regime) with a four-method regime ensemble |
| **Math** | Ridge · Huber · OLS · ElasticNet · PCA-WLS · Bai-Perron breaks · DFA-Hurst · Andrews median-unbiased AR(1) | Adaptive Kalman · 3-state HMM · GARCH-like · CUSUM · sigmoid-bounded composite |

The **Convergence layer** cross-validates both systems across four dimensions
(Direction · Breadth · Magnitude · Regime), adaptively reweights by per-dimension
clarity, runs the result through a Drift-Diffusion filter, and emits a final

> **`nishkarsh_conviction ∈ [−100, +100]`**
>
> with 95% confidence bands and a divergence classifier on top.

**Maximum conviction emerges only when both systems converge.** Neither system
alone can produce this insight.

---

## Quick Start

### Prerequisites

- **Python 3.12** or higher
- **pip** (bundled with Python)
- Internet connection (for live data from niftyindices.com and yfinance)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Data Source

Nishkarsh pulls its primary Aarambh data (Nifty 50 PE ratio, breadth metrics)
from a Google Sheet. Set the URL via environment variable:

```bash
export AARAMBH_GOOGLE_SHEETS_URL="https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit?gid=<GID>"
```

Or leave it unset — the app will prompt you for a URL in the sidebar at runtime.

### 3. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`), select a
data source in the sidebar, and click **Run Analysis**.

A full pipeline run takes ~30–50 seconds on a cold cache, ~3–5 seconds when
data is cached.

### Streamlit Cloud Deployment

1. Push to a GitHub repository.
2. Connect at [share.streamlit.io](https://share.streamlit.io).
3. **Settings → Secrets** and add:

   ```toml
   [aarambh]
   google_sheets_url = "https://docs.google.com/spreadsheets/d/.../edit?gid=..."
   ```

4. Deploy. `app.py` resolves the secret via `st.secrets["aarambh"]["google_sheets_url"]`,
   falling back to the `AARAMBH_GOOGLE_SHEETS_URL` environment variable.

---

## What You See

| Surface | What It Shows |
|---|---|
| **CONVERGENCE** tab | Headline conviction, normalised Aarambh-vs-Nirnay overlay, agreement metrics, divergence event timeline |
| **AARAMBH** tab | Base conviction with DDM bands, fair-value plot, model quality (R², MAE), breadth distribution, feature-impact evolution |
| **NIRNAY** tab | Constituent oversold / overbought distribution, signal counts, HMM regime probabilities, per-stock table |
| **DIAGNOSTICS** tab | OU half-life and θ-stability, DFA-Hurst, signal-crossover performance, feature importance, **Data Layer Health** (cache + circuit-breaker state), **Intelligence Center** (read-only profile diagnostics — Train/Val IC, learned weights, threshold cards, fANOVA factor sensitivity, saved profiles) |
| **DATA** tab | Merged time-series table + CSV export |
| **Sidebar — Model Passport** | Active profile state (Default / Calibrated / Calibrated · ⚠), Trained-on label, Train IC, Val IC, last-updated timestamp, Intelligence Mode toggle, Import / Export / Reset controls |

---

## The Convergence Score

Direction (30%) measures sign agreement between the two systems. Breadth (25%)
measures alignment between Aarambh's valuation extreme and Nirnay's oversold-%
breadth. Magnitude (25%) checks that signal strengths are comparably scaled.
Regime (20%) checks that Aarambh's OU regime classification is consistent with
the HMM regime distribution from the constituents. Each dimension produces a
clarity score; the base weights are shifted up to ±10% in favour of higher-clarity
dimensions, then renormalised. The 4-D composite is mapped to `[−100, +100]`,
classified into one of nine zones, and finally smoothed by a leaky drift-diffusion
filter with mean-reverting variance to produce the headline `nishkarsh_conviction`.

---

## Intelligence Mode

Nishkarsh v1.4 ships with **self-calibrating convergence** as the default
pipeline path. Every Run Analysis runs a single end-to-end flow that learns
the optimal convergence weights and signal thresholds for the active universe,
applies them in the same run, and persists them for the next one.

### What gets calibrated

| Parameter | Default | Calibrated range |
|---|---|---|
| `w_direction` | `0.30` | `[0.10, 0.50]` |
| `w_breadth`   | `0.25` | `[0.10, 0.50]` |
| `w_magnitude` | `0.25` | `[0.10, 0.50]` |
| `w_regime`    | `0.20` | `[0.10, 0.50]` |
| `buy_strong`     | `−0.50` | asymmetric long threshold |
| `buy_moderate`   | `−0.30` | asymmetric long threshold |
| `sell_moderate`  | `+0.30` | asymmetric short threshold |
| `sell_strong`    | `+0.50` | asymmetric short threshold |

Weights are renormalised to sum to 1; thresholds are constrained so
`buy_strong < buy_moderate < 0 < sell_moderate < sell_strong`.

### Calibration objective

Maximise the **Spearman Information Ratio** of the composite convergence
signal against forward NIFTY-50-PE returns at horizons `[3, 5, 10, 20]`
trading days, with **L2 regularisation toward uniform weights** to discourage
overfit. Validates on a **chronological 70/30 train/val split** — no
shuffling, no leakage.

The optimiser is **Optuna TPE** (Tree-structured Parzen Estimator), with
configurable trial count (default 50 — sufficient for IC convergence on
~250-day windows).

### One-flow execution

A single Run Analysis click executes:

```
1. First-pass CrossValidator  ── prior profile (or factory defaults)
2. Initial UnifiedConvictionModel fit (populates convergence series)
3. Optuna TPE calibration on fresh (convergence_df, aarambh_ts)
4. apply_calibrated_weights() ── vectorized re-weight, no re-loop
5. Conviction model re-fit on recomputed convergence
6. Normalised convergence classified with asymmetric thresholds
7. Profile persisted to ~/.cache/nishkarsh/intelligence/profiles.json
```

The Passport sidebar updates on the post-analysis rerun to show the
freshly-saved profile.

### Toggle behaviour

The **Intelligence Mode toggle** (Passport sidebar, default ON) controls
whether steps 3–6 run. When OFF, the pipeline uses the factory
`0.30 / 0.25 / 0.25 / 0.20` weights and the symmetric `±0.30 / ±0.50`
thresholds throughout — useful for A/B baseline comparisons.

### Per-universe profile keying

Profiles are keyed by `(universe · selected_index)` and versioned
(`PROFILE_VERSION = "v1-nishkarsh-convergence"`). If you switch universes
between runs, the Passport surfaces a **Profile mismatch — calibrated
weights still active** warning until you re-run.

### Progress visibility

The pipeline progress bar surfaces every Intelligence Mode sub-stage —
`Setup → Calibrating (with live trial counter) → Profile Saved → Applying
Calibrated Profile → Re-Fitting Conviction Model`. Phase completion shows
either `calibrated profile applied` or `factory defaults` so you always
know which path ran.

---

## Resilient Data Layer

Every external call sits behind three production-grade primitives:

- **Circuit breaker** (`data/circuit_breaker.py`) — `CLOSED → OPEN → HALF_OPEN`
  state machine per service, thread-safe. Two module-level breakers protect
  `yfinance` and Google Sheets respectively.
- **Retry with backoff** — exponential decorator (1s → 2s → 4s, capped at 60s,
  max 3 retries).
- **Two-tier cache** (`data/cache.py`) — memory + disk, TTL expiry, versioned
  keys (`version="v1"` bump invalidates a namespace atomically), and a
  `get_stale()` last-good-snapshot fallback used automatically when a fetch
  fails *and* the circuit is open — the UI keeps working through API outages.

Open the **Diagnostics → Data Layer Health** card for live per-namespace
cache hit rate, disk entry count, last-fetch timestamp, and per-service
circuit-breaker state.

The macro universe is **66 yfinance bond ETF tickers** (US Treasury curve,
TIPS, aggregate bonds, corporate IG / HY, mortgage-backed, municipals,
developed-markets sovereign, India fixed income, emerging markets) plus
18 commodity / FX tickers — replacing the broken Stooq endpoints that
returned HTML errors in late 2025.

---

## Divergence Types

When Aarambh and Nirnay *disagree*, the disagreement is a typed signal:

| Type | Meaning | Trader Read |
|---|---|---|
| `AARAMBH_LEADS` | Valuation is at an extreme but the constituents have not turned | Early warning — index price is stretched, breadth has not confirmed |
| `NIRNAY_LEADS` | Constituents are extreme but valuation is still neutral | Momentum-first move — breadth is confirming before the index does |
| `CONTRADICTION` | Persistent disagreement (≥5 events in a 20-day window) | Uncertain regime — reduce position size |

---

## Architecture

### Directory Structure

```
app.py                         Streamlit entrypoint and pipeline orchestrator

core/
  config.py                    Constants, thresholds, palettes (single source of truth)
  logger_config.py             Terminal logging with color and run IDs

analytics/                     Pure functions. No state. No IO. Independently testable.
  conformal.py                 Empirical-quantile prediction intervals (fat-tail adjustment)
  ddm_filter.py                Drift-Diffusion leaky integrator with mean-reverting variance
  hurst.py                     DFA Hurst exponent with ADF stationarity guard
  ou_process.py                Ornstein-Uhlenbeck with Andrews (1993) bias correction
  regime.py                    Adaptive Kalman / HMM / GARCH-like / CUSUM
  signals.py                   MSF + MMR primitives (Market Strength Factor, Macro-Micro Regime)
  structural_breaks.py         Bai-Perron multiple breakpoints (with fallback)
  utils.py                     sigmoid, zscore_clipped, ATR, soft bounds, zone classifier

data/                          Sole IO boundary. Production-grade resilience.
  fetcher.py                   Google Sheets + yfinance — circuit-protected, retry-wrapped
  constituents.py              Nifty 50 list (live from niftyindices.com + fallback)
  schema.py                    UnifiedDataset dataclass contracts
  cache.py                     Two-tier (memory + disk) TTL cache with versioned keys
                               and last-good-snapshot fallback
  circuit_breaker.py           CLOSED → OPEN → HALF_OPEN per-service + retry-with-backoff

engines/                       Stateful orchestrators. Import from analytics/.
  aarambh.py                   FairValueEngine — top-down walk-forward ensemble regression
  nirnay.py                    Per-constituent MSF + MMR + regime ensemble

convergence/                   Consumes both engines. No IO. No Streamlit.
  cross_validator.py           4-dimension convergence scoring (weights either
                               from the calibrated profile or factory defaults)
  conviction_model.py          UnifiedConvictionModel — DDM on convergence scores
  divergence_detector.py       Cross-system divergence detection and classification
  normalization.py             Shared z-score / align / classify math for the
                               Convergence cards + Unified Signal plot (single source).
                               Thresholds either calibrated (asymmetric) or default (±)
  intelligence.py              Self-calibration layer — Optuna-TPE search for
                               (weights, thresholds), Spearman-IR objective vs
                               forward returns, chronological 70/30 train/val,
                               disk persistence at
                               ~/.cache/nishkarsh/intelligence/profiles.json,
                               apply_calibrated_weights() for vectorized re-weight

ui/                            Rendering only. No math.
  theme.py                     Obsidian Quant Terminal CSS + Plotly defaults
  components.py                Reusable widgets (metric cards, signal badges, headers)
  tabs/                        One file per tab (aarambh, nirnay, convergence, diagnostics, data)

CHANGELOG.md                   Version history (Keep-a-Changelog 1.1.0)
LICENSE.md                     Proprietary license — @thebullishvalue
requirements.txt               Dependency specifications
```

### Layering Rule

```
analytics/  →  engines/  →  convergence/  →  ui/
```

Never upward, never sideways. `data/` is the only IO boundary. `ui/` reads from
`st.session_state` and never imports from `engines/` directly.

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  app.py — Streamlit Entry Point                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. DATA ACQUISITION                                      │  │
│  │     fetcher.py → Google Sheets + yfinance (84 macro       │  │
│  │       tickers, Sanket's Global Macro universe)            │  │
│  │     circuit_breaker.py → per-service fault tolerance      │  │
│  │     cache.py → two-tier TTL cache + stale-fallback        │  │
│  │     constituents.py → Nifty 50 symbols                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  2. AARAMBH — FairValueEngine                             │  │
│  │     analytics/ → conformal, ddm, ou_process, hurst,       │  │
│  │                  structural_breaks, utils                 │  │
│  │     Walk-forward ensemble regression (Ridge/Huber/OLS/    │  │
│  │     ElasticNet/PCA-WLS) with conformal prediction bounds  │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  3. NIRNAY — Constituent Regime Intelligence              │  │
│  │     analytics/ → regime (Kalman/HMM/GARCH/CUSUM),         │  │
│  │                  signals (MSF/MMR), hurst                 │  │
│  │     Per-constituent MSF + MMR analysis aggregated to      │  │
│  │     daily breadth statistics                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  4. CONVERGENCE — Cross-System Validation                 │  │
│  │     ├─ 4a. First-pass CrossValidator (prior profile)      │  │
│  │     ├─ 4b. Initial UnifiedConvictionModel fit             │  │
│  │     ├─ 4c. intelligence.py → Optuna-TPE calibration       │  │
│  │     │       (Spearman-IR · 70/30 chrono · L2 reg)         │  │
│  │     ├─ 4d. apply_calibrated_weights() vectorized re-weight│  │
│  │     ├─ 4e. Conviction model re-fit on recomputed scores   │  │
│  │     ├─ 4f. normalization → asymmetric threshold classify  │  │
│  │     └─ 4g. divergence_detector → Disagreement events      │  │
│  │     Profile persisted to disk; Passport sidebar refreshes │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  5. UI RENDERING                                          │  │
│  │     ui/tabs/ → Five tab renderers                         │  │
│  │     ui/components.py → Metric cards, signal badges        │  │
│  │     ui/theme.py → Obsidian Quant Terminal design system   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

All tunable constants live in `core/config.py`. The most consequential ones:

| Constant | Default | Affects |
|---|---|---|
| `LOOKBACK_WINDOWS` | `(5, 10, 20, 50, 100)` | Conformal z-score multi-horizon |
| `MIN_TRAIN_SIZE` | `20` | Aarambh walk-forward floor |
| `REFIT_INTERVAL` | `5` | Steps between ensemble refits |
| `DDM_LEAK_RATE` | `0.08` | DDM smoothing (both passes) |
| `OU_PROJECTION_DAYS` | `90` | Forward path length |
| `CONVICTION_STRONG / MODERATE / WEAK` | `60 / 40 / 20` | Signal classification |
| `DIV_PERSISTENCE_THRESHOLD` | `5` | Divergence event flagging |
| `CONV_WEIGHT_DIRECTION / BREADTH / MAGNITUDE / REGIME` | `0.30 / 0.25 / 0.25 / 0.20` | Convergence base weights (overridden when Intelligence Mode is ON) |
| **Intelligence Mode toggle** | `ON` | Sidebar Passport — when OFF, factory weights and symmetric thresholds are used |
| `intel_n_trials` | `50` | Optuna TPE trial count per calibration |
| `PROFILE_VERSION` | `"v1-nishkarsh-convergence"` | Schema version for persisted profiles |

---

## Performance Profile

| Phase | Cold Cache | Warm Cache |
|---|---|---|
| Data acquisition | 5–10 s | ~0 s |
| Aarambh walk-forward | 10–20 s | 10–20 s |
| Nirnay (50 stocks, sequential) | 5–10 s | 5–10 s |
| Convergence (first pass) + DDM | ~1 s | ~1 s |
| Intelligence calibration (Optuna 50 trials) | 5–15 s | 5–15 s |
| Apply + re-fit | <1 s | <1 s |
| Render | <1 s | <1 s |
| **Total (Intelligence ON)** | **35–60 s** | **20–40 s** |
| **Total (Intelligence OFF)** | **30–50 s** | **15–30 s** |

Engine outputs are **not** currently cached; every Streamlit rerun re-runs
the math. Calibration profiles ARE cached to disk and re-used as the
prior on the next run (warm path).

---

## Known Limitations

Nishkarsh is honest about what it does not yet do:

- **No demonstrated tradeable edge on PE.** This is the headline empirical
  finding, not a caveat to bury: under causal, non-overlapping, fold-gated
  evaluation, neither the Aarambh PE model nor the convergence signal shows a
  robust forecasting edge (the Aarambh signal is historically *inverted*; the
  convergence score is an *agreement* metric, not a directional one). The system
  surfaces this honestly rather than hiding it. Full detail in
  [`FINDINGS.md`](FINDINGS.md). Treat the output as context, not alpha.
- **No regression tests.** Pure-function primitives in `analytics/` are
  testable in isolation; a test harness is not yet implemented.
- **No engine output caching.** Streamlit reruns recompute the full pipeline.
- **Survivorship bias.** Nirnay runs against *today's* Nifty 50 over their
  full history. Stocks dropped from the index do not appear.
- **DDM-on-DDM lag.** Stacking the Aarambh DDM and the convergence DDM
  compounds smoothing — true half-life is closer to ~110–140 days than the
  nominal 80.
- **HMM hysteresis.** The 0.98 self-transition is sticky enough to lag
  real regime breaks.
- **GARCH parameters are fixed**, not MLE-fit per series. Treat the GARCH
  output as an EWMA-volatility proxy, not a true GARCH estimate.
- **Bai-Perron fallback.** `statsmodels` ≥ 0.15 exposes `BaiPerronTest` natively.
  Until then, a rolling-mean change-point heuristic is used as a fallback.

---

## Troubleshooting

### Google Sheets Connection

**Error:** `Failed to load data from Google Sheets`

- Verify the sheet URL is accessible (open it in your browser).
- If the sheet is private, ensure your Google account is signed in.
- Set `AARAMBH_GOOGLE_SHEETS_URL` as an environment variable or in Streamlit secrets.

### yfinance Data Gaps

**Warning:** `No macro data available`

- The macro batch covers 84 tickers (66 Global Macro bond ETFs + 18 commodity/FX).
  Individual ticker failures are tolerated; only a full-batch failure surfaces this warning.
- The data layer falls back to the **last-good snapshot** (`cache.get_stale`) when a
  fetch fails and the circuit is open — the UI keeps working through API outages.
- Macro data is supplementary; the core pipeline still runs without it.

### Circuit Breaker Open

**Warning:** `Circuit 'yfinance' is OPEN — retry in 60s`

- After 5 consecutive failures, a service's circuit opens and blocks calls for
  60 seconds before allowing a single test request (HALF-OPEN state).
- Check **Diagnostics → Data Layer Health** for current circuit state and
  failure counts.
- A successful test call closes the circuit automatically; no manual reset needed.

### Stale Cache

**Notice:** `Serving last-good snapshot`

- This is intentional. When a fetch fails *and* the circuit is open, the data
  layer returns the previous successful snapshot (no TTL check) so the UI
  remains functional. The notice is informational, not an error.

### Slow Pipeline Execution

**Issue:** Pipeline takes >60 seconds

- Nirnay processes 50 constituents sequentially. This is CPU-bound.
- On first run, yfinance downloads ~50 stock histories. Subsequent runs
  are faster due to `st.cache_data` with a 1-hour TTL.
- Consider reducing the date range if you need faster iteration.

### Structural Break Detection Warning

**Warning:** `Bai-Perron test failed, using fallback`

- The `BaiPerronTest` class is only available in unreleased statsmodels 0.15+.
- The fallback (rolling-mean change-point detection) provides heuristic results.
- Upgrade to statsmodels ≥ 0.15 when released for full Bai-Perron support.

### Streamlit Import Errors

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

```bash
pip install -r requirements.txt
```

Ensure you're using the correct Python environment. Check with:

```bash
python3 --version   # Should be 3.12+
pip3 --version      # Should match your Python version
```

---

## Academic References

Mathematical primitives draw on:

- **Andrews, D.W.K. (1993).** *Exactly median-unbiased estimation of first-order autoregressive/unit root models.* Econometrica. — Bias-corrected AR(1) coefficient.
- **Bai, J. & Perron, P. (2003).** *Computation and analysis of multiple structural change models.* J. Applied Econometrics. — Regime-break detection.
- **Peng, C.K. *et al.* (1994).** *Mosaic organization of DNA nucleotides.* Phys. Rev. E. — Detrended Fluctuation Analysis (DFA) for Hurst exponent.
- **Ratcliff, R. & McKoon, G. (2008).** *The diffusion decision model.* Neural Computation. — Drift-Diffusion Model (DDM) for sequential evidence accumulation.

---

© 2026 Nishkarsh · [@thebullishvalue](https://twitter.com/thebullishvalue) · v1.4.0
