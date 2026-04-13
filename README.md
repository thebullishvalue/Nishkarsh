# Nishkarsh

> **निष्कर्ष** *(Nishkarsha)* — *"Conclusion / Inference."*
> Two systems. One conclusion.

**Nishkarsh** is a unified quantitative convergence engine for the Nifty 50.
It runs two **orthogonal** analytical systems in parallel and treats their
agreement — and their disagreement — as the signal.

| | |
|---|---|
| **Version** | 1.2.0 — *Obsidian Quant Terminal* |
| **Stack** | Python 3.12+ · Streamlit · scikit-learn · statsmodels · Plotly |
| **Universe** | Nifty 50 (live, fetched from niftyindices.com) |
| **License** | See `LICENSE.md` |

---

## Table of Contents

- [Why Two Systems](#why-two-systems)
- [Quick Start](#quick-start)
- [What You See](#what-you-see)
- [The Convergence Score](#the-convergence-score)
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
- Internet connection (for live data from niftyindices.com, yfinance, Stooq)

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

| Tab | What It Shows |
|---|---|
| **CONVERGENCE** | Headline conviction, normalised Aarambh-vs-Nirnay overlay, agreement metrics, divergence event timeline |
| **AARAMBH** | Base conviction with DDM bands, fair-value plot, model quality (R², MAE), breadth distribution, feature-impact evolution |
| **NIRNAY** | Constituent oversold / overbought distribution, signal counts, HMM regime probabilities, per-stock table |
| **DIAGNOSTICS** | OU half-life and θ-stability, DFA-Hurst with interpretation, signal-crossover performance, feature importance |
| **DATA** | Merged time-series table + CSV export |

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

data/                          Sole IO boundary. Cached @ st.cache_data(ttl=3600).
  fetcher.py                   Google Sheets, yfinance, Stooq — unified fetcher
  constituents.py              Nifty 50 list (live from niftyindices.com + fallback)
  schema.py                    UnifiedDataset dataclass contracts
  cache.py                     TTL-based memory and disk cache

engines/                       Stateful orchestrators. Import from analytics/.
  aarambh.py                   FairValueEngine — top-down walk-forward ensemble regression
  nirnay.py                    Per-constituent MSF + MMR + regime ensemble

convergence/                   Consumes both engines. No IO. No Streamlit.
  cross_validator.py           4-dimension adaptive convergence scoring
  conviction_model.py          UnifiedConvictionModel — DDM on convergence scores
  divergence_detector.py       Cross-system divergence detection and classification

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
│  │     fetcher.py → Google Sheets + yfinance + Stooq         │  │
│  │     constituents.py → Nifty 50 symbols                    │  │
│  │     cache.py → TTL-based caching                          │  │
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
│  │     cross_validator.py → 4D adaptive-weighted scoring     │  │
│  │     conviction_model.py → DDM filtering on convergence    │  │
│  │     divergence_detector.py → Disagreement classification  │  │
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
| `CONV_WEIGHT_DIRECTION / BREADTH / MAGNITUDE / REGIME` | `0.30 / 0.25 / 0.25 / 0.20` | Convergence base weights |

---

## Performance Profile

| Phase | Cold Cache | Warm Cache |
|---|---|---|
| Data acquisition | 5–10 s | ~0 s |
| Aarambh walk-forward | 10–20 s | 10–20 s |
| Nirnay (50 stocks, sequential) | 5–10 s | 5–10 s |
| Convergence + DDM | ~1 s | ~1 s |
| Render | <1 s | <1 s |
| **Total** | **30–50 s** | **15–30 s** |

Engine outputs are **not** currently cached; every Streamlit rerun re-runs
the math.

---

## Known Limitations

Nishkarsh is honest about what it does not yet do:

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

- Stooq may block automated access during high-traffic periods.
- The system falls back to Google Sheets bond yields if available.
- Macro data is supplementary; the core pipeline still runs without it.

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

© 2026 Nishkarsh · [@thebullishvalue](https://twitter.com/thebullishvalue) · v1.2.0
