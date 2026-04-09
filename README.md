# Nishkarsh

> **निष्कर्ष** *(Nishkarsha)* — *"Conclusion / Inference."*
> Two systems. One conclusion.

**Nishkarsh** is a unified quantitative convergence engine for the Nifty 50.
It runs two **orthogonal** analytical systems in parallel and treats their
agreement (and their disagreements) as the trading signal.

| | |
|---|---|
| **Version** | 1.2.0 — *Editorial Quant Terminal* |
| **Stack** | Python 3.10+ · Streamlit · scikit-learn · statsmodels · Plotly |
| **Universe** | Nifty 50 (live, fetched from niftyindices.com) |
| **License** | See `LICENSE.md` |

---

## Why two systems?

A single quantitative model is one bet. Two **orthogonal** models that agree
are a much stronger bet — and when they disagree, the disagreement is itself
a tradable signal.

| | **Aarambh** *(आरंभ — "Beginning")* | **Nirnay** *(निर्णय — "Judgement")* |
|---|---|---|
| **Direction** | Top-down | Bottom-up |
| **Question** | *Is the index fairly valued?* | *What are the 50 constituents doing?* |
| **Method** | Walk-forward ensemble regression on the Nifty 50 PE ratio with conformal intervals, DDM smoothing, and OU mean-reversion diagnostics | Per-stock MSF (Market Strength Factor) + MMR (Macro-Micro Regime) with a four-method regime ensemble |
| **Math** | Ridge / Huber / OLS / ElasticNet / PCA-WLS · Bai-Perron breaks · DFA-Hurst · Andrews median-unbiased AR(1) | Adaptive Kalman · 3-state HMM · GARCH-like · CUSUM · sigmoid-bounded composite |

The **Convergence layer** then cross-validates them across four dimensions
(Direction · Breadth · Magnitude · Regime), adaptive-reweights by per-dimension
clarity, runs the result through a Drift-Diffusion filter, and emits a final

> **`nishkarsh_conviction ∈ [−100, +100]`**
>
> with 95% confidence bands and a divergence classifier on top.

**Maximum conviction emerges only when both systems converge.** Neither system
alone can produce this insight.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure the Aarambh data source (Google Sheet with PE / breadth columns)
export AARAMBH_GOOGLE_SHEETS_URL="https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit?gid=<GID>"

# 3. Run
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`), pick a
date range in the sidebar, click **Run Analysis**.

A full pipeline run takes ~30–50 seconds on a cold cache, ~3–5 seconds when
data is cached.

### Streamlit Cloud deployment

1. Push to a GitHub repository.
2. Connect at [share.streamlit.io](https://share.streamlit.io).
3. **Settings → Secrets** and add:

   ```toml
   [aarambh]
   google_sheets_url = "https://docs.google.com/spreadsheets/d/.../edit?gid=..."
   ```

4. Deploy. `app.py` resolves the secret via `st.secrets["aarambh"]["google_sheets_url"]`,
   falling back to the `AARAMBH_GOOGLE_SHEETS_URL` env var if the table is absent.

---

## What you see

| Tab | What it shows |
|---|---|
| **CONVERGENCE** | Headline conviction, normalised Aarambh-vs-Nirnay overlay, agreement metrics, divergence event timeline |
| **AARAMBH** | Base conviction with DDM bands, fair-value plot, model quality (R², MAE), breadth distribution, feature-impact evolution |
| **NIRNAY** | Constituent oversold / overbought distribution, signal counts, HMM regime probabilities, per-stock table |
| **DIAGNOSTICS** | OU half-life and θ-stability, DFA-Hurst with interpretation, signal-crossover performance, feature importance |
| **DATA** | Merged time-series table + CSV export |

---

## The convergence score, in one paragraph

Direction (30%) measures sign agreement between the two systems. Breadth (25%)
measures alignment between Aarambh's valuation extreme and Nirnay's oversold-%
breadth. Magnitude (25%) checks that signal strengths are comparably scaled.
Regime (20%) checks that Aarambh's OU regime classification is consistent with
the HMM regime distribution from the constituents. Each dimension produces a
clarity score; the base weights are shifted up to ±10% in favour of higher-clarity
dimensions, then renormalised. The 4-D composite is mapped to `[−100, +100]`,
classified into one of nine zones, and finally smoothed by a leaky drift-diffusion
filter with mean-reverting variance to produce the headline `nishkarsh_conviction`.

For the math itself — primitives, formulas, file pointers, citations — see
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

---

## Divergence types

When Aarambh and Nirnay *disagree*, the disagreement is a typed signal:

| Type | Meaning | Trader read |
|---|---|---|
| `AARAMBH_LEADS` | Valuation is at an extreme but the constituents have not turned | Early warning — index price is stretched, breadth has not confirmed |
| `NIRNAY_LEADS` | Constituents are extreme but valuation is still neutral | Momentum-first move — breadth is confirming before the index does |
| `CONTRADICTION` | Persistent disagreement (≥5 events in a 20-day window) | Uncertain regime — reduce position size |

---

## Project layout

```
app.py                       Streamlit entrypoint and pipeline orchestrator

core/
  config.py                  Constants, thresholds, palettes (single source of truth)
  logger_config.py           Terminal logging with color and run IDs

analytics/                   Pure functions. No state. No IO. Independently testable.
  ddm_filter.py              Drift-Diffusion leaky integrator
  ou_process.py              Ornstein-Uhlenbeck with Andrews (1993) bias correction
  hurst.py                   DFA Hurst exponent with ADF stationarity guard
  conformal.py               Empirical-quantile prediction intervals
  structural_breaks.py       Bai-Perron multiple breakpoints
  regime.py                  Adaptive Kalman / HMM / GARCH-like / CUSUM
  signals.py                 MSF + MMR primitives
  utils.py                   sigmoid, zscore_clipped, ATR, soft bounds, zone classifier

data/                        Sole IO boundary. Cached @ st.cache_data(ttl=3600).
  fetcher.py                 Google Sheets, yfinance, Stooq
  constituents.py            Nifty 50 list (live + fallback)
  schema.py                  UnifiedDataset dataclass

engines/                     Stateful orchestrators. Import from analytics/.
  aarambh.py                 FairValueEngine (top-down walk-forward)
  nirnay.py                  Per-constituent MSF + MMR + regime ensemble

convergence/                 Consumes both engines. No IO. No Streamlit.
  cross_validator.py         4-dimension adaptive convergence scoring
  conviction_model.py        UnifiedConvictionModel (DDM on convergence)
  divergence_detector.py     Cross-system divergence detection

ui/                          Rendering only. No math.
  theme.py                   Editorial Quant Terminal CSS (Fraunces + IBM Plex)
  components.py              Reusable widgets
  tabs/                      One file per tab

docs/
  ARCHITECTURE.md            System design, math primitives, data flow
  DESIGN_REVIEW.md           Multi-agent design review (known risks + roadmap)

CHANGELOG.md                 Version history (Keep-a-Changelog)
LICENSE.md                   License
requirements.txt             Pinned dependencies
```

**Layering rule:** `analytics/` → `engines/` → `convergence/` → `ui/`. Never
upward, never sideways. `data/` is the only IO boundary. `ui/` reads from
`st.session_state` and never imports from `engines/` directly.

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

Calibration history of these values is **not** currently tracked inline —
that is on the 1.3.0 roadmap (see `DESIGN_REVIEW.md` § CG-7).

---

## Performance notes

| Phase | Cold cache | Warm cache |
|---|---|---|
| Data acquisition | 5–10 s | ~0 s |
| Aarambh walk-forward | 10–20 s | 10–20 s |
| Nirnay (50 stocks, sequential) | 5–10 s | 5–10 s |
| Convergence + DDM | ~1 s | ~1 s |
| Render | <1 s | <1 s |
| **Total** | **30–50 s** | **15–30 s** |

Engine outputs are **not** currently cached; every Streamlit rerun re-runs
the math. Hotfix planned for **1.2.1** (see `DESIGN_REVIEW.md` § CG-1).

---

## Known limitations

This system is honest about what it does not yet do. From the multi-agent
review (`docs/DESIGN_REVIEW.md`):

- **No regression tests.** Pure-function primitives in `analytics/` are
  testable in isolation; harness is on the 1.3.0 roadmap.
- **No engine output caching.** Streamlit reruns recompute everything.
- **Survivorship bias.** Nirnay runs against *today's* Nifty 50 over their
  full history. Stocks dropped from the index do not appear.
- **DDM-on-DDM lag.** Stacking the Aarambh DDM and the convergence DDM
  compounds smoothing — true half-life is closer to ~110–140 days than the
  nominal 80.
- **HMM hysteresis.** The 0.98 self-transition is sticky enough to lag
  real regime breaks.
- **GARCH parameters are fixed**, not MLE-fit per series. Treat the GARCH
  output as an EWMA-volatility proxy, not a true GARCH estimate.

All of these are tracked in `DESIGN_REVIEW.md` with disposition, owner, and
target version.

---

## Documentation map

| Document | For |
|---|---|
| `README.md` *(this file)* | Users, evaluators, first-time visitors |
| `docs/ARCHITECTURE.md` | New maintainers, code reviewers |
| `docs/DESIGN_REVIEW.md` | Anyone modifying `engines/`, `convergence/`, or `analytics/` |
| `CHANGELOG.md` | Version history |

---

## Acknowledgements

Mathematical primitives draw on:

- Andrews, D.W.K. (1993). *Exactly median-unbiased estimation of first-order autoregressive/unit root models.* Econometrica.
- Bai, J. & Perron, P. (2003). *Computation and analysis of multiple structural change models.* J. Applied Econometrics.
- Peng, C.K. *et al.* (1994). *Mosaic organization of DNA nucleotides.* Phys. Rev. E. *(DFA Hurst.)*
- Ratcliff, R. & McKoon, G. (2008). *The diffusion decision model.* Neural Computation. *(DDM.)*

---

© 2026 Nishkarsh · [@thebullishvalue](https://twitter.com/thebullishvalue) · v1.2.0
