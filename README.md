# Nishkarsh v1.1.0 — Unified Nifty 50 Convergence Analysis System

**निष्कर्ष** (Nishkarsha) — "Conclusion / Inference"

> Walk-Forward Valuation + Constituent Regime Intelligence → **One Conclusion**

Nishkarsh merges two orthogonal quantitative analysis systems into a single convergence engine:

- **Aarambh (आरंभ — Top-Down):** "Is the Nifty 50 fairly valued?" — Walk-forward ensemble regression on PE ratio using macro predictors, conformal z-scores across 5 lookbacks, DDM conviction filtering, OU mean-reversion physics.
- **Nirnay (निर्णय — Bottom-Up):** "What are the 50 constituents doing?" — Per-stock MSF (price structure) + MMR (macro correlation), regime intelligence (HMM/GARCH/CUSUM/Kalman), aggregated daily.

**Maximum conviction emerges when both systems converge to the same conclusion.** Neither system alone can produce this insight.

---

## Architecture

```
config.py                     # Constants, thresholds, column mappings
logger_config.py              # Terminal logging (Pragyam-style direct console output)

data/
  fetcher.py                  # Unified data acquisition (Google Sheets, yfinance, Stooq)
  cache.py                    # TTL-based memory/disk cache
  schema.py                   # Dataclass contracts
  constituents.py             # Nifty 50 list fetching (niftyindices.com + fallback)

analytics/
  ou_process.py               # Ornstein-Uhlenbeck + Andrews MU estimator
  ddm_filter.py               # Drift-Diffusion Model filter
  hurst.py                    # Hurst exponent via DFA
  conformal.py                # Conformal prediction z-scores
  structural_breaks.py        # Bai-Perron breakpoint detection
  regime.py                   # Kalman, HMM, GARCH, CUSUM
  signals.py                  # MSF + MMR calculators
  utils.py                    # Shared math utilities

engines/
  aarambh.py                  # FairValueEngine (walk-forward regression)
  nirnay.py                   # NirnayEngine + constituent aggregation

convergence/
  cross_validator.py          # 4-dimension adaptive convergence scoring
  conviction_model.py         # UnifiedConvictionModel (DDM on convergence)
  divergence_detector.py      # Cross-system divergence detection

ui/
  app.py                      # Main Streamlit entrypoint
  theme.py                    # Shared CSS and chart theming
  components.py               # Reusable widgets
  tab_convergence.py          # Unified convergence dashboard
  tab_aarambh.py              # Aarambh fair-value views
  tab_nirnay.py               # Nirnay constituent time-series views
  tab_diagnostics.py          # ML diagnostics from both engines
  tab_data.py                 # Merged data table + CSV export

requirements.txt              # Dependencies
README.md                     # This file
CHANGELOG.md                  # Version history
```

## Key Principle

- **`analytics/`** = pure functions (no state, no pandas assumptions)
- **`engines/`** = stateful orchestrators importing from `analytics/`
- **`convergence/`** = new layer consuming both engines' outputs
- **`ui/`** = rendering only, no computation

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run ui/app.py
```

### 3. Configure

- **Sidebar:** Set Google Sheets URL for Aarambh PE/breadth data
- **Constituents:** Auto-fetched from niftyindices.com (all 50 Nifty constituents)
- **Macro Data:** Yahoo Finance (18 commodities/FX symbols) + bond yields from Google Sheets
- Click **Run Analysis** to execute the full convergence pipeline

---

## The Convergence Score

The convergence score is an **adaptive-weighted composite** of 4 dimensions:

| Dimension | Base Weight | What It Measures |
|-----------|------------|-------------------|
| Direction Agreement | 30% | Both systems pointing same way |
| Breadth Confirmation | 25% | Oversold breadth alignment |
| Magnitude Alignment | 25% | Signal strengths comparable |
| Regime Consistency | 20% | OU regime ≈ HMM regime distribution |

**Adaptive weighting:** Each dimension produces a clarity score. Dimensions with higher clarity get up to ±10% weight shift at the expense of lower-clarity dimensions.

**Nishkarsh Conviction (निष्कर्ष):** DDM-filtered convergence score with confidence bands — the unified conclusion.

---

## Divergence Types

| Type | Meaning |
|------|---------|
| `AARAMBH_LEADS` | Valuation extreme but constituents haven't turned → early warning |
| `NIRNAY_LEADS` | Constituents turning but valuation not yet extreme → momentum-first move |
| `CONTRADICTION` | Persistent disagreement → uncertain environment |

---

## Verification

1. **Math parity:** Extracted functions match originals line-by-line
2. **Engine parity:** Aarambh engine produces same output as original system
3. **Nirnay parity:** Per-stock MSF/MMR matches original Nirnay system
4. **End-to-end:** All 5 tabs render with synchronized timezone and data

---

© 2026 Nishkarsh | @thebullishvalue | v1.1.0
