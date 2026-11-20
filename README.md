<div align="center">

# NISHKARSH

**निष्कर्ष** · *conclusion, inference*

Top-Down Forecast · Bottom-Up Breadth · One Inference

[![Version](https://img.shields.io/badge/version-2.5.23-2B5FD9)](CHANGELOG.md) [![Python](https://img.shields.io/badge/python-3.13-3776AB)](requirements.txt) [![Streamlit](https://img.shields.io/badge/streamlit-1.52.2-FF4B4B)](requirements.txt) [![License](https://img.shields.io/badge/license-proprietary-6B7280)](LICENSE) [![Pragyam Family](https://img.shields.io/badge/pragyam-family-0F1217)](#the-pragyam-family)

</div>

> Aarambh forecasts the Nifty from the top down, Nirnay reads its fifty stocks from the bottom up,
> and Nishkarsh reads them as one: graded by its own walk-forward record, with the weakest link
> named.

## Overview

**Nishkarsh** is a quantitative conclusion engine for the Nifty 50. A top-down system
(**Aarambh**) forecasts the index; a bottom-up system (**Nirnay**) reads the 50
constituents. The conclusion layer combines them only as far as the data prove the
combination helps, and every run reports — with sample sizes, test statistics and
p-values — what is and is not established.

| | |
|---|---|
| **Universe** | Nifty 50 (live list from niftyindices.com; dated fallback list) |
| **Stack** | Python 3.13 · Streamlit 1.52.2 (pinned) · scikit-learn · statsmodels · numba · Plotly |

> **A note on rigor.** Nishkarsh is built to be honest about its own edge: causal
> everywhere (prefix-invariant, tested), deterministic, h-spaced significance with
> HAC and block-bootstrap inference, null error rates verified by simulation, and
> evidence attached to the exact number shown. It has **not demonstrated a tradeable
> edge on real data**: re-measured with the purged 2.0 engine on the owner's sheet, the
> earlier +0.14 directional lead (which was contaminated by look-ahead) does not
> replicate (IC −0.03, 2014–2026, 95% CI [−0.12, +0.05]). Read it as an honest
> research instrument, and read [`docs/EDGE_LEDGER.md`](docs/EDGE_LEDGER.md) first.

## What It Does

| | **AARAMBH** *(आरंभ — "Beginning")* | **NIRNAY** *(निर्णय — "Judgement")* |
|---|---|---|
| **Direction** | Top-down | Bottom-up |
| **Question** | *Where is the index likely to go over the next 10 sessions?* | *What are the 50 constituents doing?* |
| **Method** | Purged walk-forward ensemble (Ridge · Huber · ElasticNet · PCA-OLS, equal weight) forecasting forward 10-day Δlog(PE) from 20-day predictor momentum | Per-stock MSF + MMR oscillator with an online regime ensemble (adaptive Kalman · 3-state HMM · GARCH · CUSUM), aggregated to daily breadth over live, warm constituents |
| **Output** | `Forecast`, its causal z-score `zA` | Daily breadth; `zN` = causal z of the mean oscillator, read as trend (`NIRNAY_SIGN`) |

A second model that agrees is a stronger bet *only if* agreement actually predicts
better outcomes. Nishkarsh tests exactly that on every run (the **Convergence-Lift
test**) instead of assuming it. Until the test passes, Nirnay is shown as context and
the conclusion is Aarambh's.

Aarambh can also run in its original **levels** mode (fair-value regression on the PE
level, `AARAMBH_FORWARD_SIGNAL = False`). That mode is kept as *valuation context*: as
a forecaster it was falsified (Edge Ledger L-01) and the UI says so.

## Quick Start

```bash
pip install -r requirements.txt          # Python 3.13
streamlit run app.py
```

In the sidebar choose a data source:

- **Google Sheets** — set the sheet URL as a secret (`.streamlit/secrets.toml`, see
  `.streamlit/secrets.toml.example`) or `AARAMBH_GOOGLE_SHEETS_URL`; a configured URL
  is never sent to the browser. Without one you can paste a URL.
- **Upload** — a CSV/Excel export of the sheet (ISO or day-first dates both work).

Then **Run Analysis**. The model is not user-configurable: the target (`NIFTY50_PE`), the date
column and the predictors are fixed defaults. The sheet needs at least `MIN_TRAIN_SIZE + horizon + MIN_OOS_ROWS`
= 1500 + 10 + 252 usable rows of daily history (after the 20-row momentum warm-up).

**Streamlit Cloud:** push the repo, connect it at share.streamlit.io, and add the
secret `[aarambh] google_sheets_url = "…"`.

## Testing

```bash
pytest -m "not slow" -q     # 149 tests, ~1.5 min
pytest -m slow -q           # Monte-Carlo size/power, end-to-end determinism, Streamlit AppTest
```

CI (`.github/workflows/ci.yml`) runs the fast suite on every push and the slow suite
nightly. The suite enforces the purge invariant, prefix invariance, determinism,
kernel/oracle parity, null false-positive rates, end-to-end null-world honesty, the
hero's presentation rules (no call without evidence, escaping, no NaN text), the
no-pickle cache, the JSON schema of the conclusion and the package layering.

## Project Structure

```
analytics/  →  engines/  →  convergence/  →  pipeline/  →  app.py + ui/
data/  — the only IO boundary            core/  — config (validated at import) + logging
```

```
app.py              Streamlit shell: sidebar → PipelineConfig → run_pipeline → session state → tabs
pipeline/           config (fingerprint) · inputs · macro · run · report · CLI (__main__)
analytics/          causal_stats · backtest · conformal · ddm_filter · ou_process · hurst ·
                    structural_breaks (heuristic) · regime · factors · custom_features · analogs · utils
engines/            aarambh (purged walk-forward forecast) · nirnay (constituent breadth)
convergence/        conclusion (the conclusion layer) · cross_validator (agreement anatomy) · labels
data/               fetcher · cache · circuit_breaker · constituents · dates · macro_lag ·
                    sources · synthetic
ui/                 theme.css + theme.py (Graphite design system) · palette · components ·
                    copy · tabs/ (page_overview · aarambh · nirnay · convergence · precedent ·
                    system · data)
tests/              unit, statistical-regression and layering tests (pytest)
docs/               EDGE_LEDGER · ARCHITECTURE · AUDIT · FINDINGS · PERFORMANCE · CUSTOM_PREDICTORS
```

## Documentation

| Document | What it covers |
|---|---|
| [Guide](docs/GUIDE.md) | Each page, the conclusion and the Nishkarsh Oscillator, the data layer, configuration, performance, limits, troubleshooting, references |
| [Edge Ledger](docs/EDGE_LEDGER.md) | Every claim the system makes and its measured status: read this first |
| [Architecture](docs/ARCHITECTURE.md) | The layers, their boundaries and the data flow |
| [Findings](docs/FINDINGS.md) | What the measurements established |
| [Audit](docs/AUDIT.md) | The audit behind the 2.x rebuild |
| [Performance](docs/PERFORMANCE.md) | Timings, history and non-optimisations |
| [Custom predictors](docs/CUSTOM_PREDICTORS.md) | Adding a predictor to the sheet |
| [Research](docs/research/) | Pre-registrations and results: the Agreement and Nishkarsh Oscillators, Nirnay's polarity |
| [Evidence](docs/evidence/) | The run reports the ledger cites |
| [Changelog](CHANGELOG.md) | Every release, newest first |

## The Pragyam Family

Nine Streamlit applications built on one design system and one set of components, so a
panel, a table, a chart or a number reads the same in each.

| Product | What it is |
|---|---|
| [**Tattva**](https://github.com/thebullishvalue/Tattva) | Cross-asset fair value, with a call only where its own record supports one |
| **Nishkarsh** · *this repository* | Nifty 50 inference: a top-down forecast and bottom-up breadth, read as one |
| [**Arthagati**](https://github.com/thebullishvalue/Arthagati) | Valuation-anchored market sentiment, tested on data it never saw |
| [**Pragyam**](https://github.com/thebullishvalue/Pragyam) | Portfolio curation that spreads risk rather than predicting returns |
| [**Sanket**](https://github.com/thebullishvalue/Sanket) | A market signal screener: what fired, what is about to, and whether it has paid |
| [**Vaaydo**](https://github.com/thebullishvalue/Vaaydo) | An F&O trade screener on option positioning, with futures tickets |
| [**Nivesa**](https://github.com/thebullishvalue/Nivesa) | A bond portfolio ledger: cost, yield, duration and cashflows across accounts |
| [**Samhita**](https://github.com/thebullishvalue/Samhita) | A portfolio tracker: live prices, performance and risk against the benchmark |
| [**Swing**](https://github.com/thebullishvalue/Swing) | A portfolio tracker for the swing book, on the same engine as Samhita |

## License

Proprietary. © 2026 @thebullishvalue. All rights reserved. The source is visible here
for reference only: viewing it grants no right to use, copy, modify or distribute it.
See [LICENSE](LICENSE).

## Disclaimer

Nishkarsh is a research and decision-support tool, not investment advice. Nothing it shows is
a recommendation to buy or sell any security, and past or measured results do not guarantee
future ones. Every decision made with it is your own. It has not demonstrated a tradeable
edge on real data; read the [Edge Ledger](docs/EDGE_LEDGER.md) before relying on any output.

---

<div align="center"><sub>© 2026 @thebullishvalue · Part of the Pragyam product family</sub></div>
