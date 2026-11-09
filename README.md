# Nishkarsh

> **निष्कर्ष** *(Nishkarsha)* — *"Conclusion / Inference."*
> Two systems. One conclusion — **and a measurement of whether the second system earns its vote.**

**Nishkarsh** is a quantitative conclusion engine for the Nifty 50. A top-down system
(**Aarambh**) forecasts the index; a bottom-up system (**Nirnay**) reads the 50
constituents. The conclusion layer combines them only as far as the data prove the
combination helps, and every run reports — with sample sizes, test statistics and
p-values — what is and is not established.

| | |
|---|---|
| **Version** | 2.3.0 — *Nirnay read as trend (one polarity switch)* |
| **Stack** | Python 3.13 · Streamlit · scikit-learn · statsmodels · numba · Plotly |
| **Universe** | Nifty 50 (live list from niftyindices.com; dated fallback list) |
| **License** | See `LICENSE.md` |

> **A note on rigor.** Nishkarsh is built to be honest about its own edge: causal
> everywhere (prefix-invariant, tested), deterministic, h-spaced significance with
> HAC and block-bootstrap inference, null error rates verified by simulation, and
> evidence attached to the exact number shown. It has **not demonstrated a tradeable
> edge on real data**: re-measured with the purged 2.0 engine on the owner's sheet, the
> earlier +0.14 directional lead (which was contaminated by look-ahead) does not
> replicate (IC −0.03, 2014–2026, 95% CI [−0.12, +0.05]). Read it as an honest
> research instrument, and read [`docs/EDGE_LEDGER.md`](docs/EDGE_LEDGER.md) first.

---

## Contents
[Why two systems](#why-two-systems) · [Quick start](#quick-start) · [What you see](#what-you-see) ·
[The conclusion](#the-conclusion) · [The Nishkarsh Oscillator](#the-nishkarsh-oscillator--one-inference) · [Headless pipeline & run report](#headless-pipeline--run-report) ·
[Data layer](#data-layer) · [Architecture](#architecture) · [Configuration](#configuration) ·
[Testing](#testing) · [Performance](#performance) · [Known limitations](#known-limitations) ·
[Troubleshooting](#troubleshooting) · [References](#references)

---

## Why two systems

| | **AARAMBH** *(आरंभ — "Beginning")* | **NIRNAY** *(निर्णय — "Judgement")* |
|---|---|---|
| **Direction** | Top-down | Bottom-up |
| **Question** | *Where is the index likely to go over the next 10 sessions?* | *What are the 50 constituents doing?* |
| **Method** | Purged walk-forward ensemble (Ridge · Huber · ElasticNet · PCA-OLS, equal weight) forecasting forward 10-day Δlog(PE) from 20-day predictor momentum | Per-stock MSF + MMR oscillator with an online regime ensemble (adaptive Kalman · 3-state HMM · GARCH · CUSUM), aggregated to daily breadth over live, warm constituents |
| **Output** | `Forecast`, its causal z-score `zA` | Daily breadth; `zN` = causal z of −mean oscillator |

A second model that agrees is a stronger bet *only if* agreement actually predicts
better outcomes. Nishkarsh tests exactly that on every run (the **Convergence-Lift
test**) instead of assuming it. Until the test passes, Nirnay is shown as context and
the conclusion is Aarambh's.

Aarambh can also run in its original **levels** mode (fair-value regression on the PE
level, `AARAMBH_FORWARD_SIGNAL = False`). That mode is kept as *valuation context*: as
a forecaster it was falsified (Edge Ledger L-01) and the UI says so.

---

## Quick start

```bash
pip install -r requirements.txt          # Python 3.13
streamlit run app.py
```

In the sidebar choose a data source:

- **Google Sheets** — set the sheet URL as a secret (`.streamlit/secrets.toml`, see
  `.streamlit/secrets.toml.example`) or `AARAMBH_GOOGLE_SHEETS_URL`; a configured URL
  is never sent to the browser. Without one you can paste a URL.
- **Upload** — a CSV/Excel export of the sheet (ISO or day-first dates both work).
- **Synthetic demo** — a deterministic offline world; every surface is labelled
  SYNTHETIC. Useful to explore the app without data or network.

Select the target (default `NIFTY50_PE`), the date column and the predictors, then
**Run Analysis**. The sheet needs at least `MIN_TRAIN_SIZE + horizon + MIN_OOS_ROWS`
= 1500 + 10 + 252 usable rows of daily history (after the 20-row momentum warm-up).

**Streamlit Cloud:** push the repo, connect it at share.streamlit.io, and add the
secret `[aarambh] google_sheets_url = "…"`.

---

## What you see

One inference, five pages. Each page answers one question; nothing is repeated across pages and
nothing is hidden in a drop-down — every note, limit and definition sits in the section it belongs to.

| Page | The question it answers | What is on it |
|---|---|---|
| **Today** (default) | *What does Nishkarsh read today, why, and how far can I trust it?* | The read in plain words with one sentence on what drives it; how much it moved over 5 sessions, how long it has been on this side, how strong it is against its own past; the gauge (the only place the big number appears); one line on its track record; the read over time against the Nifty; one chart of what drives today's read (three pillars and their inputs on one scale); one line on what followed similar readings. |
| **Drivers** | *What is each system seeing?* | The three pillars in the same grammar — top-down (Aarambh's forecast of the Nifty), bottom-up (what the 50 stocks are doing, with a strip of every stock's oscillator and zone) and agreement (whether the two lean the same way). Each pillar over time, then its inputs as small charts. |
| **Precedent** | *What happened after readings like this?* | Past moments whose read, pillars and 5/20-day change most resembled today's (causal, one per episode), what the Nifty did 3/5/10/20 sessions later against its usual up-share with a binomial p, and those dates marked on the Nifty with what followed each shaded. |
| **Reliability** | *How much should I trust it?* | The measured status with its key numbers and yearly accuracy; the Edge Ledger for this run (every claim, its status and the number that decides it); the detailed tests (agreement lift, cost-aware backtest, Aarambh's accuracy stage by stage); the model facts; method, limits and a short glossary in plain text. |
| **Data** | *Is the data sound?* | Every source with its provenance and as-of date, the run's data notes, and the merged series (searchable, CSV export). The data-status pill in the header points here. |

Directional words (buy, sell, signal, call) appear only when the read's own pre-registered test
has passed (`VALIDATED`); until then the read is described, not recommended.

---

## The conclusion

```
zA_t   = causal z-score of Aarambh's forecast        (strictly prior mean/std, ≥ 126 obs, clip ±4)
zN_t   = causal z-score of −Avg_Signal (Nirnay)       (NaN on days without warm Nirnay data)
state  = NO_DATA | CONFIRM | CONTRADICT | NEUTRAL     (|zN| ≥ τ = 0.25 to confirm/contradict)
C_t    = 100 · tanh(zA_t / 2) · m_t                   m_t = κ_t on CONTRADICT days, else 1
tiers  = |C| ≥ 60 STRONG · ≥ 30 MODERATE · > 10 WEAK · else NO LEAN      (+ = bullish)
```

**Convergence-Lift test.** On h-spaced (non-overlapping) dates where both systems have
data: IC(Aarambh | CONFIRM) vs IC(Aarambh | CONTRADICT), compared with a one-sided
Fisher-z test (≥ 30 windows per bucket, phase range reported). Its minimum detectable
effect is shown, because with ~7 years of constituent history the test has low power —
a "no measured lift" verdict is often "not enough data", and the app says which.

**κ (the only calibrated parameter).** Re-estimated walk-forward every 126 rows on
purged history; κ < 1 only when the lift test passes, so Nirnay can only ever *reduce*
conviction on days it contradicts Aarambh. *Intelligence Mode* ON applies κ; OFF sets
κ ≡ 1 (pure Aarambh). Nothing is persisted between runs.

**Evidence and trust.** `edge` (full Aarambh history: Fisher, Newey-West HAC and
stationary-block-bootstrap IC, yearly ICs, own-momentum baseline), `edge_C` (out of
sample, on the displayed series), `backtest` (tranche and non-overlapping long/short,
0/2/5/10 bps, net Sharpe, HAC t, turnover, break-even cost) and a single trust tier:
`INVERTED · NO_PROVEN_EDGE · DIRECTIONAL_EDGE · TRADEABLE_EDGE · INSUFFICIENT_DATA`.
The edge IC is the mean over all h start offsets (every labelled day is used), tested
conservatively, and every edge read-out states its **minimum detectable IC** — with
~230 ten-day windows only ICs of roughly 0.17 or more are reliably detectable, so
"no proven edge" can mean "not enough data".

**Divergences** (one state machine): `AARAMBH_LEADS` (|zA| ≥ 1.5, Nirnay calm),
`NIRNAY_LEADS` (|zN| ≥ 1.5, Aarambh calm), `CONTRADICTION` (≥ 5 CONTRADICT days in the
last 20 trading days). Descriptive only — no edge is claimed for them.

---

## The Nishkarsh Oscillator — one inference

Everything the system computes is folded into **one causal series in −100…+100**
(+ = bullish), which the hero, the history chart and Precedent all infer from. Nirnay's
bottom-up readings follow one switch, `core.config.NIRNAY_POLARITY` ("trend" since 2.3.0, an
owner decision; the evidence under both polarities is in
[`docs/research/NIRNAY_POLARITY_CHANGE.md`](docs/research/NIRNAY_POLARITY_CHANGE.md)):

| Pillar | Inputs (oriented so + = bullish; Nirnay polarity: **trend** — strength reads bullish, `NIRNAY_POLARITY`) |
|---|---|
| **Top-down** (Aarambh) | the forecast (the conclusion over time) · forecast extension · lookback zone breadth · lookback cut crossings |
| **Bottom-up** (Nirnay) | constituent oscillator · zone breadth · regime composition · HMM state · cut crossings |
| **Agreement** | consensus × agreement ratio (the agreement anatomy) |

Each input is causally z-scored, each pillar is the equal-weight mean of its inputs
(re-standardized), and `NKO = 100·tanh(z(mean of pillars)/2)`. Nothing is fitted to
returns. When the two systems disagree their pillars cancel and the agreement pillar is
zero, so the read collapses toward 0; when they agree, it strengthens.

**Measured before promotion** (pre-registered:
[`docs/research/NISHKARSH_OSCILLATOR_PREREG.md`](docs/research/NISHKARSH_OSCILLATOR_PREREG.md);
results: [`…_RESULTS.md`](docs/research/NISHKARSH_OSCILLATOR_RESULTS.md)): on the owner's
sheet with all 50 constituents, IC +0.034 over 138 ten-day windows (p 0.35) — **UNPROVEN**.
It makes the system cohesive but is not more accurate than its best ingredient, the
signed-agreement pillar (the stand-alone Agreement Oscillator measured IC +0.09, p 0.13 —
also unproven; [`AGREEMENT_OSCILLATOR_RESULTS.md`](docs/research/AGREEMENT_OSCILLATOR_RESULTS.md)).
So the app presents NKO as **the system's read** — side, strength, what drives it — with
its measured status beside it, never as a buy/sell signal. Every run re-measures it, and
the status upgrades to VALIDATED by itself only when the pre-registered rule passes
(p < 0.05, bootstrap CI above 0, ≥ 60% of years positive).

**Precedent** matches today's NKO *state* (level, pillars, 5- and 20-day change) against
causal past states — every analog's outcome realized before today, distinct episodes —
and reports what the Nifty did next against the pool's unconditional up-share (binomial p).

---

## Headless pipeline & run report

```bash
python -m pipeline synth --seed 7 --json run.json                      # offline synthetic world
python -m pipeline run --csv sheet.csv --date-col DATE --json run.json # your sheet + live market data
python -m pipeline run --csv sheet.csv --market nse                     # constituents from NSE's official archives
python -m pipeline run --csv sheet.csv --snapshot snap/                # replay saved market data
```

Every run produces `nishkarsh.run_report/1`: provenance (version, git sha, packages),
the full configuration and its fingerprint, input hashes and date spans, per-source
data provenance (live / cache / stale / fallback / SYNTHETIC), phase timings, engine
metadata and the complete conclusion. The console SYSTEM DIAGNOSTICS block is rendered
from the report. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the Python API.

---

## Data layer

- **Sources** return a `FetchResult` with provenance; the UI and report surface stale
  snapshots, fallback constituent lists and partial batches instead of hiding them.
- **Cache**: memory + disk, TTL, versioned namespaces, parquet/JSON with atomic
  writes (no pickle), last-good snapshot fallback that survives date rollover;
  partial batches merge per symbol and never shrink a snapshot.
- **Circuit breaker** per service: consecutive-failure counting, `HALF_OPEN` admits two
  probes; retry with exponential backoff.
- **Dates** are parsed once (`data.dates.parse_dates`: ISO vs day-first detection,
  fails loudly on unparseable or out-of-order dates).
- **Cross-timezone causality**: US / EU / FX / commodity closes for date D print after
  the NSE close of D, so they are lagged one NSE session (`data.macro_lag.lag_non_ist`);
  the sheet's US yields are lagged one row.
- **Label timestamp guard**: the return label (^NSEI) must co-move with the sheet
  target at lag 0; if a ±1-session lag fits better the run warns that the edge
  read-outs may be invalid (a one-day mis-stamp can fabricate an edge).
- **Official fallback — NSE archives** (`data/nse_archive.py`): when Yahoo Finance is
  throttled, constituents and their daily OHLCV come from NSE's public bhavcopies
  (no API key; ISIN-linked across symbol renames; splits/bonuses back-adjusted with
  NSE's own adjusted previous close; cached per day). If ^NSEI is unavailable the
  return label is the sheet's own NIFTY column.
- **Macro universe**: 83 yfinance tickers (65 global bond ETFs / yield indices + 18
  commodity / FX). Supplementary: the pipeline runs without them.

---

## Architecture

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
ui/                 theme · components (hero, formatting) · charts · copy (all wording) ·
                    tabs/ (today · drivers · precedent · reliability · data)
tests/              unit, statistical-regression and layering tests (pytest)
docs/               EDGE_LEDGER · ARCHITECTURE · AUDIT · FINDINGS · PERFORMANCE · CUSTOM_PREDICTORS
```

---

## Configuration

All constants live in `core/config.py`; `validate_config()` runs at import and refuses
an incoherent configuration (for example a training window too small for the feature
count — the failure that voided 1.6.0).

| Constant | Default | Meaning |
|---|---|---|
| `AARAMBH_FORWARD_SIGNAL` | `True` | Predictive mode (False = levels / valuation context) |
| `AARAMBH_FWD_HORIZON` / `AARAMBH_FWD_MOM_K` | `10` / `20` | Forecast horizon / predictor momentum window |
| `MIN_TRAIN_SIZE` / `MAX_TRAIN_SIZE` | `1500` / `2000` | Walk-forward window, in label-complete rows |
| `AARAMBH_WF_REFIT` | `5` | Rows between refits (fixed) |
| `MIN_DATA_POINTS` / `MIN_OOS_ROWS` | `1500` / `252` | Predictor history guard / minimum out-of-sample forecasts |
| `CALIBRATION_RETURN_LABEL` | `"nsei"` | Return label: ^NSEI → target → constituent basket (with warnings) |
| `CONV2_TAU` · `CONV2_LIFT_ALPHA` · `CONV2_MIN_BUCKET` | `0.25` · `0.05` · `30` | Agreement threshold · lift significance · windows per bucket |
| `CONV2_REFIT_EVERY` / `CONV2_MIN_HISTORY` | `126` / `500` | κ walk-forward cadence / first refit |
| `CONV2_TIER_STRONG/MODERATE/WEAK` | `60/30/10` | Conclusion tiers on C |
| `EDGE_ALPHA` · `EDGE_TRADEABLE_T` | `0.05` · `2.0` | Edge significance · net-of-cost HAC t for TRADEABLE |
| `BACKTEST_COST_BPS` / `BACKTEST_REFERENCE_BPS` | `(0,2,5,10)` / `5` | Cost grid / grading cost |
| `HISTORY_YEARS` | `7` | yfinance history, anchored to 1 January |
| `NIRNAY_HMM_STD_FLOOR` · `NIRNAY_WARMUP_BARS` | `0.10` · `50` | Regime-model stability · per-stock warm-up |

---

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

---

## Performance

Measured on this release's synthetic world (~4000 sheet rows, 12 constituents,
single thread): full pipeline ≈ 21 s, of which the Aarambh walk-forward (hundreds of
purged-CV ensemble refits) ≈ 15 s, Nirnay ≈ 50 ms per stock, conclusion < 1 s. A real
sheet (~4900 rows, 50 constituents) adds roughly 3–5 s of Nirnay and the cold
yfinance download. The run report records phase timings for every run. History and
non-optimisations: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

---

## Known limitations

- **No demonstrated edge on real data.** On the owner's sheet (2026-11-08) Aarambh's
  directional IC is indistinguishable from zero and the cost-aware backtest is negative
  (Edge Ledger L-03 / L-04, evidence in `docs/evidence/`). The two-system lift (L-05)
  still needs a run with market-data access. Treat the output as a research instrument
  until a Ledger row reads PROVEN.
- **History vs predictors.** Every predictor must be present on every row, so a
  late-starting predictor truncates the whole history; the run names it and its cost
  (on the owner's sheet INIRYY costs ≈5.3 years).
- **Low power for the Convergence-Lift test** with ~7 years of constituent history;
  the minimum detectable effect is shown on every run.
- **Survivorship bias.** Nirnay uses today's index members over the whole window.
- **Sheet provenance.** The sheet's US yields are assumed to be same-day US closes and
  are lagged one row (`NON_IST_SHEET_COLUMNS`); set it to `()` if your sheet already
  lags them.
- **GARCH parameters are fixed**, not fitted per series (variance-targeted proxy).
- **Change points** are a rolling-mean heuristic, shown as a diagnostic only.

---

## Troubleshooting

- **"Need N rows"** — the sheet is shorter than `MIN_TRAIN_SIZE + horizon + MIN_OOS_ROWS`.
- **Date warnings / errors** — the day/month convention is read from the dates
  themselves (it may change part-way through a sheet). A few out-of-sequence cells
  (typos) are excluded and listed by sheet row in the run warnings — fix them in the
  sheet. An unsorted export is sorted. The run stops only when >1% of dates don't
  parse, or when out-of-order dates collide with genuine rows (an ambiguous day/month
  swap) — the message names the rows to fix.
- **Serving last-good snapshot / fallback constituents** — informational: a fetch
  failed and the last good data (or the dated fallback list, `NIFTY50_FALLBACK_AS_OF`)
  is being used; the provenance is shown in the UI and the run report.
- **Circuit open** — after 5 consecutive failures a service pauses for 60 s, then
  admits test calls.
- **Slow first run** — the Aarambh walk-forward is the floor (~15–20 s); data
  downloads are cached afterwards.

---

## References

- Kendall, M.G. (1954); Orcutt, G.H. & Winokur, H.S. (1969) — AR(1) bias correction (OU estimation).
- López de Prado, M. (2018). *Advances in Financial Machine Learning* — purged walk-forward validation.
- Newey, W.K. & West, K.D. (1987). *Econometrica* 55 — HAC standard errors.
- Politis, D.N. & Romano, J.P. (1994). *JASA* 89 — the stationary bootstrap.
- Fieller, E.C., Hartley, H.O. & Pearson, E.S. (1957). *Biometrika* 44 — variance of Fisher-z for Spearman's ρ.
- Holm, S. (1979). *Scand. J. Statistics* 6 — step-down multiple-testing adjustment.
- Chen, Y., Wiesel, A., Eldar, Y.C. & Hero, A.O. (2010). *IEEE TSP* 58 — OAS covariance shrinkage (analogs).
- Theiler, J. (1986). *Phys. Rev. A* 34 — exclusion window for nearest-neighbour analogs.
- Peng, C.K. *et al.* (1994). *Phys. Rev. E* — detrended fluctuation analysis.
- Ratcliff, R. & McKoon, G. (2008). *Neural Computation* — drift-diffusion (levels-mode conviction filter).

---

© 2026 Nishkarsh · [@thebullishvalue](https://twitter.com/thebullishvalue) · v2.3.0
