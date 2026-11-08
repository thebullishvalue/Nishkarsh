# Nishkarsh — Edge Ledger

_The single, authoritative statement of what Nishkarsh has and has not shown. Every edge
word the app or the run log prints (EDGE, TRADEABLE, confirms, adds value, durable) must be
traceable to a row here **and** to the run-time instrument that measures it on your data._

Status vocabulary:

| Status | Meaning |
|---|---|
| **PROVEN** | Passed its pre-registered test out-of-sample on real data, net of costs where relevant. |
| **PROVISIONAL** | A real lead under a valid test, but a required check is still open. |
| **CONTAMINATED** | The evidence was produced by code later found to leak; must be re-measured. |
| **UNTESTED** | No valid measurement exists yet. The instrument exists and reports on every run. |
| **FALSIFIED** | Tested properly and found absent (or inverted). Kept so it is not re-litigated. |

> **2.0.0 was built and verified on seeded synthetic worlds.** The first real-data evidence (the owner's
> sheet, 2026-11-08) is recorded in L-03 / L-04; market-data rows (L-05 onwards) still need a run with
> yfinance access. Every PROVISIONAL / UNTESTED row below is answered by the instrument named in the
> last column the first time you run 2.0.0 on the real sheet — read the **Edge Ledger** panel
> (Diagnostics tab) or the `conclusion` section of the JSON run report.

## Ledger (as of 2.0.0)

| ID | Claim | Status | Evidence so far | Run-time instrument |
|---|---|---|---|---|
| L-01 | Levels-mode PE fair value forecasts returns | **FALSIFIED** | `R² vs RW` −15.8; signal INVERTED at 252/756 (BUY preceded declines, p<0.05); the 15/30 "t≈4.8" was an overlapping-window artifact (non-overlapping t≈1.1). | Levels mode is labelled *valuation context* only. |
| L-02 | Macro panels improve Aarambh | **FALSIFIED** | R² −4 (raw) / −47 (combined PCA) / −0.78 and INVERTED (stationarized PCA). | `AARAMBH_PCA_PREDICTORS = False`. |
| L-03 | Predictive Aarambh forecast has directional IC vs the index | **NOT REPLICATED (real data, 2.0.1)** | The +0.14 of FINDINGS §6 was measured on a pre-purge engine. Re-measured on the owner's sheet with the purged 2.0 engine (label: the sheet's IST-dated NIFTY column; Aarambh-only, no constituent data in that run): default predictors, 2019–2026 OOS, IC **+0.037**, n 164, p 0.33, 95% bootstrap CI [−0.08, +0.16]; without INIRYY (history from 2008), 2014–2026 OOS, IC **−0.034**, n 298, p 0.71, CI **[−0.12, +0.05]** — the claimed +0.14 lies outside it. Yearly ICs flip sign (+0.43 2014 … −0.45 2026). Evidence: `docs/evidence/2026-11-08_real_sheet_*.json`. | `conclusion.edge` on every run; re-test with ^NSEI + Nirnay on a deploy with market data. |
| L-04 | The directional forecast survives transaction costs | **FALSIFIED on current evidence** | Net Sharpe at 5 bps −0.30 (2019–2026) and −0.46 (2014–2026); negative even at 0 bps. | `conclusion.backtest`. |
| L-05 | Nirnay adds information to Aarambh ("agreement is information") | **UNTESTED — prior negative** | Nirnay-only IC +0.027 (noise); the v1 sign-gated blend diluted Aarambh's IC by ~30–60% (council simulation: a non-informative second system costs 29–42% of the lead's IC by construction). | `conclusion.lift` — Convergence-Lift test: IC(Aarambh \| CONFIRM) − IC(Aarambh \| CONTRADICT), h-spaced, one-sided Fisher-z, with its minimum detectable effect disclosed. Nirnay only ever *shrinks* conviction (×κ ≤ 1) and only after this passes. |
| L-06 | The v1 4-dimension calibrated convergence score has an edge | **FALSIFIED (as a calibration)** | The 8 calibrated parameters are not identified: IC moves 0.008 across 200 random weight vectors vs a sampling sd of 0.06; `w_direction` has zero effect; thresholds never touch the IC; the acceptance gate accepted 9–12% of pure-noise runs. | Retired in 2.0.0. The four dimensions survive only as descriptive *agreement anatomy*. |
| L-07 | The DDM-on-DDM conviction carries the 10-day edge | **FALSIFIED (as the headline)** | Stacked-filter centroid lag ≈ 20 trading days against a 10-day horizon; out-of-sample IC of the v1 hero series +0.045–0.065 vs Aarambh-only +0.10–0.14 in matched synthetic worlds. | Retired from the conclusion path; `conclusion.chain` reports IC at each stage. |
| L-08 | Divergence types (AARAMBH_LEADS / NIRNAY_LEADS / CONTRADICTION) are informative | **UNTESTED** | v1 detector could never emit CONTRADICTION and scored missing-data days as divergences. | `conclusion.events` (rebuilt, one state machine) — descriptive only. |
| L-09 | MMR (macro → constituent) carries signal | **FALSIFIED** | `MMR Quality` ≈ 0.012; MMR is 0.83-correlated with a plain price-vs-MA z-score. | Kept as a Nirnay sub-component; not an edge claim. |
| L-10 | Precedent (analog) base rate is a stronger directional read | **UNTESTED** | The "historically stronger" claim was imported from another project (Tattva) and never measured here; on a pure null the v1 precedent row leaned directionally 41% of the time. | Precedent shows `k/n up vs unconditional %` with a binomial p; it votes in the hero only when significant. |
| L-11 | Custom engineered predictors improve the forecast | **UNTESTED** | Inert for Aarambh in predictive mode (the PCA gate is bypassed there); feed MMR only. | — |

## How a row changes status

1. Run 2.0.0 on the real sheet (Streamlit, or `python -m pipeline run --csv <export.csv> --json run.json`).
2. Read the instrument named in the row — every number comes with its `n`, test statistic and p-value.
3. Update this file **with the run report attached** (commit the JSON next to the change). A row
   moves to PROVEN only on a passed pre-registered rule:
   - **L-03** — at h = 10: HAC t ≥ 2, bootstrap CI excluding 0, ≥ 70% of yearly blocks positive,
     and IC above the own-momentum baseline.
   - **L-04** — net-of-5-bps HAC t ≥ 2 (`EDGE_TRADEABLE_T`) and break-even ≥ 2 × cost.
   - **L-05** — Convergence-Lift one-sided p < 0.05 with ≥ 30 windows per bucket **and**
     out-of-sample IC(conclusion) ≥ IC(Aarambh-only).
