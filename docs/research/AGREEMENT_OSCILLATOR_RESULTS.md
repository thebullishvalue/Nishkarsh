# Agreement Oscillator — results (2026-11-09)

Pre-registration: [`AGREEMENT_OSCILLATOR_PREREG.md`](AGREEMENT_OSCILLATOR_PREREG.md)
(committed before the run). Evidence: `docs/evidence/2026-11-09_*.json`.

**Data.** The owner's sheet; all 50 current constituents from NSE bhavcopies (1,938
trading days 2019-01 → 2026-11, ISIN-linked, split/bonus-adjusted); return label = the
sheet's NIFTY close. Engines as shipped. Overlap with a genuine Aarambh forecast:
1,640 days (2020-03-17 → 2026-11-06) → **163 non-overlapping 10-day windows**
(minimum detectable IC ≈ 0.20).

## Pre-registered family (h = 10, Holm across H1–H4)

| | Statistic | Estimate | Evidence | Holm p | Verdict |
|---|---|---|---|---|---|
| **H1** | Signed Agreement Oscillator `SAO = consensus × agreement_ratio` | IC **+0.091** | HAC t 1.92 · bootstrap 95% CI [−0.008, +0.190] · 5 of 7 calendar years positive | 0.53 (raw 0.13) | not significant |
| H2 | agreement ratio level (unsigned) | IC −0.002 | CI [−0.088, +0.084] | 1.00 | nothing |
| H3 | event study: entries into SAO ≥ +0.5 vs ≤ −0.5 | +0.25% vs +0.26% mean 10-day return | Welch t −0.01 | 1.00 | nothing |
| H4 | Nirnay breadth alone (`zN`) | IC +0.023 | CI [−0.076, +0.126] | 1.00 | nothing |

Secondary horizons (not used for decisions): SAO IC +0.051 at 5 days, +0.089 at 20 days.
Cost-aware backtest of SAO (tranche, 5 bps): net Sharpe +0.12, t 0.32, break-even 10.8 bps.

**Decision (by the pre-registered rule): rule 3 — context gauge.** The agreement ratio is
presented as the product's central *read of how much, and on which side, the two systems
agree*, with its measured status shown; it carries no buy/sell wording until it passes.

## What the data do say

The unsigned agreement level carries nothing (H2), and neither do agreement extremes
(H3). The **signed** read is where the signal sits — and the product's Convergence-Lift
test, run on the same data, explains why:

| Aarambh's directional IC when Nirnay… | IC | windows |
|---|---|---|
| confirms (same side) | **+0.084** | 52 |
| contradicts (opposite side) | **−0.261** | 74 |

Lift +0.345, one-sided p 0.033 — the first real-data evidence for the "two systems, one
conclusion" thesis: when the two systems disagree, Aarambh's forecast has tended to be
*wrong*. It is not yet established: the lift is phase-unstable across start offsets
(−0.20 … +0.35), and the walk-forward out-of-sample check has not shown the combined
conclusion beating Aarambh alone. Both H1 and the lift will be re-tested automatically
on every run; the product upgrades the oscillator's status only when its pre-registered
rule passes on the data in hand.
