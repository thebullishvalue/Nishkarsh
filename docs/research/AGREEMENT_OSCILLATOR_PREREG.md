# Agreement Oscillator — pre-registration

_Written and committed **before** the measurement was run (git history proves the order).
Purpose: decide, on real data and without data-mining, whether the agreement ratio
deserves to be Nishkarsh's focal buy/sell read. Request: "focus on the agreement ratio …
it's like an oscillator for market buy and sell … make it the focal point, but only after
measurement."_

## Data

- Sheet: the owner's Aarambh sheet (NIFTY50_PE + default predictors), as shipped.
- Constituents: the current Nifty 50 (survivorship-biased, as in the product), daily OHLCV
  from NSE bhavcopies (`data/nse_archive.py`, ISIN-linked, split/bonus-adjusted), window
  2019-01-01 → 2026-11-06.
- Return label: NIFTY 50 close (the sheet's NIFTY column, IST-dated; identical to the NSE
  index close to the cent on checked dates). Horizon **h = 10** trading days (the product
  horizon); 5 and 20 reported as secondary.
- Engines exactly as shipped (2.0.x): purged walk-forward Aarambh, Nirnay with warm-up /
  live-name breadth. No parameter of anything below is fitted to the label.

## Objects

- `agreement_ratio_t ∈ [0, 1]` — the product's agreement anatomy
  (`convergence.cross_validator.compute_agreement_anatomy`): fixed-weight mean of the
  direction / breadth / magnitude / regime agreement scores, Nirnay read contrarian.
- `consensus_t ∈ {−1, 0, +1}` — +1 when both engines lean bullish, −1 when both lean
  bearish, 0 otherwise; a lean is `sign(zA)` for Aarambh and `sign(zN)` for Nirnay
  (`zN` = causal z of −Avg_Signal), as in the conclusion layer.
- **Signed Agreement Oscillator** `SAO_t = consensus_t · agreement_ratio_t ∈ [−1, +1]`
  (+ = both systems agree on buying, − = both agree on selling, 0 = they disagree).

## Hypotheses (family of four, Holm-adjusted at α = 0.05)

| ID | Statistic | Test |
|---|---|---|
| **H1 (primary)** | IC(SAO_t, R_{t,t+10}) | one-sided, IC > 0 |
| H2 | IC(agreement_ratio_t, R_{t,t+10}) | two-sided (an unsigned level has no prior sign) |
| H3 | Event study: mean R_{t,t+10} after SAO enters ≥ +0.5 minus mean after it enters ≤ −0.5 (non-overlapping entries) | one-sided, difference > 0 (Welch t) |
| H4 | IC(zN_t, R_{t,t+10}) — Nirnay breadth alone as an oscillator | one-sided, IC > 0 |

IC = phase-averaged Spearman on h-spaced rows with the single-phase Fisher variance
(the shipped `convergence.conclusion.ic_evidence`), plus Newey-West HAC t and a seeded
stationary-block-bootstrap CI. Holm across H1–H4 decides significance.

## Promotion rule (decided now)

1. **H1 passes** (Holm p < 0.05, bootstrap CI excludes 0, ≥ 60% of calendar-year ICs
   positive) → the Signed Agreement Oscillator becomes the **focal read** of the product
   (hero + main chart). It is labelled TRADEABLE only if its cost-aware backtest
   (`analytics.backtest`, 5 bps) has net HAC t ≥ 2; otherwise DIRECTIONAL.
2. **H1 fails but H3 passes** → shown as an *event* read ("agreement extremes") with its
   measured hit rates, not as a continuous buy/sell gauge.
3. **All fail** → the agreement ratio is presented as a well-designed **context gauge**
   ("how much the two systems agree right now"), with its measured statistics in the
   evidence drawer and an explicit "not a validated buy/sell signal" caption. It is not
   promoted, and no buy/sell wording is attached to it.

H2 / H4 results are reported whatever happens; they inform but do not trigger promotion.
Secondary horizons (5, 20) are reported, never used for promotion.

## Known limitations (stated in advance)

- ~1 750 overlap days → ~175 non-overlapping 10-day windows: the minimum detectable IC
  is ≈ 0.18. A failed test may mean "not enough data", and will be reported as such.
- Survivorship: today's members over the whole window.
- No global macro panel from NSE: Nirnay's MMR leg uses the sheet's rate columns only.
