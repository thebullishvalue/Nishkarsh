# The Nishkarsh Oscillator (NKO) — pre-registration

_Committed **before** it is measured (git history proves the order). Request: "unify all
the information … into one singular inference data or oscillator; the precedent and hero
card get logically rewired to infer this singular data … so the entire system becomes
cohesive and singular in purpose."_

## Construction (fixed now; nothing is fitted to returns)

Implemented in `convergence/nishkarsh.py` — ten inputs, three pillars, one oscillator:

| Pillar | Input (oriented so + = bullish) | App panel it replaces |
|---|---|---|
| Top-down | Aarambh forecast z (`zA`) | Conclusion over time |
| Top-down | −AvgZ | Forecast extension diagnostics |
| Top-down | OversoldBreadth − OverboughtBreadth | Lookback zone breadth |
| Top-down | BuySignalBreadth − SellSignalBreadth | Lookback cut crossings |
| Bottom-up | −Avg_Signal (contrarian) | Average unified oscillator |
| Bottom-up | Oversold_Pct − Overbought_Pct | Zone breadth over time |
| Bottom-up | Regime_Bear_Pct − Regime_Bull_Pct (contrarian) | Regime composition |
| Bottom-up | avg_hmm_bear − avg_hmm_bull (contrarian) | HMM state |
| Bottom-up | (Buy_Signals − Sell_Signals) / n_live | Cut crossings over time |
| Agreement | consensus × agreement ratio | Agreement anatomy |

Each input → causal z (strictly prior, ≥ 126 obs, clip ±4); each pillar = equal-weight mean
of its inputs' z, re-standardized causally; `NKO = 100·tanh(causal_z(mean of pillars)/2)`.
Nirnay is read **contrarian** everywhere — the product's single polarity since the 2.0
audit; it is NOT chosen by looking at returns. Orientation and weights are frozen here.

## Hypothesis and rule (single primary test — no family)

- **H-NKO:** IC(NKO_t, R_{t,t+10}) > 0 on two-system dates (NIFTY 50 close, h = 10,
  phase-averaged h-spaced Spearman, single-phase Fisher variance, one-sided).
- **VALIDATED** ⇔ p < 0.05 AND stationary-block-bootstrap 95% CI excludes 0 AND ≥ 60% of
  calendar years positive. Otherwise UNPROVEN. The product applies exactly this rule on
  every run.
- Reported but never used for a decision: each pillar's IC; secondary horizons 5 and 20.
- Data: the owner's sheet + all 50 constituents from NSE bhavcopies (2019-01 → 2026-11),
  as in the agreement-oscillator measurement. Minimum detectable IC ≈ 0.20 (~163 windows).

## What the product does with it (decided now)

The hero card and the Precedent read infer from NKO alone. Precedent becomes "past dates
whose NKO state (level, pillars, 5- and 20-day change) most resembled today's, and what the
Nifty did next". Until H-NKO passes, NKO is presented as **the system's read** (side +
strength + what drives it) with its measured status, and never as a buy/sell signal.
