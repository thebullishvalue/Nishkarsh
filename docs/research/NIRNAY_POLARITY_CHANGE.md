# Nirnay polarity: contrarian → trend (2026-11-09, owner decision)

**Decision.** From 2.3.0 Nirnay's bottom-up readings are oriented **trend** (strength reads
bullish), set by the single switch `core.config.NIRNAY_POLARITY = "trend"`. The owner's
rationale: the bottom-up sign should align logically with the top-down (Aarambh) sign — a
strong market reads bullish from both sides. It flips every Nirnay-derived quantity together:
the Nirnay head `zN`, the agreement anatomy (direction, breadth, regime dimensions), the
consensus and agreement state, the Convergence-Lift test and κ, divergence typing, the signed
agreement oscillator and the Nishkarsh Oscillator's bottom-up pillar. `"contrarian"` restores
2.2.0 exactly (regression-tested).

**Council advice on record (before the change):** keep contrarian, make it visible in the copy,
and settle the polarity with a pre-registered test on independent 2010–2018 data. The owner
chose to flip now.

## Same data, both polarities (evidence: `docs/evidence/2026-11-09_*_run_report.json`)

| Measured (owner's sheet, 50 constituents from NSE bhavcopies, h = 10) | Contrarian (≤ 2.2.0) | **Trend (2.3.0)** |
|---|---|---|
| Nishkarsh Oscillator IC (n 138) | +0.034, p 0.35 | **−0.019, p 0.59** |
| — bottom-up pillar IC | +0.029 | −0.029 |
| — agreement pillar IC | +0.063 | +0.026 |
| Signed agreement oscillator IC (n 163) | +0.091, p 0.13, 5/7 years + | **+0.035, p 0.34, 2/7 years +** |
| Aarambh IC when Nirnay confirms / contradicts | +0.084 (52) / −0.261 (74) | **−0.261 (74) / +0.084 (52)** |
| Convergence-Lift | +0.345, p 0.033, passed | **−0.345, p 0.97, not passed → κ ≡ 1** |
| Status (pre-registered rule) | UNPROVEN | UNPROVEN |
| Today's read (2026-11-06) | NKO −48, moderately bearish | **NKO +62, strongly bullish** |

## What this means

- Every measured relationship is weaker under the trend reading, and none was proven under
  either. The flip **is not evidence-driven**; it is a design choice by the owner, recorded as
  such. This table is not an independent test of either polarity (the contrarian numbers were
  already known when the choice was made).
- The agreement mechanism inverts: under trend, the days the two systems "agree" are the days
  Aarambh's forecast has been worst. The lift test therefore fails and κ stays 1, so the
  conclusion is Aarambh's; Nirnay is context.
- The headline can reverse on identical data (−48 → +62 today): a broadly strong market now
  reads bullish from the bottom up instead of stretched.
- **How to settle it honestly:** pre-register "trend vs contrarian" and measure it on
  2010–2018 NSE data, which no measurement has touched (caveat: today's constituents over that
  window add survivorship bias). The switch makes either outcome a one-line change.
