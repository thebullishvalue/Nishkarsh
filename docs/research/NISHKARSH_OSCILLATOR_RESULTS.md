# Nishkarsh Oscillator — results (2026-11-09)

Pre-registration: [`NISHKARSH_OSCILLATOR_PREREG.md`](NISHKARSH_OSCILLATOR_PREREG.md)
(construction and rule committed in `c6f2b67`, before this run). Evidence:
`docs/evidence/2026-11-09_nishkarsh_oscillator_run_report.json` (the product CLI's run report,
`--market nse`, owner's sheet, all 50 constituents from NSE bhavcopies).

## H-NKO (single primary test, h = 10)

| | Estimate | Evidence | Verdict |
|---|---|---|---|
| IC(NKO, 10-day NIFTY return), two-system dates | **+0.034** | n 138 · p 0.35 · HAC t 0.60 · bootstrap CI [−0.078, +0.148] · 3 of 6 years positive · MDE 0.22 | **UNPROVEN** |

Reported, not used for decisions — pillar ICs on the same windows: top-down −0.034,
bottom-up +0.029, agreement +0.063.

## Reading

- Unifying the ten read-outs into one oscillator makes the system **cohesive** — one series
  the hero, the precedent and the history chart all infer from — but it did **not** make it
  more accurate than its best ingredient: the signed-agreement pillar (the earlier Agreement
  Oscillator, IC +0.09) remains the strongest piece, and the two engine pillars dilute it.
- The triple causal standardization (inputs → pillars → final) costs ~1.5 years of warm-up:
  the two-system series starts in 2021, so this test had 138 windows vs 163 for the
  agreement oscillator. That is a property of the frozen construction, recorded here.
- **No change was made after seeing these numbers.** Any revision (e.g. a shorter warm-up, a
  different pillar weighting, a trend reading of Nirnay) must be pre-registered as a new
  hypothesis and measured on data it was not tuned on.

The product therefore presents NKO as **the system's read** — side, strength and what drives
it — with this measured status beside it, and never as a buy/sell signal. Every run
re-measures it with the same rule; the status upgrades by itself only when the rule passes.

## Robustness notes (same day, recorded after the run — no construction change)

- **Data-source sensitivity.** The same construction read through the app (the live sheet,
  one more day to 2026-11-09, Yahoo market data and the ^NSEI label) measured IC **+0.049**
  over 139 windows (p 0.29) against **+0.034** / 138 / p 0.35 through the CLI (CSV to
  2026-11-06, NSE bhavcopies, the sheet's NIFTY label). Same verdict (UNPROVEN); the run
  report records which sources produced each number.
- **Knife-edge consensus.** The agreement pillar uses the bare *sign* of the Aarambh
  forecast z. On 2026-11-06 that z was −0.009 — effectively zero — so the pillar read −0.87
  ("both lean bearish"); one day later, with the z marginally positive, it read +0.15
  ("split"), and today's NKO moved from −48 to +1. The Nirnay side already uses a deadband
  (|zN| ≥ τ = 0.25) in the conclusion layer's agreement state; the Aarambh side does not.
  **Candidate revision (not applied):** require |zA| ≥ τ as well before a lean counts.
  It must be pre-registered as a new hypothesis and measured on fresh data before it can
  replace the frozen construction.
