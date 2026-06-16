# Nishkarsh — Research Findings & Conclusions

_Consolidated record of what was empirically established about this system's
predictive value, so it is not re-litigated. Engineering history is in
`CHANGELOG.md` (v1.4.1 → v1.4.13)._

> **Investigation status: REOPENED (2026-06-15).** Every *fair-value / levels*
> edge hypothesis remains falsified (below). But a NEW configuration —
> **predictive Aarambh mode** (forecast forward PE change from predictor
> momentum, `AARAMBH_FORWARD_SIGNAL`) — surfaced the engagement's first
> **directional IC candidate** that survives the survivorship-correct benchmark
> and does not recency-decay. It is a *lead under active test*, not yet a proven
> tradeable edge (see §6). The levels-mode conclusions are settled; the
> predictive-mode one is provisional and promising.

## Executive conclusion

**In FAIR-VALUE / levels mode, there is no robust, tradeable forecasting edge in
this stack on NIFTY-50 PE.** That mode is sound, honest **valuation / breadth /
regime *context*** — not a return signal; every levels-mode directional edge was
falsified under causal, non-overlapping evaluation.

**In PREDICTIVE mode (NEW), the picture differs.** The predictive Aarambh signal
(forward 10d Δlog-PE forecast from 20d predictor momentum) has a **directional
rank IC ≈ +0.14 against the true ^NSEI index** — 100% positive across 5 folds,
stable across 7 of 8 yearly blocks (incl. the two most recent), and ~identical
on a survivorship-correct vs survivorship-biased label. This is real directional
information the levels model structurally could not produce. It is **not yet a
confirmed tradeable edge** — significance under non-overlapping windows and a
cost/turnover backtest are still pending (§6) — but it is the first genuine lead.

The valuable outcome of this work remains a research instrument honest enough to
separate artifacts (t = 4.8 / p = 0.000) from signal — and now, to surface a
candidate worth pressure-testing rather than dismissing.

## Component-by-component verdict

### 1. Aarambh (PE fair-value engine) — NO EDGE, any configuration
- The headline `OOS R² 0.913` is a **near-unit-root persistence artifact**, not
  skill. The honest metric `R² vs RW` is deeply negative (−15.8).
- **43% of the model's explanatory power is `NIFTY50_PB` + `NIFTY50_DY`** — two
  valuation ratios of the same index. It was a near-**tautology** (PE ≈ f(PB,DY)),
  not a macro-driven fair value. Dropping PB/DY collapses R² to ≈ −4.
- Tested in every configuration:
  - **Long window (252/756):** signal **INVERTED** (BUY preceded PE declines, p<0.05).
  - **Short window (15/30):** looked great (t≈4.8) — but that was an
    **overlapping-window artifact**; under non-overlapping stats t→≈1.1, **NO EDGE**.
  - **+ macros via PCA:** raw R²=−4; unstationarized R²=−47 (blew up);
    stationarized R²=−0.78 and **INVERTED**. Macros **actively degrade** Aarambh.
  - → `AARAMBH_PCA_PREDICTORS` defaulted **OFF** (proven harmful).

### 2. MMR (macro → constituent) — ≈ ZERO
- `MMR Quality (√R² vs RW)` ≈ **0.012** mean. After de-duplicating 89 collinear
  macros into 8 causal PCA factors (removing spurious top-k selection), the
  honest macro→single-stock daily predictive power is ~nil.

### 3. Convergence — AGREEMENT, not direction (root architectural finding)
- All four convergence dimensions (`direction/breadth/magnitude/regime`) measure
  **agreement** between Aarambh and Nirnay, **not net market direction**. A
  *both-bullish* and a *both-bearish* day produce the **same** `convergence_score`
  (e.g. `regime_score = 1.0` for both) — yet they precede opposite returns.
- Calibrating this agreement score against directional returns is
  category-confused. This is the **root cause** of the calibration's symptoms:
  thin IC (~0.05–0.10) and a "top driver" that **bounces run-to-run**
  (direction → regime → magnitude) — the signature of fitting noise.
- The **Directional Convergence Test** (`directional_convergence_ic`, surfaced in
  SYSTEM DIAGNOSTICS) settles whether the *signed* convergence has any edge.
  **RESULT in LEVELS mode (decisive): NO robust directional edge.** Fold ICs
  `[−0.039, +0.126, −0.017, −0.036, +0.061]` (n=1212): 40% positive, mean +0.019
  (≈0), sign-flipping across regimes. With the fair-value Aarambh, convergence is
  a cross-system **agreement/divergence monitor**, not a timing signal.
  **In PREDICTIVE mode this flips — see §6** (same test, IC +0.13, 100% positive).
  The difference is entirely the Aarambh head, not the convergence math.

### 4. Calibration — honest, gated, but data-starved
- Real price-return label (`^NSEI`), chronological split, fold-stability gate,
  apples-to-apples incumbent comparison — all sound.
- But it generalizes off only **~2 years** of Aarambh↔Nirnay overlap (constituent
  OHLCV history), so the learned profile is regime-specific and its driver is
  unstable. "Accepted" ≠ "real edge."

### 5. Convergence score — now direction-carrying (architectural fix shipped)
- The agreement-vs-direction flaw in §3 was **fixed** (adapted from the Tattva
  fork): `convergence_score = −consensus_direction · agreement_strength · 100`
  (negative = bullish; disagree → ~0). The calibration composite and the
  zones/hero card now agree in sign. This makes the convergence score a *valid*
  directional signal to test — it did not, by itself, create edge.

### 6. Predictive Aarambh mode (NEW, 2026-06-15) — directional IC candidate ✓ lead
Config: `AARAMBH_FORWARD_SIGNAL=True`. Aarambh forecasts forward 10d Δlog-PE from
20d predictor momentum (sign-agnostic K-day diff features; PCA gate bypassed);
conviction = the forecast. Validated by a standalone reproduction that matched
the live run's basket IC (+0.128 vs logged +0.131).

- **Directional IC vs ^NSEI (survivorship-correct): mean +0.130**, 100% positive
  across 5 folds (min +0.055) — ≈ identical to the survivorship-biased basket
  (+0.128). **Survivorship is refuted**: the edge is real co-movement with the
  true index, not survivor drift.
- **Source is Aarambh, not Nirnay.** Decomposition vs ^NSEI: **Aarambh-only IC
  +0.142** (100% pos, 5/5 folds, min +0.096) vs **Nirnay-only +0.027** (60% pos,
  noise). Nirnay slightly *dilutes* the combined signal. The Aarambh-only number
  is rank-based and ~invariant to the (full-sample) normalization, so it is
  largely lookahead-free.
- **No recency decay.** 8 yearly folds: `[+0.054, +0.177, +0.284, −0.017, +0.106,
  +0.256, +0.185, +0.116]` — 7/8 positive, the **two most recent healthy**
  (+0.185, +0.116); only 2022 (global rate-shock bear) was flat. Basket and
  ^NSEI trajectories nearly identical. (The "not durable" reading from the app's
  `walk_forward_ic` was the *calibration composite* — a different, weaker signal
  — not this directional candidate.)
- **vs levels mode:** directional IC +0.14 here vs +0.019 (sign-flipping) in §3
  — predictive mode unlocked ~7× the directional signal.
- **Robust across train windows AND both benchmarks (2026-06-15 sweep).**
  Aarambh-only IC over full history (n_oos 1899–4649), 6 windows from 250/1250
  to 3000/4500, **100% positive folds in every cell, no inversion anywhere**:

  | train win | IC vs **NIFTY50_PE** | IC vs **^NSEI** |
  |---|---|---|
  | 250/1250  | +0.137 (min +0.081) | +0.136 (min +0.072) |
  | 500/1500  | +0.153 (min +0.067) | +0.151 (min +0.071) |
  | 1000/2000 | +0.154 (min +0.101) | +0.154 (min +0.129) |
  | 2000/3000 | +0.154 (min +0.118) | +0.149 (min +0.128) |
  | 3000/4500 | +0.149 (min +0.122) | +0.147 (min +0.093) |

  Two findings: (a) **PE and ^NSEI give near-identical IC** — the PE-direction
  forecast IS a price-direction forecast over 3–20d (the edge is coherent, not
  PE-only). (b) **Larger windows lift the *worst-fold* IC** (min +0.08→+0.12),
  i.e. more consistent — which is why calibration only *accepts* at MIN_TRAIN
  ≳1500. Mean IC peaks ~1000–2000/…. The user's "2000+ inverts" concern applies
  to the convergence/calibration *composite*, not the core Aarambh signal, which
  is uniformly positive. `R² vs RW` stays ≈ −4.5 throughout (magnitude is poor;
  the signal is purely directional — IC is the correct metric).

- **The "not durable" verdict was a BASKET artifact (2026-06-15 label study).**
  Running the real `ConvergenceTuner` (50 trials) under three labels:

  | label | train IC | val IC | fold stab | walk-forward (re-cal, purged OOS) |
  |---|---|---|---|---|
  | target=PE | +0.027 | **+0.148** | 100% | **100% durable** (+0.156) |
  | ^NSEI | +0.039 | **+0.133** | 100% | **100% durable** (+0.137) |
  | basket | +0.163 | +0.038 | 80% | **67% NOT durable** |

  The survivorship-biased **basket** lets the optimizer overfit survivor drift
  (train IC 0.163 ≫ val 0.038) and is what produced every prior run's
  "walk-forward NOT durable" reading — **the app was silently using basket** as
  the `target` fallback in predictive mode. With a clean label (PE or ^NSEI) the
  calibration barely fits in-sample yet generalizes strongly and is **100%
  durable** out-of-sample. → Default set to `CALIBRATION_RETURN_LABEL="nsei"`;
  predictive-mode `target` fixed to use the real PE level, never basket.

**Not yet proven tradeable — open items before claiming edge:**
1. **Non-overlapping significance.** The fold ICs use overlapping forward windows
   (inflates t-stats, not the IC point estimate). Re-test Aarambh-only with
   non-overlapping h-spaced returns for an honest t-stat.
2. **IC → P&L.** A rank IC of ~0.14 is promising but is not a backtest. Build a
   simple long/short on the Aarambh-only signal with transaction costs/turnover
   to get a net Sharpe; confirm it survives costs.
3. **Causal normalization.** The combined signal's a/n weighting uses full-sample
   z-score params (mild lookahead); make it expanding-window before trusting the
   *combined* number (Aarambh-only is already rank-robust).

## What was hardened (so the above conclusions are trustworthy)
Causal everywhere (verified prefix-invariant), deterministic (`random_state`
pins), honest significance (non-overlapping forward windows), persistence-aware
reporting (lead with `R² vs RW`), edge gating (INVERTED / NO-EDGE flags),
survivorship-correct return label, full per-run telemetry.

## Recommended next steps (in priority order)
1. **Pursue the predictive-Aarambh lead (§6) — this is now the priority.** It is
   the first directional IC that survived every honesty check so far. Close the
   three open items in §6: (a) non-overlapping significance on the Aarambh-only
   signal, (b) a cost/turnover long-short backtest (IC → net Sharpe), (c)
   expanding-window normalization for the combined signal. If (a)+(b) hold, this
   is a real, tradeable directional tilt.
2. **`convergence_score` is now directional (§5, shipped).** With predictive
   Aarambh the convergence is no longer pure agreement-monitoring — but note the
   edge is the Aarambh head; Nirnay currently dilutes it (§6), so test whether
   Nirnay should be down-weighted or dropped from the directional blend.
3. **Levels mode stays as valuation context, not a forecaster** — that PE-level
   conclusion is exhaustively falsified and unchanged.
4. **More overlap history** sharpens the §6 signal's significance; it cannot
   create one, but here there is a real candidate for it to sharpen.
