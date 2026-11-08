# Nishkarsh — Audit Register (2.0.0 council audit)

_The finding register of the 2.0.0 multi-tier council audit: six read-only specialist
auditors (Tier I), a Chair synthesis into one decision register (Tier II), parallel
builders on disjoint files (Tier III), and a verification tier (Tier IV). Every S0/S1
finding was reproduced with a seeded, offline script before it was fixed, and the fix
was re-verified the same way; the permanent versions of those checks live in `tests/`._

> **About `docs/AUDIT_2026-07.md`.** CHANGELOG 1.5.0 / 1.6.0 cite an audit register at
> that path. It was never committed (no commit and no vault archive contains it), nor
> were the `tests/verify_audit.py` / `tests/test_hero_verdict.py` files those releases
> describe. This file replaces it; the 2.0.0 test suite replaces the missing tests.

Severity: **S0** integrity (look-ahead, crash, false verdict) · **S1** statistical /
correctness · **S2** logic / coherence / vision · **S3** hygiene.

## Root cause of the release (S0)

| ID | Finding | Evidence | Resolution |
|---|---|---|---|
| A1-01 / L1 | v1.6.0 shipped `MIN/MAX_TRAIN_SIZE = 15/30` (and `MIN_DATA_POINTS = 30`) — test-mode values. At a ≤30-row window the engine's "fewer than 50 rows → drop the purge" fallback fired on **every** chunk, so the forward-label purge never applied. | Pure-noise predictors and an independent random-walk PE: forecast IC **+0.501 ± 0.013**, R² vs RW **+0.52** (5 seeds). With the purge kept: IC 0.000 ± 0.043. | 1500/2000 restored (label-complete rows); a starved chunk now yields NaN and is counted, the purge is never dropped; `validate_config()` refuses p/n-degenerate windows at import. Re-verified: IC −0.03 at 1500/2000 and −0.0003 at a deliberately tiny 30/60 window. |
| A1-02 | At 15/30 live forecasts were trained on the zero-filled, not-yet-realized label tail and repainted when labels arrived. | Live forecast moved +0.033 → +0.051 on one new day. | Fixed by A1-01; prefix-invariance is a test. |

## Aarambh engine (A1)

| ID | Sev | Finding | Resolution |
|---|---|---|---|
| A1-03 | S1 | Member-weighting "validation" slice was in-sample (fit and scored on the same rows). | Equal-weight, NaN-aware ensemble (weighting was measured to be inert). |
| A1-04 | S1 | Conformal "z" divided the 95% interval width by the IQR constant 1.35 → mis-scaled 2.7–2.9× (P(\|z\|>1) = 0.007 vs nominal 0.32). | Divides by 2·1.96; std(z) → 1.03 at long lookbacks. |
| A1-05 | S1 | DDM emitted 0.0 (and Regime "NEUTRAL") through the warm-up; adaptive bands absorbed those zeros. | Warm-up is NaN / "N/A"; quantiles on the masked series. |
| A1-06 | S1 | DFA "Hurst" on residual *levels* inverted its meaning (AR(0.9) read as trending). | DFA on differences; NaN (not a fabricated 0.5) when not computable. |
| A1-07 | S1 | OU μ biased (median 7.68 vs true 5). | Intercept consistent with the bias-corrected AR(1); μ median 4.92–5.07. |
| A1-08 | S1 | Engine EDGE/INVERTED verdict: six any-pass t-tests on raw means — fired on 19.5% of null runs (48.7% with PE drift); in forward mode the UI's "double-forward artifact" caveat was false. | Forward mode defers the edge verdict to the conclusion layer; levels mode tests the excess over the unconditional forward change, Holm-adjusted, on the displayed rule. |
| A1-09 | S1 | "Bai-Perron" never ran (absent from statsmodels 0.15); the fallback heuristic found exactly 3 breaks on 100% of iid series and still moved the training window. | Removed from the walk-forward; exposed only as `changepoints_heuristic` (diagnostic). |
| A1-10 | S2 | Forward-mode conviction stack (conformal z of the forecast → zones → DDM) zeroed 60% of rows and lagged ≥ the horizon. | Forward conviction = −100·tanh(ForecastZ/2), ForecastZ the causal z of the forecast; the conformal stack is kept as extension diagnostics. |
| A1-13/14/15 | S2 | Model stats scored placeholder labels; refit cadence derived from n (repainted at n = 1000); inner CV (LOO / 2-fold) on overlapping labels. | `label_mask` honoured; fixed `AARAMBH_WF_REFIT`; purged `TimeSeriesSplit(gap=h)`; alpha grid widened. |

## Nirnay engine (A2)

| ID | Sev | Finding | Resolution |
|---|---|---|---|
| A2-01 | S1 | Online-HMM emission std floored at 1e-4 → states collapsed onto frozen means in 50/50 stocks; a one-row window slide flipped 25% of regime labels. | Floor `NIRNAY_HMM_STD_FLOOR = 0.10` (kernel and class oracle): start-shift flips 11.6% → 0.00%. |
| A2-02 | S1 | Convergence read Nirnay contrarian in one dimension and trend-following in another (opposite on 75% of days). | One polarity (contrarian) in every anatomy dimension; the conclusion uses −Avg_Signal. |
| A2-03/13 | S1 | Pre-listing NaN rows counted as Neutral (Oversold_Pct bias −6 pp); warm-up rows entered convergence. | NaN-Close rows dropped; `Warmup` flag; breadth over `n_live` warm names; bias 0.00 pp. |
| A2-04 | S1 | US/EU/FX/commodity closes dated D (printing after the NSE close of D) were joined onto NSE date D. | `data.macro_lag.lag_non_ist` (one NSE session); sheet US yields lagged one row (`NON_IST_SHEET_COLUMNS`). |
| A2-05 | S1 | Hard-coded `df["DATE"]` crashed when the sheet's date column had another name. | Canonical dates from the selected column via `parse_dates`. |
| A2-06 | S2 | Daily-sliding 7-year start repainted expanding statistics/PCA/HMM history. | History start anchored to 1 January (`HISTORY_YEARS`). |
| A2-07/08/09/10/11 | S2 | MMR_Quality mislabelled; thresholds/tooltips described rules the code doesn't run; CUSUM fired ~20× its design rate; aggregation reported a count as a percentage; macro PCA zero-filled young columns (\|PC\| up to 116). | MSE-ratio skill; actual cuts exposed (`OS_Cut`/`OB_Cut`); CUSUM on standardized innovations; percentage fields for every regime bucket; columns enter the PCA only with real history. |
| A2-12 | S2 | Dead, divergent duplicates (`analytics/signals.py` with look-ahead math; duplicated helpers). | Deleted; `analytics/utils` is the single source. |

## Conclusion / convergence layer (A3, A6)

| ID | Sev | Finding | Resolution |
|---|---|---|---|
| A3-03 / A6-10 | S1 | The 8 calibrated parameters (4 weights, 4 thresholds) were not identified: the score is non-zero only when the engines agree in sign, so its IC is the IC of a 3-valued sign; 200 random weight vectors moved IC by 0.008 vs a 0.06 sampling sd; `w_direction` had zero effect. | v1 calibration retired. Conclusion v2 calibrates one identified parameter (κ). |
| A3-02 | S1 | ~80% of calibration rows had no Nirnay data; 3 of 4 objective folds returned a −1e6 sentinel; "train +0.03 / val +0.15, 100% durable" was overlap geometry. | Missing Nirnay is `NO_DATA`, never neutral-filled; lift tested only where both engines have data. |
| A3-04 / A6-02 | S1 | The acceptance gate accepted 9–12% of pure-noise runs (22% near the overlap gate); the incumbent was never demoted; a rejected run displayed the prior's stored IC. | No persisted profiles: κ is re-estimated walk-forward on every run; runs are idempotent. |
| A3-05 / A6-05 | S1 | "Non-overlapping" ICs used stride = min horizon (3) for horizons up to 20. | Every reported IC is h-spaced at its own horizon, with Fisher, HAC and block-bootstrap inference. |
| A3-06 / A6-03 | S1 | Thresholds learned on one signal were applied to another (HOLD share 51% vs 91%). | One signal, fixed tiers. |
| A3-07 / A6-01 | S1 | The hero's number (DDM of the composite) had no measured edge; its chip and horizon belonged to other signals. | Hero shows C with evidence computed on C at the displayed horizon. |
| A3-08/09/10 | S2 | Divergence `CONTRADICTION` unreachable; missing-data days scored as divergences; four definitions of "agreement". | One state machine (`agreement_state`), trading-day persistence, no events on `NO_DATA`. |
| A3-01 / A3-12 / A4-02 / A4-07 | S0–S1 | An imported profile with a missing key crashed every later run; NaN val IC rendered "SOLID EDGE"; a corrupt `profiles.json` silently wiped all profiles. | Profile import/persistence removed from the product. |
| A6-17 | S2 | Stacked DDM-on-DDM lag ≈ 20 trading days against a 10-day forecast. | No re-filtering of the conclusion; `conclusion.chain` reports IC at each stage. |

## Presentation truth (A4)

| ID | Sev | Finding | Resolution |
|---|---|---|---|
| A4-01 | S1 | Analog Hurst feature was 0.5 on 4 of 5 rows (fill never fired) → analog choice tracked row position mod 5. | Fixed (≈84% of analogs change). |
| A4-03 | S1 | Hero made a timed, coloured directional call while its own trust tier said NO EDGE / UNCALIBRATED, and in levels mode while INVERTED. | Hero renders the conclusion's trust tier; muted / no call without evidence. |
| A4-06 | S1 | CSV column names, profile fields and exception text reached `unsafe_allow_html` unescaped (raw `<iframe>` rendered under AppTest). | Single escape helper; no profile import. |
| A4-08 | S1 | Aarambh tab used fair-value / "primary trade signal" language in forward mode; OU half-life shown when not meaningful. | Mode-aware copy. |
| A4-14 | S1 | Precedent "≥5 distinct analogs" gate never fired; the precedent row leaned on 41% of null runs; "historically stronger" claim unsupported. | Unconditional base rate + binomial p; votes only when significant. |

## Orchestration, data & deployability (A5)

| ID | Sev | Finding | Resolution |
|---|---|---|---|
| A5-01 | S0 | Under pandas 3, `dayfirst=True` scrambled ISO dates (2500 rows → 984 kept, 900 wrong) and crashed the bond join. | `data.dates.parse_dates`: ISO vs day-first detection, fails loudly. |
| A5-02 | S0 | Cache key used `len(data)`: a different sheet of the same length returned the previous sheet's results. | Config fingerprint + content hash. |
| A5-03/04 | S1 | Intelligence toggle did not recompute; gate-skip published defaults while scoring with prior weights. | Toggle is part of the config and the key; no hidden prior. |
| A5-05..10 | S1–S2 | Refresh Data was a no-op for threaded fetches; stale fallback missed after midnight; partial batches overwrote full snapshots; breaker counted lifetime failures; HALF_OPEN refused the parallel call; the Sheets stale path aborted. | Explicit `force`; namespace-level stale lookup; per-symbol merge, never shrink; consecutive-failure breaker admitting 2 probes; provenance instead of errors. |
| A5-12 | S1 | Secret sheet URL was sent to every visitor's browser as a password field's value. | Never sent to the client. |
| A5-16 | S1 | Pickle disk cache: a planted snapshot executed code during macro backfill. | Parquet / JSON with atomic writes; legacy `.pkl` never read. |
| F5 | S2 | The whole pipeline lived inside Streamlit `main()` — nothing ran or could be tested headlessly. | `pipeline/` package, CLI, JSON run report; `app.py` is a UI shell. |

## Adversarial red team (T2, after the build)

| Attack | Result |
|---|---|
| End-to-end prefix invariance (sheet truncated by 20 / 60 rows; κ < 1 path forced) | PASS — max \|Δ\| = 0 for zA, zN, κ, C, Forecast, ForecastZ, Nirnay, macro panel. |
| Label leakage | PASS on the code; **unguarded assumption** that the label's timestamps are correct — a one-day mis-stamp scored TRADEABLE. → `label_alignment_check` guard. |
| End-to-end honesty | PASS — 0/12 no-signal worlds reached DIRECTIONAL/TRADEABLE, 0/12 passed the lift, κ = 1 on every row. **Power is low**: a planted IC of 0.15 was detected in 1/6 worlds; the engine recovers ~0.03–0.10 of it among 12 noise predictors, and ~234 windows can only detect IC ≳ 0.17. → phase-averaged IC + disclosed minimum detectable IC. |
| Guard weakening | PASS — every edge word in the report is backed by a field. |
| Determinism & fingerprint | Partial FAIL → fixed: predictor order changed results under one key (canonical order now); fingerprint now includes every `core.config` constant. |

## Documentation claims (A6)

README tab list (5 vs 6), the config table (1500/2000 vs 15/30), "84 macro tickers" (83),
"44 custom predictors → Aarambh" (up to 44 are built when every input exists, but in predictive mode none reach Aarambh — they feed MMR only), "Bai-Perron",
"95% confidence bands", the Intelligence "one-flow" steps and objective, the DDM
half-life arithmetic, `PROFILE_VERSION` "checked", "tests/verify_audit.py", and 31
"v1.4.0" module headers were all false or stale. Corrected in the 2.0.0 README,
CHANGELOG (with an Erratum for 1.5.0/1.6.0), FINDINGS status note and module docstrings.
