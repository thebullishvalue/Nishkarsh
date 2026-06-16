# Nishkarsh — Performance & Optimization Findings

_Empirical record of where a full run spends wall-clock and which optimizations
were applied, verified, or deliberately **not** done — so the decisions are not
re-litigated. Chronological history is in [`../CHANGELOG.md`](../CHANGELOG.md);
predictive-value conclusions are in [`FINDINGS.md`](FINDINGS.md)._

Measured 2026-06-16/17 on Streamlit Community Cloud (≈1 weak, shared vCPU;
run-to-run timing is variable) and a local 4-core box. Target run: NIFTY50_PE,
~4900 obs, 50 constituents.

---

## Phase wall-clock (representative cold run)

| Phase | Time | Nature |
|---|---|---|
| 1 — Data acquisition | ~1m (cold) / ~0 (warm) | yfinance network + rate-limit retries; disk-cached |
| 2 — Aarambh walk-forward | ~1m30s | Compute — HuberRegressor-dominated (~74%) ensemble across ~685 refit chunks |
| 3 — Nirnay | ~1m10s → ~36s | Per-stock recursion (~35s) + aggregation (was ~35s, now ~1s) |
| 4 — Convergence | ~6s | Compute |
| 5 — Final assembly | ~4s | — |

Warm-cache runs are far faster: the data fetches and the macro-factor PCA are
both memoized.

---

## Optimizations applied (all output-preserving)

1. **Causal macro-factor PCA → incremental covariance** (`analytics/factors.py`).
   The expanding-window PCA refit `StandardScaler`+`PCA` on the whole window at
   every block — `O(refits·n·m²)`. Rewritten to carry running `Σx` / `Σxxᵀ` and
   extend them one block per step (`O(n·m²)`), eigendecomposing the small `m×m`
   correlation matrix. **Bit-identical to ~1e-13** (verified against the prior
   algorithm, incl. zero-variance columns). Minutes → sub-second on a weak CPU.
   Memoized via `st.cache_data` on the panel content.

2. **Nirnay daily aggregation → vectorized** (`aggregate_constituent_timeseries`).
   Replaced an `O(dates × stocks)` per-cell `.loc` loop (~86k lookups) with one
   `concat` + `groupby`. **~50× faster, bit-identical** (max diff 1e-14, same
   columns/index/dtypes).

3. **Concurrent data acquisition** (`app.py`). The macro and constituent-OHLCV
   yfinance batches now download in parallel (the circuit breaker releases its
   lock during the network call; cache + breaker are `threading.Lock`-guarded;
   fetchers touch no `st.*`). Overlaps the two cold-cache fetches.

4. **`HUBER_MAX_ITER` 500 → 200** (`core/config.py`). The walk-forward Huber fits
   converge well before the cap (0 of 140 chunks hit 150 iters on real data);
   output bit-identical, just less headroom.

5. **scikit-learn 1.9 compatibility.** `ElasticNetCV(n_alphas=…)` was removed in
   sklearn 1.9 → switched to the integer-`alphas` form (`alphas=10`,
   behaviour-identical); `requirements.txt` floor raised to `scikit-learn>=1.7`.

---

## Huber A/B study (why Huber is kept)

Member-level walk-forward on the **real cached PE sheet** (2200 rows, 140
chunks, 700 OOS), swapping only the robust-regression slot:

| member | rel. cost | OOS R² | IC |
|---|---|---|---|
| **HuberRegressor** | heaviest | **0.668** | **0.832** |
| OLS / Ridge | ~100× faster | 0.17–0.18 | 0.77 |
| SGD(huber) | slower-ish *and* worse | 0.43 | 0.79 |
| TheilSen / RANSAC | 16 min+ (pathological) | — | — |

**Verdict:** Huber is simultaneously the slowest **and** the most accurate
member — its robustness is load-bearing on this outlier-heavy data (that's *why*
it's slow). Every faster alternative is materially worse. **Do not replace it.**

---

## Deliberate non-optimizations (genuine compute floors)

- **Aarambh walk-forward Huber.** Cannot be sped up while exact: warm-start gave
  no speedup and shifted coefficients ~1e-3; iteration trimming is already done
  (it converges before the cap). Parallelizing chunks won't help on a 1-core host.
- **Nirnay per-stock recursion** (Kalman → GARCH → HMM → CUSUM, per row, ×50
  stocks). A sequential online-learning recursion built on per-row sliding-window
  reductions; making them incremental changes float-point order and risks
  drifting regime labels. HMM micro-opts were applied (bit-identical) but yielded
  ~0. Process-parallelism across the 50 independent stocks gave only ~1.4× on 4
  cores (DataFrame pickling dominates) and would be worse on the cloud.
- **Data acquisition cold time** beyond the concurrent-fetch overlap is yfinance
  server latency + rate-limit backoff — external, not optimizable.

**Rule of thumb going forward:** the big wins here were *algorithmic*
(incremental covariance) and *vectorization* (aggregation), both bit-identical.
The remaining poles are genuine sequential compute or external I/O — verify any
further change is output-preserving (diff to ~1e-13) before shipping it.
