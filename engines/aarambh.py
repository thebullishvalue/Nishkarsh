"""
Nishkarsh — AARAMBH FairValueEngine: purged walk-forward ensemble for the Nifty 50 PE.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Two modes (contract C-AAR, docs/ARCHITECTURE.md):
  * FORWARD (predictive, shipped default): ``y[t]`` is the forward h-day Δlog(target);
    the engine's output IS the purged walk-forward forecast. ``ForecastZ`` is its
    causal z-score and ``ConvictionScore = ConvictionBounded = −100·tanh(ForecastZ/2)``
    (NEGATIVE = bullish). The conformal/breadth stack is kept only as "extension"
    diagnostics. The directional edge is graded downstream by the conclusion layer.
  * LEVELS (valuation context — falsified as a forecaster): ``y`` is the target level;
    residual → conformal z-scores → breadth → leaky-integrator conviction with causal
    adaptive bands, and a repaired engine-level edge test.

Imports math primitives from analytics.*; no Streamlit, no IO.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from core.config import (
    AARAMBH_INNER_CV_SPLITS,
    AARAMBH_SAMPLE_HALFLIFE,
    AARAMBH_WF_REFIT,
    CONV2_TIER_MODERATE,
    CONV2_TIER_STRONG,
    CONV2_TIER_WEAK,
    CONV2_ZCLIP,
    CONV2_ZMIN_PERIODS,
    CONVICTION_MODERATE,
    CONVICTION_STRONG,
    CONVICTION_WEAK,
    DDM_DRIFT_SCALE,
    DDM_LEAK_RATE,
    DDM_LONG_RUN_VAR,
    EDGE_ALPHA,
    HUBER_EPSILON,
    HUBER_MAX_ITER,
    LOOKBACK_WINDOWS,
    MAX_TRAIN_SIZE,
    MIN_TRAIN_SIZE,
    OU_PROJECTION_DAYS,
    RIDGE_ALPHAS,
)

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

from analytics.causal_stats import causal_z
from analytics.conformal import compute_conformal_zscores
from analytics.ddm_filter import drift_diffusion_filter
from analytics.hurst import hurst_dfa
from analytics.ou_process import ornstein_uhlenbeck_estimate
from analytics.structural_breaks import detect_structural_breaks
from analytics.utils import _apply_conviction_bounds, _classify_zones, _detect_crossover_signals

# Ensemble members (equal-weight, NaN-aware mean). "pca_ols" = OLS on the PCA
# scores of the standardized features (95% variance).
_MEMBERS = ("ridge", "huber", "elasticnet", "pca_ols")
# Minimum non-overlapping signal events for a levels-mode edge test to run.
_EDGE_MIN_EVENTS = 20
_PERF_PERIODS = (5, 10, 20)

DEFERRED_EDGE_TEXT = "DEFERRED — directional edge is graded by the conclusion layer (Edge Ledger)"


def _t_test(values: np.ndarray) -> dict:
    """One-sample two-sided t-test of mean 0. Degenerate input → p = 1 (no evidence)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 3:
        return {"mean": float(np.mean(v)) if n else float("nan"), "t_stat": 0.0, "p_value": 1.0, "n": n}
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    if not np.isfinite(sd) or sd < 1e-10:
        return {"mean": m, "t_stat": 0.0, "p_value": 1.0, "n": n}
    t = m / (sd / np.sqrt(n))
    return {"mean": m, "t_stat": float(t), "p_value": float(2.0 * stats.t.sf(abs(t), n - 1)), "n": n}


def _holm(pvals: list[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values (family = all entries)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


class _LinearFit:
    """Minimal fitted linear model (coef_, intercept_, alpha_, predict)."""

    def __init__(self, coef: np.ndarray, intercept: float, alpha: float) -> None:
        self.coef_, self.intercept_, self.alpha_ = coef, float(intercept), float(alpha)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


def _weighted_ridge_path(X: np.ndarray, y: np.ndarray, w: np.ndarray, alphas: np.ndarray):
    """sklearn ``Ridge(fit_intercept=True)`` with sample weights, for every alpha at once.
    Returns (coefs[k, p], intercepts[k])."""
    ws = w / w.sum()
    x_off = ws @ X
    y_off = float(ws @ y)
    sw = np.sqrt(w)[:, None]
    Xc = (X - x_off) * sw
    yc = (y - y_off) * sw[:, 0]
    evals, V = np.linalg.eigh(Xc.T @ Xc)
    b = V.T @ (Xc.T @ yc)
    coefs = (b[None, :] / (evals[None, :] + alphas[:, None])) @ V.T
    return coefs, y_off - coefs @ x_off


def purged_ridge_cv(X: np.ndarray, y: np.ndarray, w: np.ndarray, cv, alphas) -> _LinearFit:
    """Closed-form equivalent of ``RidgeCV(alphas, cv=cv).fit(X, y, sample_weight=w)``:
    weighted ridge per CV split, alpha = first maximiser of the mean (sample-weighted)
    test-fold R², refit on all rows with weights. Same selection rule as sklearn's
    GridSearchCV path, ~30× cheaper (verified alpha-identical, |Δcoef| ~1e-15)."""
    a = np.asarray(alphas, dtype=np.float64)
    scores = np.zeros(len(a))
    splits = list(cv.split(X))
    for tr, te in splits:
        coefs, icpt = _weighted_ridge_path(X[tr], y[tr], w[tr], a)
        pred = X[te] @ coefs.T + icpt[None, :]
        wt, yt = w[te], y[te]
        ybar = float(wt @ yt) / float(wt.sum())
        ss_tot = float(wt @ (yt - ybar) ** 2)
        ss_res = wt @ (yt[:, None] - pred) ** 2
        scores += (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.where(ss_res == 0, 1.0, 0.0)
    k = int(np.argmax(scores / max(len(splits), 1)))
    coefs, icpt = _weighted_ridge_path(X, y, w, a[k:k + 1])
    return _LinearFit(coefs[0], icpt[0], a[k])


def _spaced(idx: np.ndarray, gap: int) -> np.ndarray:
    """Greedy subset of sorted indices at least ``gap`` apart (non-overlapping windows)."""
    out, last = [], -(10 ** 9)
    for i in idx:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.asarray(out, dtype=int)


class FairValueEngine:
    """Purged walk-forward ensemble engine (Ridge / Huber / ElasticNet / PCA-OLS).

    Pipeline:
        1. Purged walk-forward: at every refit point ``t`` (fixed cadence
           ``AARAMBH_WF_REFIT``) the members are fit on the last ``max_train``
           label-complete rows strictly before ``t − purge`` (≥ ``min_train`` of
           them, else the chunk is NaN and counted as starved — the purge is never
           dropped). Inner regularization search uses
           ``TimeSeriesSplit(n_splits=AARAMBH_INNER_CV_SPLITS, gap=purge)``.
           Exponential sample weights, half-life ``AARAMBH_SAMPLE_HALFLIFE``.
        2. Equal-weight, NaN-aware ensemble mean → ``Forecast``; member dispersion
           → ``ModelSpread``; ``ForecastZ`` = ``causal_z(Forecast)``.
        3. Conformal z-scores (5 lookbacks) of the residual → zones → breadth.
        4. FORWARD: conviction = −100·tanh(ForecastZ/2), fixed CONV2 tiers.
           LEVELS: leaky-integrator conviction, causal adaptive bands.
        5. Diagnostics: OU fit, DFA Hurst of the residual's increments,
           change-point heuristic (descriptive only), forward-change table.
    """

    def __init__(self) -> None:
        self.ts_data: pd.DataFrame = pd.DataFrame()
        self.lookback_data: dict = {}
        self.model_stats: dict = {}
        self.ou_params: dict = {}
        self.ou_projection: np.ndarray = np.array([])
        self.ou_projection_upper: np.ndarray = np.array([])
        self.ou_projection_lower: np.ndarray = np.array([])
        self.hurst: float = float("nan")
        self.conviction_levels: dict = {
            "weak": float(CONVICTION_WEAK),
            "moderate": float(CONVICTION_MODERATE),
            "strong": float(CONVICTION_STRONG),
        }
        self._bands: dict = {}
        self.latest_feature_impacts: dict = {}
        self.signal_edge: dict = {}
        self.feature_impact_history: list[dict] = []
        self.theta_history: list[float] = []
        self.changepoints_heuristic: list[int] = []
        self.feature_names: list[str] = []
        self.n_samples: int = 0
        self.forward_signal: bool = False
        self.purge: int = 0
        self.min_train: int = int(MIN_TRAIN_SIZE)
        self.max_train: int = int(MAX_TRAIN_SIZE)
        self.wf_meta: dict = {}
        self.price: np.ndarray | None = None
        self.label_mask: np.ndarray | None = None
        self.y: np.ndarray = np.array([])
        self.predictions: np.ndarray = np.array([])
        self.model_spread: np.ndarray = np.array([])
        self.residuals: np.ndarray = np.array([])
        self._label_ok: np.ndarray = np.array([], dtype=bool)
        self._valid: np.ndarray = np.array([], dtype=bool)

    @property
    def break_dates(self) -> list[int]:
        """DEPRECATED alias of ``changepoints_heuristic`` (a heuristic, not a test).
        Kept only so pre-2.0 call sites do not raise; remove with app.py's console."""
        return self.changepoints_heuristic

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        progress_callback: Callable | None = None,
        forward_signal: bool = False,
        purge: int = 0,
        price: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
        min_train: int | None = None,
        max_train: int | None = None,
    ) -> "FairValueEngine":
        """Run the full walk-forward pipeline.

        forward_signal: True = PREDICTIVE mode (``y`` is the forward h-day return).
        purge: rows between the last training label and the forecast point. Must be
            ≥ the label horizon h in forward mode (label at s covers (s, s+h]); the
            training set for a forecast at t uses only rows s < t − purge.
        price: optional target LEVEL aligned with ``y`` (forward-change table and
            swing detection; in forward mode ``y`` is a return, not a price).
        label_mask: optional bool array, False where ``y`` is a placeholder (the
            unrealized forward-label tail). Such rows never enter training and are
            NaN in ``Actual`` / ``Label`` and in the model statistics.
        min_train / max_train: walk-forward window in LABEL-COMPLETE rows
            (default ``core.config`` MIN/MAX_TRAIN_SIZE). Tests pass small values.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"X must be 2-D with len(X) == len(y); got {X.shape} vs {y.shape}")
        n = len(y)
        self.min_train = int(MIN_TRAIN_SIZE if min_train is None else min_train)
        self.max_train = int(MAX_TRAIN_SIZE if max_train is None else max_train)
        if self.min_train < 1 or self.max_train < self.min_train:
            raise ValueError(f"need 1 ≤ min_train ≤ max_train; got {self.min_train}/{self.max_train}")
        p = X.shape[1]
        if self.min_train < 10 * p:
            logging.warning(
                "Aarambh: min_train=%d < 10·p=%d — degenerate p/n; forecasts are unreliable",
                self.min_train, 10 * p,
            )

        self.feature_names = list(feature_names) if feature_names else [f"X{i}" for i in range(p)]
        self.n_samples = n
        self.y = y.copy()
        self.forward_signal = bool(forward_signal)
        self.purge = max(0, int(purge))
        self.price = None
        if price is not None:
            price = np.asarray(price, dtype=np.float64)
            if len(price) == n:
                self.price = price
            else:
                logging.warning("Aarambh: price length %d != %d — ignored", len(price), n)
        if label_mask is not None:
            label_mask = np.asarray(label_mask, dtype=bool)
            if len(label_mask) != n:
                raise ValueError(f"label_mask length {len(label_mask)} != {n}")
        self.label_mask = label_mask
        self._label_ok = (label_mask.copy() if label_mask is not None else np.ones(n, dtype=bool)) & np.isfinite(y)

        # Descriptive change-point heuristic on the target level (full-sample,
        # retrospective, NOT a test; never used by the walk-forward).
        cp_series = self.price if self.price is not None else np.where(self._label_ok, y, np.nan)
        self.changepoints_heuristic = detect_structural_breaks(cp_series)

        self._walk_forward_regression(X, y, progress_callback)
        self._valid = np.isfinite(self.predictions)

        self.residuals = np.where(self._valid, self.y - self.predictions, np.nan)
        self._compute_model_stats()
        if self.forward_signal:
            # The signal IS the forecast: drive the extension stack with −forecast so
            # an expected rise maps to the oversold (bullish) pole.
            self.residuals = -self.predictions

        self._build_ts_data()
        self._compute_multi_lookback_signals()
        self._compute_breadth_metrics()
        self._compute_conviction()
        self._compute_divergences()
        self._compute_forward_changes()
        self._compute_ou_diagnostics()
        self._compute_hurst()
        self._compute_signal_edge()

        if progress_callback:
            progress_callback(1.0, "Done")
        return self

    def get_current_signal(self) -> dict:
        """Current composite signal from the latest observation (keys per C-AAR)."""
        fwd = bool(self.forward_signal)
        edge_source = "conclusion" if fwd else "engine_levels_test"
        if self.ts_data.empty:
            return {
                "signal": "HOLD", "strength": "NEUTRAL", "confidence": "N/A",
                "conviction_score": float("nan"), "conviction_upper": float("nan"),
                "conviction_lower": float("nan"),
                "regime": "N/A", "oversold_breadth": float("nan"), "overbought_breadth": float("nan"),
                "residual": float("nan"), "fair_value": float("nan"), "actual": float("nan"),
                "avg_z": float("nan"), "model_spread": float("nan"),
                "has_bullish_div": False, "has_bearish_div": False,
                "ou_half_life": float("nan"), "adf_pvalue": float("nan"), "kpss_pvalue": float("nan"),
                "hurst": float("nan"), "theta_stable": False,
                "changepoints_heuristic": [],
                "conviction_levels": dict(self.conviction_levels),
                "tradeable": False,
                "edge_assessment": DEFERRED_EDGE_TEXT if fwd else "N/A — no data",
                "edge_source": edge_source,
                "forecast_skill_vs_rw": float("nan"), "mean_reverting": False,
                "random_walk_regime": False, "half_life_meaningful": False,
                "display_strength": "NEUTRAL" if fwd else "NO EDGE",
                "has_forward_edge": False, "inverted": False, "forward_signal": fwd,
                "forecast": float("nan"), "forecast_z": float("nan"),
            }

        current = self.ts_data.iloc[-1]
        cb = float(current["ConvictionBounded"])
        lv = self.conviction_levels
        if not np.isfinite(cb):
            signal, strength = "HOLD", "NEUTRAL"
        elif cb < -lv["strong"]:
            signal, strength = "BUY", "STRONG"
        elif cb < -lv["moderate"]:
            signal, strength = "BUY", "MODERATE"
        elif cb < -lv["weak"]:
            signal, strength = "BUY", "WEAK"
        elif cb > lv["strong"]:
            signal, strength = "SELL", "STRONG"
        elif cb > lv["moderate"]:
            signal, strength = "SELL", "MODERATE"
        elif cb > lv["weak"]:
            signal, strength = "SELL", "WEAK"
        else:
            signal, strength = "HOLD", "NEUTRAL"

        osb = float(current["OversoldBreadth"])
        obb = float(current["OverboughtBreadth"])
        if not np.isfinite(cb):
            confidence = "N/A"
        elif signal == "BUY":
            confidence = "HIGH" if osb >= 80 else "MEDIUM" if osb >= 60 else "LOW"
        elif signal == "SELL":
            confidence = "HIGH" if obb >= 80 else "MEDIUM" if obb >= 60 else "LOW"
        else:
            confidence = "HIGH" if abs(cb) < lv["weak"] * 0.5 else "MEDIUM" if abs(cb) < lv["weak"] else "LOW"

        theta_stable = False
        if len(self.theta_history) >= 10:
            th = np.asarray(self.theta_history[-10:])
            theta_stable = bool(np.std(th) / max(np.mean(th), 1e-6) < 0.5)

        r2_vs_rw = float(self.model_stats.get("r2_vs_rw", float("nan")))
        adf_p = float(self.ou_params.get("adf_pvalue", float("nan")))
        h = float(self.hurst)
        mean_reverting = bool(np.isfinite(adf_p) and adf_p < 0.05 and np.isfinite(h) and h < 0.5)
        random_walk_regime = bool((np.isfinite(h) and 0.45 <= h <= 0.58) or (np.isfinite(adf_p) and adf_p > 0.05))

        edge = self.signal_edge or {}
        has_forward_edge = bool(edge.get("has_forward_edge", False))
        inverted = bool(edge.get("inverted", False))
        best_h = edge.get("best_horizon")
        tradeable = has_forward_edge and not inverted
        if fwd:
            edge_assessment = DEFERRED_EDGE_TEXT
            display_strength = strength
        else:
            a = f"Holm-adjusted p<{EDGE_ALPHA:g}, excess over the unconditional forward change"
            if inverted:
                edge_assessment = f"INVERTED — displayed BUY/SELL preceded the wrong direction ({a})"
            elif has_forward_edge:
                edge_assessment = f"EDGE — displayed signal preceded the correct direction at {best_h}d ({a})"
            elif random_walk_regime:
                edge_assessment = "NO EDGE — random-walk regime; no significant forward excess move"
            else:
                edge_assessment = "NO EDGE — no significant forward excess move on history"
            display_strength = strength if tradeable else ("INVERTED" if inverted else "NO EDGE")

        return {
            "signal": signal,
            "strength": strength,
            "confidence": confidence,
            "conviction_score": cb,
            "conviction_upper": float(current["ConvictionUpper"]),
            "conviction_lower": float(current["ConvictionLower"]),
            "regime": current["Regime"],
            "oversold_breadth": osb,
            "overbought_breadth": obb,
            "residual": float(current["Residual"]),
            "fair_value": float(current["FairValue"]),
            "actual": float(current["Actual"]),
            "avg_z": float(current["AvgZ"]),
            "model_spread": float(current["ModelSpread"]),
            "has_bullish_div": bool(current["BullishDiv"]),
            "has_bearish_div": bool(current["BearishDiv"]),
            "ou_half_life": float(self.ou_params.get("half_life", float("nan"))),
            "adf_pvalue": adf_p,
            "kpss_pvalue": float(self.ou_params.get("kpss_pvalue", float("nan"))),
            "hurst": h,
            "theta_stable": theta_stable,
            "changepoints_heuristic": list(self.changepoints_heuristic),
            "conviction_levels": dict(self.conviction_levels),
            "tradeable": bool(tradeable) and not fwd,
            "edge_assessment": edge_assessment,
            "edge_source": edge_source,
            "forecast_skill_vs_rw": r2_vs_rw,
            "mean_reverting": mean_reverting,
            "random_walk_regime": random_walk_regime,
            "half_life_meaningful": mean_reverting,
            "display_strength": display_strength,
            "has_forward_edge": has_forward_edge and not fwd,
            "inverted": inverted and not fwd,
            "forward_signal": fwd,
            "forecast": float(current["Forecast"]),
            "forecast_z": float(current["ForecastZ"]),
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_regime_stats(self) -> dict:
        ts = self.ts_data
        if ts.empty:
            return {"strongly_oversold": 0, "oversold": 0, "neutral": 0, "overbought": 0,
                    "strongly_overbought": 0, "current_regime": "N/A"}
        counts = ts.loc[ts["Valid"].astype(bool), "Regime"].value_counts()
        return {
            "strongly_oversold": int(counts.get("STRONGLY OVERSOLD", 0)),
            "oversold": int(counts.get("OVERSOLD", 0)),
            "neutral": int(counts.get("NEUTRAL", 0)),
            "overbought": int(counts.get("OVERBOUGHT", 0)),
            "strongly_overbought": int(counts.get("STRONGLY OVERBOUGHT", 0)),
            "current_regime": ts["Regime"].iloc[-1],
        }

    def get_signal_performance(self) -> dict:
        """Forward change after the DISPLAYED signal, vs the unconditional forward change.

        Event rule = the displayed classification: BUY when ConvictionBounded <
        −weak band at that row (causal adaptive band in levels mode, fixed CONV2
        tier in forward mode), SELL when > +weak band. Events are thinned to
        non-overlapping windows (≥ period apart). ``*_avg``/``*_hit`` are the raw
        forward change (SELL sign-flipped, so positive = correct); the test is a
        one-sample t on the EXCESS over the unconditional mean forward change of
        all evaluated rows (drift-adjusted). ``*_p_holm`` = Holm-adjusted across the
        6 (period × side) tests; tests with < 20 events enter the family with p=1.
        """
        ts = self.ts_data
        results: dict = {}
        if ts.empty:
            for period in _PERF_PERIODS:
                results[period] = self._empty_perf()
            return results
        cb = ts["ConvictionBounded"].to_numpy(dtype=np.float64)
        weak = self._bands.get("weak", np.full(len(ts), float(self.conviction_levels["weak"])))
        valid = ts["Valid"].to_numpy(dtype=bool) & np.isfinite(cb)
        raw_p = []
        for period in _PERF_PERIODS:
            fwd = ts[f"FwdChg_{period}"].to_numpy(dtype=np.float64)
            ev = valid & np.isfinite(fwd)
            if not ev.any():
                results[period] = self._empty_perf()
                raw_p += [1.0, 1.0]
                continue
            mu_u = float(np.mean(fwd[ev]))
            buy_idx = _spaced(np.flatnonzero(ev & (cb < -weak)), period)
            sell_idx = _spaced(np.flatnonzero(ev & (cb > weak)), period)
            buy_raw, sell_raw = fwd[buy_idx], -fwd[sell_idx]
            bt, st = _t_test(buy_raw - mu_u), _t_test(sell_raw + mu_u)
            r = {
                "uncond_avg": mu_u, "n_eval": int(ev.sum()),
                "buy_avg": float(np.mean(buy_raw)) if len(buy_raw) else 0.0,
                "buy_excess": bt["mean"] if len(buy_raw) else 0.0,
                "buy_hit": float(np.mean(buy_raw > 0)) if len(buy_raw) else 0.0,
                "buy_count": int(len(buy_raw)), "buy_t_stat": bt["t_stat"], "buy_p_value": bt["p_value"],
                "sell_avg": float(np.mean(sell_raw)) if len(sell_raw) else 0.0,
                "sell_excess": st["mean"] if len(sell_raw) else 0.0,
                "sell_hit": float(np.mean(sell_raw > 0)) if len(sell_raw) else 0.0,
                "sell_count": int(len(sell_raw)), "sell_t_stat": st["t_stat"], "sell_p_value": st["p_value"],
            }
            results[period] = r
            raw_p.append(r["buy_p_value"] if r["buy_count"] >= _EDGE_MIN_EVENTS else 1.0)
            raw_p.append(r["sell_p_value"] if r["sell_count"] >= _EDGE_MIN_EVENTS else 1.0)
        adj = _holm(raw_p)
        for k, period in enumerate(_PERF_PERIODS):
            results[period]["buy_p_holm"] = adj[2 * k]
            results[period]["sell_p_holm"] = adj[2 * k + 1]
        return results

    @staticmethod
    def _empty_perf() -> dict:
        return {"uncond_avg": float("nan"), "n_eval": 0,
                "buy_avg": 0.0, "buy_excess": 0.0, "buy_hit": 0.0, "buy_count": 0,
                "buy_t_stat": 0.0, "buy_p_value": 1.0, "buy_p_holm": 1.0,
                "sell_avg": 0.0, "sell_excess": 0.0, "sell_hit": 0.0, "sell_count": 0,
                "sell_t_stat": 0.0, "sell_p_value": 1.0, "sell_p_holm": 1.0}

    def get_feature_impact_history(self) -> pd.DataFrame:
        """Per-refit |loading| share of each feature in the PCA-OLS member (index = refit row)."""
        if not self.feature_impact_history:
            return pd.DataFrame()
        return pd.DataFrame(self.feature_impact_history)

    # ── Private: Walk-Forward Regression ──────────────────────────────────

    def _walk_forward_regression(self, X: np.ndarray, y: np.ndarray, progress_callback) -> None:
        n = self.n_samples
        mn, mx, purge, refit = self.min_train, self.max_train, self.purge, int(AARAMBH_WF_REFIT)
        self.predictions = np.full(n, np.nan)
        self.model_spread = np.full(n, np.nan)
        self.feature_impact_history = []
        self.latest_feature_impacts = {}
        self._member_failures: dict[str, int] = {}

        # Label-complete, finite rows are the only admissible training rows.
        elig_idx = np.flatnonzero(self._label_ok & np.isfinite(X).all(axis=1))
        global_weights = np.exp(-np.log(2) / float(AARAMBH_SAMPLE_HALFLIFE) * np.arange(mx - 1, -1, -1))

        chunks_total = starved = fallback = 0
        starts = range(mn + purge, n, refit)
        n_chunks = max(len(starts), 1)
        for i, t_start in enumerate(starts):
            t_end = min(t_start + refit, n)
            chunks_total += 1
            train_end = t_start - purge  # every training row s < t_start − purge
            k = int(np.searchsorted(elig_idx, train_end, side="left"))
            idx = elig_idx[max(0, k - mx):k]
            if len(idx) < mn:
                starved += 1  # never drop the purge: no forecast for this chunk
            else:
                try:
                    out = self._process_wf_chunk(t_start, t_end, X, y, idx, global_weights)
                except Exception as e:  # pragma: no cover - defensive
                    logging.warning("Walk-forward chunk failed [%d:%d]: %s", t_start, t_end, e)
                    out = None
                if out is None:
                    fallback += 1
                else:
                    preds, spreads, impacts = out
                    self.predictions[t_start:t_end] = preds
                    self.model_spread[t_start:t_end] = spreads
                    if impacts:
                        self.latest_feature_impacts = dict(sorted(impacts.items(), key=lambda kv: kv[1], reverse=True))
                        self.feature_impact_history.append({"index": int(t_start), **impacts})
            if progress_callback:
                progress_callback((i + 1) / n_chunks, f"Walking Forward... ({t_end}/{n})")

        if self._member_failures:
            logging.warning("Aarambh: ensemble member fit failures (chunks): %s", self._member_failures)
        finite = np.flatnonzero(np.isfinite(self.predictions))
        self.wf_meta = {
            "min_train": mn, "max_train": mx, "purge": purge, "refit": refit,
            "first_forecast_row": int(finite[0]) if len(finite) else None,
            "n_oos": int(len(finite)),
            "chunks_total": chunks_total,
            "chunks_skipped_starved": starved,
            "chunks_fallback": fallback,
        }

    def _process_wf_chunk(
        self, t_start: int, t_end: int, X: np.ndarray, y: np.ndarray,
        train_idx: np.ndarray, global_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict] | None:
        """Fit the ensemble on ``train_idx`` and forecast rows [t_start, t_end).
        Returns None when no member produced a forecast (counted as fallback)."""
        X_tr, y_tr = X[train_idx], y[train_idx]
        w = global_weights[-len(train_idx):]
        models, scaler, valid_cols = self._fit_ensemble(X_tr, y_tr, w)

        X_pred = X[t_start:t_end][:, valid_cols]
        m = len(X_pred)
        rows_ok = np.isfinite(X_pred).all(axis=1)
        member_preds = []
        if rows_ok.any():
            Xs = scaler.transform(X_pred[rows_ok])
            for key in _MEMBERS:
                mdl = models.get(key)
                if mdl is None:
                    continue
                try:
                    pr = mdl.predict(models["pca"].transform(Xs)) if key == "pca_ols" else mdl.predict(Xs)
                except Exception:
                    continue
                full = np.full(m, np.nan)
                full[rows_ok] = np.where(np.isfinite(pr) & (np.abs(pr) < 1e10), pr, np.nan)
                if np.isfinite(full).any():
                    member_preds.append(full)
        if not member_preds:
            return None

        P = np.vstack(member_preds)
        cnt = np.isfinite(P).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            preds = np.nanmean(P, axis=0)  # equal weights over the members that fit
            spreads = np.where(cnt >= 2, np.nanstd(P, axis=0), np.where(cnt == 1, 0.0, np.nan))
        preds = np.where(cnt > 0, preds, np.nan)

        impacts: dict = {}
        ols, pca = models.get("pca_ols"), models.get("pca")
        if ols is not None and pca is not None:
            fw = np.abs(np.dot(ols.coef_, pca.components_))
            tot = float(np.sum(fw))
            if tot > 1e-10:
                names = np.array(self.feature_names)[valid_cols]
                impacts = {str(f): float(v) for f, v in zip(names, fw / tot * 100)}
        return preds, spreads, impacts

    def _fit_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray):
        models: dict = {"ridge": None, "huber": None, "elasticnet": None, "pca": None, "pca_ols": None}
        valid_cols = np.std(X_train, axis=0) > 1e-8
        if not np.any(valid_cols):
            valid_cols = np.ones(X_train.shape[1], dtype=bool)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train[:, valid_cols])
        # Purged inner CV: validation folds lie AFTER their training folds with a
        # `purge`-row gap, so overlapping forward labels cannot leak into the
        # regularization choice (LOO on overlapping labels under-regularizes).
        cv = TimeSeriesSplit(n_splits=int(AARAMBH_INNER_CV_SPLITS), gap=self.purge)

        def _fail(key: str) -> None:
            self._member_failures[key] = self._member_failures.get(key, 0) + 1

        try:
            models["ridge"] = purged_ridge_cv(Xs, y_train, weights, cv, RIDGE_ALPHAS)
        except Exception:
            _fail("ridge")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models["huber"] = HuberRegressor(
                    epsilon=HUBER_EPSILON, max_iter=HUBER_MAX_ITER, tol=1e-3
                ).fit(Xs, y_train, sample_weight=weights)
        except Exception:
            _fail("huber")
        try:
            # random_state pins the random coordinate-descent order (determinism).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models["elasticnet"] = ElasticNetCV(
                    l1_ratio=[0.5, 0.9, 1.0], alphas=10, cv=cv, max_iter=2000, tol=1e-2,
                    selection="random", random_state=42, n_jobs=1,
                ).fit(Xs, y_train, sample_weight=weights)
        except Exception:
            _fail("elasticnet")
        try:
            pca = PCA(n_components=0.95, svd_solver="full")
            Z = pca.fit_transform(Xs)
            models["pca"] = pca
            models["pca_ols"] = LinearRegression().fit(Z, y_train, sample_weight=weights)
        except Exception:
            _fail("pca_ols")
        return models, scaler, valid_cols

    # ── Private: Analytics Pipeline ───────────────────────────────────────

    def _compute_model_stats(self) -> None:
        """OOS skill on rows with a genuine forecast AND a realized label (A1-13)."""
        oos = self._valid & self._label_ok
        y_v, p_v = self.y[oos], self.predictions[oos]
        nan = float("nan")
        r2 = rmse = mae = r2_vs_rw = nan
        if len(y_v) > 2:
            r2 = float(r2_score(y_v, p_v))
            rmse = float(np.sqrt(mean_squared_error(y_v, p_v)))
            mae = float(mean_absolute_error(y_v, p_v))
            stride = max(1, self.purge) if self.forward_signal else 1
            y_sub, p_sub = y_v[::stride], p_v[::stride]
            if self.forward_signal:
                baseline = np.zeros_like(y_sub)  # martingale null for a return forecast
            else:
                baseline = np.empty_like(y_sub)  # random walk for a level
                baseline[0] = y_sub[0]
                baseline[1:] = y_sub[:-1]
            ss_res = float(np.sum((y_sub - p_sub) ** 2))
            ss_rw = float(np.sum((y_sub - baseline) ** 2))
            r2_vs_rw = 1 - ss_res / max(ss_rw, 1e-10)
        meta = self.wf_meta
        attempted = max(meta.get("chunks_total", 0) - meta.get("chunks_skipped_starved", 0), 0)
        fb = meta.get("chunks_fallback", 0)
        sp = self.model_spread[self._valid]
        self.model_stats = {
            "r2_oos": r2, "r2_vs_rw": r2_vs_rw, "rmse_oos": rmse, "mae_oos": mae,
            "n_obs": int(len(y_v)), "n_features": len(self.feature_names),
            "avg_model_spread": float(np.nanmean(sp)) if np.isfinite(sp).any() else nan,
            "model_coverage": float(1.0 - fb / attempted) if attempted else nan,
            "n_fallback_chunks": int(fb),
            "n_starved_chunks": int(meta.get("chunks_skipped_starved", 0)),
        }

    def _build_ts_data(self) -> None:
        label = np.where(self._label_ok, self.y, np.nan)
        forecast = np.where(self._valid, self.predictions, np.nan)
        fz = causal_z(forecast, min_periods=CONV2_ZMIN_PERIODS, clip=CONV2_ZCLIP).to_numpy(dtype=np.float64)
        self.ts_data = pd.DataFrame({
            "Actual": label, "FairValue": self.predictions,
            "Residual": self.residuals, "ModelSpread": self.model_spread,
            "Valid": self._valid.copy(), "Forecast": forecast, "Label": label, "ForecastZ": fz,
        })

    def _compute_multi_lookback_signals(self) -> None:
        r = self.residuals
        n = len(r)
        self.lookback_data = {}
        for lb in LOOKBACK_WINDOWS:
            if n < lb:
                continue
            z, _, _ = compute_conformal_zscores(r, window=lb, min_periods=max(lb // 2, 5), alpha=0.05)
            zones = _classify_zones(z)
            buy, sell = _detect_crossover_signals(z)
            self.lookback_data[lb] = {"z_scores": z, "zones": zones, "buy_signals": buy, "sell_signals": sell}
            self.ts_data[f"Z_{lb}"] = z
            self.ts_data[f"Zone_{lb}"] = zones
            self.ts_data[f"Buy_{lb}"] = buy
            self.ts_data[f"Sell_{lb}"] = sell

    def _compute_breadth_metrics(self) -> None:
        n = len(self.ts_data)
        lbs = list(self.lookback_data)
        num_lb = max(len(lbs), 1)
        os_, ob, eos, eob, bc, sc = (np.zeros(n) for _ in range(6))
        zs = []
        for lb in lbs:
            d = self.lookback_data[lb]
            zones = d["zones"]
            eos += zones == "Extreme Under"
            os_ += zones == "Undervalued"
            eob += zones == "Extreme Over"
            ob += zones == "Overvalued"
            bc += d["buy_signals"]
            sc += d["sell_signals"]
            zs.append(d["z_scores"])
        # A row with no finite z in any lookback has no reading: NaN, never a
        # fabricated neutral 0.
        row_ok = np.vstack([np.isfinite(z) for z in zs]).any(axis=0) if zs else np.zeros(n, dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_z = np.nanmean(np.vstack(zs), axis=0) if zs else np.full(n, np.nan)

        def _m(a):
            return np.where(row_ok, a, np.nan)

        self.ts_data["OversoldBreadth"] = _m((os_ + eos) / num_lb * 100)
        self.ts_data["OverboughtBreadth"] = _m((ob + eob) / num_lb * 100)
        self.ts_data["ExtremeOversold"] = _m(eos / num_lb * 100)
        self.ts_data["ExtremeOverbought"] = _m(eob / num_lb * 100)
        self.ts_data["BuySignalBreadth"] = bc
        self.ts_data["SellSignalBreadth"] = sc
        self.ts_data["AvgZ"] = _m(avg_z)
        raw = (ob - os_) / num_lb * 100 + (eob - eos) / num_lb * 100 * 1.5
        self.ts_data["ConvictionRaw"] = _m(raw)

    def _compute_conviction(self) -> None:
        n = len(self.ts_data)
        if self.forward_signal:
            # FORWARD: conviction is the causally z-scored forecast itself
            # (NEGATIVE = bullish), fixed tiers, no fabricated band.
            fz = self.ts_data["ForecastZ"].to_numpy(dtype=np.float64)
            conv = -100.0 * np.tanh(fz / 2.0)
            self.ts_data["ConvictionScore"] = conv
            self.ts_data["ConvictionBounded"] = conv
            self.ts_data["ConvictionUpper"] = np.nan
            self.ts_data["ConvictionLower"] = np.nan
            self.conviction_levels = {"weak": float(CONV2_TIER_WEAK), "moderate": float(CONV2_TIER_MODERATE),
                                      "strong": float(CONV2_TIER_STRONG)}
            weak = np.full(n, float(CONV2_TIER_WEAK))
            moderate = np.full(n, float(CONV2_TIER_MODERATE))
            strong = np.full(n, float(CONV2_TIER_STRONG))
            bounded = conv
        else:
            raw = self.ts_data["ConvictionRaw"].to_numpy(dtype=np.float64)
            filtered, _, variances = drift_diffusion_filter(
                raw, leak_rate=DDM_LEAK_RATE, drift_scale=DDM_DRIFT_SCALE, long_run_var=DDM_LONG_RUN_VAR
            )
            show = self._valid & np.isfinite(raw) & np.isfinite(filtered)
            filtered = np.where(show, filtered, np.nan)
            sd = np.sqrt(np.maximum(np.where(show, variances, np.nan), 0))
            bounded = np.asarray(_apply_conviction_bounds(filtered))
            self.ts_data["ConvictionScore"] = filtered
            self.ts_data["ConvictionBounded"] = bounded
            # Heuristic band (magnitude-driven, not a calibrated interval).
            self.ts_data["ConvictionUpper"] = _apply_conviction_bounds(filtered + 1.96 * sd)
            self.ts_data["ConvictionLower"] = _apply_conviction_bounds(filtered - 1.96 * sd)
            weak, moderate, strong = self._compute_causal_conviction_levels(bounded)
            if n:
                self.conviction_levels = {"weak": float(weak[-1]), "moderate": float(moderate[-1]),
                                          "strong": float(strong[-1])}
        self._bands = {"weak": weak, "moderate": moderate, "strong": strong}

        fin = np.isfinite(bounded)
        reg = np.full(n, "N/A", dtype=object)
        reg[fin] = "NEUTRAL"
        reg[fin & (bounded > weak)] = "OVERBOUGHT"
        reg[fin & (bounded > strong)] = "STRONGLY OVERBOUGHT"
        reg[fin & (bounded < -weak)] = "OVERSOLD"
        reg[fin & (bounded < -strong)] = "STRONGLY OVERSOLD"
        self.ts_data["Regime"] = reg.tolist()

    @staticmethod
    def _compute_causal_conviction_levels(bounded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row bands = expanding 50/75/90% quantiles of |conviction| over PRIOR
        rows with a genuine reading (NaN warm-up excluded — A1-05). Config
        CONVICTION_* is the cold-start prior until 50 readings exist."""
        abs_b = pd.Series(np.abs(np.asarray(bounded, dtype=np.float64)))
        ex = abs_b.expanding(min_periods=50)
        weak = ex.quantile(0.50).shift(1).fillna(float(CONVICTION_WEAK)).clip(lower=5.0).to_numpy()
        moderate = ex.quantile(0.75).shift(1).fillna(float(CONVICTION_MODERATE)).to_numpy()
        strong = ex.quantile(0.90).shift(1).fillna(float(CONVICTION_STRONG)).to_numpy()
        moderate = np.maximum(moderate, weak + 1.0)
        strong = np.maximum(strong, moderate + 1.0)
        return weak, moderate, strong

    def _compute_divergences(self) -> None:
        n = len(self.ts_data)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        order = 5
        if n >= order * 3:
            price = self.price if self.price is not None else np.asarray(self.y)
            residual = np.asarray(self.residuals)
            # Fixed min_periods (prefix-invariant); expanding std through row i.
            ex_std = pd.Series(residual).expanding(min_periods=20).std().ffill().fillna(0.0).to_numpy()
            last_low = last_high = -1
            for i in range(order * 2, n):
                wp = price[i - 2 * order: i + 1]
                if not np.isfinite(wp).all():
                    continue
                if np.argmin(wp) == order:
                    cl = i - order
                    if last_low != -1 and price[cl] < price[last_low] and residual[cl] > residual[last_low] \
                            and residual[cl] < -ex_std[cl] * 0.5:
                        bull_div[i] = True
                    last_low = cl
                if np.argmax(wp) == order:
                    ch = i - order
                    if last_high != -1 and price[ch] > price[last_high] and residual[ch] < residual[last_high] \
                            and residual[ch] > ex_std[ch] * 0.5:
                        bear_div[i] = True
                    last_high = ch
        self.ts_data["BullishDiv"] = bull_div
        self.ts_data["BearishDiv"] = bear_div

    def _compute_forward_changes(self) -> None:
        """FORWARD-LOOKING by design: % change of the target level over the next
        5/10/20 rows (evaluation only, never an input). Uses ``price`` when given."""
        n = len(self.ts_data)
        y_arr = self.price if self.price is not None else np.asarray(self.y, dtype=np.float64)
        for period in _PERF_PERIODS:
            fwd = np.full(n, np.nan)
            if n > period:
                cur, nxt = y_arr[:-period], y_arr[period:]
                ok = np.abs(cur) > 1e-10
                seg = np.full(n - period, np.nan)
                seg[ok] = (nxt[ok] - cur[ok]) / np.abs(cur[ok]) * 100
                fwd[:-period] = seg
            self.ts_data[f"FwdChg_{period}"] = np.clip(fwd, -100, 100)

    def _compute_ou_diagnostics(self) -> None:
        """Full-sample OU fit of the residual over rows with a genuine forecast."""
        r = np.asarray(self.residuals, dtype=np.float64)
        oos_r = r[self._valid & np.isfinite(r)] if len(r) else r
        nan = float("nan")
        self.theta_history = []
        if len(oos_r) > 30:
            theta, mu, sigma = ornstein_uhlenbeck_estimate(oos_r)
            adf_pvalue = kpss_pvalue = nan
            if _HAS_STATSMODELS:
                try:
                    try:   # statsmodels ≥ 0.15: keep the tuple return explicitly
                        _adf = adfuller(oos_r, autolag="AIC", result_object=False)
                    except TypeError:   # older statsmodels has no result_object kwarg
                        _adf = adfuller(oos_r, autolag="AIC")
                    adf_pvalue = float(_adf[1])
                except Exception:
                    pass
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        kpss_pvalue = float(kpss(oos_r, regression="c", nlags="auto")[1])
                except Exception:
                    pass
            window_size = min(60, len(oos_r) // 3)
            for i in range(window_size, len(oos_r)):
                self.theta_history.append(ornstein_uhlenbeck_estimate(oos_r[i - window_size: i])[0])
            theta_std = float(np.std(self.theta_history)) if len(self.theta_history) > 1 else 0.0
            dynamic_theta = self.theta_history[-1] if self.theta_history else theta
            # Unstable rolling θ → robust median, or the base θ.
            if theta_std > theta and len(self.theta_history) >= 5:
                dynamic_theta = float(np.median(self.theta_history))
                if abs(dynamic_theta - theta) > 3.0 * max(theta, 1e-4):
                    dynamic_theta = theta
            current_r = float(oos_r[-1]) if np.isfinite(r[-1]) else nan
        else:
            theta = dynamic_theta = sigma = mu = adf_pvalue = kpss_pvalue = nan
            theta_std = 0.0
            current_r = nan

        def _hl(th):
            return float(np.log(2) / max(th, 1e-4)) if np.isfinite(th) else nan

        self.ou_params = {
            "theta": theta, "dynamic_theta": dynamic_theta, "mu": mu, "sigma": sigma,
            "half_life_base": _hl(theta), "half_life": _hl(dynamic_theta),
            "stationary_std": float(sigma / np.sqrt(2 * max(theta, 1e-4))) if np.isfinite(theta) else nan,
            "adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue, "theta_std": theta_std,
        }
        if np.isfinite(current_r) and np.isfinite(dynamic_theta):
            d = np.arange(1, OU_PROJECTION_DAYS + 1)
            self.ou_projection = mu + (current_r - mu) * np.exp(-dynamic_theta * d)
            if theta_std > 0:
                self.ou_projection_upper = mu + (current_r - mu) * np.exp(-(dynamic_theta - theta_std) * d)
                self.ou_projection_lower = mu + (current_r - mu) * np.exp(-(dynamic_theta + theta_std) * d)
            else:
                self.ou_projection_upper = self.ou_projection.copy()
                self.ou_projection_lower = self.ou_projection.copy()
        else:
            self.ou_projection = self.ou_projection_upper = self.ou_projection_lower = np.array([])

    def _compute_hurst(self) -> None:
        """DFA Hurst of the residual's INCREMENTS over valid rows (H<0.5 = the
        residual mean-reverts); NaN when not computable (A1-06)."""
        r = np.where(self._valid, self.residuals, np.nan) if len(self.residuals) else self.residuals
        d = np.diff(r) if len(r) > 1 else np.array([])
        self.hurst = float(hurst_dfa(d[np.isfinite(d)]))

    def _compute_signal_edge(self) -> None:
        """FORWARD: deferred to the conclusion layer. LEVELS: Holm across the 6
        drift-adjusted tests of the displayed signal (get_signal_performance)."""
        if self.forward_signal:
            self.signal_edge = {"has_forward_edge": False, "inverted": False, "best_horizon": None,
                                "edge_source": "conclusion"}
            return
        perf = self.get_signal_performance()
        has_edge = inverted = False
        best_h, best_p = None, 1.0
        for period in _PERF_PERIODS:
            r = perf[period]
            for side in ("buy", "sell"):
                ph = r[f"{side}_p_holm"]
                if r[f"{side}_count"] < _EDGE_MIN_EVENTS or ph >= EDGE_ALPHA:
                    continue
                if r[f"{side}_excess"] > 0:
                    has_edge = True
                    if ph < best_p:
                        best_h, best_p = period, ph
                elif r[f"{side}_excess"] < 0:
                    inverted = True
        self.signal_edge = {"has_forward_edge": bool(has_edge), "inverted": bool(inverted),
                            "best_horizon": best_h, "edge_source": "engine_levels_test",
                            "alpha": float(EDGE_ALPHA)}
