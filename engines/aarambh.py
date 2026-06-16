"""
Nishkarsh v1.4.0 — FairValueEngine: Walk-forward ensemble regression for Nifty 50 PE fair value.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

AARAMBH — Walk-forward ensemble regression on Nifty 50 PE ratio with conformal z-scores and DDM filtering.

Imports math primitives from analytics.* instead of inline definitions.
No Streamlit dependency.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from core.config import (
    LOOKBACK_WINDOWS,
    MIN_TRAIN_SIZE,
    MAX_TRAIN_SIZE,
    REFIT_INTERVAL,
    RIDGE_ALPHAS,
    HUBER_EPSILON,
    HUBER_MAX_ITER,
    OU_PROJECTION_DAYS,
    CONVICTION_STRONG,
    CONVICTION_MODERATE,
    CONVICTION_WEAK,
    DDM_LEAK_RATE,
    DDM_DRIFT_SCALE,
    DDM_LONG_RUN_VAR,
)

# Optional dependencies
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    _HAS_STATSMODELS = True
except ImportError:
    sm = None
    _HAS_STATSMODELS = False

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Math imports from package
from analytics.ou_process import ornstein_uhlenbeck_estimate, andrews_median_unbiased_ar1
from analytics.ddm_filter import drift_diffusion_filter
from analytics.hurst import hurst_dfa
from analytics.structural_breaks import detect_structural_breaks
from analytics.conformal import compute_conformal_zscores
from analytics.utils import (
    _classify_zones,
    _detect_crossover_signals,
    _compute_significance,
    _apply_conviction_bounds,
)


class FairValueEngine:
    """Walk-forward fair value engine with multi-lookback breadth analytics.

    Pipeline:
        1. Expanding-window ensemble regression (Ridge + Huber + ElasticNet + WLS)
        2. Multi-lookback conformal z-score computation and zone classification
        3. Breadth aggregation and raw conviction scoring
        4. Drift-Diffusion filtering of conviction with mean-reverting variance
        5. OU estimation with Andrews MU correction for half-life and projection
        6. Hurst exponent via DFA for mean-reversion validation
        7. Swing-based divergence detection
        8. Forward change analysis with significance testing
        9. Structural break detection for regime-aware resetting
    """

    def __init__(self) -> None:
        self.ts_data: pd.DataFrame = pd.DataFrame()
        self.lookback_data: dict = {}
        self.model_stats: dict = {}
        self.ou_params: dict = {}
        self.ou_projection: np.ndarray = np.array([])
        self.ou_projection_upper: np.ndarray = np.array([])
        self.ou_projection_lower: np.ndarray = np.array([])
        self.pivots: dict = {}
        self.residual_stats: dict = {}
        self.hurst: float = 0.5
        # Conviction bands — empirical, recomputed per fit from the realized
        # conviction distribution (market-led). Config constants are the
        # cold-start prior used until enough history exists.
        self.conviction_levels: dict = {
            "weak": float(CONVICTION_WEAK),
            "moderate": float(CONVICTION_MODERATE),
            "strong": float(CONVICTION_STRONG),
        }
        self.latest_feature_impacts: dict = {}
        # Ground-truth forward-edge assessment (set in fit). Empty until then.
        self.signal_edge: dict = {}
        self.feature_impact_history: list[dict] = []
        self.theta_history: list[float] = []
        self.break_dates: list[int] = []
        self.feature_names: list[str] = []
        self.n_samples: int = 0
        self.forward_signal: bool = False
        self.y: np.ndarray = np.array([])
        self.predictions: np.ndarray = np.array([])
        self.model_spread: np.ndarray = np.array([])
        self.residuals: np.ndarray = np.array([])

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        progress_callback: Callable | None = None,
        forward_signal: bool = False,
    ) -> "FairValueEngine":
        """Run the full walk-forward pipeline.

        forward_signal: when True (PREDICTIVE mode), ``y`` is a FORWARD return
            (target change t→t+h) forecast from lagged features, so the tradeable
            signal is the model's PREDICTION (expected forward return), not the
            fair-value residual. The conviction stack is driven by ``-prediction``
            so a positive expected forward return maps to the engine's bullish
            (oversold) pole. R²/R²-vs-RW still measure forecast skill (prediction
            vs realized forward return).
        """
        start_time = time.time()

        self.feature_names = feature_names or [f"X{i}" for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()
        self.forward_signal = forward_signal

        # Detect structural breaks
        self.break_dates = detect_structural_breaks(y)

        # Walk-forward regression
        self._walk_forward_regression(X, y, progress_callback)

        # Per-period residual = realized − predicted; model stats are computed on
        # the raw target (returns in forward mode) so R²/R²-vs-RW stay honest.
        self.residuals = self.y - self.predictions
        self._compute_model_stats()
        if forward_signal:
            # Predictive mode: the signal IS the forecast. Drive the conviction
            # stack with -prediction so a positive expected forward return maps
            # to the bullish (oversold) pole — instead of the fair-value residual.
            self.residuals = -np.nan_to_num(self.predictions, nan=0.0)
        self._compute_multi_lookback_signals()
        self._compute_breadth_metrics()
        self._compute_ddm_conviction()
        self._find_pivots()
        self._compute_divergences()
        self._compute_forward_changes()
        self._compute_ou_diagnostics()
        self._compute_hurst()
        self._compute_signal_edge()

        if progress_callback:
            progress_callback(1.0, "Done")

        return self

    def _compute_signal_edge(self) -> None:
        """Ground-truth forward-edge test from realized signal performance.

        Independent of R²/Hurst (which only describe the *level*, not whether
        the signal works). Asks the only question that matters: when the engine
        said BUY (undervalued → expect PE to rise), did PE actually rise — with
        statistical significance? And symmetrically for SELL. If neither
        direction shows significant *correct* forward movement, there is no
        tradeable edge; if a direction shows significant *wrong* movement, the
        signal is inverted (anti-predictive). Sets ``self.signal_edge``.
        """
        perf = self.get_signal_performance()
        has_edge = False
        inverted = False
        best_horizon: int | None = None
        for p in (5, 10, 20):
            r = perf.get(p, {})
            bc, sc = r.get("buy_count", 0), r.get("sell_count", 0)
            # BUY should precede a rise (buy_avg > 0); SELL is stored as -fwd,
            # so a correct SELL also has sell_avg > 0.
            if bc >= 20 and r.get("buy_avg", 0) > 0 and r.get("buy_p_value", 1) < 0.10:
                has_edge, best_horizon = True, p
            if sc >= 20 and r.get("sell_avg", 0) > 0 and r.get("sell_p_value", 1) < 0.10:
                has_edge, best_horizon = True, p
            if bc >= 20 and r.get("buy_avg", 0) < 0 and r.get("buy_p_value", 1) < 0.05:
                inverted = True
            if sc >= 20 and r.get("sell_avg", 0) < 0 and r.get("sell_p_value", 1) < 0.05:
                inverted = True
        self.signal_edge = {
            "has_forward_edge": bool(has_edge),
            "inverted": bool(inverted),
            "best_horizon": best_horizon,
        }

    def get_current_signal(self) -> dict:
        """Derive the current composite signal from the latest observation."""
        if self.ts_data.empty:
            return {
                "signal": "HOLD", "strength": "NEUTRAL", "confidence": "N/A",
                "conviction_score": 0, "conviction_upper": 0, "conviction_lower": 0,
                "regime": "NEUTRAL", "oversold_breadth": 0, "overbought_breadth": 0,
                "residual": 0, "fair_value": 0, "actual": 0, "avg_z": 0,
                "model_spread": 0, "has_bullish_div": False, "has_bearish_div": False,
                "ou_half_life": 0, "adf_pvalue": 1.0, "kpss_pvalue": 0.0, "hurst": 0.5,
                "theta_stable": True, "break_detected": False,
                "conviction_levels": dict(self.conviction_levels),
                "tradeable": False, "edge_assessment": "N/A — no data",
                "forecast_skill_vs_rw": 0.0, "mean_reverting": False,
                "random_walk_regime": True, "half_life_meaningful": False,
                "display_strength": "NEUTRAL",
                "has_forward_edge": False, "inverted": False,
            }

        current = self.ts_data.iloc[-1]
        conviction_bounded = current["ConvictionBounded"]

        lvl_strong = self.conviction_levels["strong"]
        lvl_moderate = self.conviction_levels["moderate"]
        lvl_weak = self.conviction_levels["weak"]

        if conviction_bounded < -lvl_strong:
            signal, strength = "BUY", "STRONG"
        elif conviction_bounded < -lvl_moderate:
            signal, strength = "BUY", "MODERATE"
        elif conviction_bounded < -lvl_weak:
            signal, strength = "BUY", "WEAK"
        elif conviction_bounded > lvl_strong:
            signal, strength = "SELL", "STRONG"
        elif conviction_bounded > lvl_moderate:
            signal, strength = "SELL", "MODERATE"
        elif conviction_bounded > lvl_weak:
            signal, strength = "SELL", "WEAK"
        else:
            signal, strength = "HOLD", "NEUTRAL"

        oversold_breadth = current["OversoldBreadth"]
        overbought_breadth = current["OverboughtBreadth"]

        if signal == "BUY":
            confidence = "HIGH" if oversold_breadth >= 80 else "MEDIUM" if oversold_breadth >= 60 else "LOW"
        elif signal == "SELL":
            confidence = "HIGH" if overbought_breadth >= 80 else "MEDIUM" if overbought_breadth >= 60 else "LOW"
        else:
            conviction_abs = abs(conviction_bounded)
            confidence = "HIGH" if conviction_abs < lvl_weak * 0.5 else "MEDIUM" if conviction_abs < lvl_weak else "LOW"

        theta_stable = True
        if len(self.theta_history) >= 10:
            theta_cv = np.std(self.theta_history[-10:]) / max(np.mean(self.theta_history[-10:]), 1e-6)
            theta_stable = theta_cv < 0.5

        # ── Forecast-edge assessment ──────────────────────────────────────
        # The verdict is driven by REALIZED FORWARD PERFORMANCE (does a BUY
        # actually precede a rise, significantly?), not by R²/Hurst — those
        # describe the level, not whether the signal works, and a Hurst cutoff
        # is brittle at the boundary (0.552 vs 0.55). R²-vs-RW and the
        # random-walk regime are kept as supporting context only.
        r2_vs_rw = float(self.model_stats.get("r2_vs_rw", 0.0))
        adf_p = float(self.ou_params.get("adf_pvalue", 1.0))
        mean_reverting = (adf_p < 0.05) and (self.hurst < 0.5)
        random_walk_regime = (0.45 <= self.hurst <= 0.58) or (adf_p > 0.05)

        edge = self.signal_edge or {}
        has_forward_edge = bool(edge.get("has_forward_edge", False))
        inverted = bool(edge.get("inverted", False))
        best_h = edge.get("best_horizon")

        tradeable = has_forward_edge and not inverted
        if inverted:
            edge_assessment = (
                "INVERTED — signal historically anti-predictive "
                "(BUY/SELL preceded the wrong direction, p<0.05)"
            )
        elif has_forward_edge:
            edge_assessment = f"EDGE — signal predicted the correct direction at {best_h}d (p<0.10)"
        elif random_walk_regime:
            edge_assessment = "NO EDGE — random-walk regime; no significant forward predictive power"
        else:
            edge_assessment = "NO EDGE — no significant forward predictive power on history"

        # OU half-life is meaningful only if residuals actually mean-revert.
        half_life_meaningful = bool(mean_reverting)

        # On a no-edge / inverted series, never show a confident directional
        # call — collapse the displayed strength. In PREDICTIVE mode the
        # conviction IS the directional forecast, so show its strength rather
        # than the level-test "NO EDGE" label (the directional verdict lives in
        # the convergence IC, not this magnitude flag).
        if self.forward_signal:
            display_strength = strength
        else:
            display_strength = strength if tradeable else ("INVERTED" if inverted else "NO EDGE")

        return {
            "signal": signal,
            "strength": strength,
            "confidence": confidence,
            "conviction_score": conviction_bounded,
            "conviction_upper": current["ConvictionUpper"],
            "conviction_lower": current["ConvictionLower"],
            "regime": current["Regime"],
            "oversold_breadth": oversold_breadth,
            "overbought_breadth": overbought_breadth,
            "residual": current["Residual"],
            "fair_value": current["FairValue"],
            "actual": current["Actual"],
            "avg_z": current["AvgZ"],
            "model_spread": current["ModelSpread"],
            "has_bullish_div": current["BullishDiv"],
            "has_bearish_div": current["BearishDiv"],
            "ou_half_life": self.ou_params.get("half_life", 0),
            "adf_pvalue": self.ou_params.get("adf_pvalue", 1.0),
            "kpss_pvalue": self.ou_params.get("kpss_pvalue", 0.0),
            "hurst": self.hurst,
            "theta_stable": theta_stable,
            "break_detected": len(self.break_dates) > 0,
            "conviction_levels": dict(self.conviction_levels),
            "tradeable": tradeable,
            "edge_assessment": edge_assessment,
            "forecast_skill_vs_rw": r2_vs_rw,
            "mean_reverting": bool(mean_reverting),
            "random_walk_regime": bool(random_walk_regime),
            "half_life_meaningful": half_life_meaningful,
            "display_strength": display_strength,
            "has_forward_edge": has_forward_edge,
            "inverted": inverted,
            # In predictive mode, tradeable/edge_assessment/inverted reflect the
            # LEVEL/magnitude forward-edge test, which is NOT the verdict — the
            # tradeable edge is DIRECTIONAL (graded by the convergence IC). UI/log
            # consumers branch on this flag to avoid the "valuation context"
            # framing, which only applies to fair-value (levels) mode.
            "forward_signal": bool(self.forward_signal),
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_regime_stats(self) -> dict:
        ts = self.ts_data
        regime_counts = ts["Regime"].value_counts()
        return {
            "strongly_oversold": regime_counts.get("STRONGLY OVERSOLD", 0),
            "oversold": regime_counts.get("OVERSOLD", 0),
            "neutral": regime_counts.get("NEUTRAL", 0),
            "overbought": regime_counts.get("OVERBOUGHT", 0),
            "strongly_overbought": regime_counts.get("STRONGLY OVERBOUGHT", 0),
            "current_regime": ts["Regime"].iloc[-1],
        }

    def get_signal_performance(self) -> dict:
        """Forward change analysis with significance testing."""
        ts = self.ts_data
        results = {}
        burn_in = max(MIN_TRAIN_SIZE + 50, 80)

        for period in (5, 10, 20):
            buy_changes: list[float] = []
            sell_changes: list[float] = []
            # Enforce NON-OVERLAPPING forward windows: consecutive included
            # signals must be ≥ `period` bars apart. Overlapping h-day forward
            # returns sampled at nearby signal dates are heavily autocorrelated,
            # which violates the t-test's independence assumption and inflates
            # significance by ~√h (a spurious t≈4.8 at 20d collapses to ≈1.1
            # once the ~h-fold redundancy is removed).
            last_buy = -(10 ** 9)
            last_sell = -(10 ** 9)

            for i in range(burn_in, len(ts) - period):
                score = ts["ConvictionScore"].iloc[i]
                fwd = ts.get(f"FwdChg_{period}")
                if fwd is None:
                    continue
                fwd_val = fwd.iloc[i]
                if pd.isna(fwd_val):
                    continue
                if score < -CONVICTION_MODERATE and (i - last_buy) >= period:
                    buy_changes.append(fwd_val)
                    last_buy = i
                if score > CONVICTION_MODERATE and (i - last_sell) >= period:
                    sell_changes.append(-fwd_val)
                    last_sell = i

            buy_stats = _compute_significance(buy_changes)
            sell_stats = _compute_significance(sell_changes)

            results[period] = {
                "buy_avg": buy_stats["mean"],
                "buy_hit": float(np.mean([c > 0 for c in buy_changes])) if buy_changes else 0.0,
                "buy_count": len(buy_changes),
                "buy_t_stat": buy_stats["t_stat"],
                "buy_p_value": buy_stats["p_value"],
                "sell_avg": sell_stats["mean"],
                "sell_hit": float(np.mean([c > 0 for c in sell_changes])) if sell_changes else 0.0,
                "sell_count": len(sell_changes),
                "sell_t_stat": sell_stats["t_stat"],
                "sell_p_value": sell_stats["p_value"],
            }

        return results

    def get_feature_impact_history(self) -> pd.DataFrame:
        if not self.feature_impact_history:
            return pd.DataFrame()
        return pd.DataFrame(self.feature_impact_history)

    # ── Private: Walk-Forward Regression ──────────────────────────────────

    def _walk_forward_regression(
        self, X: np.ndarray, y: np.ndarray, progress_callback,
    ) -> None:
        n = self.n_samples
        self.predictions = np.full(n, np.nan)
        self.model_spread = np.zeros(n)
        # Track how often a chunk had to fall back to the train-mean because no
        # ensemble member fit — surfaced as model_coverage in model_stats.
        self._wf_chunks_total = 0
        self._wf_chunks_fallback = 0

        for t in range(MIN_TRAIN_SIZE):
            self.predictions[t] = float(np.mean(y[:t])) if t > 0 else float(y[0])
            self.model_spread[t] = 0.0

        decay_rate = np.log(2) / 252.0
        global_weights = np.exp(-decay_rate * np.arange(MAX_TRAIN_SIZE - 1, -1, -1))
        dynamic_refit = int(np.clip(n // 200, 2, 5))

        last_models: dict = {"ridge": None, "huber": None, "ols": None, "elasticnet": None, "pca_wls": None}
        valid_cols = np.ones(X.shape[1], dtype=bool)

        total_chunks = (n - MIN_TRAIN_SIZE) // dynamic_refit + 1
        for i, t_start in enumerate(range(MIN_TRAIN_SIZE, n, dynamic_refit)):
            t_end = min(t_start + dynamic_refit, n)
            try:
                result = self._process_wf_chunk(t_start, t_end, X, y, global_weights)
                t_start_res, t_end_res, preds, spreads, models, v_cols = result
                self.predictions[t_start_res:t_end_res] = preds
                self.model_spread[t_start_res:t_end_res] = spreads
                last_models, valid_cols = models, v_cols
            except Exception as e:
                logging.warning("Walk-forward chunk failed [%d:%d]: %s", t_start, t_end, e)

            if progress_callback:
                progress_callback((i + 1) / total_chunks, f"Walking Forward... ({t_end}/{n})")

        self._compute_feature_impacts(last_models, valid_cols, n - 1)

    def _process_wf_chunk(
        self, t_start: int, t_end: int, X: np.ndarray, y: np.ndarray,
        global_weights: np.ndarray,
    ) -> tuple[int, int, np.ndarray, np.ndarray, dict, np.ndarray]:
        # Detect breaks causally on data available up to this chunk only.
        # Using the globally-detected self.break_dates would leak the future:
        # whether (and where) a break exists before t_start was decided using
        # data after t_start, which shifts the historical training window and
        # makes the backtest non-reproducible under truncation.
        valid_breaks = detect_structural_breaks(y[:t_start])
        last_break = valid_breaks[-1] if valid_breaks else 0
        max_lookback = max(0, t_start - MAX_TRAIN_SIZE)

        if last_break > max_lookback:
            start_idx = last_break
            if t_start - start_idx < MIN_TRAIN_SIZE:
                start_idx = max(0, t_start - MIN_TRAIN_SIZE)
        else:
            start_idx = max_lookback

        models, scaler, valid_cols = self._fit_ensemble(
            X[start_idx:t_start], y[start_idx:t_start], t_start, global_weights
        )

        self._wf_chunks_total += 1
        X_chunk = X[t_start:t_end]
        if len(X_chunk) == 0:
            return t_start, t_end, np.array([]), np.array([]), models, valid_cols

        val_size = min(30, max(5, int((t_start - start_idx) * 0.2)))
        X_val = X[t_start - val_size : t_start] if t_start > val_size else None
        y_val = y[t_start - val_size : t_start] if t_start > val_size else None

        preds_matrix, weights = self._predict_ensemble(
            X_chunk, models, scaler, valid_cols, t_start, X_val, y_val
        )

        if preds_matrix:
            preds_stacked = np.vstack(preds_matrix)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if len(preds_matrix) > 1 and len(weights) == len(preds_matrix) and sum(weights) > 0:
                    w = np.array(weights) / sum(weights)
                    preds = np.average(preds_stacked, axis=0, weights=w)
                else:
                    preds = np.nanmean(preds_stacked, axis=0)
                spreads = np.maximum(np.nanstd(preds_stacked, axis=0), 1e-6) if len(preds_matrix) > 1 else np.full(len(preds), 1e-6)
                nans = np.isnan(preds)
                if np.any(nans):
                    preds[nans] = float(np.mean(y[start_idx:t_start]))
                    spreads[nans] = 1e-6
        else:
            self._wf_chunks_fallback += 1
            fallback = float(np.mean(y[start_idx:t_start]))
            preds = np.full(t_end - t_start, fallback)
            spreads = np.full(t_end - t_start, 1e-6)

        return t_start, t_end, preds, spreads, models, valid_cols

    @staticmethod
    def _fit_ensemble(
        X_train: np.ndarray, y_train: np.ndarray, t: int, global_weights: np.ndarray,
    ) -> tuple[dict, any | None, np.ndarray]:
        models: dict = {"ridge": None, "huber": None, "ols": None, "elasticnet": None, "pca_wls": None}
        scaler = None
        valid_cols = np.std(X_train, axis=0) > 1e-8
        if not np.any(valid_cols):
            valid_cols = np.ones(X_train.shape[1], dtype=bool)

        X_train_clean = X_train[:, valid_cols]
        n_samples = len(y_train)
        weights = global_weights[-n_samples:]

        if _HAS_SKLEARN:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train_clean)

            try:
                ridge = RidgeCV(alphas=list(RIDGE_ALPHAS), cv=None)
                ridge.fit(X_scaled, y_train, sample_weight=weights)
                models["ridge"] = ridge
            except Exception as e:
                logging.warning("Ridge fit failed at t=%d: %s", t, e)

            try:
                huber = HuberRegressor(epsilon=HUBER_EPSILON, max_iter=HUBER_MAX_ITER, tol=1e-3)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    huber.fit(X_scaled, y_train, sample_weight=weights)
                models["huber"] = huber
            except Exception as e:
                logging.warning("Huber fit failed at t=%d: %s", t, e)

            try:
                # random_state pins the "random" coordinate-descent selection and
                # the CV folds — without it the engine is non-deterministic and the
                # signal flickers run-to-run on identical data.
                enet = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], n_alphas=10, cv=2, max_iter=2000, tol=1e-2, selection="random", random_state=42, n_jobs=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    enet.fit(X_scaled, y_train, sample_weight=weights)
                models["elasticnet"] = enet
            except Exception as e:
                logging.warning("ElasticNet fit failed at t=%d: %s", t, e)

            try:
                pca = PCA(n_components=0.95, svd_solver="full")
                X_pca = pca.fit_transform(X_scaled)
                models["pca_wls"] = pca
                ols = LinearRegression()
                ols.fit(X_pca, y_train, sample_weight=weights)
                models["ols"] = ols
            except Exception as e:
                logging.warning("PCA/OLS fit failed at t=%d: %s", t, e)

        return models, scaler, valid_cols

    def _predict_ensemble(
        self, X_pred: np.ndarray, models: dict, scaler: any | None,
        valid_cols: np.ndarray, t_start: int,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Predict the forward chunk with each ensemble member and weight by
        honest out-of-sample accuracy.

        Each member's weight is the inverse MAE of *that member's* predictions
        on the held-out validation rows ``(X_val, y_val)`` — i.e. the same
        model is asked to predict the validation window, and members that fit
        the recent past better get more weight. (The previous implementation
        compared ``y_val`` against the forward-chunk predictions, an unrelated
        time slice, making every weight meaningless.)
        """
        preds_list: list[np.ndarray] = []
        weights: list[float] = []

        if not (_HAS_SKLEARN and scaler is not None):
            return preds_list, weights

        X_pred_clean = X_pred[:, valid_cols]
        try:
            X_pred_scaled = scaler.transform(X_pred_clean)
        except Exception:
            return preds_list, weights

        # Scale the validation window once for honest per-member weighting.
        X_val_scaled: np.ndarray | None = None
        if X_val is not None and len(X_val) > 0 and y_val is not None and len(y_val) > 0:
            try:
                X_val_scaled = scaler.transform(X_val[:, valid_cols])
            except Exception:
                X_val_scaled = None

        def _member_predict(key: str, X_scaled: np.ndarray) -> np.ndarray | None:
            m = models.get(key)
            if m is None:
                return None
            try:
                if key == "ols":  # PCA → OLS pipeline
                    pca = models.get("pca_wls")
                    if pca is None:
                        return None
                    return m.predict(pca.transform(X_scaled))
                return m.predict(X_scaled)
            except Exception:
                return None

        for key in ("ridge", "huber", "elasticnet", "ols"):
            chunk_pred = _member_predict(key, X_pred_scaled)
            if chunk_pred is None:
                continue
            chunk_clean = np.where(
                np.isfinite(chunk_pred) & (np.abs(chunk_pred) < 1e10), chunk_pred, np.nan
            )
            if np.all(np.isnan(chunk_clean)):
                continue
            preds_list.append(chunk_clean)

            # Inverse-MAE weight from the SAME model on the validation window.
            weight = 1.0
            if X_val_scaled is not None:
                val_pred = _member_predict(key, X_val_scaled)
                if val_pred is not None and len(val_pred) == len(y_val) and np.all(np.isfinite(val_pred)):
                    mae = float(mean_absolute_error(y_val, val_pred))
                    weight = max(1.0 / max(mae, 1e-6), 0.01)
                else:
                    weight = 0.05
            weights.append(weight)

        return preds_list, weights

    def _compute_feature_impacts(self, models: dict, valid_cols: np.ndarray, t_index: int) -> None:
        features = np.array(self.feature_names)[valid_cols]
        wls = models.get("ols")
        pca = models.get("pca_wls")

        if wls is not None and pca is not None:
            try:
                wls_weights = wls.coef_
                feature_weights = np.dot(wls_weights, pca.components_)
                abs_weights = np.abs(feature_weights)
                total_impact = np.sum(abs_weights)
                if total_impact > 1e-10:
                    pct_impacts = (abs_weights / total_impact) * 100
                    impacts = {f: float(imp) for f, imp in zip(features, pct_impacts)}
                    self.latest_feature_impacts = dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))
                    self.feature_impact_history.append({"index": t_index, **impacts})
                    return
            except Exception:
                pass
        self.latest_feature_impacts = {}

    # ── Private: Analytics Pipeline ───────────────────────────────────────

    def _compute_model_stats(self) -> None:
        oos_mask = np.arange(self.n_samples) >= MIN_TRAIN_SIZE
        y_oos = self.y[oos_mask]
        pred_oos = self.predictions[oos_mask]
        valid = np.isfinite(pred_oos)
        y_v, p_v = y_oos[valid], pred_oos[valid]

        if len(y_v) > 2 and _HAS_SKLEARN:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            r2 = r2_score(y_v, p_v)
            rmse = float(np.sqrt(mean_squared_error(y_v, p_v)))
            mae = float(mean_absolute_error(y_v, p_v))
        else:
            ss_res = float(np.sum((y_v - p_v) ** 2))
            ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            rmse = float(np.sqrt(np.mean((y_v - p_v) ** 2)))
            mae = float(np.mean(np.abs(y_v - p_v)))

        if len(y_v) > 2:
            rw_forecast = np.empty_like(y_v)
            rw_forecast[0] = y_v[0]
            rw_forecast[1:] = y_v[:-1]
            ss_res = float(np.sum((y_v - p_v) ** 2))
            ss_rw = float(np.sum((y_v - rw_forecast) ** 2))
            r2_vs_rw = 1 - ss_res / max(ss_rw, 1e-10)
        else:
            r2_vs_rw = 0.0

        total_chunks = max(getattr(self, "_wf_chunks_total", 0), 1)
        fallback_chunks = getattr(self, "_wf_chunks_fallback", 0)
        model_coverage = 1.0 - fallback_chunks / total_chunks

        self.model_stats = {
            "r2_oos": r2, "r2_vs_rw": r2_vs_rw, "rmse_oos": rmse,
            "mae_oos": mae, "n_obs": len(y_v), "n_features": len(self.feature_names),
            "avg_model_spread": float(np.mean(self.model_spread[oos_mask])),
            # Fraction of walk-forward chunks where ≥1 ensemble member fit
            # (vs. silently falling back to the train mean). < 1.0 means the
            # "fair value" line is partly a moving average — surface it.
            "model_coverage": float(model_coverage),
            "n_fallback_chunks": int(fallback_chunks),
        }

    def _compute_multi_lookback_signals(self) -> None:
        r = self.residuals
        n = len(r)
        self.lookback_data = {}

        for lb in LOOKBACK_WINDOWS:
            if n < lb:
                continue
            min_periods = max(lb // 2, 5)
            z_scores, lower_bounds, upper_bounds = compute_conformal_zscores(
                r, window=lb, min_periods=min_periods, alpha=0.05
            )
            zones = _classify_zones(z_scores)
            buy_signals, sell_signals = _detect_crossover_signals(z_scores)
            self.lookback_data[lb] = {
                "z_scores": z_scores, "zones": zones,
                "buy_signals": buy_signals, "sell_signals": sell_signals,
                "lower_bounds": lower_bounds, "upper_bounds": upper_bounds,
            }

        self.ts_data = pd.DataFrame({
            "Actual": self.y, "FairValue": self.predictions,
            "Residual": self.residuals, "ModelSpread": self.model_spread,
        })
        for lb, data in self.lookback_data.items():
            self.ts_data[f"Z_{lb}"] = data["z_scores"]
            self.ts_data[f"Zone_{lb}"] = data["zones"]
            self.ts_data[f"Buy_{lb}"] = data["buy_signals"]
            self.ts_data[f"Sell_{lb}"] = data["sell_signals"]

    def _compute_breadth_metrics(self) -> None:
        n = len(self.ts_data)
        valid_lookbacks = [lb for lb in LOOKBACK_WINDOWS if f"Z_{lb}" in self.ts_data.columns]
        num_lb = max(len(valid_lookbacks), 1)

        oversold = np.zeros(n)
        overbought = np.zeros(n)
        extreme_os = np.zeros(n)
        extreme_ob = np.zeros(n)
        buy_count = np.zeros(n)
        sell_count = np.zeros(n)
        z_scores_list = []

        for lb in valid_lookbacks:
            zones = self.ts_data[f"Zone_{lb}"].values
            z = self.ts_data[f"Z_{lb}"].values
            extreme_os += (zones == "Extreme Under")
            oversold += (zones == "Undervalued")
            extreme_ob += (zones == "Extreme Over")
            overbought += (zones == "Overvalued")
            buy_count += self.ts_data[f"Buy_{lb}"].values
            sell_count += self.ts_data[f"Sell_{lb}"].values
            z_scores_list.append(z)

        avg_z = np.nan_to_num(np.nanmean(np.vstack(z_scores_list), axis=0), nan=0.0) if z_scores_list else np.zeros(n)

        self.ts_data["OversoldBreadth"] = (oversold + extreme_os) / num_lb * 100
        self.ts_data["OverboughtBreadth"] = (overbought + extreme_ob) / num_lb * 100
        self.ts_data["ExtremeOversold"] = extreme_os / num_lb * 100
        self.ts_data["ExtremeOverbought"] = extreme_ob / num_lb * 100
        self.ts_data["BuySignalBreadth"] = buy_count
        self.ts_data["SellSignalBreadth"] = sell_count
        self.ts_data["AvgZ"] = avg_z
        self.ts_data["ConvictionRaw"] = (
            (overbought - oversold) / num_lb * 100
            + (extreme_ob - extreme_os) / num_lb * 100 * 1.5
        )

    def _compute_ddm_conviction(self) -> None:
        raw = self.ts_data["ConvictionRaw"].values
        filtered, _gains, variances = drift_diffusion_filter(
            raw, leak_rate=DDM_LEAK_RATE, drift_scale=DDM_DRIFT_SCALE, long_run_var=DDM_LONG_RUN_VAR
        )
        ddm_std = np.sqrt(np.maximum(variances, 0))
        self.ts_data["ConvictionScore"] = filtered
        bounded = _apply_conviction_bounds(filtered)
        self.ts_data["ConvictionBounded"] = bounded
        self.ts_data["ConvictionUpper"] = _apply_conviction_bounds(filtered + 1.96 * ddm_std)
        self.ts_data["ConvictionLower"] = _apply_conviction_bounds(filtered - 1.96 * ddm_std)

        # Derive market-led conviction bands from the realized distribution,
        # causally (each bar's band uses only prior history).
        weak_arr, moderate_arr, strong_arr = self._compute_causal_conviction_levels(
            np.asarray(bounded)
        )

        regimes = []
        for i, score_bounded in enumerate(bounded):
            if score_bounded < -strong_arr[i]:
                regimes.append("STRONGLY OVERSOLD")
            elif score_bounded < -weak_arr[i]:
                regimes.append("OVERSOLD")
            elif score_bounded > strong_arr[i]:
                regimes.append("STRONGLY OVERBOUGHT")
            elif score_bounded > weak_arr[i]:
                regimes.append("OVERBOUGHT")
            else:
                regimes.append("NEUTRAL")
        self.ts_data["Regime"] = regimes

        # Latest band (expanding through the final bar) → used by
        # get_current_signal for today's decision. Using all history up to now
        # is causal for the live point estimate.
        if len(weak_arr):
            self.conviction_levels = {
                "weak": float(weak_arr[-1]),
                "moderate": float(moderate_arr[-1]),
                "strong": float(strong_arr[-1]),
            }

    @staticmethod
    def _compute_causal_conviction_levels(
        bounded: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row conviction bands from the EXPANDING |conviction| distribution.

        Replaces the fixed 60/40/20 cutoffs *and* the earlier full-sample
        percentile version (which leaked future information into historical
        labels). Each band at bar ``t`` is an expanding quantile of
        ``|conviction|`` over rows ``< t`` (shifted), so "STRONG" means strong
        relative to this market's own *prior* history and the historical Regime
        track is backtest-safe. Percentile knots (50/75/90) are structural
        choices, not magnitude thresholds. Config CONVICTION_* is the cold-start
        prior until ≥50 observations exist; a small floor and row-wise ordering
        keep the bands well-formed.
        """
        abs_b = pd.Series(np.abs(np.asarray(bounded, dtype=np.float64)))
        weak = abs_b.expanding(min_periods=50).quantile(0.50).shift(1)
        moderate = abs_b.expanding(min_periods=50).quantile(0.75).shift(1)
        strong = abs_b.expanding(min_periods=50).quantile(0.90).shift(1)
        weak_a = weak.fillna(float(CONVICTION_WEAK)).clip(lower=5.0).to_numpy()
        moderate_a = moderate.fillna(float(CONVICTION_MODERATE)).to_numpy()
        strong_a = strong.fillna(float(CONVICTION_STRONG)).to_numpy()
        # Enforce weak < moderate < strong row-wise.
        moderate_a = np.maximum(moderate_a, weak_a + 1.0)
        strong_a = np.maximum(strong_a, moderate_a + 1.0)
        return weak_a, moderate_a, strong_a

    def _compute_divergences(self) -> None:
        n = len(self.ts_data)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        order = 5
        if n < order * 3:
            self.ts_data["BullishDiv"] = bull_div
            self.ts_data["BearishDiv"] = bear_div
            return

        price = np.asarray(self.y)
        residual = np.asarray(self.residuals)
        last_low_idx = -1
        last_high_idx = -1
        expanding_std = pd.Series(residual).expanding(min_periods=min(20, max(2, len(residual) // 3))).std().ffill().fillna(0.0).values

        for i in range(order * 2, n):
            window_price = price[i - 2 * order : i + 1]
            if np.argmin(window_price) == order:
                curr_low = i - order
                if last_low_idx != -1 and price[curr_low] < price[last_low_idx] and residual[curr_low] > residual[last_low_idx]:
                    if residual[curr_low] < -expanding_std[curr_low] * 0.5:
                        bull_div[i] = True
                last_low_idx = curr_low
            if np.argmax(window_price) == order:
                curr_high = i - order
                if last_high_idx != -1 and price[curr_high] > price[last_high_idx] and residual[curr_high] < residual[last_high_idx]:
                    if residual[curr_high] > expanding_std[curr_high] * 0.5:
                        bear_div[i] = True
                last_high_idx = curr_high

        self.ts_data["BullishDiv"] = bull_div
        self.ts_data["BearishDiv"] = bear_div

    def _find_pivots(self, order: int = 5) -> None:
        r = np.asarray(self.residuals)
        n = len(r)
        conf_tops, conf_bottoms, top_vals, bottom_vals = [], [], [], []

        for i in range(order * 2, n):
            window = r[i - 2 * order : i + 1]
            if np.argmax(window) == order:
                conf_tops.append(i)
                top_vals.append(r[i - order])
            if np.argmin(window) == order:
                conf_bottoms.append(i)
                bottom_vals.append(r[i - order])

        fallback_top = float(pd.Series(r).ewm(alpha=0.05).mean().iloc[-1] + pd.Series(r).ewm(alpha=0.05).std().iloc[-1]) if n > 0 else 0.0
        fallback_bottom = float(pd.Series(r).ewm(alpha=0.05).mean().iloc[-1] - pd.Series(r).ewm(alpha=0.05).std().iloc[-1]) if n > 0 else 0.0

        self.pivots = {
            "tops": conf_tops, "bottoms": conf_bottoms,
            "avg_top": float(np.mean(top_vals)) if top_vals else fallback_top,
            "avg_bottom": float(np.mean(bottom_vals)) if bottom_vals else fallback_bottom,
        }
        self.ts_data["IsPivotTop"] = False
        self.ts_data["IsPivotBottom"] = False
        if conf_tops:
            self.ts_data.loc[conf_tops, "IsPivotTop"] = True
        if conf_bottoms:
            self.ts_data.loc[conf_bottoms, "IsPivotBottom"] = True

    def _compute_forward_changes(self) -> None:
        n = len(self.ts_data)
        y_arr = np.asarray(self.y)
        for period in (5, 10, 20):
            fwd = np.full(n, np.nan)
            y_curr = y_arr[:-period]
            y_fwd = y_arr[period:]
            valid = np.abs(y_curr) > 1e-10
            fwd[:-period][valid] = (y_fwd[valid] - y_curr[valid]) / np.abs(y_curr[valid]) * 100
            fwd = np.clip(fwd, -100, 100)
            self.ts_data[f"FwdChg_{period}"] = fwd

    def _compute_ou_diagnostics(self) -> None:
        r = self.residuals
        oos_r = r[MIN_TRAIN_SIZE:]

        if len(oos_r) > 30:
            theta, mu, sigma = ornstein_uhlenbeck_estimate(oos_r)
            try:
                adf_pvalue = float(adfuller(oos_r, autolag="AIC")[1])
            except Exception:
                adf_pvalue = 1.0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_pvalue = float(kpss(oos_r, regression="c", nlags="auto")[1])
            except Exception:
                kpss_pvalue = 0.0

            vol_multiplier = 1.0
            dynamic_theta = theta
            window_size = min(60, len(oos_r) // 3)
            self.theta_history = []
            for i in range(window_size, len(oos_r)):
                theta_roll, _, _ = ornstein_uhlenbeck_estimate(oos_r[i - window_size : i])
                self.theta_history.append(theta_roll)

            theta_std = np.std(self.theta_history) if len(self.theta_history) > 1 else 0.0
            dynamic_theta = self.theta_history[-1] if self.theta_history else theta
            # If the rolling θ is unstable (its dispersion exceeds the stable
            # base θ), the last-window estimate is unreliable — it produced an
            # implausible sub-day half-life (1.7d) in real runs. Fall back to a
            # robust estimate: the median rolling θ, or the base θ.
            if theta_std > theta and len(self.theta_history) >= 5:
                dynamic_theta = float(np.median(self.theta_history))
                if abs(dynamic_theta - theta) > 3.0 * max(theta, 1e-4):
                    dynamic_theta = theta
        else:
            theta, mu, sigma = 0.05, 0.0, max(float(np.std(r)), 1e-6)
            adf_pvalue, kpss_pvalue, vol_multiplier = 1.0, 0.0, 1.0
            dynamic_theta, theta_std = theta, 0.0

        self.ou_params = {
            "theta": theta, "dynamic_theta": dynamic_theta, "mu": mu, "sigma": sigma,
            "half_life_base": np.log(2) / max(theta, 1e-4),
            "half_life": np.log(2) / max(dynamic_theta, 1e-4),
            "stationary_std": sigma / np.sqrt(2 * max(theta, 1e-4)),
            "adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue,
            "vol_multiplier": vol_multiplier, "theta_std": theta_std,
        }

        current_r = float(r[-1])
        proj_days = np.arange(1, OU_PROJECTION_DAYS + 1)
        self.ou_projection = mu + (current_r - mu) * np.exp(-dynamic_theta * proj_days)

        if theta_std > 0:
            self.ou_projection_upper = mu + (current_r - mu) * np.exp(-(dynamic_theta - theta_std) * proj_days)
            self.ou_projection_lower = mu + (current_r - mu) * np.exp(-(dynamic_theta + theta_std) * proj_days)
        else:
            self.ou_projection_upper = self.ou_projection.copy()
            self.ou_projection_lower = self.ou_projection.copy()

    def _compute_hurst(self) -> None:
        oos_r = self.residuals[MIN_TRAIN_SIZE:]
        if len(oos_r) > 30:
            if self.ou_params.get("adf_pvalue", 1.0) > 0.05:
                self.hurst = 0.5
            else:
                self.hurst = hurst_dfa(oos_r)
        else:
            self.hurst = 0.5
