"""
Nishkarsh v1.4.0 — Regime detection: Kalman Filter, HMM, GARCH, CUSUM.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Multi-model regime classification for constituent-level analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from analytics.utils import MathUtils


# ─── Fused single-pass regime kernel (Numba) ─────────────────────────────────
# A faithful port of the object-based per-constituent loop in engines/nirnay.py
# (Kalman → GARCH → HMM → CUSUM). The object classes below are kept verbatim and
# remain the reference implementation; this kernel reproduces their output
# bit-for-bit (validated to ~1e-12) with zero per-step Python/NumPy dispatch,
# which is the bulk of the Nirnay cost (one loop per name × ~50 names × ~1700
# rows). LOGIC IS UNCHANGED from Nishkarsh: the CUSUM z-score window INCLUDES the
# current point (non-causal, as in CUSUMDetector.update) and the regime cuts use
# Nishkarsh's structural thresholds (strong = 2/3, weak = 1/3 + 0.07), NOT magic
# 0.6 / 0.4.


@njit(cache=True)
def _njit_gaussian_pdf(x: float, mean: float, std: float) -> float:
    var = std * std
    denom = (2.0 * np.pi * var) ** 0.5
    num = np.exp(-(x - mean) ** 2 / (2.0 * var))
    return num / denom


@njit(cache=True)
def _pop_var(buf: np.ndarray, lo: int, hi: int) -> float:
    """Population variance (ddof=0) of buf[lo:hi] — mirrors np.var / np.std default."""
    k = hi - lo
    if k <= 0:
        return 0.0
    s = 0.0
    for j in range(lo, hi):
        s += buf[j]
    m = s / k
    v = 0.0
    for j in range(lo, hi):
        d = buf[j] - m
        v += d * d
    return v / k


@njit(cache=True)
def _regime_loop_njit(sig: np.ndarray):
    """Single-pass port of the Nirnay regime-intelligence loop.

    Returns, per step:
        regime_code  0=NEUTRAL 1=BULL 2=BEAR 3=WEAK_BULL 4=WEAK_BEAR 5=TRANSITION
        hmm_bull, hmm_bear, confidence  (floats)
        vol_code     0=LOW 1=NORMAL 2=HIGH 3=EXTREME
        change       (0/1)
    """
    n = sig.shape[0]
    regime_code = np.zeros(n, dtype=np.int64)
    vol_code = np.zeros(n, dtype=np.int64)
    hmm_bull = np.zeros(n, dtype=np.float64)
    hmm_bear = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    change_pts = np.zeros(n, dtype=np.int64)

    # Regime cuts (structural to the 3-state space — see nirnay.py).
    uniform_p = 1.0 / 3.0
    strong_p = 2.0 * uniform_p
    weak_p = uniform_p + 0.07

    # ── Kalman state ──
    k_est = 0.0
    k_cov = 1.0
    k_proc = 0.01
    k_meas = 0.1
    innov = np.empty(n, dtype=np.float64)

    # ── GARCH state ──
    g_var = 0.04
    g_omega = 0.0001
    g_alpha = 0.1
    g_beta = 0.85
    g_ltm = 0.04
    shocks = np.empty(n, dtype=np.float64)

    # ── HMM state ──
    trans = np.array([[0.98, 0.01, 0.01],
                      [0.01, 0.98, 0.01],
                      [0.01, 0.01, 0.98]])
    em_mean = np.array([0.6, 0.0, -0.6])
    em_std = np.array([0.3, 0.25, 0.3])
    sp = np.array([0.33, 0.34, 0.33])
    obs_hist = np.empty(n, dtype=np.float64)   # observation_history (= filtered)
    state_hist = np.empty(n, dtype=np.int64)

    # ── CUSUM state ──
    pos_cusum = 0.0
    neg_cusum = 0.0
    c_thr = 4.0
    c_drift = 0.5
    cvals = np.empty(n, dtype=np.float64)
    run_mean = 0.0
    run_std = 1.0

    prev_sig = 0.0
    has_prev = False

    for i in range(n):
        s = sig[i]

        # ── Kalman update ──
        pred_est = k_est
        pred_cov = k_cov + k_proc
        innovation = s - pred_est
        innov[i] = innovation
        innov_cov = pred_cov + k_meas
        gain = pred_cov / innov_cov
        k_est = pred_est + gain * innovation
        k_cov = (1.0 - gain) * pred_cov
        ni = i + 1  # innovation_history length (capped at 50, only last 20 used)
        if ni >= 5:
            w = 20 if ni >= 20 else ni
            iv = _pop_var(innov, i + 1 - w, i + 1)
            k_meas = 0.9 * k_meas + 0.1 * iv
        filtered = k_est

        # ── GARCH update(shock) ──
        shock = (s - prev_sig) if has_prev else 0.0
        shocks[i] = shock
        new_var = g_omega + g_alpha * (shock * shock) + g_beta * g_var
        if new_var < 0.001:
            new_var = 0.001
        elif new_var > 1.0:
            new_var = 1.0
        g_var = new_var
        ns = i + 1
        if ns >= 10:
            w = 50 if ns >= 50 else ns
            realized = _pop_var(shocks, i + 1 - w, i + 1)
            g_ltm = 0.95 * g_ltm + 0.05 * realized
        # get_regime()
        cur_vol = np.sqrt(g_var)
        lt_vol = np.sqrt(g_ltm)
        ratio = cur_vol / lt_vol if lt_vol > 0 else 1.0
        if ratio < 0.6:
            vol_code[i] = 0
        elif ratio < 0.9:
            vol_code[i] = 1
        elif ratio < 1.4:
            vol_code[i] = 2
        else:
            vol_code[i] = 3

        # ── HMM update(filtered) ──
        obs_hist[i] = filtered
        e0 = _njit_gaussian_pdf(filtered, em_mean[0], em_std[0] + 1e-4)
        e1 = _njit_gaussian_pdf(filtered, em_mean[1], em_std[1] + 1e-4)
        e2 = _njit_gaussian_pdf(filtered, em_mean[2], em_std[2] + 1e-4)
        # predicted = trans.T @ sp
        p0 = trans[0, 0] * sp[0] + trans[1, 0] * sp[1] + trans[2, 0] * sp[2]
        p1 = trans[0, 1] * sp[0] + trans[1, 1] * sp[1] + trans[2, 1] * sp[2]
        p2 = trans[0, 2] * sp[0] + trans[1, 2] * sp[1] + trans[2, 2] * sp[2]
        u0 = e0 * p0
        u1 = e1 * p1
        u2 = e2 * p2
        total = u0 + u1 + u2
        if total > 1e-10:
            sp[0] = u0 / total
            sp[1] = u1 / total
            sp[2] = u2 / total
        else:
            sp[0] = 0.33
            sp[1] = 0.34
            sp[2] = 0.33
        # most likely (np.argmax → first max index)
        if sp[0] >= sp[1] and sp[0] >= sp[2]:
            ml = 0
        elif sp[1] >= sp[2]:
            ml = 1
        else:
            ml = 2
        state_hist[i] = ml
        nobs = i + 1
        # _adapt_emissions (last 50 obs) — runs when observation_history >= 10
        if nobs >= 10:
            w = 50 if nobs >= 50 else nobs
            lo = i + 1 - w
            for stt in range(3):
                cnt = 0
                ssum = 0.0
                for j in range(lo, i + 1):
                    if state_hist[j] == stt:
                        cnt += 1
                        ssum += obs_hist[j]
                if cnt >= 2:
                    nm = ssum / cnt
                    vv = 0.0
                    for j in range(lo, i + 1):
                        if state_hist[j] == stt:
                            d = obs_hist[j] - nm
                            vv += d * d
                    nsd = np.sqrt(vv / cnt)
                    if nsd < 1e-4:
                        nsd = 1e-4
                    em_mean[stt] = 0.9 * em_mean[stt] + 0.1 * nm
                    em_std[stt] = 0.9 * em_std[stt] + 0.1 * nsd
        # _adapt_transitions (last 30 states) — runs when state_history >= 5
        if nobs >= 5:
            w = 30 if nobs >= 30 else nobs
            lo = i + 1 - w
            counts = np.zeros((3, 3))
            for j in range(lo, i):
                counts[state_hist[j], state_hist[j + 1]] += 1.0
            for r in range(3):
                rs = counts[r, 0] + counts[r, 1] + counts[r, 2]
                if rs >= 2:
                    for cc in range(3):
                        np_ = (counts[r, cc] + 1.0) / (rs + 3.0)
                        trans[r, cc] = 0.8 * trans[r, cc] + 0.2 * np_

        bull_p = sp[0]
        neut_p = sp[1]
        bear_p = sp[2]

        # ── CUSUM update(filtered) — NON-causal window (includes current point),
        # exactly matching CUSUMDetector.update in Nishkarsh. ──
        cvals[i] = filtered
        nc = i + 1
        if nc >= 3:
            w = 20 if nc >= 20 else nc
            lo = i + 1 - w  # window [lo, i+1) INCLUDES the just-appended value
            csum = 0.0
            for j in range(lo, i + 1):
                csum += cvals[j]
            run_mean = csum / w
            cv = 0.0
            for j in range(lo, i + 1):
                d = cvals[j] - run_mean
                cv += d * d
            sd = np.sqrt(cv / w)
            run_std = sd if sd > 0.1 else 0.1
        z = (filtered - run_mean) / run_std
        pos_cusum = pos_cusum + z - c_drift
        if pos_cusum < 0.0:
            pos_cusum = 0.0
        neg_cusum = neg_cusum - z - c_drift
        if neg_cusum < 0.0:
            neg_cusum = 0.0
        change = (pos_cusum > c_thr) or (neg_cusum > c_thr)
        if change:
            pos_cusum = 0.0
            neg_cusum = 0.0
        change_pts[i] = 1 if change else 0

        # ── Regime classification (Nishkarsh structural thresholds) ──
        if change:
            regime_code[i] = 5  # TRANSITION
        elif bull_p > strong_p:
            regime_code[i] = 1  # BULL
        elif bear_p > strong_p:
            regime_code[i] = 2  # BEAR
        elif bull_p > weak_p:
            regime_code[i] = 3  # WEAK_BULL
        elif bear_p > weak_p:
            regime_code[i] = 4  # WEAK_BEAR
        else:
            regime_code[i] = 0  # NEUTRAL

        hmm_bull[i] = bull_p
        hmm_bear[i] = bear_p
        c = bull_p
        if bear_p > c:
            c = bear_p
        if neut_p > c:
            c = neut_p
        confidence[i] = c

        prev_sig = s
        has_prev = True

    return regime_code, hmm_bull, hmm_bear, vol_code, change_pts, confidence


_REGIME_NAMES = ("NEUTRAL", "BULL", "BEAR", "WEAK_BULL", "WEAK_BEAR", "TRANSITION")
_VOL_NAMES = ("LOW", "NORMAL", "HIGH", "EXTREME")


def run_regime_loop(unified_vals: np.ndarray):
    """Vectorized driver for the per-constituent regime loop.

    Returns ``(regimes, hmm_bulls, hmm_bears, vol_regimes, change_points,
    confidences)`` matching the object-based loop's output exactly. NaNs in
    ``unified_vals`` are treated as 0.0 (as the original loop does).
    """
    sig = np.nan_to_num(np.asarray(unified_vals, dtype=np.float64), nan=0.0)
    rc, hb, hbe, vc, cp, conf = _regime_loop_njit(sig)
    regimes = [_REGIME_NAMES[c] for c in rc]
    vol_regimes = [_VOL_NAMES[c] for c in vc]
    change_points = [bool(x) for x in cp]
    return regimes, hb, hbe, vol_regimes, change_points, conf


def _hmm_forward_step(transition_matrix: np.ndarray, state_probabilities: np.ndarray,
                      emissions: np.ndarray) -> np.ndarray:
    """One forward-algorithm step (vectorized NumPy).

    Kept as plain NumPy: the operation is a 3×3 matrix-vector product invoked
    once per bar per constituent. Numba JIT warm-up cost exceeded the per-call
    saving at this size and added a compile-time dependency for no benefit.
    """
    predicted = transition_matrix.T @ state_probabilities
    updated = emissions * predicted
    total = np.sum(updated)
    if total > 1e-10:
        return updated / total
    return np.array([0.33, 0.34, 0.33])


def _gaussian_pdf(x: float, mean: float, std: float) -> float:
    var = std ** 2
    denom = (2 * np.pi * var) ** 0.5
    num = np.exp(-(x - mean) ** 2 / (2 * var))
    return num / denom


# ─── Dataclass states ────────────────────────────────────────────────────────


@dataclass
class KalmanState:
    """Internal state of the Kalman filter."""

    estimate: float = 0.0
    error_covariance: float = 1.0
    process_variance: float = 0.01
    measurement_variance: float = 0.1


@dataclass
class HMMState:
    """Internal state of the Hidden Markov Model."""

    n_states: int = 3
    transition_matrix: np.ndarray | None = None
    emission_means: np.ndarray | None = None
    emission_stds: np.ndarray | None = None
    state_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.transition_matrix is None:
            # Extreme structural hysteresis: 98% chance to remain, punishing noise flips.
            self.transition_matrix = np.array([
                [0.98, 0.01, 0.01],
                [0.01, 0.98, 0.01],
                [0.01, 0.01, 0.98],
            ])
        if self.emission_means is None:
            self.emission_means = np.array([0.6, 0.0, -0.6])
        if self.emission_stds is None:
            self.emission_stds = np.array([0.3, 0.25, 0.3])
        if self.state_probabilities is None:
            self.state_probabilities = np.array([0.33, 0.34, 0.33])


@dataclass
class GARCHState:
    """Internal state of the GARCH volatility detector."""

    current_variance: float = 0.04
    omega: float = 0.0001
    alpha: float = 0.1
    beta: float = 0.85
    long_term_mean: float = 0.04


@dataclass
class CUSUMState:
    """Internal state of the CUSUM change point detector."""

    positive_cusum: float = 0.0
    negative_cusum: float = 0.0
    threshold: float = 4.0
    drift: float = 0.5


# ─── Adaptive Kalman Filter ──────────────────────────────────────────────────


class AdaptiveKalmanFilter:
    """Kalman filter with adaptive noise estimation for signal smoothing.

    Parameters
    ----------
    process_var : float
        Initial process variance.
    measurement_var : float
        Initial measurement variance.
    """

    def __init__(
        self, process_var: float = 0.01, measurement_var: float = 0.1
    ) -> None:
        self.state = KalmanState(
            process_variance=process_var,
            measurement_variance=measurement_var,
        )
        self.innovation_history: list[float] = []

    def update(self, measurement: float) -> float:
        """Update the filter with a new measurement.

        Returns
        -------
        float
            Filtered state estimate.
        """
        predicted_estimate = self.state.estimate
        predicted_covariance = self.state.error_covariance + self.state.process_variance

        innovation = measurement - predicted_estimate
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > 50:
            self.innovation_history.pop(0)

        innovation_cov = predicted_covariance + self.state.measurement_variance
        kalman_gain = predicted_covariance / innovation_cov

        self.state.estimate = predicted_estimate + kalman_gain * innovation
        self.state.error_covariance = (1 - kalman_gain) * predicted_covariance

        # Adaptive noise estimation
        if len(self.innovation_history) >= 5:
            innovation_var = np.var(
                self.innovation_history[-min(20, len(self.innovation_history)) :]
            )
            self.state.measurement_variance = (
                0.9 * self.state.measurement_variance + 0.1 * innovation_var
            )

        return self.state.estimate

    def get_uncertainty(self) -> float:
        """Return the standard deviation of the current estimate."""
        return float(np.sqrt(self.state.error_covariance))

    def reset(self, initial: float = 0.0) -> None:
        """Reset the filter to initial conditions."""
        self.state.estimate = initial
        self.state.error_covariance = 1.0
        self.innovation_history = []


# ─── Adaptive HMM ────────────────────────────────────────────────────────────


class AdaptiveHMM:
    """HMM for regime state estimation with online learning.

    Maintains three states: Bull (state 0), Neutral (state 1), Bear (state 2).
    Emission parameters adapt online based on recent observations.
    """

    def __init__(self) -> None:
        self.state = HMMState()
        self.observation_history: list[float] = []
        self.state_history: list[int] = []

    def _emission_prob(self, observation: float, state_idx: int) -> float:
        """Emission probability for a given state with static epsilon shrinkage."""
        std = self.state.emission_stds[state_idx] + 1e-4  # Epsilon diagonal penalty
        return _gaussian_pdf(
            observation,
            self.state.emission_means[state_idx],
            std
        )

    def _forward_step(self, observation: float) -> np.ndarray:
        """Single step of the forward algorithm vectorized via Numba."""
        emissions = np.array([self._emission_prob(observation, s) for s in range(3)])
        updated = _hmm_forward_step(
            self.state.transition_matrix,
            self.state.state_probabilities,
            emissions
        )
        self.state.state_probabilities = updated
        return updated

    def update(self, observation: float) -> dict[str, float]:
        """Update HMM with a new observation.

        Returns
        -------
        dict[str, float]
            State probabilities keyed ``"BULL"``, ``"NEUTRAL"``, ``"BEAR"``.
        """
        self.observation_history.append(observation)
        probs = self._forward_step(observation)

        most_likely = int(np.argmax(probs))
        self.state_history.append(most_likely)

        if len(self.observation_history) >= 10:
            self._adapt_emissions()
        if len(self.state_history) >= 5:
            self._adapt_transitions()

        return {"BULL": probs[0], "NEUTRAL": probs[1], "BEAR": probs[2]}

    def _adapt_emissions(self) -> None:
        """Adapt emission parameters based on recent observations."""
        recent_obs = np.asarray(self.observation_history[-50:], dtype=float)
        # Convert the state window ONCE (the old code rebuilt this array inside
        # the per-state loop, 3× per row across ~1700 rows × 50 stocks).
        recent_states = np.asarray(self.state_history[-len(recent_obs) :])

        for state_idx in range(3):
            mask = recent_states == state_idx
            if mask.sum() >= 2:
                state_obs = recent_obs[mask]
                new_mean = np.mean(state_obs)
                # np.std(x) == sqrt(mean((x − mean)²)); reuse new_mean to skip the
                # second internal mean pass np.std would do. Bit-identical result.
                new_std = max(np.sqrt(np.mean((state_obs - new_mean) ** 2)), 1e-4)

                self.state.emission_means[state_idx] = (
                    0.9 * self.state.emission_means[state_idx] + 0.1 * new_mean
                )
                self.state.emission_stds[state_idx] = (
                    0.9 * self.state.emission_stds[state_idx] + 0.1 * new_std
                )

    def _adapt_transitions(self) -> None:
        """Adapt transition matrix based on recent state transitions."""
        recent = np.asarray(self.state_history[-30:])
        counts = np.zeros((3, 3))
        if len(recent) > 1:
            # Vectorized equivalent of the per-transition Python loop.
            np.add.at(counts, (recent[:-1], recent[1:]), 1)

        for i in range(3):
            row_sum = counts[i].sum()
            if row_sum >= 2:
                new_probs = (counts[i] + 1) / (row_sum + 3)
                self.state.transition_matrix[i] = (
                    0.8 * self.state.transition_matrix[i] + 0.2 * new_probs
                )

    def get_persistence(self) -> int:
        """Return the number of consecutive steps in the current state."""
        if len(self.state_history) < 2:
            return 1
        current = self.state_history[-1]
        persistence = 1
        for i in range(len(self.state_history) - 2, -1, -1):
            if self.state_history[i] == current:
                persistence += 1
            else:
                break
        return persistence

    def reset(self) -> None:
        """Reset HMM to initial conditions."""
        self.state = HMMState()
        self.observation_history = []
        self.state_history = []


# ─── GARCH Volatility Detector ───────────────────────────────────────────────


class GARCHDetector:
    """GARCH-inspired volatility regime detection.

    Models variance as: ``σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}``
    """

    def __init__(self) -> None:
        self.state = GARCHState()
        self.shock_history: list[float] = []

    def update(self, shock: float) -> float:
        """Update with a new shock value.

        Returns
        -------
        float
            Current volatility estimate (``√σ²``).
        """
        self.shock_history.append(shock)
        shock_sq = shock**2
        new_var = (
            self.state.omega
            + self.state.alpha * shock_sq
            + self.state.beta * self.state.current_variance
        )
        new_var = np.clip(new_var, 0.001, 1.0)
        self.state.current_variance = new_var

        if len(self.shock_history) >= 10:
            realized = np.var(
                self.shock_history[-min(50, len(self.shock_history)) :]
            )
            self.state.long_term_mean = (
                0.95 * self.state.long_term_mean + 0.05 * realized
            )

        return float(np.sqrt(new_var))

    def get_regime(self) -> tuple[str, float]:
        """Classify the current volatility regime.

        Returns
        -------
        regime : str
            One of ``"LOW"``, ``"NORMAL"``, ``"HIGH"``, ``"EXTREME"``.
        sensitivity : float
            Multiplier for signal adjustment (low vol → amplify,
            high vol → dampen).
        """
        current_vol = np.sqrt(self.state.current_variance)
        long_term_vol = np.sqrt(self.state.long_term_mean)
        ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0

        if ratio < 0.6:
            return "LOW", 1.3
        if ratio < 0.9:
            return "NORMAL", 1.0
        if ratio < 1.4:
            return "HIGH", 0.8
        return "EXTREME", 0.6

    def reset(self) -> None:
        """Reset to initial conditions."""
        self.state = GARCHState()
        self.shock_history = []


# ─── CUSUM Change Point Detector ─────────────────────────────────────────────


class CUSUMDetector:
    """Cumulative Sum (CUSUM) change point detection.

    Detects shifts in the mean of a time-series by tracking
    cumulative deviations from a running mean.
    """

    def __init__(self, threshold: float = 4.0, drift: float = 0.5) -> None:
        self.state = CUSUMState(threshold=threshold, drift=drift)
        self.value_history: list[float] = []
        self.running_mean: float = 0.0
        self.running_std: float = 1.0

    def update(self, value: float) -> bool:
        """Update with a new value.

        Returns
        -------
        bool
            ``True`` if a change point is detected.
        """
        self.value_history.append(value)

        if len(self.value_history) >= 3:
            recent = self.value_history[-min(20, len(self.value_history)) :]
            self.running_mean = np.mean(recent)
            self.running_std = max(np.std(recent), 0.1)

        z = (value - self.running_mean) / self.running_std

        self.state.positive_cusum = max(
            0, self.state.positive_cusum + z - self.state.drift
        )
        self.state.negative_cusum = max(
            0, self.state.negative_cusum - z - self.state.drift
        )

        change_detected = (
            self.state.positive_cusum > self.state.threshold
            or self.state.negative_cusum > self.state.threshold
        )

        if change_detected:
            self.state.positive_cusum = 0
            self.state.negative_cusum = 0

        return change_detected

    def reset(self) -> None:
        """Reset to initial conditions."""
        self.state = CUSUMState()
        self.value_history = []
