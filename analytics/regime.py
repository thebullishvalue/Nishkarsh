"""
Nishkarsh — Regime detection: Kalman filter, online HMM, GARCH, CUSUM.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Multi-model regime classification for constituent-level analysis.

Two implementations of ONE model, kept in lockstep:

* ``_regime_loop_njit`` — the fused single-pass Numba kernel the engine runs
  (one loop per constituent; eliminates per-bar Python dispatch).
* ``AdaptiveKalmanFilter`` / ``AdaptiveHMM`` / ``GARCHDetector`` /
  ``CUSUMDetector`` driven by ``regime_loop_reference`` — the readable object
  implementation, kept as the PARITY TEST ORACLE (kernel vs classes ≤ 1e-12).

Per bar ``t`` the inputs are the constituent's Unified signal ``u_t`` ∈ [-1, 1]
and its Close log return ``r_t`` (0 on the first bar):

1. **Kalman** (local level, adaptive measurement variance) smooths ``u_t`` →
   ``f_t``; its innovation ``e_t = u_t − f_{t|t−1}`` is standardised by a
   causal EWMA (λ = 0.98) of the PAST squared innovations,
   ``ν_t = e_t / √V_{t−1}``, ``V_t = 0.98·V_{t−1} + 0.02·e_t²`` (V₀ = 0.1), so
   ν has ≈ unit variance and is ≈ iid when no change occurs.
2. **HMM** (3 states Bull / Neutral / Bear, online EM on ``f_t``) gives state
   probabilities. Emission std is floored at ``std_floor``
   (``core.config.NIRNAY_HMM_STD_FLOOR``): a 1e-4 floor let a state collapse
   onto a frozen mean, making labels path / start-date dependent (A2-01).
   A Bull > Neutral > Bear mean ordering is enforced after every adaptation
   (label-switching guard; Jasra, Holmes & Stephens 2005).
3. **GARCH(1,1), variance-targeted, on Close log returns** — ``Vol_Regime`` is
   genuine PRICE volatility: the ratio of the conditional vol to its causal
   long-run level (A2-09). Scale-free: ω = (1 − α − β)·L_{t−1}.
4. **CUSUM** (two-sided Page, k = 0.5, h = 4) on the standardised Kalman
   innovations ν_t — ≈ iid under no change, so the alarm rate is the design
   rate instead of the 1-in-8-to-18 bars it fired on the autocorrelated
   filtered level (A2-09).

Regime label: TRANSITION on a CUSUM alarm, else BULL / BEAR when that state's
probability > 2/3, WEAK_BULL / WEAK_BEAR when > 1/3 + 0.07, else NEUTRAL.
Everything is causal: the outputs at ``t`` depend on bars ≤ ``t`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from core.config import NIRNAY_HMM_STD_FLOOR

# ─── Model constants (shared by the kernel and the classes) ──────────────────
# Regime cuts are structural to the 3-state space (uniform = 1/3).
_STRONG_P = 2.0 / 3.0
_WEAK_P = 1.0 / 3.0 + 0.07
# GARCH(1,1) persistence; the long-run level L is an EWMA (0.95/0.05) of the
# trailing-50 realised variance, seeded with the expanding variance of the
# first 10 shocks.
_G_ALPHA = 0.1
_G_BETA = 0.85
_G_MIN_SHOCKS = 10
# Vol-ratio bands (current conditional vol / long-run vol), centred on 1.0 so
# "current vol ≈ its long-run level" is NORMAL (the old 0.6/0.9/1.4 bands
# labelled ratio = 1.0 HIGH). Symmetric in log for LOW/HIGH (0.8 = 1/1.25).
_VOL_LOW = 0.8
_VOL_HIGH = 1.25
_VOL_EXTREME = 1.6
# CUSUM (Page) reference value k and decision interval h, in units of ν.
_C_DRIFT = 0.5
_C_THRESHOLD = 4.0
# Innovation standardisation: EWMA of past squared innovations (λ, V₀);
# below _R_MIN the scale is floored.
_NU_LAMBDA = 0.98
_NU_VAR0 = 0.1
_R_MIN = 1e-12

_REGIME_NAMES = ("NEUTRAL", "BULL", "BEAR", "WEAK_BULL", "WEAK_BEAR", "TRANSITION")
_VOL_NAMES = ("LOW", "NORMAL", "HIGH", "EXTREME")


# ─── Fused single-pass regime kernel (Numba) ─────────────────────────────────


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
def _regime_loop_njit(sig: np.ndarray, logret: np.ndarray, std_floor: float):
    """Single-pass Kalman → GARCH → HMM → CUSUM regime loop.

    ``sig`` is the Unified signal, ``logret`` the Close log returns (same
    length; non-finite entries must already be 0.0), ``std_floor`` the HMM
    emission-std floor. Returns, per step:
        regime_code  0=NEUTRAL 1=BULL 2=BEAR 3=WEAK_BULL 4=WEAK_BEAR 5=TRANSITION
        hmm_bull, hmm_bear  (floats)
        vol_code     0=LOW 1=NORMAL 2=HIGH 3=EXTREME
        change       (0/1)
        confidence   (max state probability)
    """
    n = sig.shape[0]
    regime_code = np.zeros(n, dtype=np.int64)
    vol_code = np.zeros(n, dtype=np.int64)
    hmm_bull = np.zeros(n, dtype=np.float64)
    hmm_bear = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    change_pts = np.zeros(n, dtype=np.int64)

    # ── Kalman state ──
    k_est = 0.0
    k_cov = 1.0
    k_proc = 0.01
    k_meas = 0.1
    innov = np.empty(n, dtype=np.float64)
    nu_var = _NU_VAR0

    # ── GARCH state ──
    g_var = 0.0
    g_ltm = 0.0
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

    for i in range(n):
        s = sig[i]

        # ── Kalman update; standardised innovation uses R from BEFORE bar i ──
        pred_est = k_est
        pred_cov = k_cov + k_proc
        innovation = s - pred_est
        innov[i] = innovation
        if nu_var > _R_MIN:
            nu = innovation / np.sqrt(nu_var)
        else:
            nu = innovation / np.sqrt(_R_MIN)
        nu_var = _NU_LAMBDA * nu_var + (1.0 - _NU_LAMBDA) * innovation * innovation
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

        # ── GARCH(1,1), variance-targeted, on the Close log return ──
        shock = logret[i]
        shocks[i] = shock
        ns = i + 1
        if ns < _G_MIN_SHOCKS:
            g_ltm = _pop_var(shocks, 0, ns)
            g_var = g_ltm
        else:
            g_var = ((1.0 - _G_ALPHA - _G_BETA) * g_ltm
                     + _G_ALPHA * (shock * shock) + _G_BETA * g_var)
            w = 50 if ns >= 50 else ns
            realized = _pop_var(shocks, i + 1 - w, i + 1)
            g_ltm = 0.95 * g_ltm + 0.05 * realized
        if g_ltm > 0.0:
            ratio = np.sqrt(g_var) / np.sqrt(g_ltm)
        else:
            ratio = 1.0
        if ratio < _VOL_LOW:
            vol_code[i] = 0
        elif ratio < _VOL_HIGH:
            vol_code[i] = 1
        elif ratio < _VOL_EXTREME:
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
                    if nsd < std_floor:
                        nsd = std_floor
                    em_mean[stt] = 0.9 * em_mean[stt] + 0.1 * nm
                    em_std[stt] = 0.9 * em_std[stt] + 0.1 * nsd
            # Ordering constraint (state 0=Bull > 1=Neutral > 2=Bear, 0.01
            # margin): without it the online EM has no identifiability
            # constraint and a slow one-sided drift can label-switch the states.
            if em_mean[0] < em_mean[1] + 0.01:
                em_mean[0] = em_mean[1] + 0.01
            if em_mean[2] > em_mean[1] - 0.01:
                em_mean[2] = em_mean[1] - 0.01
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

        # ── CUSUM on the standardised innovation ──
        pos_cusum = pos_cusum + nu - _C_DRIFT
        if pos_cusum < 0.0:
            pos_cusum = 0.0
        neg_cusum = neg_cusum - nu - _C_DRIFT
        if neg_cusum < 0.0:
            neg_cusum = 0.0
        change = (pos_cusum > _C_THRESHOLD) or (neg_cusum > _C_THRESHOLD)
        if change:
            pos_cusum = 0.0
            neg_cusum = 0.0
        change_pts[i] = 1 if change else 0

        # ── Regime classification ──
        if change:
            regime_code[i] = 5  # TRANSITION
        elif bull_p > _STRONG_P:
            regime_code[i] = 1  # BULL
        elif bear_p > _STRONG_P:
            regime_code[i] = 2  # BEAR
        elif bull_p > _WEAK_P:
            regime_code[i] = 3  # WEAK_BULL
        elif bear_p > _WEAK_P:
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

    return regime_code, hmm_bull, hmm_bear, vol_code, change_pts, confidence


def _prepare_inputs(unified_vals, close_vals) -> tuple[np.ndarray, np.ndarray]:
    """Unified (NaN → 0.0) and Close log returns (first bar / non-finite → 0.0)."""
    sig = np.nan_to_num(np.asarray(unified_vals, dtype=np.float64), nan=0.0)
    close = np.asarray(close_vals, dtype=np.float64)
    if close.shape != sig.shape:
        raise ValueError(f"unified ({sig.shape}) and close ({close.shape}) lengths differ")
    logret = np.zeros_like(sig)
    if close.size > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(close[1:]) - np.log(close[:-1])
        logret[1:] = np.where(np.isfinite(lr), lr, 0.0)
    return sig, logret


def run_regime_loop(unified_vals, close_vals, std_floor: float = NIRNAY_HMM_STD_FLOOR):
    """Single entry point for the per-constituent regime loop.

    Parameters
    ----------
    unified_vals : array-like
        The constituent's Unified signal (NaN treated as 0.0).
    close_vals : array-like
        The constituent's Close, same length (drives the GARCH vol regime).
    std_floor : float
        HMM emission-std floor (default ``core.config.NIRNAY_HMM_STD_FLOOR``).

    Returns
    -------
    ``(regimes, hmm_bulls, hmm_bears, vol_regimes, change_points, confidences)``
    — label lists / float arrays, identical to ``regime_loop_reference``.
    """
    sig, logret = _prepare_inputs(unified_vals, close_vals)
    rc, hb, hbe, vc, cp, conf = _regime_loop_njit(sig, logret, float(std_floor))
    regimes = [_REGIME_NAMES[c] for c in rc]
    vol_regimes = [_VOL_NAMES[c] for c in vc]
    change_points = [bool(x) for x in cp]
    return regimes, hb, hbe, vol_regimes, change_points, conf


def regime_loop_reference(unified_vals, close_vals, std_floor: float = NIRNAY_HMM_STD_FLOOR):
    """Object-based oracle of ``_regime_loop_njit`` (parity test reference).

    Drives the four classes below bar by bar. Returns the kernel's raw tuple
    ``(regime_code, hmm_bull, hmm_bear, vol_code, change, confidence)`` as
    NumPy arrays so the two can be compared element-wise.
    """
    sig, logret = _prepare_inputs(unified_vals, close_vals)
    kf = AdaptiveKalmanFilter()
    hmm = AdaptiveHMM(std_floor=std_floor)
    garch = GARCHDetector()
    cusum = CUSUMDetector()
    vmap = {name: k for k, name in enumerate(_VOL_NAMES)}
    n = sig.shape[0]
    rc = np.zeros(n, dtype=np.int64)
    vc = np.zeros(n, dtype=np.int64)
    cp = np.zeros(n, dtype=np.int64)
    hb = np.zeros(n)
    hbe = np.zeros(n)
    conf = np.zeros(n)
    for i in range(n):
        filtered = kf.update(sig[i])
        nu = kf.last_standardized_innovation
        garch.update(logret[i])
        vol, _ = garch.get_regime()
        probs = hmm.update(filtered)
        change = cusum.update(nu)
        b, ne, be = probs["BULL"], probs["NEUTRAL"], probs["BEAR"]
        if change:
            rc[i] = 5
        elif b > _STRONG_P:
            rc[i] = 1
        elif be > _STRONG_P:
            rc[i] = 2
        elif b > _WEAK_P:
            rc[i] = 3
        elif be > _WEAK_P:
            rc[i] = 4
        vc[i] = vmap[vol]
        cp[i] = int(change)
        hb[i], hbe[i] = b, be
        conf[i] = max(b, be, ne)
    return rc, hb, hbe, vc, cp, conf


def _hmm_forward_step(transition_matrix: np.ndarray, state_probabilities: np.ndarray,
                      emissions: np.ndarray) -> np.ndarray:
    """One forward-algorithm step (plain NumPy, 3×3)."""
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
            # Strong structural hysteresis: 98% chance to remain.
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
    """Internal state of the variance-targeted GARCH(1,1) detector."""

    current_variance: float = 0.0
    alpha: float = _G_ALPHA
    beta: float = _G_BETA
    long_term_mean: float = 0.0


@dataclass
class CUSUMState:
    """Internal state of the CUSUM change point detector."""

    positive_cusum: float = 0.0
    negative_cusum: float = 0.0
    threshold: float = _C_THRESHOLD
    drift: float = _C_DRIFT


# ─── Adaptive Kalman Filter ──────────────────────────────────────────────────


class AdaptiveKalmanFilter:
    """Local-level Kalman filter with adaptive measurement-noise estimation.

    Parameters
    ----------
    process_var : float
        Process (level) variance.
    measurement_var : float
        Initial measurement variance; thereafter an EWMA (0.9/0.1) of the
        trailing-20 innovation variance.
    """

    def __init__(
        self, process_var: float = 0.01, measurement_var: float = 0.1
    ) -> None:
        self.state = KalmanState(
            process_variance=process_var,
            measurement_variance=measurement_var,
        )
        self.innovation_history: list[float] = []
        self.innovation_scale_var: float = _NU_VAR0
        self.last_standardized_innovation: float = 0.0

    def update(self, measurement: float) -> float:
        """Update the filter with a new measurement.

        Also sets ``last_standardized_innovation`` = innovation / √V, with V
        the EWMA (λ = 0.98) of squared innovations from BEFORE this bar.

        Returns
        -------
        float
            Filtered state estimate.
        """
        predicted_estimate = self.state.estimate
        predicted_covariance = self.state.error_covariance + self.state.process_variance

        innovation = measurement - predicted_estimate
        self.last_standardized_innovation = innovation / np.sqrt(max(self.innovation_scale_var, _R_MIN))
        self.innovation_scale_var = (
            _NU_LAMBDA * self.innovation_scale_var + (1.0 - _NU_LAMBDA) * innovation * innovation
        )
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
        self.innovation_scale_var = _NU_VAR0
        self.last_standardized_innovation = 0.0


# ─── Adaptive HMM ────────────────────────────────────────────────────────────


class AdaptiveHMM:
    """HMM for regime state estimation with online learning.

    Maintains three states: Bull (state 0), Neutral (state 1), Bear (state 2).
    Emission parameters adapt online based on recent observations; the
    emission std is floored at ``std_floor``.
    """

    def __init__(self, std_floor: float = NIRNAY_HMM_STD_FLOOR) -> None:
        self.state = HMMState()
        self.std_floor = float(std_floor)
        self.observation_history: list[float] = []
        self.state_history: list[int] = []

    def _emission_prob(self, observation: float, state_idx: int) -> float:
        """Gaussian emission probability (std + 1e-4 epsilon)."""
        std = self.state.emission_stds[state_idx] + 1e-4
        return _gaussian_pdf(
            observation,
            self.state.emission_means[state_idx],
            std
        )

    def _forward_step(self, observation: float) -> np.ndarray:
        """Single step of the forward algorithm."""
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
        recent_states = np.asarray(self.state_history[-len(recent_obs) :])

        for state_idx in range(3):
            mask = recent_states == state_idx
            if mask.sum() >= 2:
                state_obs = recent_obs[mask]
                new_mean = np.mean(state_obs)
                # == np.std(state_obs), reusing new_mean.
                new_std = max(np.sqrt(np.mean((state_obs - new_mean) ** 2)), self.std_floor)

                self.state.emission_means[state_idx] = (
                    0.9 * self.state.emission_means[state_idx] + 0.1 * new_mean
                )
                self.state.emission_stds[state_idx] = (
                    0.9 * self.state.emission_stds[state_idx] + 0.1 * new_std
                )

        # Ordering constraint (Bull > Neutral > Bear, 0.01 margin) — mirrors
        # the kernel exactly.
        means = self.state.emission_means
        if means[0] < means[1] + 0.01:
            means[0] = means[1] + 0.01
        if means[2] > means[1] - 0.01:
            means[2] = means[1] - 0.01

    def _adapt_transitions(self) -> None:
        """Adapt transition matrix based on recent state transitions."""
        recent = np.asarray(self.state_history[-30:])
        counts = np.zeros((3, 3))
        if len(recent) > 1:
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
    """Variance-targeted GARCH(1,1) volatility-regime detector.

    Fed Close log returns ``r_t``. For the first 10 shocks the conditional and
    long-run variances are both the expanding variance of the shocks so far;
    thereafter ``σ²_t = (1 − α − β)·L_{t−1} + α·r_t² + β·σ²_{t−1}`` and
    ``L_t = 0.95·L_{t−1} + 0.05·Var(last 50 shocks)``. Scale-free: the regime
    is the ratio ``σ_t / √L_t``.
    """

    def __init__(self) -> None:
        self.state = GARCHState()
        self.shock_history: list[float] = []

    def update(self, shock: float) -> float:
        """Update with a new shock (log return).

        Returns
        -------
        float
            Current conditional volatility estimate (``√σ²``).
        """
        self.shock_history.append(shock)
        ns = len(self.shock_history)
        if ns < _G_MIN_SHOCKS:
            self.state.long_term_mean = float(np.var(self.shock_history))
            self.state.current_variance = self.state.long_term_mean
        else:
            self.state.current_variance = (
                (1.0 - self.state.alpha - self.state.beta) * self.state.long_term_mean
                + self.state.alpha * shock ** 2
                + self.state.beta * self.state.current_variance
            )
            realized = np.var(self.shock_history[-min(50, ns):])
            self.state.long_term_mean = (
                0.95 * self.state.long_term_mean + 0.05 * realized
            )
        return float(np.sqrt(self.state.current_variance))

    def get_regime(self) -> tuple[str, float]:
        """Classify the current volatility regime.

        Returns
        -------
        regime : str
            ``"LOW"`` (ratio < 0.8), ``"NORMAL"`` (< 1.25), ``"HIGH"`` (< 1.6)
            or ``"EXTREME"``, where ratio = current vol / long-run vol.
        sensitivity : float
            Multiplier for signal adjustment (low vol → amplify,
            high vol → dampen).
        """
        long_term = self.state.long_term_mean
        if long_term > 0.0:
            ratio = np.sqrt(self.state.current_variance) / np.sqrt(long_term)
        else:
            ratio = 1.0

        if ratio < _VOL_LOW:
            return "LOW", 1.3
        if ratio < _VOL_HIGH:
            return "NORMAL", 1.0
        if ratio < _VOL_EXTREME:
            return "HIGH", 0.8
        return "EXTREME", 0.6

    def reset(self) -> None:
        """Reset to initial conditions."""
        self.state = GARCHState()
        self.shock_history = []


# ─── CUSUM Change Point Detector ─────────────────────────────────────────────


class CUSUMDetector:
    """Two-sided Page CUSUM on a standardised input.

    ``update(z)`` expects ``z`` ≈ iid N(0, 1) under no change (the engine
    feeds the standardised Kalman innovation). An alarm fires when either
    cumulative sum exceeds ``threshold`` (h); both sums then reset.
    """

    def __init__(self, threshold: float = _C_THRESHOLD, drift: float = _C_DRIFT) -> None:
        self.state = CUSUMState(threshold=threshold, drift=drift)

    def update(self, z: float) -> bool:
        """Update with a new standardised value.

        Returns
        -------
        bool
            ``True`` if a change point is detected.
        """
        self.state.positive_cusum = max(
            0.0, self.state.positive_cusum + z - self.state.drift
        )
        self.state.negative_cusum = max(
            0.0, self.state.negative_cusum - z - self.state.drift
        )

        change_detected = (
            self.state.positive_cusum > self.state.threshold
            or self.state.negative_cusum > self.state.threshold
        )

        if change_detected:
            self.state.positive_cusum = 0.0
            self.state.negative_cusum = 0.0

        return change_detected

    def reset(self) -> None:
        """Reset to initial conditions."""
        self.state = CUSUMState(threshold=self.state.threshold, drift=self.state.drift)
