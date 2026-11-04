"""
Nishkarsh — Leaky-integrator ("drift-diffusion") filter with a heuristic band.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

ANALYTICS — Causal smoothing of a bounded evidence series. Pure function.

What it is (exactly):
  * state:   ``s_t = (1−λ)·s_{t−1} + κ·e_t``  (λ = ``leak_rate``, κ = ``drift_scale``)
    — an exponential smoother with steady-state GAIN κ/λ (1.875 at the defaults
    0.15/0.08, i.e. a constant input of 10 settles at 18.75, not 10) and mean lag
    (1−λ)/λ bars (11.5 at λ=0.08).
  * "variance": ``v_t = (1−λ)·v_{t−1} + λ·σ²_LR + min(0.5·κ·|e_t|, 0.5·σ²_LR)``.
    This depends only on the MAGNITUDE of recent evidence, not on its
    consistency (a constant +40 and an alternating ±40 give the same width), and
    it is not a posterior variance estimated from data. ``±1.96·sqrt(v)`` is
    therefore a heuristic band, NOT a calibrated confidence interval, and must
    be labelled as such wherever it is shown.
"""

from __future__ import annotations

import numpy as np


def drift_diffusion_filter(
    observations: np.ndarray,
    leak_rate: float = 0.08,
    drift_scale: float = 0.15,
    long_run_var: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal leaky integrator with a magnitude-driven heuristic band.

    The filter starts at the FIRST finite observation (state seeded to that
    observation, variance to ``long_run_var``). Rows before it are warm-up and
    are returned as NaN — never a fabricated neutral 0.0. A non-finite
    observation after the start contributes zero evidence (the state decays by
    ``1−λ``); callers that must not display a reading on such rows mask them.

    Parameters
    ----------
    observations : np.ndarray
        Evidence series (NaN allowed).
    leak_rate : float
        λ — memory decay per step.
    drift_scale : float
        κ — weight of each new observation.
    long_run_var : float
        σ²_LR — level the heuristic variance reverts to.

    Returns
    -------
    filtered : np.ndarray
        Filtered state (NaN before the first finite observation).
    gains : np.ndarray
        Always zeros — the filter has no data-dependent gain; retained only for
        call-site compatibility of the 3-tuple.
    variances : np.ndarray
        Heuristic variance (NaN before the first finite observation).
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    filtered = np.full(n, np.nan)
    variances = np.full(n, np.nan)
    gains = np.zeros(n)
    if n == 0:
        return filtered, gains, variances

    finite = np.isfinite(obs)
    if not finite.any():
        return filtered, gains, variances

    i0 = int(np.argmax(finite))
    state = float(obs[i0])
    var_est = float(long_run_var)
    filtered[i0] = state
    variances[i0] = var_est

    for i in range(i0 + 1, n):
        evidence = obs[i] if finite[i] else 0.0
        drift = evidence * drift_scale
        state = state * (1 - leak_rate) + drift
        var_est = (
            var_est * (1 - leak_rate)
            + leak_rate * long_run_var
            + min(abs(drift) * 0.5, long_run_var * 0.5)
        )
        var_est = max(var_est, 1e-6)
        filtered[i] = state
        variances[i] = var_est

    return filtered, gains, variances
