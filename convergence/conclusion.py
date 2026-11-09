"""
Nishkarsh — the Conclusion layer (Convergence v2). "Two systems. One conclusion."
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

Pure and headless (no IO, no Streamlit, no hidden state): every quantity is
recomputed from the data in hand, so runs are idempotent.

  zA_t  = causal_z(head)_t        head = Aarambh Forecast (forward mode) or
                                  −ConvictionRaw (levels mode), on Valid rows only,
                                  over the ENTIRE Aarambh history. + = bullish.
  zN_t  = causal_z(−Avg_Signal)_t over Nirnay's own (warm) history; NaN where Nirnay
                                  has no data for the date (NEVER forward-filled).
  state ∈ {CONFIRM, CONTRADICT, NEUTRAL, NO_DATA}   (fixed deadband CONV2_TAU, not fitted)
  C_t   = 100·tanh(zA_t/2)·m_t,   m_t = κ_t on CONTRADICT days, else 1.  + = BULLISH.
  κ_t   — the ONLY fitted quantity: refit walk-forward every CONV2_REFIT_EVERY rows
          after CONV2_MIN_HISTORY rows of Nirnay overlap, on rows whose labels were
          realized before the refit date (s < T_k − h), and < 1 ONLY when that refit's
          Convergence-Lift test passes. ``kappa_enabled=False`` ⇒ κ ≡ 1 (pure Aarambh).

Nirnay can therefore only SHRINK conviction, and only after the data show that
Aarambh is more accurate when the two systems agree than when they contradict
(h-spaced non-overlapping samples, one-sided Fisher-z, minimum bucket sizes, the
80%-power minimum detectable effect disclosed). Every edge word the UI may print is
backed by a field of ``ConclusionResult.to_dict()`` (schema ``nishkarsh.conclusion/1``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _sstats

from analytics.backtest import backtest
from analytics.causal_stats import (
    causal_z,
    fisher_z,
    newey_west_t,
    norm_sf,
    spearman,
    stationary_block_bootstrap_ic,
)
from core.config import (
    AARAMBH_FWD_MOM_K,
    BACKTEST_COST_BPS,
    BACKTEST_REFERENCE_BPS,
    CONV2_DIV_CALM,
    CONV2_DIV_EXTREME,
    CONV2_DIV_PERSIST,
    CONV2_LIFT_ALPHA,
    CONV2_MIN_BUCKET,
    CONV2_MIN_HISTORY,
    CONV2_REFIT_EVERY,
    CONV2_TAU,
    CONV2_TIER_MODERATE,
    CONV2_TIER_STRONG,
    CONV2_TIER_WEAK,
    CONV2_ZCLIP,
    CONV2_ZMIN_PERIODS,
    EDGE_ALPHA,
    EDGE_TRADEABLE_T,
)
from convergence.labels import _clean_levels

SCHEMA = "nishkarsh.conclusion/1"
CONFIRM, CONTRADICT, NEUTRAL, NO_DATA = "CONFIRM", "CONTRADICT", "NEUTRAL", "NO_DATA"
STATES = (CONFIRM, CONTRADICT, NEUTRAL, NO_DATA)
AARAMBH_LEADS, NIRNAY_LEADS, CONTRADICTION = "AARAMBH_LEADS", "NIRNAY_LEADS", "CONTRADICTION"
DIVERGENCE_TYPES = (AARAMBH_LEADS, NIRNAY_LEADS, CONTRADICTION)
TRUST_TIERS = ("INVERTED", "NO_PROVEN_EDGE", "DIRECTIONAL_EDGE", "TRADEABLE_EDGE", "INSUFFICIENT_DATA")
FRAME_COLUMNS = ("zA", "zN", "state", "kappa", "lift_passed", "C", "C_aarambh", "tier")
EDGE_KEYS = ("signal", "ic", "ic_phase0", "n", "t", "p", "mde", "hac_t", "boot_ci", "significant",
             "phase_min", "phase_max", "yearly", "baseline_momentum_ic", "available")
BACKTEST_KEYS = ("gross_sharpe", "net_sharpe", "t", "ann_return", "turnover_per_year", "breakeven_bps", "n")
# Schema constant (the key is literally "same_sign_rate_60d"): window and its minimum coverage.
AGREEMENT_WINDOW = 60
AGREEMENT_MIN_OBS = AGREEMENT_WINDOW // 2
# Calendar-mismatch tolerance when aligning the label's closes onto Aarambh dates
# (an Aarambh date with no label close takes the previous close at most this many
# rows back; never beyond the label's last date).
PRICE_FFILL_LIMIT = 5
POWER_TARGET = 0.80
_STATE_WORD = {CONFIRM: "agrees", CONTRADICT: "disagrees", NEUTRAL: "is neutral"}


# ════════════════════════════════════════════════════════════════════════════
# Heads, alignment, agreement state
# ════════════════════════════════════════════════════════════════════════════

def _norm_dates(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce")).normalize()


def direction_head(aarambh_ts: pd.DataFrame, forward_mode: bool) -> pd.Series:
    """Aarambh's directional head on Valid rows with a finite value (+ = bullish).

    Forward mode: ``Forecast`` (forecast forward Δlog target). Levels mode:
    ``−ConvictionRaw`` (engine convention − = bullish, flipped)."""
    if aarambh_ts is None or len(aarambh_ts) == 0:
        return pd.Series(dtype="float64")
    a = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
    col = "Forecast" if forward_mode else "ConvictionRaw"
    if col not in a.columns:
        raise KeyError(f"run_conclusion: aarambh_ts lacks '{col}' (contract C-AAR, "
                       f"{'forward' if forward_mode else 'levels'} mode)")
    v = pd.to_numeric(a[col], errors="coerce").astype("float64")
    if not forward_mode:
        v = -v
    keep = np.isfinite(v.to_numpy())
    if "Valid" in a.columns:
        keep &= a["Valid"].fillna(False).astype(bool).to_numpy()
    out = v[keep]
    out.name = "head"
    return out


def nirnay_head(nirnay_daily: pd.DataFrame | None) -> pd.Series:
    """Nirnay's directional head ``−Avg_Signal`` on its own warm dates (+ = bullish,
    contrarian: an oversold average oscillator reads bullish). Normalized dates."""
    if nirnay_daily is None or len(nirnay_daily) == 0 or "Avg_Signal" not in nirnay_daily.columns:
        return pd.Series(dtype="float64")
    n = nirnay_daily.copy()
    n.index = _norm_dates(n.index)
    n = n[~n.index.isna()]
    n = n[~n.index.duplicated(keep="last")].sort_index()
    if "n_live" in n.columns:
        n = n[pd.to_numeric(n["n_live"], errors="coerce").fillna(0) > 0]
    v = -pd.to_numeric(n["Avg_Signal"], errors="coerce").astype("float64")
    return v[np.isfinite(v.to_numpy())]


def nirnay_z_on(index: pd.Index, nirnay_daily: pd.DataFrame | None) -> pd.Series:
    """zN over Nirnay's own history, mapped onto ``index`` by date — no fill."""
    nh = nirnay_head(nirnay_daily)
    if len(nh) == 0:
        return pd.Series(np.nan, index=index, dtype="float64")
    zn = causal_z(nh, CONV2_ZMIN_PERIODS, CONV2_ZCLIP)
    vals = zn.reindex(_norm_dates(index)).to_numpy()
    return pd.Series(vals, index=index, dtype="float64")


def align_log_price(price_levels: pd.Series | None, index: pd.Index) -> pd.Series:
    """log label price on ``index`` dates (causal: previous close ≤ PRICE_FFILL_LIMIT
    rows back; NaN outside the label's own date span)."""
    s = _clean_levels(price_levels)
    if s is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    lp = np.log(s)
    lp.index = lp.index.normalize()
    lp = lp[~lp.index.duplicated(keep="last")]
    key = _norm_dates(index)
    u = lp.index.union(key)
    al = lp.reindex(u).ffill(limit=PRICE_FFILL_LIMIT).reindex(key).to_numpy(dtype="float64", copy=True)
    al[(key > lp.index.max()) | (key < lp.index.min())] = np.nan
    return pd.Series(al, index=index, dtype="float64")


def agreement_state(zA: pd.Series, zN: pd.Series, tau: float = CONV2_TAU) -> pd.Series:
    """THE single agreement definition (used by C, the UI and divergence typing).

    NO_DATA if zN is NaN; CONFIRM if |zN| ≥ tau and sign(zN) = sign(zA); CONTRADICT if
    |zN| ≥ tau and the signs differ; NEUTRAL otherwise (incl. zA = 0). None where zA is
    NaN (no Aarambh conclusion yet)."""
    a = zA.to_numpy(dtype="float64")
    nz = zN.reindex(zA.index).to_numpy(dtype="float64")
    out = np.full(len(a), NEUTRAL, dtype=object)
    strong = np.abs(nz) >= tau
    sa, sn = np.sign(a), np.sign(nz)
    out[strong & (sa != 0) & (sa == sn)] = CONFIRM
    out[strong & (sa != 0) & (sa == -sn)] = CONTRADICT
    out[~np.isfinite(nz)] = NO_DATA
    out[~np.isfinite(a)] = None
    return pd.Series(out, index=zA.index, dtype=object)


def tier_of(c: float) -> str:
    c = abs(float(c)) if c is not None and np.isfinite(c) else 0.0
    if c >= CONV2_TIER_STRONG:
        return "STRONG"
    if c >= CONV2_TIER_MODERATE:
        return "MODERATE"
    if c > CONV2_TIER_WEAK:
        return "WEAK"
    return "NO LEAN"


def direction_of(c: float) -> str:
    if c is None or not np.isfinite(c):
        return "NEUTRAL"
    return "BULLISH" if c > CONV2_TIER_WEAK else "BEARISH" if c < -CONV2_TIER_WEAK else "NEUTRAL"


# ════════════════════════════════════════════════════════════════════════════
# Convergence-Lift test
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LiftResult:
    available: bool
    ic_confirm: float
    ic_contradict: float
    n_confirm: int
    n_contradict: int
    lift: float
    z: float
    p_one_sided: float          # H1: IC|CONFIRM > IC|CONTRADICT
    lift_phase_min: float
    lift_phase_max: float
    kappa: float                # shrink for CONTRADICT days implied by this test (1 unless passed)
    passed: bool
    mde: float                  # 80%-power minimum detectable Fisher-z lift at these bucket sizes


def _mde(n_c: int, n_k: int, alpha: float) -> float:
    if n_c <= 3 or n_k <= 3:
        return float("nan")
    se = np.sqrt(1.06 / (n_c - 3) + 1.06 / (n_k - 3))
    return float((_sstats.norm.ppf(1 - alpha) + _sstats.norm.ppf(POWER_TARGET)) * se)


def convergence_lift(zA: pd.Series, state: pd.Series, fwd_ret: pd.Series, h: int,
                     alpha: float = CONV2_LIFT_ALPHA, min_bucket: int = CONV2_MIN_BUCKET) -> LiftResult:
    """IC(zA, R_h | CONFIRM) − IC(zA, R_h | CONTRADICT) on h-spaced (non-overlapping)
    labelled rows, phase 0 from the first labelled row. One-sided Fisher-z test of two
    independent Spearman correlations; passes iff p < alpha, IC|CONFIRM > 0 and both
    buckets hold ≥ min_bucket windows. Lift range across all h phases is reported."""
    df = pd.DataFrame({"a": zA, "s": state, "r": fwd_ret})
    df = df[df["a"].notna() & df["r"].notna() & df["s"].notna()]

    def _at(ph: int):
        d = df.iloc[ph::h]
        c, k = d[d["s"] == CONFIRM], d[d["s"] == CONTRADICT]
        return spearman(c["a"], c["r"]), spearman(k["a"], k["r"]), len(c), len(k)

    ic_c, ic_k, n_c, n_k = _at(0)
    mde = _mde(n_c, n_k, alpha)
    nan = float("nan")
    if n_c < min_bucket or n_k < min_bucket or not (np.isfinite(ic_c) and np.isfinite(ic_k)):
        return LiftResult(False, ic_c, ic_k, n_c, n_k, nan, nan, nan, nan, nan, 1.0, False, mde)
    lifts = []
    for ph in range(h):
        a_, b_, *_ = _at(ph)
        if np.isfinite(a_) and np.isfinite(b_):
            lifts.append(a_ - b_)
    zc, vc = fisher_z(ic_c, n_c)
    zk, vk = fisher_z(ic_k, n_k)
    z = (zc - zk) / np.sqrt(vc + vk)
    p = norm_sf(z)
    passed = bool(p < alpha and ic_c > 0)
    kappa = float(np.clip(max(ic_k, 0.0) / ic_c, 0.0, 1.0)) if passed else 1.0
    return LiftResult(True, ic_c, ic_k, n_c, n_k, float(ic_c - ic_k), float(z), float(p),
                      float(np.min(lifts)) if lifts else nan, float(np.max(lifts)) if lifts else nan,
                      kappa, passed, mde)


# ════════════════════════════════════════════════════════════════════════════
# Walk-forward conclusion
# ════════════════════════════════════════════════════════════════════════════

def conclusion_walk_forward(zA: pd.Series, zN: pd.Series, fwd_ret: pd.Series, h: int, *,
                            kappa_enabled: bool = True, tau: float = CONV2_TAU,
                            refit_every: int = CONV2_REFIT_EVERY, min_history: int = CONV2_MIN_HISTORY,
                            alpha: float = CONV2_LIFT_ALPHA, min_bucket: int = CONV2_MIN_BUCKET,
                            ) -> tuple[pd.DataFrame, list[tuple[int, LiftResult]]]:
    """Per-date conclusion frame + the list of (refit row T_k, LiftResult).

    Refits at T_k = (first row with finite zN) + min_history, then every refit_every
    rows; refit k trains on rows s < T_k − h (labels realized before T_k) and its κ
    applies to rows [T_k, T_{k+1}). κ = 1 before the first refit, when a refit's test
    fails, and everywhere when ``kappa_enabled`` is False. Prefix-invariant."""
    idx = zA.index
    n = len(idx)
    zN = zN.reindex(idx)
    state = agreement_state(zA, zN, tau)
    kappa = np.ones(n)
    passed = np.zeros(n, dtype=bool)
    refits: list[tuple[int, LiftResult]] = []
    have_n = np.isfinite(zN.to_numpy(dtype="float64"))
    if have_n.any():
        k_start = int(np.argmax(have_n)) + int(min_history)
        for T in range(k_start, n, int(refit_every)):
            tr = slice(0, max(T - h, 0))
            res = convergence_lift(zA.iloc[tr], state.iloc[tr], fwd_ret.iloc[tr], h, alpha, min_bucket)
            kappa[T:T + refit_every] = res.kappa if kappa_enabled else 1.0
            passed[T:T + refit_every] = res.passed
            refits.append((T, res))
    st = state.to_numpy()
    m = np.where(st == CONTRADICT, kappa, 1.0)
    base = 100.0 * np.tanh(zA.to_numpy(dtype="float64") / 2.0)
    C = base * m
    frame = pd.DataFrame({"zA": zA.to_numpy(dtype="float64"), "zN": zN.to_numpy(dtype="float64"),
                          "state": st, "kappa": kappa, "lift_passed": passed,
                          "C": C, "C_aarambh": base}, index=idx)
    frame["state"] = pd.Series(st, index=idx, dtype=object)   # None where zA is NaN
    frame["tier"] = pd.Series([tier_of(c) for c in C], index=idx, dtype=object)
    return frame[list(FRAME_COLUMNS)], refits


# ════════════════════════════════════════════════════════════════════════════
# Divergence typing (one state machine, day-indexed persistence)
# ════════════════════════════════════════════════════════════════════════════

def divergence_events(frame: pd.DataFrame, tau: float = CONV2_TAU, extreme: float = CONV2_DIV_EXTREME,
                      calm: float = CONV2_DIV_CALM, persist: tuple[int, int] = CONV2_DIV_PERSIST) -> pd.DataFrame:
    """Typed divergence events on the conclusion frame (descriptive only).

    AARAMBH_LEADS: |zA| ≥ extreme and |zN| < tau.  NIRNAY_LEADS: |zN| ≥ extreme and
    |zA| < calm.  CONTRADICTION: state CONTRADICT, both |z| ≥ calm, and CONTRADICT on
    ≥ persist[0] of the last persist[1] TRADING DAYS (frame rows, today included).
    ``persistent``: the event's own condition held on ≥ persist[0] of the last
    persist[1] trading days (always True for CONTRADICTION, by definition).
    No event on NO_DATA days (zN missing) or where zA is missing."""
    cols = ["type", "zA", "zN", "persistent"]
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="Date"))
    k_min, win = int(persist[0]), int(persist[1])
    a, z = frame["zA"].astype("float64"), frame["zN"].astype("float64")
    have = a.notna() & z.notna()
    al = have & (a.abs() >= extreme) & (z.abs() < tau)
    nl = have & (z.abs() >= extreme) & (a.abs() < calm)
    is_k = (frame["state"] == CONTRADICT)
    k_cnt = is_k.astype(float).rolling(win, min_periods=1).sum()
    ct = have & is_k & (a.abs() >= calm) & (z.abs() >= calm) & (k_cnt >= k_min)
    al_p = al.astype(float).rolling(win, min_periods=1).sum() >= k_min
    nl_p = nl.astype(float).rolling(win, min_periods=1).sum() >= k_min
    typ = pd.Series(None, index=frame.index, dtype=object)
    typ[al] = AARAMBH_LEADS
    typ[nl] = NIRNAY_LEADS
    typ[ct] = CONTRADICTION
    pers = (al & al_p) | (nl & nl_p) | ct
    ev = pd.DataFrame({"type": typ, "zA": a, "zN": z, "persistent": pers.astype(bool)})
    ev = ev[typ.notna()]
    ev.index.name = "Date"
    return ev[cols]


# ════════════════════════════════════════════════════════════════════════════
# Edge evidence
# ════════════════════════════════════════════════════════════════════════════

from scipy.stats import norm as _sstats_norm
_sstats_norm_ppf = _sstats_norm.ppf


def _hspaced_ic(sig: pd.Series, fwd_ret: pd.Series, h: int) -> tuple[float, int]:
    df = pd.DataFrame({"s": sig, "r": fwd_ret}).dropna()
    d = df.iloc[::h]
    return spearman(d["s"], d["r"]), int(len(d))


def ic_evidence(signal: pd.Series, fwd_ret: pd.Series, h: int, *, signal_name: str,
                momentum: pd.Series | None = None, alpha: float = EDGE_ALPHA,
                min_n: int = CONV2_MIN_BUCKET, bootstrap: bool = True) -> dict:
    """Directional evidence of ``signal`` (+ = bullish) for the h-row forward return.

    ``ic``: mean of the h phase-wise Spearman ICs (each on h-spaced, non-overlapping
    rows starting at offset k = 0..h−1), so every labelled row is used; ``ic_phase0``
    is the offset-0 IC alone. ``n``: windows per phase (offset 0). ``t``/``p``: Fisher
    t of ``ic`` with the SINGLE-phase variance 1.06/(n−3) — conservative, because the
    mean of h equally variable phase ICs has variance ≤ that of any one of them —
    one-sided p (H1: IC > 0). (Red-team T2: scoring one offset alone discarded
    (h−1)/h of the data and made the verdict hinge on an arbitrary start date.)
    ``hac_t``: Newey–West(h) t of the DAILY rank IC (mean of standardized rank
    products over all labelled rows). ``boot_ci``: 95% stationary-block-bootstrap CI
    of the daily rank IC (mean block 2h, seeded). ``phase_min/max``: IC range across
    the h start offsets. ``yearly``: daily rank IC per calendar year (years with
    ≥ min_n labelled rows). ``baseline_momentum_ic``: same h-spaced rows, the label's
    own trailing momentum as the signal. ``available``: n ≥ min_n and IC finite."""
    df = pd.DataFrame({"s": signal, "r": fwd_ret})
    if momentum is not None:
        df["m"] = momentum
    df = df[df["s"].notna() & df["r"].notna()]
    d = df.iloc[::h]
    ic0 = spearman(d["s"], d["r"])
    n = int(len(d))
    ph_all = [spearman(df["s"].iloc[k::h], df["r"].iloc[k::h]) for k in range(h)] if len(df) else []
    ph_all = [x for x in ph_all if np.isfinite(x)]
    ic = float(np.mean(ph_all)) if len(ph_all) == h else ic0
    available = bool(n >= min_n and np.isfinite(ic))
    zf, vf = fisher_z(ic, n)
    t = zf / np.sqrt(vf) if np.isfinite(vf) and vf > 0 else float("nan")
    p = norm_sf(t)
    hac_t = float("nan")
    boot = None
    phase_min = phase_max = float("nan")
    yearly: dict[str, float] = {}
    if len(df) >= max(min_n, h + 5):
        rs = df["s"].rank().to_numpy(dtype="float64", copy=True)
        rr = df["r"].rank().to_numpy(dtype="float64", copy=True)
        rs = (rs - rs.mean()) / rs.std()
        rr = (rr - rr.mean()) / rr.std()
        hac_t = newey_west_t(rs * rr, lags=h)
        if bootstrap:
            lo, hi = stationary_block_bootstrap_ic(df["s"], df["r"], mean_block=2 * h, alpha=alpha,
                                                   min_obs=min_n)
            boot = [lo, hi] if np.isfinite(lo) and np.isfinite(hi) else None
        if ph_all:
            phase_min, phase_max = float(min(ph_all)), float(max(ph_all))
        years = pd.DatetimeIndex(df.index).year
        for y, g in df.groupby(years):
            if len(g) >= min_n:
                v = spearman(g["s"], g["r"])
                if np.isfinite(v):
                    yearly[str(int(y))] = float(v)
    base_ic = float("nan")
    if momentum is not None and "m" in df.columns:
        dm = df[df["m"].notna()]
        bph = [spearman(dm["m"].iloc[k::h], dm["r"].iloc[k::h]) for k in range(h)] if len(dm) else []
        bph = [x for x in bph if np.isfinite(x)]
        base_ic = float(np.mean(bph)) if bph else float("nan")   # same phase-averaged estimator
    # Power disclosure: the smallest IC this test detects with 80% power at one-sided
    # alpha, given n windows (Fisher-z scale) — a "no proven edge" verdict below this
    # is "not enough data", not "no edge".
    mde = float(np.tanh((_sstats_norm_ppf(1.0 - alpha) + _sstats_norm_ppf(0.80)) * np.sqrt(1.06 / (n - 3)))) if n > 3 else float("nan")
    return {"signal": signal_name, "ic": ic, "ic_phase0": ic0, "n": n, "t": t, "p": p, "mde": mde,
            "hac_t": hac_t, "boot_ci": boot,
            "significant": bool(available and np.isfinite(p) and p < alpha and ic > 0),
            "phase_min": phase_min, "phase_max": phase_max, "yearly": yearly,
            "baseline_momentum_ic": base_ic, "available": available}


def _bt_view(b: dict) -> dict:
    return {k: b.get(k) for k in BACKTEST_KEYS}


# ════════════════════════════════════════════════════════════════════════════
# Read-outs (pure text; the hero only renders them)
# ════════════════════════════════════════════════════════════════════════════

def _fmt(x, spec: str) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return format(v, spec) if np.isfinite(v) else "n/a"


def trust_readout(graded: dict, basis: str, bt_ref: dict, bt_nonoverlap: dict, h: int,
                  ref_bps: float = BACKTEST_REFERENCE_BPS) -> dict:
    """Trust tier of the DISPLAYED series at horizon h (see C-V2 / A3 v2-6)."""
    num = (f"{basis} IC {_fmt(graded.get('ic'), '+.3f')} (t {_fmt(graded.get('t'), '.1f')}, "
           f"n {int(graded.get('n') or 0)}, p {_fmt(graded.get('p'), '.3f')})")
    bt = (f"net Sharpe {_fmt(bt_ref.get('net_sharpe'), '+.2f')} @{ref_bps:g}bps "
          f"(HAC t {_fmt(bt_ref.get('t'), '.1f')}; non-overlap t {_fmt(bt_nonoverlap.get('t'), '.1f')})")
    t = graded.get("t")
    t = float(t) if t is not None and np.isfinite(t) else float("nan")
    if not graded.get("available"):
        return {"tier": "INSUFFICIENT_DATA",
                "text": (f"INSUFFICIENT DATA — {int(graded.get('n') or 0)} non-overlapping {h}d windows "
                         f"(need ≥ {CONV2_MIN_BUCKET}); no edge can be graded.")}
    ic = float(graded["ic"])
    if ic < 0 and np.isfinite(t) and _sstats.norm.cdf(t) < EDGE_ALPHA:
        return {"tier": "INVERTED",
                "text": f"INVERTED — {num}: the signal has pointed the wrong way; no directional call."}
    if not graded.get("significant"):
        return {"tier": "NO_PROVEN_EDGE",
                "text": (f"NO PROVEN EDGE — {num}; direction is unvalidated "
                         f"(this sample can only detect IC ≥ {_fmt(graded.get('mde'), '.2f')} reliably).")}
    t_tr, t_no = bt_ref.get("t"), bt_nonoverlap.get("t")
    if (t_tr is not None and t_no is not None and t_tr >= EDGE_TRADEABLE_T and t_no >= EDGE_TRADEABLE_T):
        return {"tier": "TRADEABLE_EDGE", "text": f"TRADEABLE EDGE — {num} · {bt}."}
    return {"tier": "DIRECTIONAL_EDGE",
            "text": f"DIRECTIONAL EDGE — {num} · {bt}: not tradeable net of costs."}


def nirnay_readout(single_system: bool, today_state, lift: LiftResult, kappa_today: float,
                   kappa_enabled: bool, oos: dict) -> dict:
    """Exact Nirnay line (A3 v2-6). 'CONFIRMS' / 'confirming' appear ONLY when the
    latest refit's lift test passed AND the out-of-sample condition holds."""
    if single_system:
        return {"state": NO_DATA,
                "text": "Single-system mode (Nirnay unavailable) — conclusion is Aarambh-only; Convergence Lift N/A."}
    if today_state not in (CONFIRM, CONTRADICT, NEUTRAL):
        return {"state": NO_DATA, "text": "Nirnay: no data for this date — conclusion is Aarambh-only."}
    w = _STATE_WORD[today_state]
    L, p = _fmt(lift.lift, "+.3f"), lift.p_one_sided
    if not lift.available:
        txt = (f"Nirnay {w} — context only: no measured lift (only {lift.n_confirm}/{lift.n_contradict} "
               f"agree/disagree windows; need ≥ {CONV2_MIN_BUCKET} each). Conviction is Aarambh's.")
    elif not lift.passed:
        txt = (f"Nirnay {w} — context only: agreement has not been shown to improve Aarambh's accuracy "
               f"(lift {L}, p {_fmt(p, '.2f')}; detectable lift ≥ {_fmt(lift.mde, '.2f')}). "
               f"Conviction is Aarambh's.")
    elif not kappa_enabled:
        txt = (f"Nirnay {w} — lift {L} (p {_fmt(p, '.3f')}) passed, but Intelligence Mode is OFF "
               f"(κ ≡ 1). Conviction is Aarambh's.")
    elif today_state == CONFIRM and oos.get("adds_value"):
        txt = (f"Nirnay CONFIRMS — IC {_fmt(lift.ic_confirm, '+.3f')} when confirming vs "
               f"{_fmt(lift.ic_contradict, '+.3f')} when contradicting (lift {L}, p {_fmt(p, '.3f')}); "
               f"out-of-sample IC {_fmt(oos.get('ic_C'), '+.3f')} vs Aarambh-only {_fmt(oos.get('ic_A'), '+.3f')}.")
    elif today_state == CONFIRM:
        txt = (f"Nirnay agrees — lift {L} (p {_fmt(p, '.3f')}) passed in-sample, but out-of-sample IC "
               f"has not improved (C {_fmt(oos.get('ic_C'), '+.3f')} vs Aarambh {_fmt(oos.get('ic_A'), '+.3f')}). "
               f"Conviction is Aarambh's.")
    elif today_state == CONTRADICT:
        txt = f"Nirnay CONTRADICTS — conviction ×{_fmt(kappa_today, '.2f')} (measured lift {L}, p {_fmt(p, '.3f')})."
    else:
        txt = f"Nirnay is neutral (|zN| < {CONV2_TAU:g}) — conviction is Aarambh's."
    if lift.available and lift.passed and np.isfinite(lift.lift_phase_min) and lift.lift_phase_min <= 0:
        txt += (f" Phase-unstable: lift ranges {_fmt(lift.lift_phase_min, '+.3f')}…"
                f"{_fmt(lift.lift_phase_max, '+.3f')} across start offsets.")
    return {"state": today_state, "text": txt}


# ════════════════════════════════════════════════════════════════════════════
# JSON safety
# ════════════════════════════════════════════════════════════════════════════

def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    if o is None or isinstance(o, str):
        return o
    if o is pd.NA or o is pd.NaT:
        return None
    return str(o)


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ConclusionResult:
    frame: pd.DataFrame
    events: pd.DataFrame
    payload: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """JSON-safe summary (schema ``nishkarsh.conclusion/1``); floats finite or None."""
        return copy.deepcopy(self.payload)


def _lift_dict(L: LiftResult) -> dict:
    return {"available": L.available, "ic_confirm": L.ic_confirm, "ic_contradict": L.ic_contradict,
            "n_confirm": L.n_confirm, "n_contradict": L.n_contradict, "lift": L.lift, "z": L.z,
            "p_one_sided": L.p_one_sided, "lift_phase_min": L.lift_phase_min,
            "lift_phase_max": L.lift_phase_max, "kappa": L.kappa, "passed": L.passed, "mde": L.mde}


def run_conclusion(aarambh_ts: pd.DataFrame, nirnay_daily: pd.DataFrame | None, price_levels: pd.Series | None,
                   *, forward_mode: bool, h: int, kappa_enabled: bool = True, label_src: str = "") -> ConclusionResult:
    """Compute THE conclusion and its evidence (contract C-V2).

    aarambh_ts: engine.ts_data (DatetimeIndex; Valid, Forecast | ConvictionRaw, ConvictionBounded).
    nirnay_daily: C-NIR daily frame (absent dates = NO_DATA; may be empty → single-system mode).
    price_levels: level series of the resolved return label (``convergence.labels``).
    """
    h = int(h)
    if h < 1:
        raise ValueError("run_conclusion: h must be >= 1")
    head = direction_head(aarambh_ts, forward_mode)
    head = head[~pd.isna(head.index)]
    zA = causal_z(head, CONV2_ZMIN_PERIODS, CONV2_ZCLIP)
    zA.name = "zA"
    idx = zA.index
    zN = nirnay_z_on(idx, nirnay_daily)
    single_system = not bool(np.isfinite(zN.to_numpy()).any())
    logp = align_log_price(price_levels, idx)
    R = logp.shift(-h) - logp
    mom = logp - logp.shift(int(AARAMBH_FWD_MOM_K))

    frame, refits = conclusion_walk_forward(zA, zN, R, h, kappa_enabled=kappa_enabled)
    events = divergence_events(frame)
    C, CA = frame["C"], frame["C_aarambh"]

    # ── Edge of Aarambh's head over its FULL valid history ──────────────────
    edge = ic_evidence(zA, R, h, signal_name="aarambh_head", momentum=mom)
    # ── Edge of the displayed series C, out-of-sample from the first κ refit ──
    first_T = refits[0][0] if refits else 0
    oos_rows = idx[first_T:]
    edge_C = ic_evidence(C.loc[oos_rows], R.loc[oos_rows], h, signal_name="C" if refits else "C (all rows; κ never refit)",
                         momentum=mom.loc[oos_rows])

    # ── Lift (latest refit = the test in force today) ───────────────────────
    if refits:
        lift = refits[-1][1]
    else:
        full = convergence_lift(zA, frame["state"], R, h)
        nan = float("nan")
        lift = LiftResult(False, nan, nan, full.n_confirm, full.n_contradict, nan, nan, nan, nan, nan,
                          1.0, False, full.mde)

    # ── OOS read-out: C vs Aarambh-only on h-spaced rows from the first refit ─
    if refits:
        d = pd.DataFrame({"C": C, "A": CA, "r": R}).iloc[first_T:].dropna().iloc[::h]
        ic_C, ic_A = spearman(d["C"], d["r"]), spearman(d["A"], d["r"])
        differs = bool((d["C"] != d["A"]).any())
        adds = bool(kappa_enabled and lift.passed and np.isfinite(ic_C) and np.isfinite(ic_A)
                    and ic_C >= ic_A and differs)
        oos = {"ic_C": ic_C, "ic_A": ic_A, "n": int(len(d)), "adds_value": adds}
    else:
        oos = {"ic_C": None, "ic_A": None, "n": 0, "adds_value": False}

    # ── Cost-aware backtest of the displayed series C ───────────────────────
    live = C.notna()
    pos = (C[live] / 100.0)
    by_cost = {str(float(c)): _bt_view(backtest(pos, logp[live], h, c, "tranche")) for c in BACKTEST_COST_BPS}
    ref_key = str(float(BACKTEST_REFERENCE_BPS))
    nonov_ref = _bt_view(backtest(pos, logp[live], h, BACKTEST_REFERENCE_BPS, "nonoverlap"))

    # ── Signal chain ───────────────────────────────────────────────────────
    # (all stages on the SAME rows: where every stage is finite, h-spaced)
    stages = (("forecast" if forward_mode else "valuation_head", head.reindex(idx)),
              ("engine_conviction", _engine_conviction(aarambh_ts, idx)), ("C", C))
    common = R.notna()
    for _, sig in stages:
        common &= sig.notna()
    chain = []
    for stage, sig in stages:
        ic_s, n_s = _hspaced_ic(sig[common], R[common], h)
        chain.append({"stage": stage, "ic": ic_s, "n": n_s})

    # ── Trust tier: graded on the DISPLAYED series ─────────────────────────
    c_equiv = not bool(((C != CA) & C.notna() & CA.notna()).any())
    graded, basis = (edge, "Aarambh head") if c_equiv else (edge_C, "Conclusion C (out-of-sample)")
    trust = trust_readout(graded, basis, by_cost.get(ref_key, {}), nonov_ref, h)

    # ── Latest / agreement / divergence ────────────────────────────────────
    if len(frame):
        last = frame.iloc[-1]
        today_state = last["state"]
        latest = {"date": pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d"), "C": last["C"],
                  "C_aarambh": last["C_aarambh"], "zA": last["zA"], "zN": last["zN"], "state": today_state,
                  "kappa": last["kappa"], "tier": tier_of(last["C"]), "direction": direction_of(last["C"])}
        tail = frame.iloc[-AGREEMENT_WINDOW:]
        both = tail["zA"].notna() & tail["zN"].notna()
        rate = (float((np.sign(tail.loc[both, "zA"]) == np.sign(tail.loc[both, "zN"])).mean())
                if int(both.sum()) >= AGREEMENT_MIN_OBS else None)
        win = int(CONV2_DIV_PERSIST[1])
        recent = events[events.index.isin(frame.index[-win:])]
        div_recent = {t: int((recent["type"] == t).sum()) for t in DIVERGENCE_TYPES}
    else:
        today_state, rate, win = None, None, int(CONV2_DIV_PERSIST[1])
        latest = {"date": None, "C": None, "C_aarambh": None, "zA": None, "zN": None, "state": None,
                  "kappa": 1.0, "tier": "NO LEAN", "direction": "NEUTRAL"}
        div_recent = {t: 0 for t in DIVERGENCE_TYPES}
    div_recent["window_days"] = win
    nirnay = nirnay_readout(single_system, today_state, lift, latest["kappa"], kappa_enabled, oos)

    payload = {
        "schema": SCHEMA, "as_of": latest["date"], "horizon": h, "label_src": str(label_src),
        "forward_mode": bool(forward_mode), "kappa_enabled": bool(kappa_enabled), "single_system": single_system,
        "latest": latest,
        "edge": {k: edge[k] for k in EDGE_KEYS},
        "edge_C": {k: edge_C[k] for k in EDGE_KEYS},
        "lift": _lift_dict(lift),
        "oos": oos,
        "backtest": {"mode": "tranche", "reference_bps": float(BACKTEST_REFERENCE_BPS), "by_cost": by_cost,
                     "nonoverlap_reference": nonov_ref},
        "chain": chain,
        "trust": trust,
        "nirnay": nirnay,
        "agreement": {"today_state": today_state, "same_sign_rate_60d": rate},
        "divergence_recent": div_recent,
    }
    return ConclusionResult(frame=frame, events=events, payload=_json_safe(payload))


def _engine_conviction(aarambh_ts: pd.DataFrame, idx: pd.Index) -> pd.Series:
    """Engine conviction flipped to + = bullish (−ConvictionBounded, else −ConvictionScore)."""
    a = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
    for col in ("ConvictionBounded", "ConvictionScore"):
        if col in a.columns:
            return -pd.to_numeric(a[col], errors="coerce").astype("float64").reindex(idx)
    return pd.Series(np.nan, index=idx, dtype="float64")


# ════════════════════════════════════════════════════════════════════════════
# Agreement Oscillator — the signed read of how much, and on which side, the two
# systems agree (docs/research/AGREEMENT_OSCILLATOR_*.md). Measured on every run;
# its status is VALIDATED only when the pre-registered H1 rule passes on the data
# in hand — until then it is a context gauge with no buy/sell wording.
# ════════════════════════════════════════════════════════════════════════════

OSCILLATOR_STATUSES = ("VALIDATED", "UNPROVEN", "INSUFFICIENT_DATA")


def oscillator_series(frame: pd.DataFrame, anatomy: pd.DataFrame | None) -> pd.DataFrame:
    """``agreement_ratio`` (anatomy, [0,1]), ``consensus`` (+1 both lean bullish, −1 both
    bearish, 0 split; NaN without Nirnay data) and ``SAO = consensus · agreement_ratio``
    on the conclusion frame's dates. Causal: every input is causal at its date."""
    out = pd.DataFrame(index=frame.index)
    agr = (anatomy["agreement_ratio"].reindex(frame.index) if anatomy is not None and len(anatomy)
           and "agreement_ratio" in anatomy.columns else pd.Series(np.nan, index=frame.index))
    za, zn = frame["zA"], frame["zN"]
    la, ln = np.sign(za), np.sign(zn)
    cons = pd.Series(np.where((la == ln) & (la != 0), la, 0.0), index=frame.index)
    cons[za.isna() | zn.isna()] = np.nan
    out["agreement_ratio"] = agr
    out["consensus"] = cons
    out["SAO"] = cons * agr
    return out


def oscillator_readout(sao: pd.Series, price_levels: pd.Series | None, h: int,
                       alpha: float = EDGE_ALPHA) -> dict:
    """Evidence and status of the Signed Agreement Oscillator at horizon ``h``.

    VALIDATED ⇔ one-sided p < alpha (IC > 0, phase-averaged h-spaced, conservative
    single-phase variance) AND the block-bootstrap CI excludes 0 AND ≥ 60% of calendar
    years show a positive IC — the pre-registered H1 rule."""
    s = sao.dropna()
    latest = float(s.iloc[-1]) if len(s) else None
    base = {"value": latest, "as_of": str(s.index[-1].date()) if len(s) else None,
            "side": ("BUY-SIDE" if latest and latest > 0 else "SELL-SIDE" if latest and latest < 0 else "SPLIT")
            if latest is not None else None,
            "strength": abs(latest) if latest is not None else None}
    logp = align_log_price(price_levels, sao.index) if price_levels is not None and len(price_levels) else None
    if logp is None or s.empty:
        return base | {"status": "INSUFFICIENT_DATA", "evidence": None,
                       "text": "Agreement Oscillator: not enough two-system data to measure it."}
    e = ic_evidence(sao, logp.shift(-h) - logp, h, signal_name="SAO", alpha=alpha)
    yearly = e.get("yearly") or {}
    yr_pos = (sum(v > 0 for v in yearly.values()) / len(yearly)) if yearly else None
    boot = e.get("boot_ci")
    if not e.get("available"):
        status = "INSUFFICIENT_DATA"
    elif (np.isfinite(e["p"]) and e["p"] < alpha and boot and boot[0] > 0 and yr_pos is not None and yr_pos >= 0.6):
        status = "VALIDATED"
    else:
        status = "UNPROVEN"
    num = (f"IC {_fmt(e.get('ic'), '+.3f')} over {int(e.get('n') or 0)} non-overlapping {h}-day windows "
           f"(p {_fmt(e.get('p'), '.2f')}; positive in {int(round((yr_pos or 0) * len(yearly)))}/{len(yearly)} years)")
    if status == "VALIDATED":
        text = f"Validated directional read — {num}."
    elif status == "UNPROVEN":
        text = (f"Not yet a validated buy/sell signal — {num}; this sample can only confirm an IC of "
                f"{_fmt(e.get('mde'), '.2f')} or more.")
    else:
        text = f"Not enough two-system history to measure the oscillator ({int(e.get('n') or 0)} windows)."
    ev = {k: e.get(k) for k in ("ic", "n", "p", "mde", "hac_t", "boot_ci", "yearly")}
    ev["yearly_positive_share"] = yr_pos
    return base | {"status": status, "evidence": ev, "text": text}


def attach_oscillator(result: ConclusionResult, anatomy: pd.DataFrame | None,
                      price_levels: pd.Series | None, h: int) -> ConclusionResult:
    """Add the oscillator columns to ``result.frame`` and ``"oscillator"`` to its payload."""
    osc = oscillator_series(result.frame, anatomy)
    for c in osc.columns:
        result.frame[c] = osc[c]
    result.payload["oscillator"] = _json_safe(oscillator_readout(osc["SAO"], price_levels, h))
    return result
