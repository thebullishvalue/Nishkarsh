"""
Nishkarsh — Nirnay Engine: per-stock MSF + MMR with regime intelligence.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

NIRNAY — per-constituent MSF + MMR oscillators with a Kalman/HMM/GARCH/CUSUM
regime ensemble, aggregated into daily breadth over the LIVE (real-bar,
warmed-up) names only. Everything is causal: a row's values depend on that
stock's bars up to and including that row.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from analytics.regime import run_regime_loop
from analytics.utils import (
    calculate_atr,
    causal_gram_schmidt_orthogonalize,
    sigmoid,
    zscore_clipped,
)
from core.config import (
    NIRNAY_HMM_STD_FLOOR,
    NIRNAY_OVERBOUGHT,
    NIRNAY_OVERSOLD,
    NIRNAY_WARMUP_BARS,
)


# ─── Market Strength Factor ──────────────────────────────────────────────────


def calculate_msf(
    df: pd.DataFrame,
    length: int = 20,
    roc_len: int = 14,
    clip: float = 3.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Market Strength Factor from OHLCV data.

    Combines four components that are **explicitly orthogonalized via
    Gram-Schmidt** before being summed, so the composite does not double-count
    the inputs they share (momentum, trend and vol-adjusted momentum are all
    functions of ``close.diff(5)``):

    - **Momentum**: Rate of change z-score (primary; preserved unchanged)
    - **Trend**: Multi-timeframe composite (trend diff + momentum accel
      + volume-adjusted momentum + mean reversion), less its momentum overlap
    - **Microstructure**: Volume-weighted direction vs impact, residualized
    - **Flow**: Accumulation/distribution ratio + volatility-adaptive regime
      counting, residualized

    Each component's significant-bar and band logic is volatility-adaptive
    (no fixed percentage cut-offs).

    Returns
    -------
    msf_signal, micro_norm, momentum_norm, flow_norm
        The diagnostic component series returned are the *pre*-orthogonalization
        values (for display); ``msf_signal`` is the orthogonalized composite.
        ``flow_norm`` is the flow component that enters the composite
        (accumulation ratio + volatility-adaptive regime count, /√2).
    """
    close = df["Close"]

    # Momentum
    roc_raw = close.pct_change(roc_len, fill_method=None)
    roc_z = zscore_clipped(roc_raw, length, clip)
    momentum_norm = sigmoid(roc_z, 1.5)

    # Microstructure
    intrabar_dir = (df["High"] + df["Low"]) / 2 - df["Open"]
    vol_ma = df["Volume"].rolling(length).mean()
    vol_ratio = (df["Volume"] / vol_ma).fillna(1.0)
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    micro_raw = vw_direction - vw_impact
    micro_z = zscore_clipped(micro_raw, length, clip)
    micro_norm = sigmoid(micro_z, 1.5)

    # Trend
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, length, clip)
    mom_accel_raw = close.diff(5).diff(5)
    mom_accel_z = zscore_clipped(mom_accel_raw, length, clip)
    atr = calculate_atr(df, 14)
    vol_adj_mom_raw = close.diff(5) / atr
    vol_adj_mom_z = zscore_clipped(vol_adj_mom_raw, length, clip)
    mean_rev_z = zscore_clipped(close - trend_slow, length, clip)
    composite_trend_z = (
        trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z
    ) / np.sqrt(4.0)
    composite_trend_norm = sigmoid(composite_trend_z, 1.5)

    # Flow
    typical_price = (df["High"] + df["Low"] + close) / 3
    mf = typical_price * df["Volume"]
    mf_pos = np.where(close > close.shift(1), mf, 0)
    mf_neg = np.where(close < close.shift(1), mf, 0)
    mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(length).mean()
    mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(length).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
    accum_ratio = accum_ratio.fillna(0.5)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    pct_change = close.pct_change(fill_method=None)
    # Significant-bar threshold is volatility-adaptive, not a fixed 0.33%: a bar
    # counts as up/down only if it exceeds a quarter of the stock's own recent
    # daily sigma (causal, shifted). This adapts per-name and per-regime instead
    # of applying one hardcoded percentage to every constituent.
    # No bfill / full-series-std fill on the warm-up: both would leak future
    # volatility into the first bars' significant-bar cut. Warm-up rows fall
    # through to bar_cut = 0.0 (any nonzero move counts) until 5 real bars of
    # realized vol exist — a neutral prior, not a look-ahead.
    bar_vol = (
        pct_change.rolling(length, min_periods=5).std().shift(1)
    )
    bar_vol = bar_vol.replace(0, np.nan)
    bar_cut = (0.25 * bar_vol).fillna(0.0)
    regime_signals = np.select(
        [pct_change > bar_cut, pct_change < -bar_cut], [1, -1], default=0
    )
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_z = zscore_clipped(regime_raw, length, clip)
    regime_norm = sigmoid(regime_z, 1.5)

    # Combine — causal Gram-Schmidt orthogonalize the four components first.
    # They share inputs (momentum, trend and vol-adjusted momentum are all
    # functions of close.diff(5)), so a plain weighted sum double-counts that
    # overlap. The causal variant (a) estimates the decorrelation basis from
    # past rows only — historical MSF is backtest-safe — and (b) keeps each
    # residual at its natural magnitude, so a redundant component contributes
    # little instead of having its noise rescaled to full weight.
    flow_norm = (accum_norm + regime_norm) / np.sqrt(2.0)
    comp_matrix = np.column_stack([
        momentum_norm.fillna(0.0).to_numpy(),          # primary: raw momentum
        composite_trend_norm.fillna(0.0).to_numpy(),   # trend, less the momentum overlap
        micro_norm.fillna(0.0).to_numpy(),             # microstructure residual
        flow_norm.fillna(0.0).to_numpy(),              # flow residual
    ])
    ortho = causal_gram_schmidt_orthogonalize(comp_matrix)
    msf_raw = ortho.sum(axis=1)
    msf_signal = pd.Series(sigmoid(msf_raw, 1.0), index=df.index)

    return msf_signal, micro_norm, momentum_norm, flow_norm


# ─── Macro-Micro Regime ──────────────────────────────────────────────────────


def calculate_mmr(
    df: pd.DataFrame,
    length: int = 20,
    num_vars: int = 5,
    macro_columns: list[str] | None = None,
) -> tuple[pd.Series, list[dict[str, Any]], pd.Series]:
    """Calculate Macro-Micro Regime via rolling R²-weighted level fits.

    For every macro column a 20-bar (``length``) univariate level fit of Close
    on the macro is estimated from rows < t (slope = r·σy/σx, intercept from
    the prior-window means); at each row the ``num_vars`` fits with the
    highest prior r² are blended (r²-weighted) into a fitted level ŷ_t. The
    MMR signal is the causal z of ``Close − ŷ`` squashed to [-1, 1].

    ``mmr_quality`` = √max(0, 1 − MSE(Close − ŷ) / MSE(ΔClose)) over the
    trailing ``length`` rows: the skill of the "20d level-fit vs RW" (random
    walk "today ≈ yesterday"), MSE-based so a biased fit is not scored as
    perfect (A2-07). 0 = no better than a random walk.

    Returns
    -------
    mmr_signal, driver_details, mmr_quality
    """
    if macro_columns is None:
        macro_columns = []
    available_macros = [v for v in macro_columns if v in df.columns]
    target = df["Close"]

    if len(df) < length + 10 or not available_macros:
        return (pd.Series(0.0, index=df.index), [], pd.Series(0.0, index=df.index))

    # No bfill on any shifted rolling stat: backfilling the warm-up rows with
    # the first post-warm-up value fills them from a FUTURE window (for
    # roll_corr that was ~`length` rows of look-ahead). Warm-up rows stay NaN
    # and fall through to the y_mean fallback in the driver selection below.
    y_mean = target.rolling(length, min_periods=1).mean().shift(1)
    y_std = target.rolling(length, min_periods=1).std().shift(1)

    preds_list = []
    r2_list = []

    # Per-driver causal rolling level fits (coefficients from rows < t)
    for ticker in available_macros:
        x = df[ticker].ffill().fillna(0)
        x_mean = x.rolling(length, min_periods=1).mean().shift(1)
        x_std = x.rolling(length, min_periods=1).std().shift(1)

        # Pearson correlation shifted (only prior data used to estimate the
        # relationship). rolling.corr emits ±inf (not NaN) on near-zero-
        # variance windows — routine with pegged/constant series (e.g.
        # LIQUIDBEES.NS) or forward-filled holiday runs — and catastrophic
        # cancellation can also yield finite |r| > 1. Both are numerically
        # meaningless for a Pearson r and, squared, +inf would ALWAYS win the
        # top-N driver argsort below (it passed the old ~isnan mask), letting
        # an information-free column hijack a driver slot. Sanitize at source:
        # non-finite → NaN (no information), then clip to the valid range.
        roll_corr = (
            x.rolling(length, min_periods=length).corr(target)
             .replace([np.inf, -np.inf], np.nan)
             .clip(-1.0, 1.0)
             .shift(1)
             .fillna(0)
        )
        slope = roll_corr * (y_std / x_std.replace(0, np.nan)).fillna(0)
        intercept = y_mean - (slope * x_mean)

        pred = (slope * x) + intercept
        r2 = roll_corr**2

        preds_list.append(pred)
        r2_list.append(r2)

    all_preds_arr = np.column_stack([p.to_numpy(dtype=np.float64) for p in preds_list])
    all_r2_arr = np.column_stack([r.to_numpy(dtype=np.float64) for r in r2_list])

    # Causally select the top `num_vars` drivers per row (vectorized). A driver
    # is eligible only if both its r² and its prediction are finite (an inf r²
    # from a pegged column would otherwise always win). Ineligible entries are
    # masked to −inf so a STABLE argsort of the full row reproduces the stable
    # argsort of the compacted eligible vector exactly (ties → lower column
    # index first), then the last `num_vars` are the top drivers.
    valid = np.isfinite(all_r2_arr) & np.isfinite(all_preds_arr)
    masked = np.where(valid, all_r2_arr, -np.inf)
    top = np.argsort(masked, axis=1, kind="stable")[:, -num_vars:]
    r2_sel = np.take_along_axis(all_r2_arr, top, axis=1)
    preds_sel = np.take_along_axis(all_preds_arr, top, axis=1)
    y_mean_arr = y_mean.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        r2_sum = r2_sel.sum(axis=1)
        ok = (valid.sum(axis=1) >= num_vars) & np.isfinite(r2_sum) & (r2_sum > 1e-6)
        blended = (preds_sel * r2_sel).sum(axis=1) / np.where(ok, r2_sum, 1.0)
    y_predicted = np.where(ok, blended, y_mean_arr)

    deviation = target - pd.Series(y_predicted, index=df.index)
    mmr_z = zscore_clipped(deviation, length, 3.0)
    mmr_signal = sigmoid(mmr_z, 1.5)

    # Skill of the 20d level fit vs a RANDOM WALK ("today ≈ yesterday"),
    # MSE-based (mean of squares, not variance: a fit that is always 5 points
    # off has zero residual variance but a large MSE — A2-07):
    #   skill = 1 − MSE(Close − ŷ) / MSE(ΔClose)   over the trailing `length` rows
    #   MMR_Quality = √clip(skill, 0, 1)
    min_p = max(length // 2, 5)
    fit_mse = (deviation ** 2).rolling(length, min_periods=min_p).mean()
    rw_mse = (target.diff() ** 2).rolling(length, min_periods=min_p).mean().replace(0, np.nan)
    skill = (1.0 - fit_mse / rw_mse).clip(lower=0.0, upper=1.0)
    mmr_quality = np.sqrt(skill.fillna(0.0))

    # For display purposes (not trading logic), get the trailing global top
    # drivers. Ranked by |r| but the SIGNED correlation is reported — an
    # inversely-related driver previously displayed as positively correlated.
    # Non-finite correlations (pegged/constant columns) are excluded.
    driver_details = []
    if len(df) > length:
        # corrwith legitimately warns (0/0) on a constant column — expected,
        # and the resulting NaN is dropped below; suppression is scoped to
        # exactly this call (no blanket filter).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            trailing_corr = (
                df[available_macros].iloc[-length:]
                .corrwith(target.iloc[-length:])
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
        for ticker in trailing_corr.abs().sort_values(ascending=False).head(num_vars).index:
            driver_details.append({
                "Symbol": ticker,
                "Correlation": round(float(trailing_corr[ticker]), 4),
            })

    return mmr_signal, driver_details, mmr_quality


# ─── Full Analysis Pipeline ──────────────────────────────────────────────────

# Columns run_full_analysis adds to the input frame (in order).
_OUTPUT_COLUMNS = (
    "MSF", "Micro", "Momentum", "Flow", "MMR", "MMR_Quality",
    "Unified", "Unified_Osc", "MSF_Osc", "MMR_Osc", "MSF_Weight", "MMR_Weight",
    "Agreement", "OS_Cut", "OB_Cut", "Buy_Signal", "Sell_Signal", "Bullish_Div",
    "Bearish_Div", "Condition", "Warmup",
    "Regime", "HMM_Bull", "HMM_Bear", "Vol_Regime", "Change_Point", "Confidence",
)
_BOOL_OUTPUTS = ("Buy_Signal", "Sell_Signal", "Bullish_Div", "Bearish_Div", "Warmup", "Change_Point")


def run_full_analysis(
    df: pd.DataFrame,
    length: int,
    roc_len: int,
    regime_sensitivity: float,
    base_weight: float,
    macro_columns: list[str] | None = None,
    num_vars: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the complete Nirnay pipeline on a single stock DataFrame.

    Rows whose ``Close`` is NaN (pre-listing prefix of a union-indexed
    download, or an inner missing bar) are DROPPED first: the output holds the
    stock's genuine bars only, so phantom rows can neither be counted nor pin
    the expanding quantiles (A2-03). The caller's frame is not mutated.

    Steps
    -----
    1. MSF and MMR signals
    2. Adaptive clarity-based weights, blend, agreement multiplier
    3. Condition (Oversold / Overbought / Neutral) against causal per-stock
       cuts — the expanding 30th / 70th percentile of this stock's own
       oscillator over the bars BEFORE t (``OS_Cut`` / ``OB_Cut`` record the
       cut actually applied on each row; ~30% / 30% of live stock-days are
       flagged by construction). The config ±``NIRNAY_OVERSOLD`` prior applies
       only until ``NIRNAY_WARMUP_BARS`` prior bars exist (or on a degenerate
       flat band).
    4. ``Warmup`` = the first ``NIRNAY_WARMUP_BARS`` genuine bars; excluded
       from the daily aggregation.
    5. Regime ensemble (Kalman → GARCH on Close log returns → HMM → CUSUM on
       standardized Kalman innovations), see ``analytics.regime``.
    """
    if macro_columns is None:
        macro_columns = []

    # Genuine bars only, unique sorted DatetimeIndex. A duplicate date would
    # make every downstream `.loc[date]` return a frame instead of a row.
    df = df.loc[df["Close"].notna()]
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index().copy()

    if df.empty:
        empty = pd.DataFrame(
            {c: pd.Series(dtype=object if c in ("Condition", "Regime", "Vol_Regime")
                          else bool if c in _BOOL_OUTPUTS else float)
             for c in _OUTPUT_COLUMNS},
            index=df.index,
        )
        return pd.concat([df, empty], axis=1), []

    msf, micro, momentum, flow = calculate_msf(df, length, roc_len)
    mmr, drivers, mmr_quality = calculate_mmr(
        df, length, num_vars=num_vars, macro_columns=macro_columns
    )
    df = pd.concat(
        [df, pd.DataFrame({"MSF": msf, "Micro": micro, "Momentum": momentum, "Flow": flow,
                           "MMR": mmr, "MMR_Quality": mmr_quality}, index=df.index)],
        axis=1,
    )

    # Adaptive weighting based on signal clarity
    msf_clarity = df["MSF"].abs()
    mmr_clarity = df["MMR"].abs()
    msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
    mmr_clarity_scaled = (mmr_clarity * df["MMR_Quality"]).pow(regime_sensitivity)
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001

    msf_w_adaptive = msf_clarity_scaled / clarity_sum
    mmr_w_adaptive = mmr_clarity_scaled / clarity_sum

    msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
    mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
    w_sum = msf_w_final + mmr_w_final
    msf_w_norm = msf_w_final / w_sum
    mmr_w_norm = mmr_w_final / w_sum

    unified_signal = (msf_w_norm * df["MSF"]) + (mmr_w_norm * df["MMR"])

    # Agreement multiplier amplifies aligned signals, dampens conflicts
    agreement = df["MSF"] * df["MMR"]
    agree_strength = agreement.abs()
    multiplier = np.where(
        agreement > 0,
        1.0 + 0.2 * agree_strength,
        1.0 - 0.1 * agree_strength,
    )

    unified = np.asarray((unified_signal * multiplier).clip(-1.0, 1.0))
    unified_osc = unified * 10.0
    msf_osc = df["MSF"].to_numpy() * 10.0
    mmr_osc = df["MMR"].to_numpy() * 10.0
    close_arr = df["Close"].to_numpy(dtype=np.float64)
    agreement_arr = agreement.to_numpy()

    # Causal per-stock bands: expanding quantiles of THIS name's own history,
    # shifted one bar (cut at t uses bars < t only). The first adaptive cut
    # needs NIRNAY_WARMUP_BARS prior bars, i.e. exactly the Warmup rows run on
    # the config cold-start prior.
    osc_s = pd.Series(unified_osc, index=df.index)
    os_cut_arr = (
        osc_s.expanding(min_periods=NIRNAY_WARMUP_BARS).quantile(0.30).shift(1)
        .fillna(float(NIRNAY_OVERSOLD)).to_numpy()
    )
    ob_cut_arr = (
        osc_s.expanding(min_periods=NIRNAY_WARMUP_BARS).quantile(0.70).shift(1)
        .fillna(float(NIRNAY_OVERBOUGHT)).to_numpy()
    )
    # Guard degenerate (flat) windows where the band would collapse.
    _flat = (ob_cut_arr - os_cut_arr) < 1e-6
    os_cut_arr = np.where(_flat, float(NIRNAY_OVERSOLD), os_cut_arr)
    ob_cut_arr = np.where(_flat, float(NIRNAY_OVERBOUGHT), ob_cut_arr)

    agree_s = pd.Series(agreement_arr, index=df.index)
    agree_cut_arr = (
        agree_s.expanding(min_periods=20).quantile(0.75).shift(1)
        .fillna(0.3).clip(lower=0.05).to_numpy()
    )

    strong_agreement = agreement_arr > agree_cut_arr
    buy_signal = strong_agreement & (unified_osc < os_cut_arr)
    sell_signal = strong_agreement & (unified_osc > ob_cut_arr)

    # Divergence detection (shift(1) ↔ prepend NaN, drop last)
    prev_unified_osc = np.concatenate(([np.nan], unified_osc[:-1]))
    prev_close = np.concatenate(([np.nan], close_arr[:-1]))
    with np.errstate(invalid="ignore"):  # NaN comparisons → False, silently
        osc_rising = unified_osc > prev_unified_osc
        price_falling = close_arr < prev_close
        osc_falling = unified_osc < prev_unified_osc
        price_rising = close_arr > prev_close
    bullish_div = osc_rising & price_falling & (unified_osc < os_cut_arr)
    bearish_div = osc_falling & price_rising & (unified_osc > ob_cut_arr)

    condition = np.where(
        unified_osc < os_cut_arr,
        "Oversold",
        np.where(unified_osc > ob_cut_arr, "Overbought", "Neutral"),
    )
    warmup = np.arange(len(df)) < NIRNAY_WARMUP_BARS

    regimes, hmm_bulls, hmm_bears, vol_regimes, change_points, confidences = (
        run_regime_loop(unified, close_arr, std_floor=NIRNAY_HMM_STD_FLOOR)
    )

    # One block-build + one merge (per-column inserts fragment the frame).
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Unified": unified,
                    "Unified_Osc": unified_osc,
                    "MSF_Osc": msf_osc,
                    "MMR_Osc": mmr_osc,
                    "MSF_Weight": msf_w_norm.to_numpy(),
                    "MMR_Weight": mmr_w_norm.to_numpy(),
                    "Agreement": agreement_arr,
                    "OS_Cut": os_cut_arr,
                    "OB_Cut": ob_cut_arr,
                    "Buy_Signal": buy_signal,
                    "Sell_Signal": sell_signal,
                    "Bullish_Div": bullish_div,
                    "Bearish_Div": bearish_div,
                    "Condition": condition,
                    "Warmup": warmup,
                    "Regime": regimes,
                    "HMM_Bull": hmm_bulls,
                    "HMM_Bear": hmm_bears,
                    "Vol_Regime": vol_regimes,
                    "Change_Point": change_points,
                    "Confidence": confidences,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )
    return df, drivers


# ─── Constituent Aggregation ─────────────────────────────────────────────────

_BULL_LABELS = ("BULL", "WEAK_BULL")
_BEAR_LABELS = ("BEAR", "WEAK_BEAR")

# Daily frame schema (aggregate_constituent_timeseries), in order.
_DAILY_COLUMNS = (
    "n_universe", "n_live", "coverage",
    "Oversold", "Overbought", "Neutral", "Buy_Signals", "Sell_Signals",
    "Total_Analyzed", "Avg_Signal", "Signal_Sum", "Dispersion",
    "Bull_Div", "Bear_Div", "Regime_Bull", "Regime_Bear", "Regime_Transition",
    "Vol_High", "Vol_Low", "Change_Points",
    "Oversold_Pct", "Overbought_Pct", "Neutral_Pct",
    "Regime_Bull_Pct", "Regime_Bear_Pct", "Regime_Neutral_Pct", "Regime_Transition_Pct",
    "Vol_High_Pct", "avg_hmm_bull", "avg_hmm_bear", "avg_msf_osc", "avg_mmr_osc",
)


def _live_rows(df: pd.DataFrame) -> pd.DataFrame:
    """A constituent's LIVE rows: genuine bar (finite Close), past warm-up,
    finite oscillator; one row per calendar day (tz dropped, normalized)."""
    sub = df
    if isinstance(sub.columns, pd.MultiIndex):
        sub = sub.copy()
        sub.columns = [c[0] for c in sub.columns]
    mask = pd.Series(True, index=sub.index)
    if "Close" in sub.columns:
        mask &= pd.to_numeric(sub["Close"], errors="coerce").notna()
    if "Warmup" in sub.columns:
        mask &= ~sub["Warmup"].fillna(True).astype(bool)
    else:  # frame from an older engine: derive warm-up from genuine-bar order
        mask &= mask.cumsum() > NIRNAY_WARMUP_BARS
    osc = pd.to_numeric(sub.get("Unified_Osc", pd.Series(np.nan, index=sub.index)), errors="coerce")
    mask &= np.isfinite(osc)
    sub = sub.loc[mask.to_numpy()]
    idx = pd.DatetimeIndex(sub.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    sub = sub.set_axis(idx.normalize(), axis=0)
    if sub.index.has_duplicates:
        sub = sub[~sub.index.duplicated(keep="last")]
    return sub


def aggregate_constituent_timeseries(
    constituent_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-constituent Nirnay results into daily breadth.

    Only LIVE names enter a day: a genuine bar that day, past the
    ``NIRNAY_WARMUP_BARS`` warm-up, finite oscillator. Missing / pre-listing /
    warm-up names are excluded — never neutral-filled (A2-03/13).

    Columns (DatetimeIndex ``Date``; days with ``n_live == 0`` are DROPPED, so
    absence of a date means NO_DATA downstream):

    - ``n_universe`` — constituents supplied to this run (constant per run);
      ``n_live`` — live names that day, the DENOMINATOR of every pct / mean;
      ``coverage`` = n_live / n_universe; ``Total_Analyzed`` = n_live.
    - Counts: ``Oversold``, ``Overbought``, ``Neutral``, ``Buy_Signals``,
      ``Sell_Signals``, ``Bull_Div``, ``Bear_Div``, ``Regime_Bull``,
      ``Regime_Bear``, ``Regime_Transition``, ``Vol_High`` (HIGH+EXTREME),
      ``Vol_Low``, ``Change_Points``.
    - Pct of n_live: ``Oversold_Pct``, ``Overbought_Pct``, ``Neutral_Pct``,
      ``Regime_Bull_Pct`` (BULL + WEAK_BULL), ``Regime_Bear_Pct``
      (BEAR + WEAK_BEAR), ``Regime_Neutral_Pct``, ``Regime_Transition_Pct``
      (the four regime pcts sum to 100), ``Vol_High_Pct``.
    - Means over live names: ``Avg_Signal`` (Unified_Osc, ±10 scale),
      ``avg_hmm_bull``, ``avg_hmm_bear``, ``avg_msf_osc``, ``avg_mmr_osc``;
      ``Signal_Sum`` (sum of Unified_Osc); ``Dispersion`` = cross-sectional
      population std (ddof=0) of Unified_Osc.
    """
    empty = pd.DataFrame({c: pd.Series(dtype=float) for c in _DAILY_COLUMNS},
                         index=pd.DatetimeIndex([], name="Date"))
    if not constituent_results:
        return empty
    n_universe = len(constituent_results)

    parts: list[pd.DataFrame] = []
    for df in constituent_results.values():
        if df is None or len(df) == 0:
            continue
        sub = _live_rows(df)
        if sub.empty:
            continue

        def _num(col: str, default: float) -> np.ndarray:
            if col in sub.columns:
                return pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
            return np.full(len(sub), default)

        def _flag(col: str) -> np.ndarray:
            if col in sub.columns:
                return sub[col].fillna(False).astype(bool).to_numpy(dtype=np.int64)
            return np.zeros(len(sub), dtype=np.int64)

        cond = sub["Condition"].astype(str) if "Condition" in sub.columns else pd.Series("Neutral", index=sub.index)
        reg = sub["Regime"].astype(str) if "Regime" in sub.columns else pd.Series("NEUTRAL", index=sub.index)
        vol = sub["Vol_Regime"].astype(str) if "Vol_Regime" in sub.columns else pd.Series("NORMAL", index=sub.index)
        is_bull = reg.isin(_BULL_LABELS)
        is_bear = reg.isin(_BEAR_LABELS)
        is_tr = reg == "TRANSITION"
        osc = _num("Unified_Osc", np.nan)
        parts.append(pd.DataFrame(
            {
                "n_live": 1,
                "Signal_Sum": osc,
                "osc_sq": osc * osc,
                "Oversold": (cond == "Oversold").to_numpy(dtype=np.int64),
                "Overbought": (cond == "Overbought").to_numpy(dtype=np.int64),
                "Neutral": (~cond.isin(("Oversold", "Overbought"))).to_numpy(dtype=np.int64),
                "Buy_Signals": _flag("Buy_Signal"),
                "Sell_Signals": _flag("Sell_Signal"),
                "Bull_Div": _flag("Bullish_Div"),
                "Bear_Div": _flag("Bearish_Div"),
                "Regime_Bull": is_bull.to_numpy(dtype=np.int64),
                "Regime_Bear": is_bear.to_numpy(dtype=np.int64),
                "Regime_Transition": is_tr.to_numpy(dtype=np.int64),
                "Regime_Neutral": (~is_bull & ~is_bear & ~is_tr).to_numpy(dtype=np.int64),
                "Vol_High": vol.isin(("HIGH", "EXTREME")).to_numpy(dtype=np.int64),
                "Vol_Low": (vol == "LOW").to_numpy(dtype=np.int64),
                "Change_Points": _flag("Change_Point"),
                "avg_hmm_bull": _num("HMM_Bull", np.nan),
                "avg_hmm_bear": _num("HMM_Bear", np.nan),
                "avg_msf_osc": _num("MSF_Osc", np.nan),
                "avg_mmr_osc": _num("MMR_Osc", np.nan),
            },
            index=sub.index,
        ))
    if not parts:
        return empty

    big = pd.concat(parts, axis=0)
    grp = big.groupby(level=0, sort=True)
    count_cols = ["n_live", "Signal_Sum", "osc_sq", "Oversold", "Overbought", "Neutral",
                  "Buy_Signals", "Sell_Signals", "Bull_Div", "Bear_Div", "Regime_Bull",
                  "Regime_Bear", "Regime_Transition", "Regime_Neutral", "Vol_High",
                  "Vol_Low", "Change_Points"]
    agg = grp[count_cols].sum()
    means = grp[["Signal_Sum", "avg_hmm_bull", "avg_hmm_bear", "avg_msf_osc", "avg_mmr_osc"]].mean()
    disp = grp["Signal_Sum"].std(ddof=0)

    agg = agg[agg["n_live"] > 0]
    n = agg["n_live"].astype(np.int64)
    out = pd.DataFrame(index=pd.DatetimeIndex(agg.index, name="Date"))
    out["n_universe"] = np.int64(n_universe)
    out["n_live"] = n
    out["coverage"] = n / float(n_universe)
    for c in ("Oversold", "Overbought", "Neutral", "Buy_Signals", "Sell_Signals"):
        out[c] = agg[c].astype(np.int64)
    out["Total_Analyzed"] = n
    out["Avg_Signal"] = means["Signal_Sum"].reindex(out.index)
    out["Signal_Sum"] = agg["Signal_Sum"]
    out["Dispersion"] = disp.reindex(out.index)
    for c in ("Bull_Div", "Bear_Div", "Regime_Bull", "Regime_Bear", "Regime_Transition",
              "Vol_High", "Vol_Low", "Change_Points"):
        out[c] = agg[c].astype(np.int64)
    pct = 100.0 / n
    out["Oversold_Pct"] = agg["Oversold"] * pct
    out["Overbought_Pct"] = agg["Overbought"] * pct
    out["Neutral_Pct"] = agg["Neutral"] * pct
    out["Regime_Bull_Pct"] = agg["Regime_Bull"] * pct
    out["Regime_Bear_Pct"] = agg["Regime_Bear"] * pct
    out["Regime_Neutral_Pct"] = agg["Regime_Neutral"] * pct
    out["Regime_Transition_Pct"] = agg["Regime_Transition"] * pct
    out["Vol_High_Pct"] = agg["Vol_High"] * pct
    for c in ("avg_hmm_bull", "avg_hmm_bear", "avg_msf_osc", "avg_mmr_osc"):
        out[c] = means[c].reindex(out.index)
    return out[list(_DAILY_COLUMNS)]
