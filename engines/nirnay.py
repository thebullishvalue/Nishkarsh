"""
Nishkarsh v1.4.0 — Nirnay Engine: Per-stock MSF + MMR with regime intelligence.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

NIRNAY — Per-constituent MSF + MMR analysis with HMM/GARCH/CUSUM regime intelligence aggregation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytics.regime import (
    AdaptiveKalmanFilter,
    AdaptiveHMM,
    GARCHDetector,
    CUSUMDetector,
)
from analytics.utils import causal_gram_schmidt_orthogonalize
from core.config import (
    NIRNAY_OVERSOLD,
    NIRNAY_OVERBOUGHT,
    NIRNAY_STRONG_BUY,
    NIRNAY_STRONG_SELL,
)


# ─── Utility functions ───────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray | float, scale: float = 1.0) -> np.ndarray | float:
    """Sigmoid transformation bounding to [-1, 1].

    Formula: ``2 / (1 + exp(-x/scale)) - 1`` (original Nirnay formula).
    """
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0


def _zscore_clipped(series: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
    """Rolling causal z-score with outlier clipping. Uses shift(1) to prevent today's outlier from biasing the denominator."""
    series_filled = series.ffill().fillna(0)
    roll_mean = series_filled.rolling(window=window, min_periods=1).mean().shift(1).bfill()
    roll_std = series_filled.rolling(window=window, min_periods=1).std().shift(1).bfill()
    z = (series_filled - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)


def _calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Exponential moving average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


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
    msf_signal, micro_norm, momentum_norm, accum_norm
        The diagnostic component series returned are the *pre*-orthogonalization
        values (for display); ``msf_signal`` is the orthogonalized composite.
    """
    close = df["Close"]

    # Momentum
    roc_raw = close.pct_change(roc_len, fill_method=None)
    roc_z = _zscore_clipped(roc_raw, length, clip)
    momentum_norm = _sigmoid(roc_z, 1.5)

    # Microstructure
    intrabar_dir = (df["High"] + df["Low"]) / 2 - df["Open"]
    vol_ma = df["Volume"].rolling(length).mean()
    vol_ratio = (df["Volume"] / vol_ma).fillna(1.0)
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    micro_raw = vw_direction - vw_impact
    micro_z = _zscore_clipped(micro_raw, length, clip)
    micro_norm = _sigmoid(micro_z, 1.5)

    # Trend
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = _zscore_clipped(trend_fast - trend_slow, length, clip)
    mom_accel_raw = close.diff(5).diff(5)
    mom_accel_z = _zscore_clipped(mom_accel_raw, length, clip)
    atr = _calculate_atr(df, 14)
    vol_adj_mom_raw = close.diff(5) / atr
    vol_adj_mom_z = _zscore_clipped(vol_adj_mom_raw, length, clip)
    mean_rev_z = _zscore_clipped(close - trend_slow, length, clip)
    composite_trend_z = (
        trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z
    ) / np.sqrt(4.0)
    composite_trend_norm = _sigmoid(composite_trend_z, 1.5)

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
    bar_vol = (
        pct_change.rolling(length, min_periods=5).std().shift(1).bfill()
    )
    bar_vol = bar_vol.fillna(pct_change.std()).replace(0, np.nan)
    bar_cut = (0.25 * bar_vol).fillna(0.0)
    regime_signals = np.select(
        [pct_change > bar_cut, pct_change < -bar_cut], [1, -1], default=0
    )
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_z = _zscore_clipped(regime_raw, length, clip)
    regime_norm = _sigmoid(regime_z, 1.5)

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
    msf_signal = pd.Series(_sigmoid(msf_raw, 1.0), index=df.index)

    return msf_signal, micro_norm, momentum_norm, accum_norm


# ─── Macro-Micro Regime ──────────────────────────────────────────────────────


def calculate_mmr(
    df: pd.DataFrame,
    length: int = 20,
    num_vars: int = 5,
    macro_columns: list[str] | None = None,
) -> tuple[pd.Series, list[dict[str, Any]], pd.Series]:
    """Calculate Macro-Micro Regime via rolling R²-weighted regression.

    Finds the top ``num_vars`` macro indicators most correlated with price,
    builds a weighted composite prediction, and measures the deviation of
    actual price from that prediction.

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

    y_mean = target.rolling(length, min_periods=1).mean().shift(1).bfill()
    y_std = target.rolling(length, min_periods=1).std().shift(1).bfill()

    preds_list = []
    r2_list = []
    
    # Vectorized causal rolling computations
    for ticker in available_macros:
        x = df[ticker].ffill().fillna(0)
        x_mean = x.rolling(length, min_periods=1).mean().shift(1).bfill()
        x_std = x.rolling(length, min_periods=1).std().shift(1).bfill()
        
        # Pearson correlation shifted (only prior data used to estimate relationship)
        roll_corr = x.rolling(length, min_periods=length).corr(target).shift(1).bfill().fillna(0)
        slope = roll_corr * (y_std / x_std.replace(0, np.nan)).fillna(0)
        intercept = y_mean - (slope * x_mean)
        
        pred = (slope * x) + intercept
        r2 = roll_corr**2
        
        preds_list.append(pred)
        r2_list.append(r2)

    all_preds = pd.concat(preds_list, axis=1)
    all_r2 = pd.concat(r2_list, axis=1)
    
    # Causally select top `num_vars` drivers per row!
    all_preds_arr = all_preds.values
    all_r2_arr = all_r2.values
    
    n_rows = len(df)
    y_predicted = np.empty(n_rows, dtype=np.float64)

    for i in range(n_rows):
        row_r2 = all_r2_arr[i]
        valid_mask = ~np.isnan(row_r2)
        if np.sum(valid_mask) < num_vars:
            y_predicted[i] = y_mean.iloc[i]
            continue

        top_indices = np.argsort(row_r2[valid_mask])[-num_vars:]
        top_real_indices = np.where(valid_mask)[0][top_indices]

        r2_sel = row_r2[top_real_indices]
        preds_sel = all_preds_arr[i, top_real_indices]

        r2_sum = np.sum(r2_sel)
        if r2_sum > 1e-6:
            y_predicted[i] = np.sum(preds_sel * r2_sel) / r2_sum
        else:
            y_predicted[i] = y_mean.iloc[i]

    deviation = target - pd.Series(y_predicted, index=df.index)
    mmr_z = _zscore_clipped(deviation, length, 3.0)
    mmr_signal = _sigmoid(mmr_z, 1.5)

    # Rolling out-of-sample skill of the composite fair-value model, benchmarked
    # against a RANDOM WALK (naive last-value predictor) rather than the raw
    # price-level variance. R² vs price-level is dominated by trend and inflates
    # toward 1 for any model that merely tracks the drift; R² vs random walk
    # asks the honest question — does the macro model beat "tomorrow ≈ today"?
    #   R²_rw = 1 - Var(target - ŷ) / Var(target - target₋₁)
    min_p = max(length // 2, 5)
    resid_var = deviation.rolling(length, min_periods=min_p).var()
    rw_resid = target.diff()
    rw_var = rw_resid.rolling(length, min_periods=min_p).var().replace(0, np.nan)
    r2_roll = (1.0 - resid_var / rw_var).clip(lower=0.0, upper=1.0)
    mmr_quality = np.sqrt(r2_roll.fillna(0.0))

    # For display purposes (not trading logic), get the trailing global top drivers
    driver_details = []
    if len(df) > length:
        trailing_corr = df[available_macros].iloc[-length:].corrwith(target.iloc[-length:]).abs().sort_values(ascending=False)
        for ticker in trailing_corr.head(num_vars).index:
            driver_details.append({
                "Symbol": ticker,
                "Correlation": round(float(trailing_corr[ticker]), 4),
            })

    return mmr_signal, driver_details, mmr_quality


# ─── Full Analysis Pipeline ──────────────────────────────────────────────────


def run_full_analysis(
    df: pd.DataFrame,
    length: int,
    roc_len: int,
    regime_sensitivity: float,
    base_weight: float,
    macro_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the complete Nirnay pipeline on a single stock DataFrame.

    Steps
    -----
    1. Calculate MSF and MMR signals
    2. Compute adaptive clarity-based weights
    3. Blend signals with agreement multiplier
    4. Classify conditions (Oversold/Overbought/Neutral)
    5. Run regime intelligence loop (Kalman → GARCH → HMM → CUSUM)
    """
    if macro_columns is None:
        macro_columns = []

    # Guarantee a unique, sorted DatetimeIndex. A duplicate date (yfinance
    # occasionally returns one; a join against a duplicated macro index can also
    # introduce them) would make every downstream `.loc[date]` return a frame
    # instead of a row, corrupting the daily aggregation.
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    df["MSF"], df["Micro"], df["Momentum"], df["Flow"] = calculate_msf(df, length, roc_len)
    df["MMR"], drivers, df["MMR_Quality"] = calculate_mmr(
        df, length, num_vars=5, macro_columns=macro_columns
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

    # Compute all derived columns as local arrays first, then join in ONE
    # block-build via pd.concat. Note: df.assign() also triggers the
    # PerformanceWarning on newer pandas because internally it does a per-kwarg
    # column insert loop. pd.concat with a fresh inner DataFrame builds the
    # twelve new columns as a single block and merges them in one operation.
    unified = np.asarray((unified_signal * multiplier).clip(-1.0, 1.0))
    unified_osc = unified * 10.0
    msf_osc = df["MSF"].to_numpy() * 10.0
    mmr_osc = df["MMR"].to_numpy() * 10.0
    close_arr = df["Close"].to_numpy()

    agreement_arr = agreement.to_numpy() if hasattr(agreement, "to_numpy") else np.asarray(agreement)

    # Market-led, CAUSAL oscillator bands: expanding-window quantiles of THIS
    # name's own oscillator / agreement history, shifted one bar so the cut-point
    # at t uses only information available *before* t — no look-ahead, so the
    # historical Condition / Buy / Sell / Divergence flags are backtest-safe.
    # Config NIRNAY_* / 0.3 are the cold-start prior until ≥50 (20) bars exist.
    osc_s = pd.Series(unified_osc, index=df.index)
    os_cut_arr = (
        osc_s.expanding(min_periods=50).quantile(0.30).shift(1)
        .fillna(float(NIRNAY_OVERSOLD)).to_numpy()
    )
    ob_cut_arr = (
        osc_s.expanding(min_periods=50).quantile(0.70).shift(1)
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

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Unified": unified,
                    "Unified_Osc": unified_osc,
                    "MSF_Osc": msf_osc,
                    "MMR_Osc": mmr_osc,
                    "MSF_Weight": msf_w_norm,
                    "MMR_Weight": mmr_w_norm,
                    "Agreement": agreement.to_numpy() if hasattr(agreement, "to_numpy") else np.asarray(agreement),
                    "Buy_Signal": buy_signal,
                    "Sell_Signal": sell_signal,
                    "Bullish_Div": bullish_div,
                    "Bearish_Div": bearish_div,
                    "Condition": condition,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Regime intelligence loop
    hmm = AdaptiveHMM()
    garch = GARCHDetector()
    cusum = CUSUMDetector()
    kalman = AdaptiveKalmanFilter()

    regimes: list[str] = []
    hmm_bulls: list[float] = []
    hmm_bears: list[float] = []
    vol_regimes: list[str] = []
    change_points: list[bool] = []
    confidences: list[float] = []
    signal_history: list[float] = []

    unified_vals = df["Unified"].values

    # HMM confidence cuts tied to the model's uniform baseline (3 states → 1/3
    # each) rather than magic 0.6/0.4: "strong" = twice the uniform prior,
    # "weak" = a margin above it. Structural to the state space, not a market
    # magnitude threshold.
    uniform_p = 1.0 / 3.0
    strong_p = 2.0 * uniform_p
    weak_p = uniform_p + 0.07

    for i in range(len(df)):
        sig = unified_vals[i] if not np.isnan(unified_vals[i]) else 0.0

        # Kalman smoothing
        filtered = kalman.update(sig)

        # GARCH volatility regime
        shock = sig - signal_history[-1] if signal_history else 0.0
        garch.update(shock)
        vol_regime, _ = garch.get_regime()

        # HMM state estimation
        hmm_probs = hmm.update(filtered)
        change = cusum.update(filtered)

        bull_p = hmm_probs["BULL"]
        bear_p = hmm_probs["BEAR"]

        if change:
            regime = "TRANSITION"
        elif bull_p > strong_p:
            regime = "BULL"
        elif bear_p > strong_p:
            regime = "BEAR"
        elif bull_p > weak_p:
            regime = "WEAK_BULL"
        elif bear_p > weak_p:
            regime = "WEAK_BEAR"
        else:
            regime = "NEUTRAL"

        regimes.append(regime)
        hmm_bulls.append(bull_p)
        hmm_bears.append(bear_p)
        vol_regimes.append(vol_regime)
        change_points.append(change)
        confidences.append(max(bull_p, bear_p, hmm_probs["NEUTRAL"]))
        signal_history.append(sig)

    # Join the six regime-intelligence columns as ONE block via pd.concat.
    # df.assign() also fragments under newer pandas because it inserts kwargs
    # one-by-one internally; pd.concat with a pre-built inner DataFrame avoids
    # that entirely (single block-build, single merge).
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
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


def aggregate_constituent_timeseries(
    constituent_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-constituent Nirnay results into daily statistics.

    Produces the same column schema as the original Nirnay system:
    Oversold/Overbought counts and percentages, signal counts, regime
    distributions, and average oscillator values.
    """
    if not constituent_results:
        return pd.DataFrame()

    # Vectorized aggregation. The previous implementation looped dates × stocks
    # with a per-cell df.loc[date] label lookup (≈ 1700 × 50 ≈ 86k lookups, each
    # followed by ~15 row.get() calls) — ~13s locally / ~35s on a weak CPU. This
    # stacks every constituent into one frame and groups by date, which is the
    # same daily counts/means but ~50x faster (bit-identical output to ~1e-14).
    _num_defaults = {"Unified_Osc": 0.0, "MSF_Osc": 0.0, "MMR_Osc": 0.0,
                     "HMM_Bull": 0.33, "HMM_Bear": 0.33}
    _str_defaults = {"Condition": "Neutral", "Regime": "NEUTRAL", "Vol_Regime": "NORMAL"}
    _bool_cols = ["Buy_Signal", "Sell_Signal", "Bullish_Div", "Bearish_Div", "Change_Point"]
    _cols = list(_num_defaults) + list(_str_defaults) + _bool_cols

    parts: list[pd.DataFrame] = []
    for sym, df in constituent_results.items():
        sub = df[[c for c in _cols if c in df.columns]].copy()
        # A duplicate date would otherwise double-count that stock for the day;
        # collapse to the last row (matches the old isinstance→iloc[-1] guard).
        if sub.index.has_duplicates:
            sub = sub[~sub.index.duplicated(keep="last")]
        for c in _cols:  # fill any columns a constituent is missing with defaults
            if c not in sub.columns:
                sub[c] = {**_num_defaults, **_str_defaults, **{b: False for b in _bool_cols}}[c]
        parts.append(sub)

    big = pd.concat(parts, axis=0)
    cond = big["Condition"].astype(str)
    reg = big["Regime"].astype(str)
    vol = big["Vol_Regime"].astype(str)
    is_bull = reg.str.contains("BULL")
    is_bear = reg.str.contains("BEAR")
    is_volhigh = vol.isin(("HIGH", "EXTREME"))

    f = pd.DataFrame(index=big.index)
    f["Total_Analyzed"] = 1
    f["Signal_Sum"] = pd.to_numeric(big["Unified_Osc"], errors="coerce").fillna(0.0).to_numpy()
    f["Oversold"] = (cond == "Oversold").to_numpy(dtype=int)
    f["Overbought"] = (cond == "Overbought").to_numpy(dtype=int)
    f["Neutral"] = (~cond.isin(("Oversold", "Overbought"))).to_numpy(dtype=int)
    f["Buy_Signals"] = big["Buy_Signal"].astype(bool).to_numpy(dtype=int)
    f["Sell_Signals"] = big["Sell_Signal"].astype(bool).to_numpy(dtype=int)
    f["Bull_Div"] = big["Bullish_Div"].astype(bool).to_numpy(dtype=int)
    f["Bear_Div"] = big["Bearish_Div"].astype(bool).to_numpy(dtype=int)
    # Priority matches the original if/elif chain: BULL > BEAR > TRANSITION > else.
    f["Regime_Bull"] = is_bull.to_numpy(dtype=int)
    f["Regime_Bear"] = (~is_bull & is_bear).to_numpy(dtype=int)
    f["Regime_Transition"] = (~is_bull & ~is_bear & (reg == "TRANSITION")).to_numpy(dtype=int)
    f["Regime_Neutral"] = (~is_bull & ~is_bear & (reg != "TRANSITION")).to_numpy(dtype=int)
    f["Vol_High"] = is_volhigh.to_numpy(dtype=int)
    f["Vol_Low"] = ((~is_volhigh) & (vol == "LOW")).to_numpy(dtype=int)
    f["Change_Points"] = big["Change_Point"].astype(bool).to_numpy(dtype=int)
    f["avg_hmm_bull"] = pd.to_numeric(big["HMM_Bull"], errors="coerce").fillna(0.33).to_numpy()
    f["avg_hmm_bear"] = pd.to_numeric(big["HMM_Bear"], errors="coerce").fillna(0.33).to_numpy()
    f["avg_msf_osc"] = pd.to_numeric(big["MSF_Osc"], errors="coerce").fillna(0.0).to_numpy()
    f["avg_mmr_osc"] = pd.to_numeric(big["MMR_Osc"], errors="coerce").fillna(0.0).to_numpy()

    grp = f.groupby(f.index, sort=True)
    sum_cols = ["Oversold", "Overbought", "Neutral", "Buy_Signals", "Sell_Signals",
                "Total_Analyzed", "Avg_Signal", "Signal_Sum", "Bull_Div", "Bear_Div",
                "Regime_Bull", "Regime_Bear", "Regime_Neutral", "Regime_Transition",
                "Vol_High", "Vol_Low", "Change_Points"]
    agg = grp[["Total_Analyzed", "Signal_Sum", "Oversold", "Overbought", "Neutral",
               "Buy_Signals", "Sell_Signals", "Bull_Div", "Bear_Div", "Regime_Bull",
               "Regime_Bear", "Regime_Neutral", "Regime_Transition", "Vol_High",
               "Vol_Low", "Change_Points"]].sum()
    means = grp[["avg_hmm_bull", "avg_hmm_bear", "avg_msf_osc", "avg_mmr_osc"]].mean()

    n = agg["Total_Analyzed"]
    out = pd.DataFrame(index=agg.index)
    out["Oversold"] = agg["Oversold"]
    out["Overbought"] = agg["Overbought"]
    out["Neutral"] = agg["Neutral"]
    out["Buy_Signals"] = agg["Buy_Signals"]
    out["Sell_Signals"] = agg["Sell_Signals"]
    out["Total_Analyzed"] = n
    out["Avg_Signal"] = (agg["Signal_Sum"] / n).where(n > 0, 0.0)
    out["Signal_Sum"] = agg["Signal_Sum"]
    out["Bull_Div"] = agg["Bull_Div"]
    out["Bear_Div"] = agg["Bear_Div"]
    out["Regime_Bull"] = agg["Regime_Bull"]
    out["Regime_Bear"] = agg["Regime_Bear"]
    out["Regime_Neutral"] = agg["Regime_Neutral"]
    out["Regime_Transition"] = agg["Regime_Transition"]
    out["Vol_High"] = agg["Vol_High"]
    out["Vol_Low"] = agg["Vol_Low"]
    out["Change_Points"] = agg["Change_Points"]
    out["Oversold_Pct"] = agg["Oversold"] / n * 100
    out["Overbought_Pct"] = agg["Overbought"] / n * 100
    out["Neutral_Pct"] = agg["Neutral"] / n * 100
    out["Regime_Bull_Pct"] = agg["Regime_Bull"] / n * 100
    out["Regime_Bear_Pct"] = agg["Regime_Bear"] / n * 100
    out["Vol_High_Pct"] = agg["Vol_High"] / n * 100
    out["avg_hmm_bull"] = means["avg_hmm_bull"]
    out["avg_hmm_bear"] = means["avg_hmm_bear"]
    out["avg_msf_osc"] = means["avg_msf_osc"]
    out["avg_mmr_osc"] = means["avg_mmr_osc"]

    out.index = [d.date() if hasattr(d, "date") else d for d in out.index]
    out.index.name = "Date"
    return out
