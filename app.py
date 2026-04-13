"""
Nishkarsh v1.2.0 — Main Streamlit entrypoint.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

NISHKARSH — Two systems. One conclusion. Walk-forward valuation + constituent regime intelligence unified by adaptive convergence.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Warning suppression ──────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*YF.download.*")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
pd.options.mode.chained_assignment = None

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── UI ───────────────────────────────────────────────────────────────────────
from ui.theme import inject_css, VERSION, PRODUCT_NAME, COMPANY, progress_bar
from ui.tabs.tab_convergence import render_convergence_tab
from ui.components import (
    render_header,
    render_info_box,
    render_nishkarsh_signal_card,
    render_warning_box,
    render_metric_card,
    render_chart_skeleton,
    render_collapsible_section,
    render_collapsible_section_close,
    section_gap,
)
from ui.tabs.tab_aarambh import render_aarambh_tab
from ui.tabs.tab_nirnay import render_nirnay_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_aarambh_data, fetch_constituent_ohlcv, fetch_macro_live, load_google_sheet
from data.constituents import fetch_nifty50_constituents

# ── Engines ──────────────────────────────────────────────────────────────────
from engines.aarambh import FairValueEngine
from engines.nirnay import run_full_analysis, aggregate_constituent_timeseries

# ── Convergence ──────────────────────────────────────────────────────────────
from convergence.cross_validator import CrossValidator
from convergence.conviction_model import UnifiedConvictionModel
from convergence.divergence_detector import CrossSystemDivergenceDetector

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id, Colors
from core.config import LOOKBACK_WINDOWS, MIN_DATA_POINTS, STALENESS_DAYS, COLOR_RED


# ── Secret Management ────────────────────────────────────────────────────────

def _get_sheet_url() -> str:
    """Get Google Sheets URL from secrets, env vars, or fallback."""
    try:
        url = st.secrets.get("aarambh", {}).get("google_sheets_url", "")
        if url:
            return url
    except Exception:
        pass
    env_url = os.environ.get("AARAMBH_GOOGLE_SHEETS_URL", "")
    if env_url:
        return env_url
    return ""


# ─── UI Rendering helpers ────────────────────────────────────────────────────

def _render_header() -> None:
    render_header(
        title=f"{PRODUCT_NAME}",
        tagline="Walk-Forward Valuation + Constituent Regime Intelligence  |  Quantitative Convergence"
    )


def _render_landing_page() -> None:
    """Render the landing page with three system cards."""
    section_gap()
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.markdown("""
        <div class='system-card aarambh'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
                AARAMBH
            </h3>
            <p>Walk-forward ensemble regression on Nifty 50 PE ratio with conformal z-scores and DDM filtering.</p>
            <div class='spec'>
                <span>Ensemble:</span> Ridge + Huber + ENet + WLS<br>
                <span>Validation:</span> Walk-forward OOS<br>
                <span>Bounds:</span> Conformal prediction
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='system-card nirnay'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                NIRNAY
            </h3>
            <p>Per-constituent MSF + MMR analysis with HMM/GARCH/CUSUM regime intelligence aggregation.</p>
            <div class='spec'>
                <span>Regime:</span> HMM Probabilities<br>
                <span>Projection:</span> 90D Path + Bands<br>
                <span>Trend:</span> DFA Hurst exponent
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='system-card convergence'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
                CONVERGENCE
            </h3>
            <p>Adaptive-weighted composite of 4 dimensions: Direction, Breadth, Magnitude, Regime — with DDM.</p>
            <div class='spec'>
                <span>Scope:</span> Multi-temporal<br>
                <span>Smoothing:</span> Leaky DDM<br>
                <span>Range:</span> Soft \u00b1100 limit
            </div>
        </div>
        """, unsafe_allow_html=True)
    section_gap()
    st.markdown("""
    <div class='landing-prompt'>
        <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            AWAITING DATA
        </h4>
        <p>Load data via the <strong>Sidebar</strong> (CSV/Excel or Google Sheet).<br>
           Select a <strong>Target</strong> and <strong>Predictors</strong>, then execute <strong>Run Analysis</strong> to initialize both engines.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_primary_signal(nishkarsh_result, agreement, aarambh_signal) -> None:
    """Render unified Nishkarsh convergence signal card."""
    if nishkarsh_result:
        conv = nishkarsh_result.nishkarsh_conviction
        sig = nishkarsh_result.nishkarsh_signal
        upper = nishkarsh_result.confidence_upper
        lower = nishkarsh_result.confidence_lower
        agreement_text = "Strong" if agreement > 0.7 else "Moderate" if agreement > 0.5 else "Weak"
        explanation = (
            f"Conviction {conv:+.0f} ({sig}). "
            f"Confidence band: [{lower:.0f}, {upper:.0f}]. "
            f"Agreement: {agreement:.0%} ({agreement_text}). "
            f"{'Both systems aligned.' if agreement > 0.6 else 'Mixed signals \u2014 wait for clearer convergence.'}"
        )
    else:
        conv = aarambh_signal.get("conviction_score", 0)
        sig = aarambh_signal.get("signal", "HOLD")
        explanation = f"Conviction: {conv:+.0f} ({sig})."

    render_nishkarsh_signal_card(
        signal=sig,
        conviction=conv,
        agreement=agreement,
        explanation=explanation,
    )
    section_gap()


def _render_footer() -> None:
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    st.markdown(
        f'<div class="app-footer">'
        f'<div class="content">'
        f'\u00a9 {ist_now.year} <strong>{PRODUCT_NAME}</strong> &nbsp;\u00b7&nbsp; {COMPANY} &nbsp;\u00b7&nbsp; v{VERSION} &nbsp;\u00b7&nbsp; {ist_now.strftime("%Y-%m-%d %H:%M:%S IST")}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title=f"{PRODUCT_NAME} | Unified Convergence",
        layout="wide", initial_sidebar_state="collapsed",
    )
    inject_css()

    # ─── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center;padding:0.75rem 0 1rem 0;">
            <div style="font-family:var(--display);font-size:1.5rem;font-weight:700;color:var(--amber);letter-spacing:0.06em;">NISHKARSH</div>
            <div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.65rem;margin-top:0.2rem;letter-spacing:0.08em;text-transform:uppercase;">निष्कर्ष | Unified Convergence</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">Data Source</div>', unsafe_allow_html=True)
        data_source = st.radio(
            "Source", ["Google Sheets", "Upload"],
            horizontal=True, label_visibility="collapsed", index=0,
        )

        df = None
        has_data = "data" in st.session_state and "run_analysis" in st.session_state

        if data_source == "Upload":
            uploaded_file = st.file_uploader("CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
                if not has_data and st.button("Run Analysis", type="primary"):
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.session_state["data"] = df
                    st.session_state["run_analysis"] = True
                    st.rerun()
            else:
                st.info("Upload a CSV or Excel file to begin analysis")
        else:
            default_url = _get_sheet_url()
            has_secret = bool(default_url)
            sheet_url = st.text_input(
                "Sheet URL",
                value=default_url if has_secret else "",
                type="password" if has_secret else "default",
                label_visibility="collapsed",
                placeholder="Enter Google Sheets URL or set AARAMBH_GOOGLE_SHEETS_URL env var",
            )
            if not sheet_url and has_secret:
                sheet_url = default_url
            if not has_data and st.button("Run Analysis", type="primary"):
                if not sheet_url:
                    st.error("Please provide a Google Sheets URL or set the AARAMBH_GOOGLE_SHEETS_URL environment variable.")
                    return
                with st.spinner("Loading data and running analysis..."):
                    df, error = load_google_sheet(sheet_url)
                    if error:
                        st.error(f"Failed: {error}")
                        return
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.session_state["data"] = df
                    st.session_state["run_analysis"] = True
                    st.rerun()
            if "data" in st.session_state and "run_analysis" in st.session_state:
                df = st.session_state["data"]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ─── Landing page if no data loaded ──────────────────────────────────
    if df is None:
        _render_header()
        _render_landing_page()
        _render_footer()
        return

    # ─── Sidebar: Model Configuration ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Model Configuration</div>', unsafe_allow_html=True)

        default_target = "NIFTY50_PE" if "NIFTY50_PE" in numeric_cols else numeric_cols[0]
        active_target_state = st.session_state.get("active_target", default_target)
        if active_target_state not in numeric_cols:
            active_target_state = numeric_cols[0]

        target_col = st.selectbox("Target Variable", numeric_cols, index=numeric_cols.index(active_target_state))

        date_candidates = [c for c in all_cols if "date" in c.lower()]
        default_date = date_candidates[0] if date_candidates else "None"
        active_date_state = st.session_state.get("active_date_col", default_date)
        if active_date_state not in ["None"] + all_cols:
            active_date_state = "None"

        date_col = st.selectbox("Date Column", ["None"] + all_cols, index=(["None"] + all_cols).index(active_date_state))

        available = [c for c in numeric_cols if c != target_col]
        valid_defaults = [p for p in ("AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH", "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO", "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB") if p in available]

        if "active_features" not in st.session_state:
            st.session_state["active_features"] = tuple(valid_defaults or available[:3])

        with st.expander("Predictor Columns", expanded=False):
            st.caption("Select predictors, then click Apply to recompute.")
            staging_features = st.multiselect(
                "Predictor Columns", options=available,
                default=[f for f in st.session_state["active_features"] if f in available],
                label_visibility="collapsed",
            )
            if not staging_features:
                st.warning("Select at least one predictor.")
                staging_features = [f for f in st.session_state["active_features"] if f in available] or available[:3]

            staging_set = set(staging_features)
            active_set = set(st.session_state["active_features"])
            has_pred_changes = staging_set != active_set
            has_other_changes = (target_col != active_target_state) or (date_col != active_date_state)
            has_changes = has_pred_changes or has_other_changes

            if has_pred_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                parts = []
                if added:
                    parts.append(f"+{len(added)} added")
                if removed:
                    parts.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(parts)}")
            elif has_other_changes:
                st.caption("Pending: Target/Date changes")

            if st.button("Apply Configuration" if has_changes else "No changes", disabled=not has_changes, type="primary" if has_changes else "secondary"):
                if has_changes:
                    st.session_state["active_target"] = target_col
                    st.session_state["active_features"] = tuple(staging_features)
                    st.session_state["active_date_col"] = date_col
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.rerun()

            if len(st.session_state["active_features"]) != len(available):
                st.info(f"Active: {len(st.session_state['active_features'])}/{len(available)} predictors")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if "run_analysis" in st.session_state and st.session_state.get("run_analysis"):
            if st.button("Reset Analysis", type="secondary"):
                st.session_state.pop("data", None)
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state.pop("run_analysis", None)
                st.session_state.pop("nishkarsh_result", None)
                st.rerun()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size:0.75rem;margin:0;color:var(--ink-tertiary);line-height:1.6;'>
                <strong style="color:var(--ink-secondary);">Version:</strong> {VERSION}<br>
                <strong style="color:var(--ink-secondary);">Engine:</strong> Nishkarsh Convergence<br>
                <strong style="color:var(--ink-secondary);">Data:</strong> Google Sheets + yfinance
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ─── Resolve active configuration ──────────────────────────────────────
    active_target = st.session_state.get("active_target", target_col)
    active_features = list(st.session_state.get("active_features", [c for c in numeric_cols if c != active_target][:3]))
    active_date = st.session_state.get("active_date_col", date_col)

    # ─── Data staleness warning ────────────────────────────────────────────
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors="coerce").dropna()
            if len(dates) > 0:
                latest_date = dates.max().to_pydatetime()
                if latest_date.tzinfo is not None:
                    latest_date = latest_date.replace(tzinfo=None)
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                data_age = (now_utc - latest_date).days
                if data_age > STALENESS_DAYS:
                    render_warning_box(
                        title="Stale Data",
                        content=f"Last data point is from {latest_date.strftime('%d %b %Y')} ({data_age} days ago). Analysis may be outdated."
                    )
        except Exception:
            pass

    # ─── Clean & Fit Engine ────────────────────────────────────────────────
    cols = [active_target] + active_features + ([active_date] if active_date != "None" and active_date in df.columns else [])
    data = df[[c for c in cols if c in df.columns]].copy()
    if active_date != "None" and active_date in data.columns:
        data[active_date] = pd.to_datetime(data[active_date], errors="coerce")
        data = data.dropna(subset=[active_date]).sort_values(active_date)
    for col in [active_target] + active_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[[active_target] + active_features] = data[[active_target] + active_features].ffill().bfill()
    data = data.dropna(subset=[active_target] + active_features).reset_index(drop=True)
    if len(data) < MIN_DATA_POINTS:
        st.error(f"Need {MIN_DATA_POINTS}+ data points for walk-forward analysis.")
        return
    active_features = [f for f in active_features if f in data.columns]
    if not active_features:
        st.error("No valid features found after data cleaning.")
        return

    X, y = data[active_features].values, data[active_target].values
    cache_key = f"{active_target}|{'|'.join(sorted(active_features))}|{len(data)}"
    if st.session_state.get("engine_cache") != cache_key:
        if "engine" in st.session_state:
            del st.session_state["engine"]

        # ════ RUN HEADER ════
        console.header("NISHKARSH — Unified Convergence Analysis", f"v{VERSION}")
        console.main_header("ANALYSIS CONFIGURATION", {
            "Run ID": generate_run_id(),
            "Target": active_target,
            "Predictors": f"{len(active_features)} columns",
            "Date Range": f"{data.shape[0]} observations",
        })

        # Custom styled progress container
        progress_container = st.empty()

        # ── Phase 1: Data Loading ─────────────────────────────────────────
        console.start_phase("DATA ACQUISITION", 1, 5)
        progress_bar(progress_container, 2, "Fetching constituents", "niftyindices.com · CSV parse")

        console.section("Constituent Fetch")
        constituents, src_msg = fetch_nifty50_constituents()
        console.item("Source", src_msg)
        console.item("Count", len(constituents))
        console.item("Symbols", f"{constituents[0]}, {constituents[1]}, {constituents[2]}...")
        console.success(f"Fetched {len(constituents)} Nifty 50 constituents")
        progress_bar(progress_container, 5, "Fetching macro data", "Stooq yields · Yahoo Finance FX · Commodities")

        console.section("Macro Data")
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=365 + 365)
        macro_df = fetch_macro_live(start_date, end_date)
        console.item("Date Range", f"{start_date.date()} to {end_date.date()}")
        if not macro_df.empty:
            console.item("YF Columns", f"{len(macro_df.columns)} symbols")
            console.item("Rows", len(macro_df))
            console.success(f"Macro data: {len(macro_df.columns)} symbols × {len(macro_df)} rows")
        else:
            console.warning("No macro data available")
        progress_bar(progress_container, 10, "Fetching OHLCV data", f"yfinance · {len(constituents)} constituents")

        console.section("Constituent OHLCV")
        constituent_ohlcv = {}
        if constituents:
            constituent_ohlcv = fetch_constituent_ohlcv(constituents, start_date, end_date)
            console.item("Requested", len(constituents))
            console.item("Downloaded", len(constituent_ohlcv))
            if constituent_ohlcv:
                sample = list(constituent_ohlcv.items())[0]
                console.item("Sample", f"{sample[0]}: {len(sample[1])} rows")
            console.success(f"OHLCV data for {len(constituent_ohlcv)} constituents")
        progress_bar(progress_container, 15, "Assembling macro indicators", "Bond yields from Google Sheets")

        console.section("Nirnay Macro Assembly")
        nirnay_macro_df = macro_df.copy() if macro_df is not None and not macro_df.empty else pd.DataFrame()
        nirnay_bond_cols = ["IN10Y", "IN02Y", "IN30Y", "US10Y", "US30Y", "US02Y"]
        available_bond_cols = [c for c in nirnay_bond_cols if c in df.columns]
        if available_bond_cols:
            bond_data = df[available_bond_cols].copy()
            if "DATE" in df.columns:
                bond_data.index = pd.to_datetime(df["DATE"])
            nirnay_macro_df = nirnay_macro_df.join(bond_data, how="outer").sort_index().ffill()
            console.item("YF Symbols", len(macro_df.columns) if not macro_df.empty else 0)
            console.item("Bond Yields", f"{len(available_bond_cols)} ({', '.join(available_bond_cols)})")
            console.success(f"Combined macro: {len(nirnay_macro_df.columns)} indicators × {len(nirnay_macro_df)} rows")
        macro_cols_list = list(nirnay_macro_df.columns) if not nirnay_macro_df.empty else []
        console.end_phase("DATA ACQUISITION")
        progress_bar(progress_container, 20, "Data acquisition complete", f"{len(constituent_ohlcv)} constituents · {len(nirnay_macro_df.columns)} macros")

        # ── Phase 2: Aarambh FairValueEngine ─────────────────────────────
        console.start_phase("AARAMBH ENGINE", 2, 5)
        progress_bar(progress_container, 20, "Running Aarambh engine", f"Walk-forward · {len(active_features)} predictors · {len(data)} rows")

        console.section("Engine Configuration")
        console.item("Target", active_target)
        console.item("Features", f"{len(active_features)}: {', '.join(active_features[:5])}...")
        console.item("Observations", f"{len(data)} rows")
        console.item("Min Train Size", MIN_DATA_POINTS)
        console.item("Lookback Windows", f"{LOOKBACK_WINDOWS}")

        console.section("Walk-Forward Regression")
        engine = FairValueEngine()
        engine.fit(X, y, feature_names=active_features, progress_callback=lambda pct, msg: progress_bar(progress_container, int(20 + pct * 20), "Running Aarambh engine", msg))

        sig = engine.get_current_signal()
        stats = engine.get_model_stats()
        console.section("Engine Results")
        console.item("Signal", f"{sig['signal']} ({sig['strength']})")
        console.item("Conviction", f"{sig['conviction_score']:+.0f}")
        console.item("OOS R²", f"{stats['r2_oos']:.3f}")
        console.item("R² vs RW", f"{stats['r2_vs_rw']:+.3f}")
        console.item("OU Half-Life", f"{sig['ou_half_life']:.0f}d")
        console.item("Hurst", f"{sig['hurst']:.2f}")
        console.item("Oversold Breadth", f"{sig['oversold_breadth']:.0f}%")
        console.success(f"Aarambh engine complete | {len(engine.ts_data)} output rows")
        console.end_phase("AARAMBH ENGINE")
        progress_bar(progress_container, 40, "Aarambh engine complete", f"Signal: {sig['signal']} ({sig['strength']}) · Conviction: {sig['conviction_score']:+.0f}")

        # ── Phase 3: Nirnay Constituent Analysis ──────────────────────────
        console.start_phase("NIRNAY ENGINE", 3, 5)
        progress_bar(progress_container, 42, "Running Nirnay engine", f"MSF+MMR+Regime · {len(constituent_ohlcv)} constituents")

        nirnay_daily = pd.DataFrame()
        nirnay_constituent_dfs = {}

        if constituent_ohlcv:
            total = len(constituent_ohlcv)
            console.section("Per-Stock Analysis")
            console.item("Constituents", total)
            console.item("MSF Length", 20)
            console.item("ROC Length", 14)
            console.item("Regime Sensitivity", 1.5)
            console.item("Base Weight", 0.6)
            console.item("Macro Columns", len(macro_cols_list))

            for i, (sym, ohlcv_df) in enumerate(constituent_ohlcv.items()):
                try:
                    merged = ohlcv_df.copy()
                    if nirnay_macro_df is not None and not nirnay_macro_df.empty:
                        merged = merged.join(nirnay_macro_df, how="left")
                        merged[macro_cols_list] = merged[macro_cols_list].ffill()

                    n_rows = len(merged)
                    has_macro = len([c for c in macro_cols_list if c in merged.columns])

                    result_df, _ = run_full_analysis(
                        merged, length=20, roc_len=14,
                        regime_sensitivity=1.5, base_weight=0.6,
                        macro_columns=macro_cols_list,
                    )
                    nirnay_constituent_dfs[sym] = result_df

                    last_row = result_df.iloc[-1]
                    osc = last_row.get('Unified_Osc', 0)
                    cond = last_row.get('Condition', 'N/A')
                    regime = last_row.get('Regime', 'N/A')
                    console.detail(f"[{i+1}/{total}] {sym}: osc={osc:+.1f} [{cond}] regime={regime} rows={n_rows} macros={has_macro}")

                    pct_val = int(45 + (i + 1) / total * 30)
                    progress_bar(progress_container, pct_val, f"Analyzing {sym}", f"osc={osc:+.1f} [{cond}] regime={regime}")

                except Exception as e:
                    console.failure(f"{sym}", str(e))

            if nirnay_constituent_dfs:
                console.section("Aggregation")
                nirnay_daily = aggregate_constituent_timeseries(nirnay_constituent_dfs)
                console.item("Trading Days", len(nirnay_daily))
                if len(nirnay_daily) > 0:
                    last = nirnay_daily.iloc[-1]
                    console.item("Avg Signal", f"{last.get('Avg_Signal', 0):+.2f}")
                    console.item("Oversold %", f"{last.get('Oversold_Pct', 0):.0f}%")
                    console.item("Overbought %", f"{last.get('Overbought_Pct', 0):.0f}%")
                    console.item("Buy Signals", int(last.get('Buy_Signals', 0)))
                    console.item("Sell Signals", int(last.get('Sell_Signals', 0)))
                console.success(f"Nirnay aggregation: {len(nirnay_daily)} trading days")

        console.end_phase("NIRNAY ENGINE")
        progress_bar(progress_container, 75, "Nirnay engine complete", f"{len(nirnay_constituent_dfs)} stocks · {len(nirnay_daily)} trading days")

        # ── Phase 4: Convergence ──────────────────────────────────────────
        console.start_phase("CONVERGENCE", 4, 5)
        progress_bar(progress_container, 78, "Computing convergence", "Cross-validation · DDM filtering")

        console.section("Cross-Validation Setup")
        validator = CrossValidator()
        divergence_detector = CrossSystemDivergenceDetector()

        aarambh_ts = engine.ts_data.copy()
        if active_date != "None" and active_date in data.columns:
            aarambh_ts["Date"] = pd.to_datetime(data[active_date].values)
            aarambh_ts = aarambh_ts.set_index("Date")
        else:
            aarambh_ts["Date"] = np.arange(len(aarambh_ts))
        aarambh_ts = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
        console.item("Aarambh Dates", len(aarambh_ts))

        nirnay_by_date = {}
        if not nirnay_daily.empty:
            nirnay_unique = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")]
            for idx in nirnay_unique.index:
                key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
                nirnay_by_date[key] = nirnay_unique.loc[idx]
            console.item("Nirnay Dates", len(nirnay_by_date))

        console.section("Daily Convergence Scoring")
        overlap_count = 0
        total_dates = len(aarambh_ts.index)
        for i, ts_idx in enumerate(aarambh_ts.index):
            ts_date = ts_idx.date() if hasattr(ts_idx, "date") else pd.Timestamp(ts_idx).date()
            date_str = str(ts_date)
            row_a = aarambh_ts.loc[ts_idx]
            if isinstance(row_a, pd.DataFrame):
                row_a = row_a.iloc[-1]
            aarambh_sig = {
                "conviction_score": float(row_a.get("ConvictionBounded", 0)),
                "oversold_breadth": float(row_a.get("OversoldBreadth", 50)),
                "regime": str(row_a.get("Regime", "NEUTRAL")),
            }
            if date_str in nirnay_by_date:
                row_n = nirnay_by_date[date_str]
                nirnay_stats = {
                    "oversold_pct": float(row_n.get("Oversold_Pct", 50)),
                    "overbought_pct": float(row_n.get("Overbought_Pct", 50)),
                    "avg_unified_osc": float(row_n.get("Avg_Signal", 0)),
                    "regime_bull": float(row_n.get("Regime_Bull_Pct", 33)),
                    "regime_weak_bull": 0,
                    "regime_bear": float(row_n.get("Regime_Bear_Pct", 33)),
                    "regime_weak_bear": 0,
                    "regime_neutral": float(row_n.get("Regime_Neutral", 34)),
                    "num_constituents": int(row_n.get("Total_Analyzed", 0)),
                }
                overlap_count += 1
            else:
                nirnay_stats = {
                    "oversold_pct": 50, "overbought_pct": 50, "avg_unified_osc": 0,
                    "regime_bull": 33, "regime_weak_bull": 0, "regime_bear": 33,
                    "regime_weak_bear": 0, "regime_neutral": 34, "num_constituents": 0,
                }
            validator.compute_convergence(aarambh_sig, nirnay_stats, date_str)
            divergence_detector.detect(aarambh_sig, nirnay_stats, date_str)

            if (i + 1) % 10 == 0 or i == total_dates - 1:
                pct_val = int(78 + (i + 1) / total_dates * 7)
                progress_bar(progress_container, pct_val, "Computing convergence", f"{i + 1}/{total_dates} dates scored")

        console.item("Total Aarambh Dates", len(aarambh_ts))
        console.item("Overlap Dates", overlap_count)
        console.success(f"Convergence scoring complete")

        console.section("Conviction Model")
        progress_bar(progress_container, 86, "Fitting conviction model", "Unified conviction series")
        convergence_df = validator.get_convergence_series()
        conviction_model = UnifiedConvictionModel()
        results = conviction_model.fit(
            convergence_df["convergence_score"].tolist(),
            convergence_df.index.tolist(),
        )
        if results:
            latest = results[-1]
            console.item("Nishkarsh Conviction", f"{latest.nishkarsh_conviction:+.0f}")
            console.item("Signal", latest.nishkarsh_signal)
            console.item("Band", f"[{latest.confidence_lower:.0f}, {latest.confidence_upper:.0f}]")
        console.success(f"Unified conviction: {len(results)} scores computed")

        console.section("Divergence Detection")
        progress_bar(progress_container, 88, "Detecting divergences", "Cross-system divergence analysis")
        events = divergence_detector.get_events()
        console.item("Total Events", len(events))
        if not events.empty:
            event_types = events['divergence_type'].value_counts()
            for etype, count in event_types.items():
                console.item(f"  {etype}", count)
        console.success(f"Divergence analysis complete")

        console.end_phase("CONVERGENCE")
        progress_bar(progress_container, 90, "Convergence complete", f"{overlap_count} overlap dates · {len(events)} divergence events")

        # ── Phase 5: Final Assembly ───────────────────────────────────────
        console.start_phase("FINAL ASSEMBLY", 5, 5)
        progress_bar(progress_container, 92, "Storing results", "Session state · Cache")
        console.section("Session State")

        st.session_state["engine"] = engine
        st.session_state["engine_cache"] = cache_key
        st.session_state["aarambh_ts"] = aarambh_ts
        st.session_state["nirnay_daily"] = nirnay_daily
        st.session_state["nirnay_constituent_dfs"] = nirnay_constituent_dfs
        st.session_state["convergence_df"] = convergence_df
        st.session_state["divergence_events"] = events
        st.session_state["nishkarsh_result"] = results[-1] if results else None
        st.session_state["last_agreement"] = convergence_df["agreement_ratio"].iloc[-1] if not convergence_df.empty else 0

        console.item("Aarambh Engine", "✅ Cached")
        console.item("Nirnay Daily", f"✅ {len(nirnay_daily)} rows")
        console.item("Constituent Results", f"✅ {len(nirnay_constituent_dfs)} stocks")
        console.item("Convergence DF", f"✅ {len(convergence_df)} rows")
        console.item("Nishkarsh Result", f"✅ {results[-1].nishkarsh_signal if results else 'N/A'}")

        console.end_phase("FINAL ASSEMBLY")

        console.summary("RUN SUMMARY", {
            "Total Phases": "5/5 complete",
            "Aarambh Rows": len(engine.ts_data),
            "Nirnay Stocks": len(nirnay_constituent_dfs),
            "Nirnay Trading Days": len(nirnay_daily),
            "Convergence Scores": len(convergence_df),
            "Overlap Dates": overlap_count,
            "Divergence Events": len(events),
            "Status": "SUCCESS",
        })

        console.line('═', 70)
        console._write(f"  {Colors.BOLD}{Colors.GREEN}Analysis Complete{Colors.RESET}")
        console.line('═', 70)
        console._write()

        progress_bar(progress_container, 100, "Analysis complete", f"Nishkarsh: {results[-1].nishkarsh_signal if results else 'N/A'}")
        time.sleep(0.25)
        progress_container.empty()
        st.session_state["run_requested"] = True
        st.rerun()

    engine: FairValueEngine = st.session_state["engine"]
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    regime_stats = engine.get_regime_stats()
    ts = engine.ts_data.copy()
    if active_date != "None" and active_date in data.columns:
        ts["Date"] = pd.to_datetime(data[active_date].values)
    else:
        ts["Date"] = np.arange(len(ts))
    if "aarambh_ts" not in st.session_state:
        st.session_state["aarambh_ts"] = ts.copy()

    nishkarsh_result = st.session_state.get("nishkarsh_result")
    agreement = st.session_state.get("last_agreement", 0)

    # ─── Primary Signal (Above Tabs, Always Visible) ───────────────────────
    _render_primary_signal(nishkarsh_result, agreement, signal)

    # ─── Sidebar Discovery Hint ─────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-hint" onclick="document.querySelector('[data-testid=stSidebarCollapse]').click()" title="Open sidebar for configuration">
            <svg class="sidebar-hint-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="15 18 9 12 15 6"></polyline>
            </svg>
            <span class="sidebar-hint-label">CONFIGURE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Timeframe Filter — with robust persistence ───────────────────────
    if 'tf_selected' not in st.session_state:
        st.session_state.tf_selected = '6M'
    TIMEFRAMES = {'3M': 63, '6M': 126, '1Y': 252, '2Y': 504, 'ALL': None}

    tf_cols = st.columns(len(TIMEFRAMES), gap="small")
    for i, tf in enumerate(TIMEFRAMES.keys()):
        with tf_cols[i]:
            btn_type = "primary" if st.session_state.tf_selected == tf else "secondary"
            if st.button(tf, key=f"tf_{tf}", type=btn_type, width='stretch'):
                st.session_state.tf_selected = tf
                st.rerun()
    selected_tf = st.session_state.tf_selected

    # Ensure timeframe survives config changes by always applying it
    ts_filtered = ts.copy()
    if selected_tf != "ALL":
        if active_date != "None" and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            from pandas import DateOffset
            max_date = ts["Date"].max()
            offsets = {"3M": DateOffset(months=3), "6M": DateOffset(months=6), "1Y": DateOffset(years=1), "2Y": DateOffset(years=2)}
            cutoff = max_date - offsets.get(selected_tf, DateOffset(years=1))
            ts_filtered = ts[ts["Date"] >= cutoff]
        else:
            from core.config import TIMEFRAME_TRADING_DAYS
            n_days = TIMEFRAME_TRADING_DAYS.get(selected_tf, 252)
            ts_filtered = ts.iloc[max(0, len(ts) - n_days):]
    x_axis = ts_filtered["Date"]
    x_title = "Date" if active_date != "None" else "Index"

    # ─── Keyboard Shortcuts Hint ─────────────────────────────────────────
    from streamlit.components.v1 import html as st_html
    st_html(
        """
        <div class="kbd-shortcuts" id="kbd-shortcuts">
            <div class="shortcut-row"><kbd>1</kbd>–<kbd>5</kbd> Switch tabs</div>
            <div class="shortcut-row"><kbd>R</kbd> Run analysis</div>
            <div class="shortcut-row"><kbd>?</kbd> Toggle shortcuts</div>
        </div>
        <script>
        (function() {
            var shortcutsVisible = false;
            var kbdEl = document.getElementById('kbd-shortcuts');
            function showKbd() {
                if (kbdEl) { kbdEl.classList.toggle('visible'); shortcutsVisible = !shortcutsVisible; }
            }
            document.addEventListener('keydown', function(e) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
                if (e.key === '?') { e.preventDefault(); showKbd(); return; }
                if (shortcutsVisible) return;
                var tabKeys = Object();
                tabKeys['1'] = 0; tabKeys['2'] = 1; tabKeys['3'] = 2; tabKeys['4'] = 3; tabKeys['5'] = 4;
                if (e.key in tabKeys) {
                    e.preventDefault();
                    var tabs = document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs[tabKeys[e.key]]) tabs[tabKeys[e.key]].click();
                }
                if (e.key === 'r' || e.key === 'R') {
                    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                        var runBtn = document.querySelector('button[kind="primary"]');
                        if (runBtn) runBtn.click();
                    }
                }
            });
        })();
        </script>
        """,
        height=0,
    )

    # ─── Theme Toggle ────────────────────────────────────────────────────
    from ui.components import render_theme_toggle
    render_theme_toggle()

    # ─── Tabs with Lazy Loading + Error Boundaries ─────────────────────────
    # Track which tabs have been rendered (lazy loading)
    if 'rendered_tabs' not in st.session_state:
        st.session_state.rendered_tabs = set()

    # Get current active tab from URL hash or default to 0
    active_tab_idx = 0  # Streamlit doesn't expose active tab index directly, so we render on demand

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "CONVERGENCE", "AARAMBH", "NIRNAY", "DIAGNOSTICS", "DATA",
    ])

    # Error boundary wrapper
    def _safe_render(name, render_fn):
        """Render a tab with graceful error handling."""
        try:
            render_fn()
        except Exception as e:
            st.markdown(
                f'<div style="background:rgba(251,113,133,0.05);border:1px solid rgba(251,113,133,0.2);'
                f'border-radius:var(--r-md);padding:var(--sp-6);margin:var(--sp-4) 0;">'
                f'<div style="font-family:var(--display);font-size:0.72rem;font-weight:700;color:var(--rose);'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:var(--sp-3);">'
                f'Error in {name}</div>'
                f'<div style="font-family:var(--data);font-size:0.8rem;color:var(--ink-secondary);line-height:1.6;">'
                f'{str(e)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab1:
        st.session_state.rendered_tabs.add(0)
        _safe_render("Convergence", lambda: render_convergence_tab(ts_filtered))
    with tab2:
        st.session_state.rendered_tabs.add(1)
        _safe_render("Aarambh", lambda: render_aarambh_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target))
    with tab3:
        st.session_state.rendered_tabs.add(2)
        _safe_render("Nirnay", lambda: render_nirnay_tab())
    with tab4:
        st.session_state.rendered_tabs.add(3)
        _safe_render("Diagnostics", lambda: render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats))
    with tab5:
        st.session_state.rendered_tabs.add(4)
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    _render_footer()


if __name__ == "__main__":
    main()
