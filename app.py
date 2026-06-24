"""
Nishkarsh v1.4.0 — Main Streamlit entrypoint.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

NISHKARSH — Two systems. One conclusion. Walk-forward valuation + constituent regime intelligence unified by adaptive convergence.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os

# ── BLAS thread pinning (MUST run before numpy/sklearn import) ────────────────
# The walk-forward fits hundreds of small models sequentially. On Streamlit
# Community Cloud the container is throttled to ~1 shared vCPU but the host
# reports many logical CPUs, so OpenBLAS/MKL spawn one thread per reported core
# and thrash — turning each tiny PCA/Ridge solve into a thread-contention storm
# (the #1 reason the walk-forward is far slower on cloud than locally). One
# thread per process is strictly faster for many-small-matrix workloads here.
# os.environ.setdefault → respects any explicit override from the environment.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ── Numba cache OUTSIDE the app tree (MUST run before numba is imported) ──────
# @njit(cache=True) kernels write .nbc/.nbi artifacts. If those land in the app
# directory (default: <module>/__pycache__), Streamlit's file watcher treats each
# write as a source change and reruns the script — restarting the whole pipeline
# mid-compile. Point Numba's cache at the home cache dir (writable, NOT watched).
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "nishkarsh", "numba"),
)

import json
import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
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
# Force PROJECT_ROOT to the FRONT of sys.path (ahead of site-packages) so the
# project's own packages (analytics, core, data, …) always win over any
# same-named package that happens to be installed in the environment. The
# project dirs carry __init__.py so they resolve as regular packages.
PROJECT_ROOT = Path(__file__).resolve().parent
_pr = str(PROJECT_ROOT)
if _pr in sys.path:
    sys.path.remove(_pr)
sys.path.insert(0, _pr)

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
from ui.tabs.tab_precedent import render_precedent_tab
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
from core.config import (
    LOOKBACK_WINDOWS, MIN_DATA_POINTS, MIN_TRAIN_SIZE, MAX_TRAIN_SIZE,
    STALENESS_DAYS, COLOR_RED,
    AARAMBH_PCA_PREDICTORS, AARAMBH_PCA_N_COMPONENTS,
    CUSTOM_PREDICTORS_ENABLED, PCA_PASSTHROUGH, PCA_PASSTHROUGH_ENABLED,
    CALIBRATION_RETURN_LABEL,
    AARAMBH_FORWARD_SIGNAL, AARAMBH_FWD_HORIZON, AARAMBH_FWD_MOM_K,
    PRECEDENT_HOLD_HORIZONS,
)

# Effective passthrough columns — empty when the master toggle is off, so all
# downstream call sites and the cache key stay in sync with one switch.
_PASSTHROUGH = tuple(PCA_PASSTHROUGH) if PCA_PASSTHROUGH_ENABLED else ()


# ─── Per-config result cache ─────────────────────────────────────────────────
# The full result of an analysis is the set of session-state keys below. We
# snapshot them per cache_key so revisiting a previously-computed config (e.g.
# the user toggles a predictor set off and back on) restores instantly instead
# of recomputing the whole 5-phase pipeline. Bounded (LRU) to cap memory.
_BUNDLE_KEYS = (
    "engine", "aarambh_ts", "nirnay_daily", "nirnay_constituent_dfs",
    "convergence_df", "divergence_events", "nishkarsh_result", "last_agreement",
    "nishkarsh_conv_normalized",
    "intelligence_active_weights", "intelligence_active_thresholds",
    "intelligence_active_profile",
)
_RESULTS_CACHE_MAX = 3  # keep the last N configs


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
        tagline="Walk-Forward Valuation + Constituent Regime Intelligence  |  Unified Convergence"
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


def _render_primary_signal(nishkarsh_norm, agreement, aarambh_signal) -> None:
    """Render the hero Nishkarsh convergence signal card.

    The conviction value now reflects the **normalized convergence** shown in
    the Unified Signal plot \u2014 i.e. the average of the two systems' z-scored
    (and clipped) signals, in ``[-1, +1]``. The signal classification and the
    interpretation paragraph follow the same scale.
    """
    if nishkarsh_norm:
        conv = nishkarsh_norm["value"]
        sig = nishkarsh_norm["signal"]
        a_norm = nishkarsh_norm["aarambh_norm"]
        n_norm = nishkarsh_norm["nirnay_norm"]
        agreement_text = "Strong" if agreement > 0.7 else "Moderate" if agreement > 0.5 else "Weak"
        explanation = (
            f"Normalized convergence {conv:+.2f} ({sig}). "
            f"Aarambh contribution: {a_norm:+.2f}; Nirnay contribution: {n_norm:+.2f}. "
            f"Agreement: {agreement:.0%} ({agreement_text}). "
            f"{'Both systems aligned.' if agreement > 0.6 else 'Mixed signals \u2014 wait for clearer convergence.'}"
        )
    else:
        conv = float(aarambh_signal.get("conviction_score", 0))
        sig = aarambh_signal.get("signal", "HOLD")
        explanation = f"Conviction: {conv:+.2f} ({sig})."

    render_nishkarsh_signal_card(
        signal=sig,
        conviction=conv,
        agreement=agreement,
        explanation=explanation,
    )
    section_gap()


def _render_model_passport_sidebar(current_universe: str, current_index: str | None = None) -> None:
    """Sidebar Passport — visible in every mode.

    Faithful port of Sanket's `_render_model_passport_sidebar`, adapted to
    Nishkarsh's (universe, index) keying. Surfaces:
      • Profile state (Default / Calibrated / Calibrated · ⚠ on mismatch)
      • Trained-on label · Train IC · Val IC · Updated timestamp
      • Universe-mismatch warning when the saved profile was fit on a
        different universe than the active sidebar selection
      • Import / Export / Reset controls

    Caller must be inside a ``with st.sidebar:`` context.
    """
    from convergence import intelligence as intel

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Model Passport</div>', unsafe_allow_html=True)

    # Intelligence-mode toggle. Default ON. When OFF, factory weights are
    # used regardless of any saved profile.
    intelligence_mode = st.toggle(
        "Intelligence Mode",
        value=bool(st.session_state.get("intelligence_mode", True)),
        help=(
            "When ON, Nishkarsh uses the persisted calibrated profile for the "
            "selected universe (if one exists). When OFF, Nishkarsh runs on "
            "the factory 0.30 / 0.25 / 0.25 / 0.20 dimension weights and "
            "symmetric ±0.3 / ±0.5 thresholds."
        ),
        key="passport_intel_toggle",
    )
    st.session_state["intelligence_mode"] = intelligence_mode

    # What profile (if any) is saved for THIS universe?
    saved_profile = intel.load_profile_for(current_universe, current_index)

    # Status card values
    if intelligence_mode and saved_profile is not None:
        cal_universe = saved_profile.universe
        cal_index    = saved_profile.selected_index
        cal_label    = cal_index or cal_universe or "—"
        cur_label    = current_index or current_universe or "—"
        universe_mismatch = cal_label != "—" and cur_label != "—" and cal_label != cur_label
        train_v = float(saved_profile.train_ic or 0.0)
        val_v   = float(saved_profile.val_ic or 0.0)
        train_str = f"{train_v:+.3f}"
        val_str   = f"{val_v:+.3f}"
        updated   = saved_profile.timestamp or "—"
        train_color = "var(--emerald)" if train_v > 0 else "var(--rose)"
        val_color   = "var(--emerald)" if val_v   > 0 else "var(--rose)"
        if universe_mismatch:
            profile_label = "Calibrated · ⚠"
            card_class = "warning"
        else:
            profile_label = "Calibrated"
            card_class = "success" if (val_v > 0 and train_v > 0) else "warning"
    elif not intelligence_mode:
        cal_label = "—"
        profile_label = "Default · Off"
        train_str = val_str = updated = "—"
        train_color = val_color = "var(--ink-secondary)"
        card_class = "neutral"
        universe_mismatch = False
    else:
        cal_label = "—"
        profile_label = "Default"
        train_str = val_str = updated = "—"
        train_color = val_color = "var(--ink-secondary)"
        card_class = "neutral"
        universe_mismatch = False

    def _trim(s: str, n: int = 22) -> str:
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    cal_label_disp = _trim(cal_label)

    st.markdown(f"""
    <div class="metric-card {card_class}" style="
            min-height:auto;
            padding:0.85rem 0.95rem;
            margin-bottom:0.7rem;
            animation:none;">
        <h4 style="margin:0 0 0.3rem 0;">Profile</h4>
        <h2 style="font-size:1.05rem; margin:0 0 0.7rem 0; letter-spacing:-0.01em;">{profile_label}</h2>
        <div style="display:flex; flex-direction:column; gap:0.32rem;
                    padding-top:0.55rem;
                    border-top:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Trained on</span>
                <span style="color:var(--ink-secondary); font-weight:500; max-width:62%; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{cal_label_disp}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Train IC</span>
                <span style="color:{train_color}; font-weight:600;">{train_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Val IC</span>
                <span style="color:{val_color}; font-weight:600;">{val_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.6rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Updated</span>
                <span style="color:var(--ink-secondary);">{updated}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if universe_mismatch:
        st.markdown(f"""
        <div style="font-family:var(--data); font-size:0.62rem; color:var(--amber);
                    background:rgba(212,168,83,0.08);
                    border:1px solid rgba(212,168,83,0.22);
                    border-radius:6px; padding:0.55rem 0.65rem;
                    margin-bottom:0.7rem; line-height:1.45;">
            <span style="font-weight:700;">Profile mismatch — calibrated weights still active.</span><br>
            Profile fit on <b>{_trim(cal_label, 28)}</b><br>
            Active universe is <b>{_trim(current_index or current_universe, 28)}</b><br>
            <span style="color:var(--ink-tertiary);">Weights learned for one universe do not generalise.
            Reset to defaults or run a new calibration for the current selection.</span>
        </div>
        """, unsafe_allow_html=True)

    # Import / Export / Reset controls
    with st.expander("↑ Import Profile", expanded=False):
        uploaded = st.file_uploader(
            " ", type=["json"], label_visibility="collapsed", key="passport_uploader",
        )
        if uploaded is not None:
            try:
                payload = json.load(uploaded)
                if isinstance(payload, dict) and "weights" in payload:
                    imported = intel.IntelligenceProfile.from_dict(payload)
                    intel.save_profile(imported)
                    st.toast(f"Profile imported · {imported.universe}", icon="✅")
                    st.rerun()
                else:
                    st.error("Import failed: file is not a valid profile dict (missing 'weights').")
            except Exception as e:
                st.error(f"Import failed: {e}")

    if saved_profile is not None:
        export_payload = saved_profile.to_dict()
        ts_slug = (saved_profile.timestamp or "").split(" ")[0] or "snapshot"
        fname = f"nishkarsh_profile_{_trim((saved_profile.selected_index or saved_profile.universe or 'profile').replace(' ', '_'), 30)}_{ts_slug}.json"
        st.download_button(
            "↓ Export Profile",
            data=json.dumps(export_payload, indent=2, default=str),
            file_name=fname,
            mime="application/json",
            width="stretch",
            key="passport_export",
        )
        if st.button("↺ Reset to Defaults", width="stretch", key="passport_reset"):
            intel.delete_profile(saved_profile.universe, saved_profile.selected_index)
            # Streamlit's toast `icon=` only accepts emojis from a curated
            # whitelist; "↺" (U+21BA, our "Reset" mark used on the button) is
            # rejected. Drop the icon — the "↺" stays on the button text where
            # the user actually sees it.
            st.toast("Profile reset.")
            st.rerun()


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
# CACHED ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _cached_macro_factors(macro_df, n_components, passthrough, stationarize, return_loadings):
    """Cache the deterministic causal macro-factor PCA on the panel's content.

    The factors are a pure function of (macro_df, params). The macro panel comes
    from the disk-cached fetches and is independent of target/feature selection,
    so this computes once per data snapshot and is reused across analysis re-runs
    (changing target/predictors hits the cache). Hashing the panel costs ~15 ms —
    negligible versus the minutes the PCA takes on a cold run. st.cache_data
    returns a fresh copy each call, so downstream mutation can't corrupt the entry.
    """
    from analytics.factors import build_causal_macro_factors
    return build_causal_macro_factors(
        macro_df,
        n_components=n_components,
        passthrough=list(passthrough) if passthrough else None,
        stationarize=stationarize,
        return_loadings=return_loadings,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title=f"NISHKARSH | Unified Convergence",
        page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
        layout="wide", initial_sidebar_state="collapsed",
    )
    inject_css()

    # ─── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center;padding:0.5rem 0 0.75rem 0;">
            <div style="font-family:var(--display);font-size:1.35rem;font-weight:700;color:var(--amber);letter-spacing:0.04em;">NISHKARSH</div>
            <div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.6rem;margin-top:0.1rem;letter-spacing:0.06em;text-transform:uppercase;">निष्कर्ष | Unified Convergence</div>
        </div>
        <hr style="margin: 0.5rem 0; opacity: 0.1;">
        """,
            unsafe_allow_html=True,
        )

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

        # Target / Date changes apply IMMEDIATELY (no staging). Changing the
        # target and seeing nothing happen — because the Apply button was buried
        # in the collapsed Predictor expander — was a discoverability trap.
        # Predictor edits still stage behind Apply (you toggle several at once).
        if target_col != active_target_state or date_col != active_date_state:
            st.session_state["active_target"] = target_col
            st.session_state["active_date_col"] = date_col
            # A column can't be both the target and a predictor.
            if "active_features" in st.session_state:
                st.session_state["active_features"] = tuple(
                    f for f in st.session_state["active_features"] if f != target_col
                )
            st.session_state.pop("engine", None)
            st.session_state.pop("engine_cache", None)
            st.rerun()

        available = [c for c in numeric_cols if c != target_col]
        valid_defaults = [p for p in ("AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH", "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO", "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB") if p in available]

        if "active_features" not in st.session_state:
            st.session_state["active_features"] = tuple(valid_defaults or available[:3])

        with st.expander("Predictor Columns", expanded=False):
            st.caption("Showing the predictors this run used — select, then Apply to recompute.")
            # Key the widget to the ACTIVE (run) predictor set. Streamlit
            # persists widget state across reruns and ignores `default=`, so
            # when active_features changes programmatically (e.g. a target
            # change drops a predictor), a plain multiselect would keep showing
            # its stale selection. Re-keying on the active set forces the widget
            # to re-sync to what the system actually ran with, while local
            # staging edits (which don't change active_features until Apply)
            # keep a stable key and persist.
            _pred_key = "pred_select::" + "|".join(sorted(st.session_state["active_features"]))
            staging_features = st.multiselect(
                "Predictor Columns", options=available,
                default=[f for f in st.session_state["active_features"] if f in available],
                label_visibility="collapsed",
                key=_pred_key,
            )
            if not staging_features:
                st.warning("Select at least one predictor.")
                staging_features = [f for f in st.session_state["active_features"] if f in available] or available[:3]

            staging_set = set(staging_features)
            active_set = set(st.session_state["active_features"])
            has_pred_changes = staging_set != active_set

            if has_pred_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                parts = []
                if added:
                    parts.append(f"+{len(added)} added")
                if removed:
                    parts.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(parts)}")

            if st.button("Apply Predictors" if has_pred_changes else "No changes", disabled=not has_pred_changes, type="primary" if has_pred_changes else "secondary"):
                if has_pred_changes:
                    st.session_state["active_features"] = tuple(staging_features)
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.rerun()

            if len(st.session_state["active_features"]) != len(available):
                st.info(f"Active: {len(st.session_state['active_features'])}/{len(available)} predictors")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if "run_analysis" in st.session_state and st.session_state.get("run_analysis"):
            if st.button("Reset Analysis", type="secondary", width="stretch"):
                st.session_state.pop("data", None)
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state.pop("run_analysis", None)
                st.session_state.pop("nishkarsh_result", None)
                st.rerun()

            # Force a live re-pull of the market data (macro + constituent OHLCV), then
            # recompute — for when the data is stale/partial. Reset = re-run on cached
            # data (fast); Refresh = re-fetch live + re-run (slower). Snapshot-preserving:
            # if the live pull fails (rate-limit / circuit open), the cache's stale
            # fallback keeps the app working on last-good data instead of going empty.
            if st.button("Refresh Data", type="secondary", width="stretch"):
                from data.cache import begin_force_refresh
                begin_force_refresh()   # next fetches bypass TTL; disk snapshot kept
                for _k in ("engine", "engine_cache", "aarambh_engine", "aarambh_fit_key",
                           "wf_results", "results_cache", "nishkarsh_result"):
                    st.session_state.pop(_k, None)
                st.rerun()
            st.caption("Reset = re-run on cached data · Refresh = re-fetch live, then re-run")

        # ── Model Passport (Sanket-style) ──────────────────────────────
        # Surfaces the active calibrated profile (Intelligence Mode).
        _current_universe = "NIFTY 50"
        _current_index = st.session_state.get("nishkarsh_index", _current_universe)
        _render_model_passport_sidebar(_current_universe, _current_index)

        st.markdown('<hr style="margin: 1rem 0 0.75rem 0; opacity: 0.05;">', unsafe_allow_html=True)
        st.markdown(
            '<div class="system-spec">'
            f'<div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">{VERSION}</span></div>'
            '<div class="spec-row"><span class="spec-label">Engine</span><span class="spec-value">Convergence</span></div>'
            '<div class="spec-row"><span class="spec-label">Data</span><span class="spec-value">Sheets + yfinance</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ─── Resolve active configuration ──────────────────────────────────────
    active_target = st.session_state.get("active_target", target_col)
    active_features = list(st.session_state.get("active_features", [c for c in numeric_cols if c != active_target][:3]))
    active_date = st.session_state.get("active_date_col", date_col)

    # ─── Data staleness warning ────────────────────────────────────────────
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors="coerce", dayfirst=True).dropna()
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
        data[active_date] = pd.to_datetime(data[active_date], errors="coerce", dayfirst=True)
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

    if AARAMBH_FORWARD_SIGNAL:
        # ── Predictive representation (experimental) ──────────────────────
        # Features X[t] = trailing AARAMBH_FWD_MOM_K-day CHANGE of each predictor
        # (momentum, known at t). Target y[t] = forward AARAMBH_FWD_HORIZON-day
        # log-change of the target (t→t+h). The last h rows have no realized
        # future — kept as live forecasts (target filled 0 so the regression
        # doesn't choke; the signal there IS the prediction). Drop only the
        # warmup head with unformed momentum.
        #
        # NOTE: predictor momentum uses a sign-agnostic K-day difference (NOT a
        # log return) because Nishkarsh predictors include breadth ratios and
        # rate spreads that can be ≤0 — log would be invalid and wipe the panel.
        # The engine's StandardScaler normalizes the differing feature scales.
        # The target (PE) is strictly positive, so its forward return is a clean
        # log change.
        _lvl = data[[active_target] + active_features].astype(float)
        _tlog = np.log(_lvl[active_target].where(_lvl[active_target] > 0))
        _fwd = _tlog.shift(-AARAMBH_FWD_HORIZON) - _tlog
        _mom = _lvl[active_features].diff(AARAMBH_FWD_MOM_K).replace([np.inf, -np.inf], np.nan)
        _valid = _mom.notna().all(axis=1).to_numpy()
        data = data.loc[_valid].reset_index(drop=True)
        X = _mom.loc[_valid].to_numpy(dtype=np.float64)
        y = np.nan_to_num(_fwd.loc[_valid].to_numpy(dtype=np.float64), nan=0.0)
        cache_key = f"fwd{AARAMBH_FWD_HORIZON}m{AARAMBH_FWD_MOM_K}|{active_target}|{'|'.join(sorted(active_features))}|{len(data)}"
    else:
        X, y = data[active_features].values, data[active_target].values
        cache_key = f"{active_target}|{'|'.join(sorted(active_features))}|{len(data)}|pca{int(AARAMBH_PCA_PREDICTORS)}|cf{int(CUSTOM_PREDICTORS_ENABLED)}|pt{'-'.join(_PASSTHROUGH)}"
    if st.session_state.get("engine_cache") != cache_key:
        # ── Restore from the per-config result cache if this exact config was
        # already computed this session (e.g. the user toggled a predictor set
        # off and came back) — full reuse, no recompute. ────────────────────
        _rcache = st.session_state.setdefault("results_cache", {})
        if cache_key in _rcache:
            for _bk, _bv in _rcache[cache_key].items():
                st.session_state[_bk] = _bv
            _rcache[cache_key] = _rcache.pop(cache_key)  # mark most-recently-used
            st.session_state["engine_cache"] = cache_key
            console.header("NISHKARSH — Cached Result Restored", f"v{VERSION}")
            console.success(f"Restored {active_target} from session cache — no recompute")
            st.rerun()
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
        progress_bar(progress_container, 2, "Fetching Constituents", "niftyindices.com · CSV parse")

        console.section("Constituent Fetch")
        constituents, src_msg = fetch_nifty50_constituents()
        console.item("Source", src_msg)
        console.item("Count", len(constituents))
        console.item("Symbols", f"{constituents[0]}, {constituents[1]}, {constituents[2]}...")
        console.success(f"Fetched {len(constituents)} Nifty 50 constituents")
        progress_bar(progress_container, 5, "Fetching Macro Data", "yfinance · Global Macro ETFs · FX · Commodities")

        console.section("Macro Data")
        # History window for the yfinance side (constituents + macro). A longer
        # window grows the Aarambh↔Nirnay overlap (was 2y → ~10%; 5y → ~25%),
        # which stabilizes the calibration sample. NOTE: the constituent set is
        # TODAY's index membership, so a longer window adds survivorship bias —
        # 7y is the deliberate ceiling. It will NOT create edge (PE/convergence
        # are non-forecastable here); it only tightens the calibration readout.
        # Beyond ~7y, survivorship distortion outweighs the marginal sample gain.
        _HISTORY_YEARS = 7
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=365 * _HISTORY_YEARS)
        # The macro and constituent-OHLCV pulls are independent yfinance batches.
        # Fire them concurrently: the circuit breaker releases its lock during the
        # network call and the disk cache is lock-guarded, so the two cold fetches
        # overlap instead of running back-to-back (saves the macro fetch's wall
        # time on a cache-miss run; warm-cache runs are unaffected). Results are
        # collected via .result() below, preserving the original logging order.
        _io_pool = ThreadPoolExecutor(max_workers=2)
        _macro_future = _io_pool.submit(fetch_macro_live, start_date, end_date)
        _ohlcv_future = (
            _io_pool.submit(fetch_constituent_ohlcv, constituents, start_date, end_date)
            if constituents else None
        )
        macro_df = _macro_future.result()
        console.item("History Window", f"{_HISTORY_YEARS}y ({start_date.date()} → {end_date.date()}) · ⚠ survivorship-biased to current constituents")
        console.item("Date Range", f"{start_date.date()} to {end_date.date()}")
        if not macro_df.empty:
            console.item("YF Columns", f"{len(macro_df.columns)} symbols")
            console.item("Rows", len(macro_df))
            console.success(f"Macro data: {len(macro_df.columns)} symbols × {len(macro_df)} rows")
        else:
            console.warning("No macro data available")
        progress_bar(progress_container, 10, "Fetching OHLCV Data", f"yfinance · {len(constituents)} constituents")

        console.section("Constituent OHLCV")
        constituent_ohlcv = {}
        if _ohlcv_future is not None:
            constituent_ohlcv = _ohlcv_future.result()  # started concurrently above
            console.item("Requested", len(constituents))
            console.item("Downloaded", len(constituent_ohlcv))
            if constituent_ohlcv:
                sample = list(constituent_ohlcv.items())[0]
                console.item("Sample", f"{sample[0]}: {len(sample[1])} rows")
            console.success(f"OHLCV data for {len(constituent_ohlcv)} constituents")
        _io_pool.shutdown(wait=False)  # both fetches collected; release the pool
        progress_bar(progress_container, 15, "Assembling Macro Indicators", "Bond Yields from Google Sheets")

        console.section("Nirnay Macro Assembly")
        nirnay_macro_df = macro_df.copy() if macro_df is not None and not macro_df.empty else pd.DataFrame()
        nirnay_bond_cols = ["IN10Y", "IN02Y", "IN30Y", "US10Y", "US30Y", "US02Y"]
        available_bond_cols = [c for c in nirnay_bond_cols if c in df.columns]
        if available_bond_cols:
            bond_data = df[available_bond_cols].copy()
            if "DATE" in df.columns:
                bond_data.index = pd.to_datetime(df["DATE"], dayfirst=True)
            nirnay_macro_df = nirnay_macro_df.join(bond_data, how="outer").sort_index().ffill()
            console.item("YF Symbols", len(macro_df.columns) if not macro_df.empty else 0)
            console.item("Bond Yields", f"{len(available_bond_cols)} ({', '.join(available_bond_cols)})")
            console.success(f"Combined macro: {len(nirnay_macro_df.columns)} indicators × {len(nirnay_macro_df)} rows")
        # Guarantee a unique macro index — an outer-join against a duplicated
        # bond-yield date (or a tz-duplicated yfinance row) would otherwise
        # multiply rows when this panel is left-joined into each constituent.
        if not nirnay_macro_df.empty and nirnay_macro_df.index.duplicated().any():
            nirnay_macro_df = nirnay_macro_df[~nirnay_macro_df.index.duplicated(keep="last")]
        macro_cols_list = list(nirnay_macro_df.columns) if not nirnay_macro_df.empty else []

        # ── Custom engineered predictors ─────────────────────────────────
        # Yield spreads, real rates, credit/commodity ratios, FX momentum,
        # cross-asset composites — all causal & stationary (see
        # docs/CUSTOM_PREDICTORS.md). Built from the macro panel + the sheet's
        # rate/breadth/valuation columns, then appended to the macro panel so
        # BOTH the MMR factor gate and Aarambh's combined PCA ingest them. The
        # PE/PB/DY-embedding features are dropped from Aarambh only (they embed
        # its target → leakage); they stay in MMR (target = stock price).
        _leakage_cols: frozenset = frozenset()
        if CUSTOM_PREDICTORS_ENABLED:
            try:
                from analytics.custom_features import build_custom_features, AARAMBH_LEAKAGE_EXCLUDE
                _leakage_cols = AARAMBH_LEAKAGE_EXCLUDE
                _feat_src = nirnay_macro_df.copy() if not nirnay_macro_df.empty else pd.DataFrame()
                _sheet_extra = ["REPO", "INIRYY", "AD_RATIO", "REL_AD_RATIO", "REL_BREADTH",
                                "NIFTY50_PE", "NIFTY50_DY", "NIFTY50_PB",
                                "IN10Y", "IN02Y", "IN30Y", "US10Y", "US02Y", "US30Y"]
                # Only the sheet columns NOT already in the macro panel — the bond
                # yields (IN10Y…) are in both, and joining duplicates raises.
                _avail_extra = [
                    c for c in _sheet_extra
                    if c in df.columns and c not in getattr(_feat_src, "columns", [])
                ]
                if _avail_extra and "DATE" in df.columns:
                    _extra = df[_avail_extra].copy()
                    _extra.index = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
                    _extra = _extra[~_extra.index.isna()]
                    _extra = _extra[~_extra.index.duplicated(keep="last")].sort_index()
                    _feat_src = _feat_src.join(_extra, how="outer").sort_index().ffill() if not _feat_src.empty else _extra
                _custom_df = build_custom_features(_feat_src)
                if not _custom_df.empty and not nirnay_macro_df.empty:
                    _custom_on_macro = _custom_df.reindex(nirnay_macro_df.index, method="ffill")
                    _new_cols = [c for c in _custom_on_macro.columns if c not in nirnay_macro_df.columns]
                    if _new_cols:
                        nirnay_macro_df = nirnay_macro_df.join(_custom_on_macro[_new_cols], how="left")
                        macro_cols_list = list(nirnay_macro_df.columns)
                        _n_leak = len([c for c in _new_cols if c in _leakage_cols])
                        console.item(
                            "Custom Predictors",
                            f"{len(_new_cols)} engineered (causal, stationary) — "
                            f"all → MMR · {len(_new_cols) - _n_leak} → Aarambh ({_n_leak} PE-leakage excluded)",
                        )
            except Exception as _cf_e:
                console.warning(f"custom features failed: {_cf_e}")
        else:
            console.item("Custom Predictors", "disabled (CUSTOM_PREDICTORS_ENABLED = False)")

        # Compress the wide, collinear macro panel into a few ORTHOGONAL causal
        # factors (expanding-window PCA). MMR then selects from de-duplicated
        # drivers instead of 90 near-identical bond/FX series — cuts selection
        # variance and the spurious "top correlated macro" problem. Falls back
        # to the raw panel if PCA is unavailable or the panel is small.
        # Keep a reference to the FULL raw macro panel before MMR-factor
        # replacement — Aarambh's combined-predictor PCA (below) uses it.
        _raw_macro_panel = nirnay_macro_df.copy() if not nirnay_macro_df.empty else pd.DataFrame()
        _n_raw_macros = len(macro_cols_list)
        if _n_raw_macros > 12:
            _macro_factors = _cached_macro_factors(
                nirnay_macro_df, n_components=8, passthrough=tuple(_PASSTHROUGH),
                stationarize=False, return_loadings=False,
            )
            if not _macro_factors.empty and _macro_factors.shape[1] >= 2:
                nirnay_macro_df = _macro_factors
                macro_cols_list = list(_macro_factors.columns)
                _pt = [c for c in _PASSTHROUGH if c in macro_cols_list]
                console.item("Macro Factors", f"{len(macro_cols_list) - len(_pt)} causal PCA factors"
                             + (f" + {len(_pt)} passthrough ({', '.join(_pt)})" if _pt else "")
                             + f" ← {_n_raw_macros} raw macros")
            else:
                console.item("Macro Factors", f"PCA unavailable — using {_n_raw_macros} raw macros")
        console.end_phase("DATA ACQUISITION")
        progress_bar(progress_container, 20, "Data Acquisition Complete", f"{len(constituent_ohlcv)} Constituents · {len(nirnay_macro_df.columns)} Macros")

        # ── Phase 2: Aarambh FairValueEngine ─────────────────────────────
        console.start_phase("AARAMBH ENGINE", 2, 5)
        progress_bar(progress_container, 20, "Running Aarambh Engine", f"Walk-Forward · {len(active_features)} Predictors · {len(data)} Rows")

        console.section("Engine Configuration")
        console.item("Target", active_target)
        console.item("Features", f"{len(active_features)}: {', '.join(active_features[:5])}...")
        console.item("Observations", f"{len(data)} rows")
        # MIN_DATA_POINTS is the data *requirement*, not the training window —
        # the walk-forward actually trains on a MIN_TRAIN_SIZE→MAX_TRAIN_SIZE
        # expanding window. Report both honestly (the old "Min Train Size: 1500"
        # label conflated the two).
        console.item("Min Data Required", MIN_DATA_POINTS)
        console.item("Train Window", f"{MIN_TRAIN_SIZE}→{MAX_TRAIN_SIZE} (expanding, capped)")
        console.item("Lookback Windows", f"{LOOKBACK_WINDOWS}")

        # ── Combined-predictor PCA gate ─────────────────────────────────
        # Push (sheet predictors + the full macro panel) through the same causal
        # expanding-window PCA as MMR and train Aarambh on the orthogonal
        # factors. Macro columns reindexed onto the Aarambh dates; pre-yfinance
        # history is flat-filled, so old-date factors are sheet-driven.
        _aar_feature_names = active_features
        if (AARAMBH_PCA_PREDICTORS and not AARAMBH_FORWARD_SIGNAL and not _raw_macro_panel.empty
                and active_date != "None" and active_date in data.columns):
            try:
                _dates = pd.to_datetime(data[active_date])
                _sheet = data[active_features].reset_index(drop=True)
                _macro_al = (
                    _raw_macro_panel.sort_index()
                    .reindex(_dates.values, method="ffill")
                    .reset_index(drop=True)
                )
                _combined = pd.concat([_sheet, _macro_al], axis=1)
                # The bond yields (IN10Y, IN30Y, US10Y…) live in BOTH the sheet
                # predictors and the macro panel — drop the duplicate columns so
                # they aren't double-counted in the PCA.
                _combined = _combined.loc[:, ~_combined.columns.duplicated()]
                # Drop PE/PB/DY-embedding custom features from Aarambh — they
                # contain the target (PE) → leakage. (They remain in MMR.)
                _drop_leak = [c for c in _combined.columns if c in _leakage_cols]
                if _drop_leak:
                    _combined = _combined.drop(columns=_drop_leak)
                _combined.index = pd.RangeIndex(len(_combined))
                _ncomp = int(min(AARAMBH_PCA_N_COMPONENTS, _combined.shape[1]))
                # stationarize=True: the macro panel holds non-stationary price
                # levels; without rolling-z + clip they make the level regression
                # extrapolate (R² → −47). Stationarizing bounds every input.
                _aar_factors, _aar_loadings = _cached_macro_factors(
                    _combined, n_components=_ncomp, passthrough=tuple(_PASSTHROUGH),
                    stationarize=True, return_loadings=True,
                )
                if not _aar_factors.empty and _aar_factors.shape[1] >= 2:
                    X = _aar_factors.fillna(0.0).to_numpy(dtype=np.float64)
                    _aar_feature_names = list(_aar_factors.columns)
                    _macro_start = _raw_macro_panel.dropna(how="all").index.min()
                    _ptA = [c for c in _PASSTHROUGH if c in _aar_feature_names]
                    console.item(
                        "Predictor PCA",
                        f"{X.shape[1] - len(_ptA)} causal factors"
                        + (f" + {len(_ptA)} passthrough ({', '.join(_ptA)})" if _ptA else "")
                        + f" ← {len(active_features)} sheet + {_raw_macro_panel.shape[1]} macro predictors (stationarized)",
                    )
                    console.item(
                        "  Macro real-history",
                        f"from {pd.Timestamp(_macro_start).date() if pd.notna(_macro_start) else 'n/a'} "
                        f"(earlier dates: sheet-driven, macros flat-filled)",
                    )
                    # Name each top factor by its dominant inputs (interpretability)
                    if not _aar_loadings.empty:
                        for _pc in list(_aar_loadings.index)[:5]:
                            _top = _aar_loadings.loc[_pc].abs().sort_values(ascending=False).head(3)
                            console.item(f"  {_pc} ≈", ", ".join(str(c) for c in _top.index))
            except Exception as _pe:
                console.warning(f"predictor PCA failed: {_pe} — using raw {len(active_features)} predictors")

        if AARAMBH_FORWARD_SIGNAL:
            console.item(
                "Mode",
                f"PREDICTIVE · forecast {AARAMBH_FWD_HORIZON}d forward Δ{active_target} "
                f"from {AARAMBH_FWD_MOM_K}d momentum ({len(active_features)} raw predictors, PCA gate bypassed)",
            )

        console.section("Walk-Forward Regression")

        # The walk-forward refits the ensemble across hundreds of expanding-window
        # chunks (HuberRegressor dominates the cost). On a constrained host this
        # runs for minutes emitting nothing to the console, which looks frozen.
        # Emit a throttled heartbeat (~every 4s) so the log shows it is alive,
        # without spamming one line per chunk. The Streamlit progress bar is still
        # updated on every chunk as before.
        _aar_last_log = [0.0]

        def _aar_progress(pct: float, msg: str) -> None:
            progress_bar(progress_container, int(20 + pct * 20), "Running Aarambh Engine", msg)
            now = time.time()
            if pct >= 1.0 or now - _aar_last_log[0] >= 4.0:
                _aar_last_log[0] = now
                console.detail(f"Walk-forward {pct * 100:3.0f}% · {msg}")

        # Reuse an already-fit engine for this exact config if a prior (possibly
        # interrupted) execution in THIS session already produced one. engine_cache
        # is only set at the end of Phase 5, so a Streamlit rerun mid-pipeline
        # (yfinance retry, cloud reconnect, stray interaction) would otherwise
        # re-enter this block and re-run the expensive walk-forward. Keyed by
        # cache_key → identical inputs → identical fit, so reuse is safe.
        if (st.session_state.get("aarambh_fit_key") == cache_key
                and isinstance(st.session_state.get("aarambh_engine"), FairValueEngine)):
            engine = st.session_state["aarambh_engine"]
            console.item("Walk-Forward", "reused cached fit (resumed run)")
            progress_bar(progress_container, 40, "Aarambh Engine Reused", "Cached walk-forward fit")
        else:
            engine = FairValueEngine()
            engine.fit(X, y, feature_names=_aar_feature_names, forward_signal=AARAMBH_FORWARD_SIGNAL, progress_callback=_aar_progress)
            # Carry the raw target LEVEL on the engine output (forward-signal mode
            # otherwise leaves only return-scale columns). Used by the Precedent tab
            # for the analog price path / forward returns. Length-guarded — ts_data
            # rows align 1:1 with the _valid-filtered `data` used to build X/y.
            if len(engine.ts_data) == len(data):
                engine.ts_data["Price"] = data[active_target].values
            st.session_state["aarambh_engine"] = engine
            st.session_state["aarambh_fit_key"] = cache_key

        sig = engine.get_current_signal()
        stats = engine.get_model_stats()
        console.section("Engine Results")
        console.item("Signal", f"{sig['signal']} ({sig['display_strength']})")
        console.item("Conviction", f"{sig['conviction_score']:+.0f}")
        # Lead with the honest skill metric, not the persistence-inflated R².
        if AARAMBH_FORWARD_SIGNAL:
            console.item("Mode", f"PREDICTIVE · forecasting {AARAMBH_FWD_HORIZON}d forward Δlog({active_target})")
            console.item("R² vs RW (skill)", f"{stats['r2_vs_rw']:+.3f}  ← forecast vs naive; the magnitude metric")
            console.item("OOS R² (forecast)", f"{stats['r2_oos']:.3f}  (returns forecast — magnitude R²≈0 is NORMAL; the edge is in IC → see Intelligence / Walk-Forward IC below)")
        else:
            console.item("Mode", "FAIR-VALUE · regressing the target level (mean-reversion)")
            console.item("R² vs RW (skill)", f"{stats['r2_vs_rw']:+.3f}  ← negative = worse than naive forecast")
            console.item("OOS R² (vs mean)", f"{stats['r2_oos']:.3f}  ⚠ persistence-inflated on PE")
        if AARAMBH_FORWARD_SIGNAL:
            # The engine's edge_assessment/tradeable come from the level/magnitude
            # forward-edge test, which is not the verdict in predictive mode (the
            # tradeable edge is DIRECTIONAL — measured downstream by IC).
            console.item("Magnitude Edge", f"{sig['edge_assessment']}  (level test — NOT the predictive verdict)")
            console.item("Directional verdict", "→ Directional Convergence Test + Walk-Forward IC (below)")
        else:
            console.item("EDGE", sig['edge_assessment'])
            console.item("Tradeable", sig['tradeable'])
        console.item("OU Half-Life", f"{sig['ou_half_life']:.0f}d" + ("" if sig['half_life_meaningful'] else "  ⚠ not meaningful (residuals not mean-reverting)"))
        console.item("Hurst", f"{sig['hurst']:.2f}")
        console.item("Oversold Breadth", f"{sig['oversold_breadth']:.0f}%")
        if AARAMBH_FORWARD_SIGNAL:
            console.item("Note", "Predictive mode — magnitude forecast is weak by nature (R²≈0 is expected); the tradeable edge is DIRECTIONAL, graded by the IC sections below.")
        elif sig.get('inverted'):
            console.warning("Aarambh: signal is INVERTED on history (BUY/SELL preceded the WRONG direction, p<0.05) — do NOT trade as-is")
        elif not sig['tradeable']:
            console.warning("Aarambh: NO FORECAST EDGE on this series — signal is valuation context, not a tradeable call")
        console.success(f"Aarambh engine complete | {len(engine.ts_data)} output rows")
        console.end_phase("AARAMBH ENGINE")
        progress_bar(progress_container, 40, "Aarambh Engine Complete", f"Signal: {sig['signal']} ({sig['strength']}) · Conviction: {sig['conviction_score']:+.0f}")

        # ── Phase 3: Nirnay Constituent Analysis ──────────────────────────
        console.start_phase("NIRNAY ENGINE", 3, 5)
        progress_bar(progress_container, 42, "Running Nirnay Engine", f"MSF+MMR+Regime · {len(constituent_ohlcv)} Constituents")

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
                    progress_bar(progress_container, pct_val, f"Analyzing {sym}", f"Osc={osc:+.1f} [{cond}] Regime={regime}")

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
        progress_bar(progress_container, 75, "Nirnay Engine Complete", f"{len(nirnay_constituent_dfs)} Stocks · {len(nirnay_daily)} Trading Days")

        # ── Phase 4: Convergence ──────────────────────────────────────────
        console.start_phase("CONVERGENCE", 4, 5)
        progress_bar(progress_container, 78, "Computing Convergence", "Cross-Validation · DDM Filtering")

        console.section("Cross-Validation Setup")
        # ── Intelligence Mode — PRIOR profile resolution ─────────────────
        # The first convergence pass uses the PRIOR profile (saved from a
        # previous run, if any). After this pass we auto-calibrate on the
        # fresh data and apply the new weights in-place (see Phase 4.5).
        # When the Intelligence Mode toggle is OFF, both passes use the
        # factory defaults — no calibration runs.
        from convergence import intelligence as _intel_mod
        _intel_universe = "NIFTY 50"
        _intel_index = st.session_state.get("nishkarsh_index", _intel_universe)
        _intel_enabled = bool(st.session_state.get("intelligence_mode", True))
        if _intel_enabled:
            _prior_w, _prior_t, _prior_profile = _intel_mod.resolve_active(_intel_universe, _intel_index)
        else:
            _prior_w, _prior_t, _prior_profile = (
                _intel_mod.DEFAULT_WEIGHTS.copy(),
                _intel_mod.DEFAULT_THRESHOLDS.copy(),
                None,
            )
        if _prior_profile is not None:
            console.item(
                "Prior profile",
                f"✅ {_prior_profile.universe} · val IC {_prior_profile.val_ic:+.3f} · "
                f"trained {_prior_profile.timestamp}",
            )
        else:
            console.item("Prior profile", "None (first run / no profile)")
        console.item(
            "Intelligence Mode",
            "ON (auto-calibrate after convergence)" if _intel_enabled else "OFF (defaults locked)",
        )

        # First-pass validator uses prior weights when available.
        _validator_weights = _prior_w if _prior_profile is not None else None
        validator = CrossValidator(active_weights=_validator_weights)
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
                progress_bar(progress_container, pct_val, "Computing Convergence", f"{i + 1}/{total_dates} Dates Scored")

        console.item("Total Aarambh Dates", len(aarambh_ts))
        console.item("Overlap Dates", overlap_count)
        console.success(f"Convergence scoring complete")

        # ── 4a. First-pass conviction model ─────────────────────────────
        # First-pass DDM filter on the convergence_score from the first
        # validator pass. Labeled "first-pass" only when Intelligence Mode
        # is ON (a second pass will follow); just "Conviction Model" otherwise.
        _first_pass_label = "First-Pass Conviction Model" if _intel_enabled else "Conviction Model"
        console.section("Conviction Model (initial pass)" if _intel_enabled else "Conviction Model")
        progress_bar(
            progress_container, 83, _first_pass_label,
            "DDM Filter · Prior Weights" if (_intel_enabled and _prior_profile is not None) else "DDM Filter · Default Weights",
        )
        convergence_df = validator.get_convergence_series()
        conviction_model = UnifiedConvictionModel()
        results = conviction_model.fit(
            convergence_df["convergence_score"].tolist(),
            convergence_df.index.tolist(),
        )
        if results:
            latest = results[-1]
            _pre_label = "DDM Conviction (pre-cal)" if _intel_enabled else "DDM Conviction"
            _sig_label = "DDM Signal (pre-cal)" if _intel_enabled else "DDM Signal"
            console.item(_pre_label, f"{latest.nishkarsh_conviction:+.0f}")
            console.item(_sig_label, latest.nishkarsh_signal)
        console.success(f"Initial conviction: {len(results)} scores computed")

        # ── 4b. AUTO-CALIBRATION (Intelligence Mode) ────────────────────
        # Runs Optuna TPE on the fresh convergence_df + aarambh_ts. The
        # search learns optimal (weights, thresholds) for this universe,
        # persists them to disk, and we immediately re-apply them below
        # so the user's signals reflect the calibrated state on THIS run.
        _final_profile: _intel_mod.IntelligenceProfile | None = None
        _cal_frame = None  # tuner.full_frame, captured for the walk-forward IC diagnostic
        # Default label description for the diagnostics directional test, which
        # runs even when intelligence is OFF (calibration block below overrides).
        _label_src = (f"Target fwd change ({active_target})"
                      if CALIBRATION_RETURN_LABEL == "target" else CALIBRATION_RETURN_LABEL)
        if _intel_enabled:
            console.section("Intelligence Calibration")
            _n_trials = int(st.session_state.get("intel_n_trials", 50))
            progress_bar(
                progress_container, 84,
                "Intelligence Mode · Setup",
                f"Building Tuner · 70/30 Chronological Split · {_n_trials} Trials",
            )
            try:
                # Calibration return label — what the convergence signal is
                # optimized to predict. Driven by CALIBRATION_RETURN_LABEL
                # (core/config.py) so the calibration AND the directional test
                # below stay coherent with the selected engine target:
                #   "target" → the selected target's (e.g. NIFTY50_PE) fwd change
                #   "nsei"   → NIFTY 50 index (^NSEI) forward price return
                #   "basket" → EW constituent basket (survivorship-biased)
                # "nsei"/"basket" degrade to each other, then to "target".
                _ret_levels = None
                _label_src = "PE proxy (fallback)"

                def _basket_levels():
                    _b = _intel_mod.build_index_return_levels(constituent_ohlcv)
                    return _b if (_b is not None and len(_b)) else None

                if CALIBRATION_RETURN_LABEL == "nsei":
                    try:
                        _idx_ohlcv = fetch_constituent_ohlcv(["^NSEI"], start_date, end_date)
                        _idx_df = _idx_ohlcv.get("^NSEI") if _idx_ohlcv else None
                        if _idx_df is not None and len(_idx_df) and "Close" in _idx_df.columns:
                            _ret_levels = _idx_df["Close"]
                            _label_src = "NIFTY 50 index (^NSEI) fwd return"
                    except Exception as _ix_e:
                        console.warning(f"^NSEI index fetch failed ({_ix_e}); using basket fallback")
                    if _ret_levels is None or len(_ret_levels) == 0:
                        _ret_levels = _basket_levels()
                        if _ret_levels is not None:
                            _label_src = "Constituent EW basket (survivorship-biased)"
                elif CALIBRATION_RETURN_LABEL == "basket":
                    _ret_levels = _basket_levels()
                    if _ret_levels is not None:
                        _label_src = "Constituent EW basket (survivorship-biased)"

                # "target" mode, or any fallthrough: optimize against the SELECTED
                # target's forward change — the strongest calibration label in the
                # 2026-06-15 study (val IC +0.148, 100% durable walk-forward).
                if _ret_levels is None or len(_ret_levels) == 0:
                    if not AARAMBH_FORWARD_SIGNAL and "Actual" in aarambh_ts.columns:
                        # Levels mode: aarambh_ts["Actual"] IS the target level.
                        _ta = aarambh_ts["Actual"].dropna()
                        if len(_ta):
                            _ret_levels = _ta
                            _label_src = f"Target fwd change ({active_target})"
                    elif (AARAMBH_FORWARD_SIGNAL and active_date != "None"
                          and active_date in data.columns and active_target in data.columns):
                        # Predictive mode: "Actual" is already a forward RETURN, so
                        # use the real target LEVEL series (PE) instead of the old
                        # basket fallback — the basket is survivorship-biased and was
                        # the WORST label (val IC +0.038, NOT durable; it overfits
                        # survivor drift and produced the spurious "not durable"
                        # verdicts). The real PE level is the best label.
                        _tl = pd.Series(
                            pd.to_numeric(data[active_target], errors="coerce").values,
                            index=pd.to_datetime(data[active_date], errors="coerce", dayfirst=True),
                        )
                        _tl = _tl[~_tl.index.isna()].dropna()
                        if len(_tl):
                            _ret_levels = _tl
                            _label_src = f"Target fwd change ({active_target})"
                    if _ret_levels is None or len(_ret_levels) == 0:
                        _ret_levels = _basket_levels()
                        if _ret_levels is not None:
                            _label_src = "Constituent EW basket (last-resort fallback)"
                tuner = _intel_mod.ConvergenceTuner(
                    convergence_df, aarambh_ts,
                    universe=_intel_universe, selected_index=_intel_index,
                    return_levels=_ret_levels,
                )
                console.item("Return Label", _label_src)
                console.item("TPE Trials", _n_trials)
                _purge_rows = len(tuner.full_frame) - len(tuner.train_frame) - len(tuner.val_frame)
                console.item("Train / Val Split", f"{int(tuner.train_frac*100)}/{int((1-tuner.train_frac)*100)} chronological · {_purge_rows}-row purge gap (= max horizon)")
                console.item("Objective", f"purged {tuner.n_cv_folds}-fold CV IC (mean − 0.5·std), L2={tuner.l2_alpha}")
                console.item("Horizons", " · ".join(str(h) for h in tuner.horizons))
                console.item("Train Rows", len(tuner.train_frame))
                console.item("Val Rows", len(tuner.val_frame))

                def _cal_cb(trial_num: int, total: int, best: float) -> None:
                    # Map trial progress to 84% → 90% on the global bar.
                    # Linear interpolation gives smooth motion through the loop.
                    pct = int(84 + (trial_num / max(1, total)) * 6)
                    progress_bar(
                        progress_container, pct,
                        "Intelligence Mode · Calibrating",
                        f"Optuna Trial {trial_num}/{total} · Best Score {best:+.4f}",
                    )

                _candidate, _ = tuner.optimize(n_trials=_n_trials, progress_callback=_cal_cb)
                tuner.evaluate_validation()
                _cv = tuner.cross_validated_ic(folds=5)   # multi-fold stability
                _candidate = tuner._make_profile()
                _cal_frame = tuner.full_frame             # for walk-forward IC diagnostic

                # Gate persistence on out-of-sample edge AND fold stability. A
                # profile that overfits (strong train IC, weak/negative val IC),
                # is unstable across regimes, or regresses against the incumbent
                # is NOT saved or applied — we keep prior / defaults instead.
                # Re-score the incumbent on THIS run's validation frame so the
                # regression check is apples-to-apples (the stored incumbent IC
                # may have been measured on a smaller/older sample — e.g. a 2y
                # window — and isn't comparable to a 5y candidate).
                _incumbent_cmp = _prior_profile
                if _prior_profile is not None:
                    try:
                        import dataclasses as _dc
                        _inc_ic_now = tuner.score_on_validation(
                            _prior_profile.weights, _prior_profile.thresholds
                        )
                        if np.isfinite(_inc_ic_now):
                            _incumbent_cmp = _dc.replace(_prior_profile, val_ic=float(_inc_ic_now))
                            console.item(
                                "Incumbent IC (re-scored on current val)",
                                f"{_inc_ic_now:+.4f}  (stored {_prior_profile.val_ic:+.4f})",
                            )
                    except Exception as _inc_e:
                        console.warning(f"incumbent re-score failed: {_inc_e}")
                _accept, _reason = _intel_mod.is_profile_acceptable(_candidate, _incumbent_cmp)
                console.item("Train IC", f"{_candidate.train_ic:+.4f}")
                console.item("Val IC",   f"{_candidate.val_ic:+.4f}")
                _fold_str = ", ".join(f"{x:+.3f}" for x in _candidate.cv_fold_ics) or "n/a"
                console.item("Fold ICs", f"[{_fold_str}]")
                console.item("Fold Stability", f"{_candidate.cv_fraction_positive*100:.0f}% positive · mean {_candidate.cv_ic_mean:+.3f} · min {_candidate.cv_ic_min:+.3f} · std {_candidate.cv_ic_std:.3f}")
                if _candidate.sensitivity:
                    _top3 = sorted(_candidate.sensitivity.items(), key=lambda kv: -kv[1])[:3]
                    console.item("Top drivers", " · ".join(f"{k} {v:.0f}%" for k, v in _top3))

                if _accept:
                    _intel_mod.save_profile(_candidate)
                    _final_profile = _candidate
                    progress_bar(
                        progress_container, 90,
                        "Intelligence Mode · Profile Saved",
                        f"Train IC {_candidate.train_ic:+.3f} · Val IC {_candidate.val_ic:+.3f}",
                    )
                    console.success(
                        f"Calibration accepted · {_reason} · "
                        f"persisted to disk ({_intel_universe} · {_intel_index})"
                    )
                else:
                    _final_profile = _prior_profile
                    progress_bar(
                        progress_container, 90,
                        "Intelligence Mode · Calibration Rejected",
                        f"{_reason} · keeping prior / defaults",
                    )
                    console.warning(
                        f"Calibration rejected · {_reason} — keeping prior profile / defaults"
                    )
            except Exception as _cal_e:
                console.warning(f"Calibration failed: {_cal_e} — falling back to prior profile / defaults")
                _final_profile = _prior_profile

        # ── 4c. APPLY calibrated weights + thresholds to current run ────
        # Either we just calibrated (use _final_profile) or Intelligence
        # Mode is OFF (use defaults). Vectorized recomputation of
        # convergence_score from existing dim_* columns — no need to
        # re-loop CrossValidator over every date.
        if _final_profile is not None:
            progress_bar(
                progress_container, 91,
                "Applying Calibrated Profile",
                "Re-Weighting Convergence · Vectorized Recompute",
            )
            convergence_df = _intel_mod.apply_calibrated_weights(
                convergence_df, _final_profile.weights,
            )
            progress_bar(
                progress_container, 92,
                "Re-Fitting Conviction Model",
                "Post-Calibration DDM Pass",
            )
            # Re-fit the conviction model with the new convergence_score
            conviction_model = UnifiedConvictionModel()
            results = conviction_model.fit(
                convergence_df["convergence_score"].tolist(),
                convergence_df.index.tolist(),
            )
            console.section("Conviction Model (post-cal)")
            if results:
                latest = results[-1]
                console.item("DDM Conviction", f"{latest.nishkarsh_conviction:+.0f}")
                console.item("DDM Signal", latest.nishkarsh_signal)
                console.item("DDM Band", f"[{latest.confidence_lower:.0f}, {latest.confidence_upper:.0f}]")
            console.success(f"Re-fit complete with calibrated profile")
            _active_w = _final_profile.weights
            _active_t = _final_profile.thresholds
        else:
            # Intelligence Mode OFF — record the default state.
            _active_w = _intel_mod.DEFAULT_WEIGHTS.copy()
            _active_t = _intel_mod.DEFAULT_THRESHOLDS.copy()

        # Publish to session state so the Passport sidebar + Convergence cards
        # see the calibrated state immediately on the next rerun.
        st.session_state["intelligence_active_weights"] = _active_w
        st.session_state["intelligence_active_thresholds"] = _active_t
        st.session_state["intelligence_active_profile"] = (
            _final_profile.to_dict() if _final_profile is not None else None
        )

        # ── 4d. Normalized convergence with calibrated thresholds ───────
        from convergence.normalization import compute_normalized_convergence
        _nishkarsh_norm = compute_normalized_convergence(
            aarambh_ts, nirnay_daily, thresholds=_active_t,
        )
        if _nishkarsh_norm:
            console.section("Normalized Convergence (UI display)")
            console.item("Conviction", f"{_nishkarsh_norm['value']:+.2f}")
            console.item("Signal", _nishkarsh_norm['signal'])
            console.item("  Aarambh contribution", f"{_nishkarsh_norm['aarambh_norm']:+.2f}")
            console.item("  Nirnay contribution",  f"{_nishkarsh_norm['nirnay_norm']:+.2f}")

        console.section("Divergence Detection")
        progress_bar(progress_container, 93, "Detecting Divergences", "Cross-System Disagreement Analysis")
        events = divergence_detector.get_events()
        console.item("Total Events", len(events))
        if not events.empty:
            event_types = events['divergence_type'].value_counts()
            for etype, count in event_types.items():
                console.item(f"  {etype}", count)
        console.success(f"Divergence analysis complete")

        console.end_phase("CONVERGENCE")
        _conv_complete_sub = (
            f"{overlap_count} Overlap Dates · {len(events)} Divergence Events · "
            f"{'Calibrated Profile Applied' if (_intel_enabled and _final_profile is not None) else 'Factory Defaults'}"
        )
        progress_bar(progress_container, 94, "Convergence Phase Complete", _conv_complete_sub)

        # ── Phase 5: Final Assembly ───────────────────────────────────────
        console.start_phase("FINAL ASSEMBLY", 5, 5)
        progress_bar(progress_container, 95, "Storing Results", "Session State · Cache")
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

        # Reuse the normalized convergence computed in the Conviction Model
        # section above — single source of truth from convergence/normalization.py,
        # shared with the metric cards and the Unified Signal plot.
        st.session_state["nishkarsh_conv_normalized"] = _nishkarsh_norm

        # Display signal = what the UI cards show (normalized if available,
        # else fall back to the DDM-derived signal).
        display_signal = (
            _nishkarsh_norm["signal"] if _nishkarsh_norm
            else (results[-1].nishkarsh_signal if results else "N/A")
        )

        console.item("Aarambh Engine", "✅ Cached")
        console.item("Nirnay Daily", f"✅ {len(nirnay_daily)} rows")
        console.item("Constituent Results", f"✅ {len(nirnay_constituent_dfs)} stocks")
        console.item("Convergence DF", f"✅ {len(convergence_df)} rows")
        console.item("Nishkarsh Result", f"✅ {display_signal}")

        console.end_phase("FINAL ASSEMBLY")

        # ════════════════════════════════════════════════════════════════
        #  SYSTEM DIAGNOSTICS — full per-run telemetry. A single copy of this
        #  block is intended to be sufficient to evaluate and tune the system
        #  end-to-end without the raw data. Each subsection is independently
        #  guarded so a single failure never aborts the run.
        # ════════════════════════════════════════════════════════════════
        console.section("SYSTEM DIAGNOSTICS", "DIAGNOSTICS")
        try:
            _sig = engine.get_current_signal()
            _ms = engine.get_model_stats()
            _ou = engine.ou_params
            _regst = engine.get_regime_stats()
        except Exception as _de:
            _sig, _ms, _ou, _regst = {}, {}, {}, {}
            console.warning(f"diagnostics: engine snapshot failed: {_de}")

        # ── 1. Data Quality ──────────────────────────────────────────────
        try:
            console.section("Data Quality")
            _tgt = data[active_target] if active_target in data.columns else pd.Series(dtype=float)
            console.item("Target", f"{active_target} · {len(data)} obs")
            if active_date != "None" and active_date in data.columns:
                _dt = pd.to_datetime(data[active_date], errors="coerce")
                console.item("Target Date Range", f"{_dt.min()} → {_dt.max()}")
            console.item("Target NaN", f"{int(_tgt.isna().sum())} ({_tgt.isna().mean()*100:.1f}%)")
            _nan_by_feat = {f: float(data[f].isna().mean()*100) for f in active_features if f in data.columns}
            _worst = sorted(_nan_by_feat.items(), key=lambda kv: -kv[1])[:5]
            console.item("Predictors", f"{len(active_features)} · worst NaN%: " +
                         (", ".join(f"{k} {v:.0f}%" for k, v in _worst if v > 0) or "none"))
            console.item("Macro Symbols", f"{macro_df.shape[1] if hasattr(macro_df,'shape') else 0} fetched")
            if constituent_ohlcv:
                _rows = [len(v) for v in constituent_ohlcv.values() if v is not None]
                if _rows:
                    console.item("Constituent Rows", f"min {min(_rows)} · median {int(np.median(_rows))} · max {max(_rows)}")
            _cov = (overlap_count / total_dates * 100) if total_dates else 0.0
            console.item("Convergence Coverage", f"{overlap_count}/{total_dates} dates ({_cov:.1f}%) have BOTH engines — "
                         f"the other {100-_cov:.1f}% are Aarambh-only (Nirnay neutral prior)")
        except Exception as _de:
            console.warning(f"diagnostics: data-quality failed: {_de}")

        # ── 2. Aarambh Engine ────────────────────────────────────────────
        try:
            console.section("Aarambh Engine — Model & Edge")
            if AARAMBH_FORWARD_SIGNAL:
                console.item("OOS R² (forecast)", f"{_ms.get('r2_oos',0):.4f}  (returns forecast — magnitude R²≈0 is NORMAL; edge is in IC)")
            else:
                console.item("OOS R² (vs mean)", f"{_ms.get('r2_oos',0):.4f}  ⚠ persistence-inflated on near-unit-root PE")
            console.item("R² vs Random Walk", f"{_ms.get('r2_vs_rw',0):+.3f}  ← magnitude skill (negative = worse than naive)")
            console.item("RMSE / MAE (OOS)", f"{_ms.get('rmse_oos',0):.4f} / {_ms.get('mae_oos',0):.4f}")
            console.item("Model Spread (avg)", f"{_ms.get('avg_model_spread',0):.4f}")
            console.item("Model Coverage", f"{_ms.get('model_coverage',0)*100:.1f}% chunks fit ({_ms.get('n_fallback_chunks',0)} fell back to mean)")
            if AARAMBH_FORWARD_SIGNAL:
                console.item("Magnitude Edge", f"{_sig.get('edge_assessment','?')}  (level test, tradeable={_sig.get('tradeable')} — NOT the predictive verdict; see Directional/Walk-Forward IC)")
            else:
                console.item("EDGE", f"{_sig.get('edge_assessment','?')}  →  tradeable={_sig.get('tradeable')}")
            if AARAMBH_FORWARD_SIGNAL:
                console.item("Regime memory", f"Hurst {_sig.get('hurst',0.5):.3f}  ⚠ describes the FORECAST series (−prediction), not mean-reverting residuals — OU/half-life below are not applicable in predictive mode")
            else:
                console.item("Regime memory", f"Hurst {_sig.get('hurst',0.5):.3f} · mean_reverting={_sig.get('mean_reverting')} · random_walk={_sig.get('random_walk_regime')}")
            console.item("OU θ / dyn-θ", f"{_ou.get('theta',0):.4f} / {_ou.get('dynamic_theta',0):.4f} (θ-std {_ou.get('theta_std',0):.4f}, stable={_sig.get('theta_stable')})")
            _hl = _ou.get('half_life',0)
            console.item("OU Half-Life", f"{_hl:.1f}d  " + ("(meaningful)" if _sig.get('half_life_meaningful') else "⚠ NOT meaningful — residuals do not mean-revert exploitably"))
            console.item("Forward Edge", f"has_edge={_sig.get('has_forward_edge')} · inverted={_sig.get('inverted')} (from realized signal performance, below)")
            console.item("ADF / KPSS p", f"{_ou.get('adf_pvalue',1):.4f} / {_ou.get('kpss_pvalue',0):.4f}")
            console.item("Structural Breaks", f"{len(engine.break_dates)} (global view): {engine.break_dates[:8]}")
            console.item("Signal", f"{_sig.get('signal')} / {_sig.get('strength')} (display: {_sig.get('display_strength')}) · conf {_sig.get('confidence')}")
            console.item("Conviction", f"{_sig.get('conviction_score',0):+.1f} [band {_sig.get('conviction_lower',0):+.0f}, {_sig.get('conviction_upper',0):+.0f}] · regime {_sig.get('regime')}")
            _cl = _sig.get('conviction_levels', {})
            console.item("Adaptive Bands", f"weak {_cl.get('weak',0):.1f} · moderate {_cl.get('moderate',0):.1f} · strong {_cl.get('strong',0):.1f} (empirical, causal)")
            console.item("Breadth", f"oversold {_sig.get('oversold_breadth',0):.0f}% · overbought {_sig.get('overbought_breadth',0):.0f}% · avgZ {_sig.get('avg_z',0):+.2f}")
            _fi = list(engine.latest_feature_impacts.items())[:6]
            if _fi:
                console.item("Top Feature Impacts", " · ".join(f"{k} {v:.0f}%" for k, v in _fi))
        except Exception as _de:
            console.warning(f"diagnostics: aarambh failed: {_de}")

        # ── 3. Signal Performance (forward-change significance) ───────────
        try:
            _perf = engine.get_signal_performance()
            console.section("Aarambh Signal Performance (forward Δ, OOS)")
            if AARAMBH_FORWARD_SIGNAL:
                console.detail(
                    "⚠ NOT meaningful in PREDICTIVE mode: the target is already a forward "
                    "return, so 'forward Δ of the target' is a double-forward quantity — the "
                    "avg% below are artifacts, not tradeable returns. Use the Directional / "
                    "Walk-Forward IC sections for the predictive-mode edge read."
                )
            for _p in (5, 10, 20):
                r = _perf.get(_p, {})
                console.detail(
                    f"{_p}d → BUY n={r.get('buy_count',0)} avg {r.get('buy_avg',0):+.2f}% "
                    f"hit {r.get('buy_hit',0)*100:.0f}% t={r.get('buy_t_stat',0):+.2f} p={r.get('buy_p_value',1):.3f}  |  "
                    f"SELL n={r.get('sell_count',0)} avg {r.get('sell_avg',0):+.2f}% "
                    f"hit {r.get('sell_hit',0)*100:.0f}% t={r.get('sell_t_stat',0):+.2f} p={r.get('sell_p_value',1):.3f}"
                )
        except Exception as _de:
            console.warning(f"diagnostics: signal-performance failed: {_de}")

        # ── 4. Nirnay Aggregate ──────────────────────────────────────────
        try:
            console.section("Nirnay Engine — Aggregate")
            if nirnay_constituent_dfs:
                _mmrq = [float(d["MMR_Quality"].iloc[-1]) for d in nirnay_constituent_dfs.values()
                         if "MMR_Quality" in d.columns and len(d)]
                if _mmrq:
                    console.item("MMR Quality (√R² vs RW)", f"mean {np.mean(_mmrq):.3f} · min {min(_mmrq):.3f} · max {max(_mmrq):.3f}")
                _last_regimes = [str(d["Regime"].iloc[-1]) for d in nirnay_constituent_dfs.values() if "Regime" in d.columns and len(d)]
                if _last_regimes:
                    _rc = pd.Series(_last_regimes).value_counts().to_dict()
                    console.item("Regime Distribution (latest)", " · ".join(f"{k}:{v}" for k, v in _rc.items()))
                _last_vol = [str(d["Vol_Regime"].iloc[-1]) for d in nirnay_constituent_dfs.values() if "Vol_Regime" in d.columns and len(d)]
                if _last_vol:
                    _vc = pd.Series(_last_vol).value_counts().to_dict()
                    console.item("Vol Regime (latest)", " · ".join(f"{k}:{v}" for k, v in _vc.items()))
                _oscs = [float(d["Unified_Osc"].iloc[-1]) for d in nirnay_constituent_dfs.values() if "Unified_Osc" in d.columns and len(d)]
                if _oscs:
                    console.item("Oscillator (latest)", f"mean {np.mean(_oscs):+.2f} · std {np.std(_oscs):.2f} · range [{min(_oscs):+.1f}, {max(_oscs):+.1f}]")
            if not nirnay_daily.empty:
                _lastnd = nirnay_daily.iloc[-1]
                console.item("Latest Day", f"OS {_lastnd.get('Oversold_Pct',0):.0f}% · OB {_lastnd.get('Overbought_Pct',0):.0f}% · "
                             f"Buy {int(_lastnd.get('Buy_Signals',0))} · Sell {int(_lastnd.get('Sell_Signals',0))} · ChgPts {int(_lastnd.get('Change_Points',0))}")
        except Exception as _de:
            console.warning(f"diagnostics: nirnay failed: {_de}")

        # ── 5. Convergence ───────────────────────────────────────────────
        try:
            console.section("Convergence — Dimensions & Zones")
            if not convergence_df.empty:
                # Full-history dim means are diluted by the ~90% of dates with no
                # Nirnay data (neutral 0.5 prior). Report BOTH full-history and the
                # overlap window (where both engines have real data) so the dilution
                # is visible rather than hidden.
                _ovl = convergence_df.tail(max(overlap_count, 1))
                for _dim in ("dim_direction", "dim_breadth", "dim_magnitude", "dim_regime"):
                    if _dim in convergence_df.columns:
                        console.item(_dim, f"full mean {convergence_df[_dim].mean():.3f} · overlap mean {_ovl[_dim].mean():.3f} · last {convergence_df[_dim].iloc[-1]:.3f}")
                if "consensus_direction" in convergence_df.columns:
                    _cd = convergence_df["consensus_direction"]
                    _nbull = int((_cd > 0).sum()); _nbear = int((_cd < 0).sum()); _ndiv = int((_cd == 0).sum())
                    _tot = max(len(_cd), 1)
                    console.item(
                        "Consensus Direction",
                        f"bull {_nbull} ({_nbull/_tot*100:.0f}%) · bear {_nbear} ({_nbear/_tot*100:.0f}%) · "
                        f"divergent {_ndiv} ({_ndiv/_tot*100:.0f}%) · last {_cd.iloc[-1]:+.0f}  "
                        f"← orients the score (the #1 fix)",
                    )
                if "convergence_score" in convergence_df.columns:
                    cs = convergence_df["convergence_score"]
                    console.item("Convergence Score", f"mean {cs.mean():+.1f} · std {cs.std():.1f} · last {cs.iloc[-1]:+.1f}  (negative = bullish)")
                if "agreement_ratio" in convergence_df.columns:
                    console.item("Agreement Ratio", f"mean {convergence_df['agreement_ratio'].mean():.3f} · last {convergence_df['agreement_ratio'].iloc[-1]:.3f}")
                if "convergence_zone" in convergence_df.columns:
                    _zc = convergence_df["convergence_zone"].value_counts().to_dict()
                    console.item("Zone Distribution", " · ".join(f"{k}:{v}" for k, v in list(_zc.items())[:7]))
                if "lead_lag_indicator" in convergence_df.columns:
                    _llc = convergence_df["lead_lag_indicator"].value_counts().to_dict()
                    console.item("Lead-Lag", " · ".join(f"{k}:{v}" for k, v in _llc.items()))
        except Exception as _de:
            console.warning(f"diagnostics: convergence failed: {_de}")

        # Captured for the predictive-mode Final Verdict headline (the directional
        # IC + walk-forward durability ARE the edge verdict in predictive mode,
        # not the engine's level/magnitude tradeable flag).
        _dir_ic = _dir_pos = _wf_durable = _wf_mean = _wf_ics_cap = None
        _dc = None

        # ── 5b. DIRECTIONAL convergence test ─────────────────────────────
        # Independent cross-check: rebuilds the signal from the raw engine
        # outputs (the *signed* normalized convergence) rather than reusing the
        # production convergence_score, and measures its IC against forward
        # returns across folds — the decisive "is there any real timing edge"
        # question, as an unbiased second opinion on the scored composite.
        try:
            _dret = None
            try:
                _dret = _ret_levels  # label levels from calibration, if available
            except NameError:
                _dret = None
            # Intelligence OFF → no calibration label. Stay coherent with the
            # configured mode: "target" uses the selected target's levels, else
            # the constituent basket.
            if _dret is None or len(_dret) == 0:
                if (CALIBRATION_RETURN_LABEL == "target" and not AARAMBH_FORWARD_SIGNAL
                        and "Actual" in aarambh_ts.columns):
                    _dret = aarambh_ts["Actual"].dropna()
                else:
                    _dret = _intel_mod.build_index_return_levels(constituent_ohlcv)
                    _label_src = "Constituent EW basket (survivorship-biased)"
            _dc = _intel_mod.directional_convergence_ic(aarambh_ts, nirnay_daily, _dret)
            console.section(f"Directional Convergence Test (signed signal vs {_label_src})")
            if _dc:
                _fi = ", ".join(f"{x:+.3f}" for x in _dc["fold_ics"])
                console.item("Fold ICs", f"[{_fi}]  (n={_dc['n']})")
                console.item("Stability", f"{_dc['fraction_positive']*100:.0f}% positive · mean {_dc['mean']:+.3f} · min {_dc['min']:+.3f}")
                _verdict = ("REAL directional edge candidate" if (_dc["fraction_positive"] >= 0.8 and _dc["mean"] > 0.03)
                            else "no robust directional edge (flat / sign-flipping folds)")
                console.item("Verdict", _verdict)
                _dir_ic, _dir_pos = _dc["mean"], _dc["fraction_positive"]
            else:
                console.item("Status", "insufficient aligned data for the test")
        except Exception as _de:
            console.warning(f"diagnostics: directional test failed: {_de}")

        # ── 5b. Walk-Forward IC (re-calibrated durability grade) ─────────
        # Unlike the directional test (one fixed signal across folds) and the
        # calibration's own fold stability (one fixed param set), this RE-
        # OPTIMIZES the weights/thresholds on each expanding train block and
        # scores the next purged OOS block — so it grades whether the edge
        # survives recalibration or just rode one lucky parameterization.
        try:
            if _cal_frame is not None and len(_cal_frame) >= 250:
                console.section("Walk-Forward IC (re-calibrated per window · purged OOS)")
                _wf = _intel_mod.walk_forward_ic(_cal_frame)
                if _wf:
                    _ics = [w["ic"] for w in _wf if not np.isnan(w["ic"])]
                    _seq = ", ".join(f"{x:+.3f}" for x in _ics)
                    console.item("Window OOS ICs", f"[{_seq}]  (n={len(_wf)})")
                    if _ics:
                        _arr = np.array(_ics, dtype=float)
                        _fp = float((_arr > 0).mean())
                        console.item("Durability", f"{_fp*100:.0f}% positive · mean {_arr.mean():+.3f} · min {_arr.min():+.3f}")
                        _wf_durable = bool(_fp >= 0.8 and _arr.mean() > 0.02)
                        _wf_mean = float(_arr.mean())
                        _wf_ics_cap = list(_ics)
                        _wf_verdict = ("durable edge across windows" if _wf_durable
                                       else "not durable (edge does not survive recalibration)")
                        console.item("Verdict", _wf_verdict)
                else:
                    console.item("Status", "insufficient history for walk-forward windows")
        except Exception as _we:
            console.warning(f"diagnostics: walk-forward IC failed: {_we}")

        # Stash the predictive-edge diagnostics for the Diagnostics tab so it
        # surfaces the IC / durability verdict (not just the levels metrics).
        st.session_state["intelligence_diag"] = {
            "forward_signal": bool(AARAMBH_FORWARD_SIGNAL),
            "label_src": _label_src,
            "directional_ic": (_dir_ic if _dir_ic is not None else None),
            "directional_pos": (_dir_pos if _dir_pos is not None else None),
            "directional_folds": (_dc.get("fold_ics") if _dc else None),
            "wf_ics": _wf_ics_cap,
            "wf_durable": _wf_durable,
            "wf_mean": _wf_mean,
        }

        # ── 6. Calibration Profile ───────────────────────────────────────
        try:
            console.section("Calibration Profile")
            if _final_profile is not None:
                console.item("Label Source", f"{_label_src} [{_final_profile.label_kind}]")
                console.item("Sample", f"train {_final_profile.n_train_dates} · val {_final_profile.n_val_dates} · trials {_final_profile.n_trials}")
                console.item("IC", f"train {_final_profile.train_ic:+.4f} · val {_final_profile.val_ic:+.4f}")
                if _final_profile.cv_fold_ics:
                    console.item("Fold ICs", "[" + ", ".join(f"{x:+.3f}" for x in _final_profile.cv_fold_ics) + "]")
                    console.item("Fold Stability", f"{_final_profile.cv_fraction_positive*100:.0f}% positive · mean {_final_profile.cv_ic_mean:+.3f} · min {_final_profile.cv_ic_min:+.3f} · std {_final_profile.cv_ic_std:.3f}")
                _w = _final_profile.weights
                console.item("Weights", " · ".join(f"{k.replace('w_','')} {v:.3f}" for k, v in _w.items()))
                _t = _final_profile.thresholds
                console.item("Thresholds", " · ".join(f"{k} {v:+.3f}" for k, v in _t.items()))
                if _final_profile.sensitivity:
                    console.item("Param Importance", " · ".join(f"{k} {v:.0f}%" for k, v in sorted(_final_profile.sensitivity.items(), key=lambda kv: -kv[1])))
                console.item("Prior val IC", f"{(_prior_profile.val_ic if _prior_profile else float('nan')):+.4f}")
            else:
                console.item("Status", "No calibrated profile (Intelligence OFF or calibration rejected)")
        except Exception as _de:
            console.warning(f"diagnostics: calibration failed: {_de}")

        # ── 7. Final Nishkarsh Verdict ───────────────────────────────────
        try:
            console.section("Nishkarsh — Final Verdict")
            console.item("Normalized Conviction", f"{(_nishkarsh_norm['value'] if _nishkarsh_norm else 0):+.3f}")
            console.item("Signal", display_signal)
            if results:
                _l = results[-1]
                console.item("DDM Conviction", f"{_l.nishkarsh_conviction:+.1f} [band {_l.confidence_lower:+.0f}, {_l.confidence_upper:+.0f}]")
            if AARAMBH_FORWARD_SIGNAL:
                # Predictive mode: the verdict is DIRECTIONAL (IC + walk-forward),
                # NOT the engine's level/magnitude tradeable flag.
                console.item("Directional IC", f"{_dir_ic:+.3f} ({(_dir_pos or 0)*100:.0f}% folds positive)" if _dir_ic is not None else "n/a")
                console.item("Walk-Forward", (("durable" if _wf_durable else "not durable") + (f" (mean {_wf_mean:+.3f})" if _wf_mean is not None else "")) if _wf_durable is not None else "n/a")
                if _dir_ic is not None and _dir_ic > 0.03 and (_dir_pos or 0) >= 0.8 and _wf_durable:
                    _headline = (f"DIRECTIONAL EDGE candidate — IC {_dir_ic:+.3f}, walk-forward durable "
                                 "(rank-IC evidence; pending non-overlapping significance + cost backtest)")
                elif _dir_ic is not None and _dir_ic > 0.03 and (_dir_pos or 0) >= 0.8:
                    _headline = f"directional IC {_dir_ic:+.3f} but NOT durable OOS — provisional, do not trade yet"
                else:
                    _headline = "no robust directional edge (predictive mode)"
            else:
                console.item("Aarambh Tradeable", _sig.get("tradeable"))
                console.item("Edge Verdict", _sig.get("edge_assessment", "?"))
                if _sig.get("inverted"):
                    _headline = "SIGNAL INVERTED — historically anti-predictive; do NOT trade as-is"
                elif not _sig.get("tradeable"):
                    _headline = "NO FORECAST EDGE — treat as valuation context only"
                else:
                    _headline = "edge present (signal predicted correct direction, p<0.10)"
            console.item("HEADLINE", f"{display_signal} · {_headline}")
        except Exception as _de:
            console.warning(f"diagnostics: verdict failed: {_de}")

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

        # Snapshot this config's full result into the bounded per-config cache so
        # revisiting it later (predictor toggle-back) restores instantly. LRU-evict
        # to keep memory bounded.
        _rcache = st.session_state.setdefault("results_cache", {})
        _rcache.pop(cache_key, None)
        _rcache[cache_key] = {bk: st.session_state.get(bk) for bk in _BUNDLE_KEYS}
        while len(_rcache) > _RESULTS_CACHE_MAX:
            _rcache.pop(next(iter(_rcache)))

        progress_bar(progress_container, 100, "Analysis Complete", f"Nishkarsh: {display_signal}")
        # Clear the transient progress card immediately — no blocking sleep on
        # Streamlit's single script thread. The completion state is surfaced in
        # the rendered result, not a lingering banner.
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

    nishkarsh_norm = st.session_state.get("nishkarsh_conv_normalized")
    agreement = st.session_state.get("last_agreement", 0)

    # ─── Primary Signal (Above Tabs, Always Visible) ───────────────────────
    _render_primary_signal(nishkarsh_norm, agreement, signal)

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "CONVERGENCE", "AARAMBH", "NIRNAY", "PRECEDENT", "DIAGNOSTICS", "DATA",
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
        _safe_render("Nirnay", lambda: render_nirnay_tab(selected_tf))
    with tab4:
        st.session_state.rendered_tabs.add(3)
        # Full-history `ts` (not the timeframe-filtered view) so the analog matcher
        # sees the whole series. Horizons follow the Aarambh forecast lens.
        _safe_render("Precedent", lambda: render_precedent_tab(
            ts, active_target, tuple(PRECEDENT_HOLD_HORIZONS), AARAMBH_FWD_MOM_K, AARAMBH_FWD_HORIZON))
    with tab5:
        st.session_state.rendered_tabs.add(4)
        _safe_render("Diagnostics", lambda: render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats))
    with tab6:
        st.session_state.rendered_tabs.add(5)
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    _render_footer()


if __name__ == "__main__":
    main()
