"""
Nishkarsh v2 — Streamlit UI shell (ruling R6).
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

The whole analysis lives in the headless ``pipeline`` package; this file only collects the
selections, calls ``pipeline.run.run_pipeline`` and renders the C-PIPE session keys.
"""

from __future__ import annotations

import os

# ── BLAS thread pinning + numba cache: MUST precede the numpy / numba imports ──
# One BLAS thread is fastest for the walk-forward's many small solves (the pipeline also
# wraps every run in threadpool_limits(1)); the numba cache lives outside the watched tree
# so Streamlit's file watcher never reruns the script mid-compile.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nishkarsh", "numba"))

import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# Narrow, source-specific filters only (no blanket FutureWarning filter — A5-20).
warnings.filterwarnings("ignore", message=".*YF.download.*")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (  # noqa: E402
    AARAMBH_FWD_HORIZON, COMPANY, DEFAULT_PREDICTORS, TIMEFRAME_TRADING_DAYS, VERSION,
)
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.inputs import InputError, sheet_content_hash  # noqa: E402
from pipeline.run import SESSION_KEYS, run_pipeline  # noqa: E402
from ui.components import data_notes, esc, pretty_date, render_error  # noqa: E402
from ui.copy import (  # noqa: E402
    INTEL_HELP_PLAIN, LANDING_FOOTNOTE, LANDING_INTRO, LANDING_STEPS, PRODUCT_TAGLINE, PRODUCT_TITLE, SYNTH_BAR,
)
from ui.tabs.tab_data import render_data_tab  # noqa: E402
from ui.tabs.tab_drivers import render_drivers_tab  # noqa: E402
from ui.tabs.tab_precedent import render_precedent_tab  # noqa: E402
from ui.tabs.tab_reliability import render_reliability_tab  # noqa: E402
from ui.tabs.tab_today import render_today_tab  # noqa: E402
from ui.theme import inject_css, progress_bar  # noqa: E402

SOURCES = ("Google Sheets", "Upload", "Synthetic demo")
TAB_NAMES = ("Today", "Drivers", "Precedent", "Reliability", "Data")
SYNTH_SEED = 7
_RESULTS_CACHE_MAX = 3
# Every key derived from a loaded sheet / run. Reset clears ALL of them (A5-02).
_DERIVED_KEYS = SESSION_KEYS + ("data", "run_analysis", "results_cache", "_result_key", "_sheet_info",
                                "_sources_kind", "_synthetic_sources", "_force_next", "_today",
                                "precedent_summary", "_prec_key", "tf_selected")
_INTEL_HELP = INTEL_HELP_PLAIN


def _clear_derived(keep_data: bool = False) -> None:
    for k in _DERIVED_KEYS:
        if keep_data and k in ("data", "run_analysis", "_sheet_info", "_sources_kind", "_synthetic_sources", "_today"):
            continue
        st.session_state.pop(k, None)


def _secret_sheet_url() -> str:
    """Configured sheet URL (secrets / env). NEVER rendered into the page (A5-12)."""
    try:
        url = st.secrets.get("aarambh", {}).get("google_sheets_url", "")
        if url:
            return str(url)
    except Exception:  # noqa: BLE001 — no secrets file
        pass
    return os.environ.get("AARAMBH_GOOGLE_SHEETS_URL", "")


def _load_sheet(url: str, force: bool):
    from data.fetcher import fetch_sheet
    return fetch_sheet(url, force=force)


# ─── static pages ────────────────────────────────────────────────────────────
def _render_landing_page() -> None:
    steps = "".join(f'<div class="nk-step"><div class="nk-step-n">{i}</div><h4>{esc(t)}</h4><p>{esc(d)}</p></div>'
                    for i, (t, d) in enumerate(LANDING_STEPS, start=1))
    st.markdown(
        f'<div class="nk-landing"><div class="nk-brand-lg">{esc(PRODUCT_TITLE)}<span>निष्कर्ष</span></div>'
        f'<h1>{esc(PRODUCT_TAGLINE)}</h1><p class="nk-lede">{esc(LANDING_INTRO)}</p>'
        f'<div class="nk-steps">{steps}</div><p class="nk-landing-foot">{esc(LANDING_FOOTNOTE)}</p></div>',
        unsafe_allow_html=True)


def _render_footer(as_of: str = "") -> None:
    tail = f" · as of {as_of}" if as_of and as_of != "—" else ""
    st.markdown(f'<div class="nk-footer">Not investment advice · {esc(PRODUCT_TITLE)} v{esc(VERSION)}{esc(tail)} · '
                f'{esc(COMPANY)}</div>', unsafe_allow_html=True)


def _render_synthetic_bar() -> None:
    st.markdown(f'<div class="nk-synth-bar">{esc(SYNTH_BAR)}</div>', unsafe_allow_html=True)


def _render_topbar(rc: dict, con: dict) -> str:
    """Brand + as-of on the left, the data status pill and the page-wide date range on the right."""
    ss = st.session_state
    frames = list(TIMEFRAME_TRADING_DAYS) + ["ALL"]
    if ss.get("tf_selected") not in frames:
        ss["tf_selected"] = "6M"
    left, mid, right = st.columns([2.5, 1.05, 1.3], vertical_alignment="center")
    as_of = pretty_date(con.get("as_of") or rc.get("as_of"))
    left.markdown(
        f'<div class="nk-topbar"><div class="nk-wordmark">{esc(PRODUCT_TITLE)}<span>निष्कर्ष</span></div>'
        f'<div class="nk-context">Nifty 50 · as of {esc(as_of)}</div></div>', unsafe_allow_html=True)
    n = len(data_notes(rc))
    pill = ('<div class="nk-pill ok"><span class="nk-dot"></span>Data current</div>' if not n else
            f'<div class="nk-pill warn"><span class="nk-dot"></span>{n} data note{"s" if n != 1 else ""} · see Data</div>')
    mid.markdown(pill, unsafe_allow_html=True)
    with right:
        st.segmented_control("Date range", frames, key="tf_selected", required=True, label_visibility="collapsed",
                             format_func=lambda f: "All" if f == "ALL" else f, width="stretch")
    return ss.get("tf_selected") or "6M"


# ─── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Nishkarsh", page_icon="◐", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    ss = st.session_state

    synthetic = ss.get("_sources_kind") == "Synthetic demo"
    with st.sidebar:
        st.markdown(f'<div class="nk-side-brand">{esc(PRODUCT_TITLE)}<span>v{esc(VERSION)}</span></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="nk-side-label">Data source</div>', unsafe_allow_html=True)
        source = st.radio("Source", SOURCES, horizontal=False, label_visibility="collapsed", key="data_source")
        if ss.get("_sources_kind") and ss["_sources_kind"] != source:
            _clear_derived()                       # switching source discards everything derived
        has_data = "data" in ss and ss.get("run_analysis")

        if source == "Upload":
            up = st.file_uploader("CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed")
            if up is not None and not has_data and st.button("Run analysis", type="primary", width="stretch"):
                try:
                    df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read the file: {e}")
                    return
                from data.sources import FetchResult
                ss.update(data=df, run_analysis=True, _sources_kind=source,
                          _sheet_info=FetchResult(None, "live", None, [], f"upload {up.name}"))
                st.rerun()
            elif up is None and not has_data:
                st.caption("Upload a CSV or Excel file to begin.")
        elif source == "Google Sheets":
            secret = _secret_sheet_url()
            if secret:
                st.caption("Using the configured sheet.")
                url = secret
            else:
                url = st.text_input("Sheet URL", value="", label_visibility="collapsed",
                                    placeholder="Paste a Google Sheets link")
            if (not has_data and st.button("Run analysis", type="primary", width="stretch")) or (
                    has_data and ss.get("_force_next") and ss.get("_sources_kind") == source):
                if not url:
                    st.error("Paste a Google Sheets link (or configure one in secrets).")
                    return
                with st.spinner("Loading sheet…"):
                    r = _load_sheet(url, force=bool(ss.get("_force_next")))
                if r.value is None:
                    st.error(f"Sheet load failed: {r.note or 'no data'}")
                    return
                from data.sources import FetchResult
                ss.update(data=r.value, run_analysis=True, _sources_kind=source,
                          _sheet_info=FetchResult(None, r.provenance, r.as_of, list(r.missing), r.note))
                if not has_data:
                    st.rerun()
        else:
            st.caption("A seeded, simulated market — not market data.")
            if not has_data and st.button("Run analysis", type="primary", width="stretch"):
                from data.sources import FetchResult
                from data.synthetic import synthetic_world
                df, src = synthetic_world(SYNTH_SEED)
                info = src.sheet()
                ss.update(data=df, run_analysis=True, _sources_kind=source, _synthetic_sources=src,
                          _today=str(pd.Timestamp(df["DATE"].iloc[-1]).date()),
                          _sheet_info=FetchResult(None, info.provenance, info.as_of, [], info.note))
                st.rerun()
        if has_data:
            b1, b2 = st.columns(2, gap="small")
            if b1.button("Refresh data", type="secondary", width="stretch", disabled=synthetic,
                         help="Re-fetch live data (bypassing caches), then re-run."):
                _clear_derived(keep_data=True)
                ss["_force_next"] = True
                st.rerun()
            if b2.button("Reset", type="secondary", width="stretch", help="Discard the data and all results."):
                _clear_derived()
                st.rerun()
    settings_box = st.sidebar.container()

    df = ss.get("data") if ss.get("run_analysis") else None
    synthetic = ss.get("_sources_kind") == "Synthetic demo"
    if df is None:
        _render_landing_page()
        _render_footer()
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                    or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    date_cols = [c for c in df.columns if "date" in str(c).lower()] or [c for c in df.columns if c not in numeric_cols]
    if len(numeric_cols) < 2 or not date_cols:
        st.error("The sheet needs a date column and at least 2 numeric columns.")
        return

    with settings_box, st.expander("Model settings", expanded=False):
        num_choices = [c for c in numeric_cols if c not in date_cols]
        tgt0 = ss.get("active_target") if ss.get("active_target") in num_choices else (
            "NIFTY50_PE" if "NIFTY50_PE" in num_choices else num_choices[0])
        target = st.selectbox("Target series", num_choices, index=num_choices.index(tgt0))
        d0 = ss.get("active_date_col") if ss.get("active_date_col") in date_cols else date_cols[0]
        date_col = st.selectbox("Date column", date_cols, index=date_cols.index(d0))
        if target != ss.get("active_target") or date_col != ss.get("active_date_col"):
            ss["active_target"], ss["active_date_col"] = target, date_col
            if "active_features" in ss:
                ss["active_features"] = tuple(f for f in ss["active_features"] if f != target)
        available = [c for c in num_choices if c != target]
        if not ss.get("active_features"):
            ss["active_features"] = tuple(p for p in DEFAULT_PREDICTORS if p in available) or tuple(available[:3])
        key = "pred_select::" + "|".join(sorted(ss["active_features"]))
        staged = st.multiselect("Predictors", available, default=[f for f in ss["active_features"] if f in available],
                                key=key, help="The inputs the displayed run used — edit, then apply.")
        changed = bool(staged) and set(staged) != set(ss["active_features"])
        if st.button("Apply predictors" if changed else "Predictors up to date", disabled=not changed,
                     type="primary" if changed else "secondary", width="stretch"):
            ss["active_features"] = tuple(staged)
            st.rerun()
        kappa = st.toggle("Intelligence mode", value=True, help=_INTEL_HELP, key="intel_toggle")

    features = tuple(f for f in ss["active_features"] if f in available)
    try:
        cfg = PipelineConfig.build(today=ss.get("_today"), target=target, features=features, date_col=date_col,
                                   kappa_enabled=bool(kappa))
    except ValueError as e:
        st.error(str(e))
        return
    result_key = f"{cfg.fingerprint()}|{sheet_content_hash(df)}|{ss.get('_sources_kind')}"
    cache = ss.setdefault("results_cache", {})

    if ss.get("_result_key") != result_key:
        if result_key in cache:
            ss.update(cache.pop(result_key))
            cache[result_key] = {k: ss[k] for k in SESSION_KEYS}
            ss["_result_key"] = result_key
        else:
            if synthetic:
                sources = ss.get("_synthetic_sources")
                if sources is None:
                    from data.synthetic import synthetic_world
                    sources = synthetic_world(SYNTH_SEED)[1]
                    ss["_synthetic_sources"] = sources
            else:
                from data.nse_archive import AutoSources   # Yahoo first, NSE archives on failure
                sources = AutoSources()
            slot = st.empty()

            def _progress(pct, title, sub=""):
                progress_bar(slot, int(pct), title, sub)   # progress_bar escapes

            try:
                result = run_pipeline(df, cfg, sources, progress=_progress, force=bool(ss.pop("_force_next", False)),
                                      sheet_info=ss.get("_sheet_info"), today=ss.get("_today"))
            except InputError as e:
                slot.empty()
                st.error(str(e))
                return
            slot.empty()
            bundle = result.to_session_state()
            ss.update(bundle)
            cache[result_key] = bundle
            while len(cache) > _RESULTS_CACHE_MAX:
                cache.pop(next(iter(cache)))
            ss["_result_key"] = result_key
            st.rerun()

    _render_results(ss, synthetic)


def _render_results(ss, synthetic: bool) -> None:
    rc = ss["run_config"]
    engine, aarambh_ts, con = ss["engine"], ss["aarambh_ts"], ss.get("conclusion") or {}
    target = rc["target"]
    forward = bool(rc.get("forward_mode", con.get("forward_mode", True)))
    tf = _render_topbar(rc, con)
    if synthetic:
        _render_synthetic_bar()

    ts = aarambh_ts.reset_index(drop=True)
    ts_f = ts if tf == "ALL" else ts.iloc[max(0, len(ts) - int(TIMEFRAME_TRADING_DAYS[tf])):]
    # One date window for every page (TIMEFRAME_TRADING_DAYS rows of the Aarambh frame).
    start = None if tf == "ALL" or not len(ts_f) else pd.Timestamp(ts_f["Date"].iloc[0])
    horizon = int(con.get("horizon") or AARAMBH_FWD_HORIZON)

    def _safe(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a page failure must not take the app down
            render_error(name, e)

    tabs = st.tabs(list(TAB_NAMES))
    with tabs[0]:
        _safe("Today", lambda: render_today_tab(start, tf))
    with tabs[1]:
        _safe("Drivers", lambda: render_drivers_tab(start, tf))
    with tabs[2]:
        _safe("Precedent", lambda: render_precedent_tab(horizon))
    with tabs[3]:
        _safe("Reliability", lambda: render_reliability_tab(engine.get_current_signal() if not forward else None))
    with tabs[4]:
        _safe("Data", lambda: render_data_tab(ts_f, ts, target, forward))
    _render_footer(pretty_date(con.get("as_of") or rc.get("as_of")))


if __name__ == "__main__":
    main()
