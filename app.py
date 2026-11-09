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
from datetime import datetime, timedelta, timezone  # noqa: E402
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
    AARAMBH_FWD_HORIZON, COMPANY, DEFAULT_PREDICTORS, PRODUCT_NAME, TIMEFRAME_TRADING_DAYS, VERSION,
)
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.inputs import InputError, sheet_content_hash  # noqa: E402
from pipeline.run import SESSION_KEYS, run_pipeline  # noqa: E402
from ui.components import build_hero_verdict, esc, render_header, render_hero_card, section_gap  # noqa: E402
from ui.copy import LANDING_CARDS, TAGLINE, TAGLINE_EXPLAINER  # noqa: E402
from ui.tabs.tab_aarambh import render_aarambh_tab  # noqa: E402
from ui.tabs.tab_convergence import render_convergence_tab  # noqa: E402
from ui.tabs.tab_data import render_data_tab  # noqa: E402
from ui.tabs.tab_diagnostics import render_diagnostics_tab  # noqa: E402
from ui.tabs.tab_nirnay import render_nirnay_tab  # noqa: E402
from ui.tabs.tab_precedent import render_precedent_tab  # noqa: E402
from ui.theme import inject_css, progress_bar  # noqa: E402

SOURCES = ("Google Sheets", "Upload", "Synthetic demo")
SYNTH_LABEL = "SYNTHETIC DEMO DATA"
SYNTH_SEED = 7
_RESULTS_CACHE_MAX = 3
# Every key derived from a loaded sheet / run. Reset clears ALL of them (A5-02).
_DERIVED_KEYS = SESSION_KEYS + ("data", "run_analysis", "results_cache", "_result_key", "_sheet_info",
                                "_sources_kind", "_synthetic_sources", "_force_next", "_today",
                                "precedent_summary", "_prec_key", "tf_selected")
_INTEL_HELP = (
    "ON: on days Nirnay CONTRADICTS Aarambh, the conclusion's conviction is multiplied by κ ≤ 1 — "
    "but only if the walk-forward Convergence-Lift test passes on this data (κ is re-estimated every run "
    "from the data in hand; nothing is stored). If the test does not pass, κ = 1 and ON equals OFF. "
    "Nirnay can never add conviction or flip the direction. OFF: κ ≡ 1 — the conclusion is pure Aarambh."
)


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
    section_gap()
    for col, (cls, title, text, spec) in zip(st.columns(3, gap="small"), LANDING_CARDS):
        rows = "".join(f"<span>{esc(k)}:</span> {esc(v)}<br>" for k, v in spec)
        col.markdown(f"<div class='system-card {esc(cls)}'><h3>{esc(title)}</h3><p>{esc(text)}</p>"
                     f"<div class='spec'>{rows}</div></div>", unsafe_allow_html=True)
    section_gap()
    st.markdown(f"<div class='landing-prompt'><h4>TWO SYSTEMS. ONE CONCLUSION.</h4><p>{esc(TAGLINE_EXPLAINER)}</p>"
                "<h4>AWAITING DATA</h4><p>Load data in the <strong>Sidebar</strong> (Google Sheet, CSV/Excel upload, "
                "or the clearly labelled synthetic demo), choose a <strong>Target</strong> and <strong>Predictors"
                "</strong>, then <strong>Run Analysis</strong>.</p></div>", unsafe_allow_html=True)


def _render_footer() -> None:
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    st.markdown(f'<div class="app-footer"><div class="content">© {ist_now.year} <strong>{esc(PRODUCT_NAME)}</strong>'
                f' · {esc(COMPANY)} · v{esc(VERSION)} · {ist_now:%Y-%m-%d %H:%M:%S} IST</div></div>',
                unsafe_allow_html=True)


def _render_synthetic_banner() -> None:
    st.error(f"**{SYNTH_LABEL}** — a seeded, simulated world (seed {SYNTH_SEED}). Nothing on this page is "
             "about the real market.")


def _render_verdict(ss) -> None:
    """Hero: renders the conclusion layer's verdict (C-V2) — no statistics computed here."""
    prec = ss.get("precedent") or {}
    signal = ss["engine"].get_current_signal() if ss.get("engine") is not None else {}
    render_hero_card(build_hero_verdict(ss.get("conclusion"), signal, prec.get("hero"), ss.get("run_config")))
    section_gap()


# ─── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="NISHKARSH | Unified Analysis", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    ss = st.session_state

    with st.sidebar:
        st.markdown(f'<div style="text-align:center;padding:0.5rem 0 0.75rem 0;"><div style="font-family:var(--display);'
                    f'font-size:1.35rem;font-weight:700;color:var(--amber);">{esc(PRODUCT_NAME.upper())}</div>'
                    f'<div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.6rem;">निष्कर्ष · v{esc(VERSION)}'
                    f'</div></div>', unsafe_allow_html=True)
        st.caption(TAGLINE_EXPLAINER)
        st.markdown('<div class="sidebar-title">Data Source</div>', unsafe_allow_html=True)
        source = st.radio("Source", SOURCES, horizontal=False, label_visibility="collapsed", key="data_source")
        if ss.get("_sources_kind") and ss["_sources_kind"] != source:
            _clear_derived()                       # switching source discards everything derived
        has_data = "data" in ss and ss.get("run_analysis")

        if source == "Upload":
            up = st.file_uploader("CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed")
            if up is not None and not has_data and st.button("Run Analysis", type="primary"):
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
                st.info("Upload a CSV or Excel file to begin.")
        elif source == "Google Sheets":
            secret = _secret_sheet_url()
            if secret:
                st.caption("Using configured sheet.")
                url = secret
            else:
                url = st.text_input("Sheet URL", value="", label_visibility="collapsed",
                                    placeholder="Google Sheets URL (or set AARAMBH_GOOGLE_SHEETS_URL)")
            if (not has_data and st.button("Run Analysis", type="primary")) or (has_data and ss.get("_force_next")
                                                                                 and ss.get("_sources_kind") == source):
                if not url:
                    st.error("Provide a Google Sheets URL or configure one in secrets.")
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
            st.warning(f"**{SYNTH_LABEL}** — offline, seeded simulation. Not market data.")
            if not has_data and st.button("Run Analysis", type="primary"):
                from data.sources import FetchResult
                from data.synthetic import synthetic_world
                df, src = synthetic_world(SYNTH_SEED)
                info = src.sheet()
                ss.update(data=df, run_analysis=True, _sources_kind=source, _synthetic_sources=src,
                          _today=str(pd.Timestamp(df["DATE"].iloc[-1]).date()),
                          _sheet_info=FetchResult(None, info.provenance, info.as_of, [], info.note))
                st.rerun()
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    df = ss.get("data") if ss.get("run_analysis") else None
    synthetic = ss.get("_sources_kind") == "Synthetic demo"
    if df is None:
        render_header(PRODUCT_NAME.upper(), TAGLINE)
        _render_landing_page()
        _render_footer()
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                    or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    date_cols = [c for c in df.columns if "date" in str(c).lower()] or [c for c in df.columns if c not in numeric_cols]
    if len(numeric_cols) < 2 or not date_cols:
        st.error("The sheet needs a date column and at least 2 numeric columns.")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Model Configuration</div>', unsafe_allow_html=True)
        num_choices = [c for c in numeric_cols if c not in date_cols]
        tgt0 = ss.get("active_target") if ss.get("active_target") in num_choices else (
            "NIFTY50_PE" if "NIFTY50_PE" in num_choices else num_choices[0])
        target = st.selectbox("Target Variable", num_choices, index=num_choices.index(tgt0))
        d0 = ss.get("active_date_col") if ss.get("active_date_col") in date_cols else date_cols[0]
        date_col = st.selectbox("Date Column (required)", date_cols, index=date_cols.index(d0))
        if target != ss.get("active_target") or date_col != ss.get("active_date_col"):
            ss["active_target"], ss["active_date_col"] = target, date_col
            if "active_features" in ss:
                ss["active_features"] = tuple(f for f in ss["active_features"] if f != target)
        available = [c for c in num_choices if c != target]
        if not ss.get("active_features"):
            ss["active_features"] = tuple(p for p in DEFAULT_PREDICTORS if p in available) or tuple(available[:3])
        with st.expander("Predictor Columns", expanded=False):
            st.caption("The predictors the displayed run used — edit, then Apply.")
            key = "pred_select::" + "|".join(sorted(ss["active_features"]))
            staged = st.multiselect("Predictors", available, default=[f for f in ss["active_features"] if f in available],
                                    label_visibility="collapsed", key=key)
            changed = bool(staged) and set(staged) != set(ss["active_features"])
            if st.button("Apply Predictors" if changed else "No changes", disabled=not changed,
                         type="primary" if changed else "secondary"):
                ss["active_features"] = tuple(staged)
                st.rerun()
        st.markdown('<div class="sidebar-title">Intelligence</div>', unsafe_allow_html=True)
        kappa = st.toggle("Intelligence (κ shrink)", value=True, help=_INTEL_HELP, key="intel_toggle")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        if st.button("Reset Analysis", type="secondary", width="stretch"):
            _clear_derived()
            st.rerun()
        if st.button("Refresh Data", type="secondary", width="stretch", disabled=synthetic):
            _clear_derived(keep_data=True)
            ss["_force_next"] = True
            st.rerun()
        st.caption("Reset = discard data and all results · Refresh = re-fetch live data (bypassing caches), then re-run")

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
    render_header(PRODUCT_NAME.upper() + (f" · {SYNTH_LABEL}" if synthetic else ""),
                  f"{target} · {len(rc['features'])} predictors · as of {rc.get('as_of') or 'n/a'} · {TAGLINE}")
    if synthetic:
        _render_synthetic_banner()
    warns = [w for w in rc.get("warnings") or [] if not str(w).startswith("SYNTHETIC")]
    if warns:
        with st.expander(f"Data & provenance warnings ({len(warns)})", expanded=any(
                k in w for w in warns for k in ("Stale", "STALE", "FALLBACK", "EMPTY", "failed"))):
            for w in warns:
                st.warning(w)
    _render_verdict(ss)

    signal, model_stats, regime_stats = engine.get_current_signal(), engine.get_model_stats(), engine.get_regime_stats()
    ts = aarambh_ts.reset_index(drop=True)
    if "tf_selected" not in ss:
        ss["tf_selected"] = "6M"
    frames = list(TIMEFRAME_TRADING_DAYS) + ["ALL"]
    for col, tf in zip(st.columns(len(frames), gap="small"), frames):
        if col.button(tf, key=f"tf_{tf}", type="primary" if ss["tf_selected"] == tf else "secondary", width="stretch"):
            ss["tf_selected"] = tf
            st.rerun()
    tf = ss["tf_selected"]
    ts_f = ts if tf == "ALL" else ts.iloc[max(0, len(ts) - int(TIMEFRAME_TRADING_DAYS[tf])):]
    # One date window for every tab (TIMEFRAME_TRADING_DAYS rows of the Aarambh frame).
    start = None if tf == "ALL" or not len(ts_f) else pd.Timestamp(ts_f["Date"].iloc[0])
    horizon = int(con.get("horizon") or AARAMBH_FWD_HORIZON)

    def _safe(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a tab failure must not take the page down
            st.markdown(f'<div class="warning-box"><div><div class="title">Error in {esc(name)}</div>'
                        f'<div class="content">{esc(e)}</div></div></div>', unsafe_allow_html=True)

    tabs = st.tabs(["CONVERGENCE", "AARAMBH", "NIRNAY", "PRECEDENT", "DIAGNOSTICS", "DATA"])
    with tabs[0]:
        _safe("Convergence", lambda: render_convergence_tab(start))
    with tabs[1]:
        _safe("Aarambh", lambda: render_aarambh_tab(engine, ts_f, ts, signal, model_stats, regime_stats, target,
                                                    forward, con))
    with tabs[2]:
        _safe("Nirnay", lambda: render_nirnay_tab(start))
    with tabs[3]:
        _safe("Precedent", lambda: render_precedent_tab(ts, target, horizon, forward))
    with tabs[4]:
        _safe("Diagnostics", lambda: render_diagnostics_tab(engine, signal))
    with tabs[5]:
        _safe("Data", lambda: render_data_tab(ts_f, ts, target, forward))
    _render_footer()


if __name__ == "__main__":
    main()
