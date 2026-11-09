"""
Nishkarsh — Main Streamlit entrypoint.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

NISHKARSH — one read of the Nifty 50, graded. Aarambh forecasts it from the top down,
Nirnay reads its fifty stocks from the bottom up, and the Nishkarsh Oscillator reads them as
one — graded by its own walk-forward record, with the weakest link named.

The whole analysis lives in the headless ``pipeline`` package; this file only collects the
data source, calls ``pipeline.run.run_pipeline`` and renders the C-PIPE session keys through
Tattva 2.8.0's shell ("Graphite"): appearance first → one progress slot → the rail →
cold start or the run → the page shell (tape → command bar → notice rail) →
``st.navigation`` pages → footer. Pages are thin wrappers over the one run; none recompute.

Usage:
    streamlit run app.py
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

import html  # noqa: E402
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
    COMPANY, DEFAULT_PREDICTORS, HISTORY_YEARS, NIRNAY_POLARITY, PRODUCT_NAME, TIMEFRAME_TRADING_DAYS, VERSION,
)
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.inputs import InputError, sheet_content_hash  # noqa: E402
from pipeline.run import SESSION_KEYS, run_pipeline  # noqa: E402
from ui.components import (  # noqa: E402
    _d, _list, _str, asof_date, num, panel, render_control_hint, render_header, render_kpi_strip, render_nav_brand,
    render_notice_rail, render_rail_readout, render_section_header, render_ticker, render_top_bar,
    render_warning_box, signed,
)
from ui.tabs.page_aarambh import render_aarambh_page  # noqa: E402
from ui.tabs.page_convergence import render_convergence_page  # noqa: E402
from ui.tabs.page_data import render_data_page  # noqa: E402
from ui.tabs.page_nirnay import render_nirnay_page  # noqa: E402
from ui.tabs.page_overview import read_verdict, render_overview_page  # noqa: E402
from ui.tabs.page_precedent import render_precedent_page  # noqa: E402
from ui.tabs.page_system import render_system_page  # noqa: E402
from ui.theme import inject_css, progress_bar  # noqa: E402

SOURCES = ("Google Sheets", "Upload")
INSTRUMENT = "NIFTY 50"
_RESULTS_CACHE_MAX = 3
# Every key derived from a loaded sheet / run. Start over clears ALL of them (A5-02).
_DERIVED_KEYS = SESSION_KEYS + ("data", "run_analysis", "results_cache", "_result_key", "_sheet_info",
                                "_sources_kind", "_sources_override", "_force_next", "_today",
                                "precedent_summary", "_prec_key", "tf_selected", "_resolved")

#: The pages: the read (Overview); the engines led by Convergence — where the two systems meet and
#: the read itself is drawn — then its top-down source (Aarambh), its bottom-up source (Nirnay)
#: and the base rate (Precedent); then the system's own grading and the record. (section, title, icon,
#: url_path, render function, windowed). Windowed pages carry the chart window in their
#: primary panel and show "Window" in the command bar.
PAGES = (
    ("", "Overview", ":material/dashboard:", "overview", render_overview_page, False),
    ("Engines", "Convergence", ":material/merge_type:", "convergence", render_convergence_page, True),
    ("Engines", "Aarambh", ":material/monitoring:", "aarambh", render_aarambh_page, True),
    ("Engines", "Nirnay", ":material/hub:", "nirnay", render_nirnay_page, True),
    ("Engines", "Precedent", ":material/history:", "precedent", render_precedent_page, False),
    ("System", "System", ":material/monitor_heart:", "system", render_system_page, False),
    ("System", "Data", ":material/table_chart:", "data", render_data_page, True),
)

#: The page favicon — Tattva's mark pattern (a ring and a rising line), in the Paper cobalt.
_FAVICON = ("data:image/svg+xml;utf8,"
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
            "<circle cx='12' cy='12' r='10' fill='none' stroke='%232B5FD9' stroke-width='2'/>"
            "<path d='M8 14l3-5 2 3 3-4' fill='none' stroke='%232B5FD9' stroke-width='2' "
            "stroke-linecap='round' stroke-linejoin='round'/></svg>")


def _clear_derived(keep_data: bool = False) -> None:
    for k in _DERIVED_KEYS:
        if keep_data and k in ("data", "run_analysis", "_sheet_info", "_sources_kind", "_sources_override", "_today"):
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


# ─── Appearance ──────────────────────────────────────────────────────────────
#: The DURABLE record of the appearance choice — a plain session key, never a widget key.
#: Streamlit garbage-collects the state of any widget not instantiated during a run; the
#: appearance control sits at the foot of the rail, so every run that reruns before reaching
#: it (Run Analysis, Start over, Refresh) would drop it and flip the theme back. A plain key
#: is never collected.
_THEME_CHOICE = "theme_choice"

#: The two appearances. Paper (light, for reading and print) LEADS, and the order is the
#: default: ``theme_choice()`` falls back to APPEARANCES[0] for any unset or unrecognised
#: value, so first-in-tuple IS first-run — one fact, so the toggle and the default agree.
APPEARANCES = ("Paper", "Slate")


def theme_choice() -> str:
    """The appearance the user last chose, always one of ``APPEARANCES``."""
    choice = st.session_state.get(_THEME_CHOICE)
    return choice if choice in APPEARANCES else APPEARANCES[0]


def _render_appearance_control() -> None:
    """The theme switch — LAST control in the rail, pinned to its foot by CSS
    (``.st-key-appearance``). Rendered exactly once per run (cold or loaded branch)."""
    with st.container(key="appearance"):
        st.markdown('<div class="sidebar-title">Appearance</div>', unsafe_allow_html=True)
        mode = st.segmented_control(
            "Appearance", list(APPEARANCES), key="theme_mode",
            default=theme_choice(), label_visibility="collapsed",
            help="Slate — dark, for working. Paper — light, for reading and print.",
        )
        # Mirror the widget into the DURABLE key and rerun, so the stylesheet at the top of
        # main() is re-injected with the new value (no half-themed run).
        if mode is not None and mode != theme_choice():
            st.session_state[_THEME_CHOICE] = mode
            st.rerun()


# ─── Cold start ──────────────────────────────────────────────────────────────
def _render_header() -> None:
    """Cold-start masthead only; once a run is loaded the command bar carries identity."""
    render_header(PRODUCT_NAME, "Top-Down Forecast · Bottom-Up Breadth · One Inference")


#: The three systems, as the cold-start screen describes them — data, not markup, rendered
#: through one template so the panels cannot drift apart in structure.
_SYSTEM_PANELS = (
    ("aarambh", "Aarambh", "The top-down forecast",
     "Forecasts the Nifty's next ten sessions from your sheet's valuation, rates and breadth "
     "series with a purged walk-forward ensemble, refitted on a schedule so no forecast ever sees "
     "its own outcome — then asks how stretched that forecast is across lookbacks.",
     (("Estimator", "Purged walk-forward ensemble"),
      ("Horizon", "10 sessions"),
      ("Pillar", "Forecast · extension · breadth"))),
    ("nirnay", "Nirnay", "The fifty stocks, bottom up",
     "Reads every Nifty 50 constituent through its own oscillator and a regime ensemble, then "
     "counts how many stand on each side. Its polarity is one switch, set by the owner and "
     "recorded with its evidence.",
     (("Signal", "MSF + MMR oscillator"),
      ("Breadth", "Share of stocks per zone"),
      ("Regime", "Kalman · HMM · GARCH · CUSUM"))),
    ("convergence", "Convergence", "One read, measured",
     "Scores the two systems against each other across direction, breadth, magnitude and regime, "
     "folds both and their agreement into one causal oscillator, and grades it the way every "
     "engine is graded — walk-forward, before any claim is made.",
     (("Fusion", "Aarambh × Nirnay"),
      ("Read", "Mean of three pillars, ±100"),
      ("Graded", "Walk-forward IC"))),
)


def _render_landing_page() -> None:
    """Cold start — a description of the product, built from the product's own parts:
    the claim, then section headers + a KPI strip + panels + one outcome grid."""
    st.markdown(
        '<div class="lede">'
        '<div class="lede-claim">Aarambh forecasts the Nifty from the top down, Nirnay reads its fifty '
        'stocks from the bottom up, and Nishkarsh reads them as one — graded by its own walk-forward '
        'record, with the weakest link named.</div>'
        '<div class="lede-cta">Pick a data source in the rail, then <strong>Run Analysis</strong>.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_section_header("Coverage", icon="layers")
    render_kpi_strip(
        [
            {"label": "Nifty 50 Constituents", "value": "50",
             "subtext": "Every current member read bottom up — today's list applied to the past, which "
                        "flatters the bottom-up history (survivorship)"},
            {"label": "Inputs → Pillars", "value": "10 → 3",
             "subtext": "Four top-down inputs, five bottom-up and one for agreement, each standardized "
                        "against its own past; no weight or sign fitted to returns"},
            {"label": "Daily History Per Run", "value": f"~{int(HISTORY_YEARS)}y",
             "subtext": "Every published value uses only data available on its own date, so re-running "
                        "extends the record instead of rewriting it"},
        ],
        max_cols=3, key="landing-coverage",
    )
    render_section_header("Systems", icon="cpu")
    cols = st.columns(3, gap="small")
    for col, (cls, name, kicker, body, specs) in zip(cols, _SYSTEM_PANELS):
        with col:
            with panel(f"landing-{cls}", name, context=kicker):
                st.markdown(
                    f'<div class="panel-copy">{html.escape(body)}</div>'
                    '<div class="panel-specs">'
                    + "".join(f'<div class="lookback-row"><span class="lbl">{html.escape(k)}</span>'
                              f'<span class="val">{html.escape(v)}</span></div>' for k, v in specs)
                    + "</div>",
                    unsafe_allow_html=True,
                )
    render_section_header("What a run returns", icon="target")
    outcomes = (
        ("One read", "Strongly bullish to strongly bearish, from three pillars — and the one thing "
                     "limiting how far it can be trusted, named."),
        ("A measured edge", "The read and each pillar scored on non-overlapping windows at the read's "
                            "horizon: the honest answer to whether it has anticipated the Nifty."),
        ("A base rate", "What the Nifty did after the most similar past reads — no model fitted to "
                        "returns, matched on the read's own state."),
        ("The record", "Every series behind the read, joined by date and exportable."),
    )
    st.markdown(
        '<div class="outcome-grid">'
        + "".join(f'<div class="outcome"><div class="o-t">{html.escape(t)}</div>'
                  f'<div class="o-d">{html.escape(d)}</div></div>' for t, d in outcomes)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    st.markdown(
        f'<div class="app-footer"><div class="content">'
        f'© {ist_now.year} <strong>{html.escape(PRODUCT_NAME)}</strong> &nbsp;·&nbsp; '
        f'{html.escape(COMPANY)} &nbsp;·&nbsp; v{html.escape(VERSION)} &nbsp;·&nbsp; '
        f'{ist_now.strftime("%Y-%m-%d %H:%M:%S IST")}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ─── Rail: Model readout ─────────────────────────────────────────────────────
def _render_model_readout() -> None:
    """What the run reached, not a control: the pillar weights, the read's walk-forward edge,
    its positive periods and Nirnay's polarity. Nothing about the model is configurable."""
    st.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)
    if not _d(st.session_state.get("conclusion")):
        render_control_hint("Run an analysis to populate.")
        return
    v = read_verdict()
    k = _d(v.get("kpi"))
    rows = [(f"{lab} weight", f"{w:.2f}", "accent")
            for lab, w in (("Top-down", num(_d(k.get("pillar_weights")).get("top_down"))),
                           ("Bottom-up", num(_d(k.get("pillar_weights")).get("bottom_up"))),
                           ("Agreement", num(_d(k.get("pillar_weights")).get("agreement"))))
            if w is not None]
    t = _d(v.get("trust"))
    ic = num(t.get("oos_ic"))
    if ic is not None:
        rows.append(("Read edge (IC)", signed(ic, "+.3f"), "long" if ic > 0 else "short"))
        nb = int(num(t.get("n_blocks")) or 0)
        if nb:
            rows.append(("Periods +", f"{int(num(t.get('positive_blocks')) or 0)}/{nb}", ""))
    rows.append(("Nirnay polarity", NIRNAY_POLARITY.capitalize(), ""))
    render_rail_readout(rows)


# ─── Inputs: the defaults the model runs on (not user-configurable) ──────────
def _resolve_inputs(df: pd.DataFrame) -> dict | str:
    """Target, date column and predictors — the defaults the retired "Model settings" controls
    defaulted to (rule 28: the model is not user-configurable). Returns an error string when the
    sheet cannot carry a run."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
               or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    date_cols = [c for c in df.columns if "date" in str(c).lower()] or [c for c in df.columns if c not in numeric]
    if len(numeric) < 2 or not date_cols:
        return "The sheet needs a date column and at least 2 numeric columns."
    date_col = date_cols[0]
    choices = [c for c in numeric if c not in date_cols]
    if not choices:
        return "The sheet needs at least 2 numeric columns besides the date."
    target = "NIFTY50_PE" if "NIFTY50_PE" in choices else choices[0]
    available = [c for c in choices if c != target]
    features = tuple(p for p in DEFAULT_PREDICTORS if p in available) or tuple(available[:3])
    return {"target": target, "date_col": date_col, "features": features}


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="NISHKARSH | Nifty 50 Inference", page_icon=_FAVICON, layout="wide",
                       initial_sidebar_state="expanded")
    # Resolve the appearance BEFORE anything is styled, from the DURABLE key — so CSS, charts,
    # tables and iframes agree on one value for the whole run.
    st.session_state["theme"] = "light" if theme_choice() == "Paper" else "dark"
    inject_css(theme=st.session_state["theme"])
    # Read by default_chart_context: every panel's context line reads "NIFTY 50 · 6M · units".
    st.session_state["active_target"] = INSTRUMENT
    # ONE main-area progress slot, used from the first click (no sidebar spinners).
    progress_container = st.empty()
    ss = st.session_state

    # ─── The control rail: Instrument → Model → Session → Appearance ──────
    # st.navigation (called once a run exists) pins its page nav to the TOP of the sidebar;
    # the brand is emitted here and lifted above it by CSS (`.nav-brand`).
    with st.sidebar:
        render_nav_brand()
        st.markdown('<div class="sidebar-title">Instrument</div>', unsafe_allow_html=True)
        source = st.radio("Data source", SOURCES, key="data_source")
        if ss.get("_sources_kind") and ss["_sources_kind"] != source:
            _clear_derived()                       # switching source discards everything derived
        has_data = "data" in ss and bool(ss.get("run_analysis"))

        if source == "Upload":
            up = st.file_uploader("Sheet file", type=["csv", "xlsx"])
            if up is not None and not has_data and st.button("Run Analysis", type="primary", width="stretch"):
                progress_bar(progress_container, 2, "Reading Uploaded Sheet", up.name)
                try:
                    df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                except Exception as e:  # noqa: BLE001
                    progress_container.empty()
                    st.error(f"Could not read the file: {e}")
                    return
                from data.sources import FetchResult
                ss.update(data=df, run_analysis=True, _sources_kind=source,
                          _sheet_info=FetchResult(None, "live", None, [], f"upload {up.name}"))
                st.rerun()
            elif up is None and not has_data:
                render_control_hint("CSV or Excel · one row per date")
        elif source == "Google Sheets":
            secret = _secret_sheet_url()
            if secret:
                render_control_hint("Configured sheet · CSV export")
                url = secret
            else:
                url = st.text_input("Sheet URL", value="", placeholder="Paste a Google Sheets link")
                render_control_hint("Shared Google Sheet · CSV export")
            refreshing = has_data and ss.get("_force_next") and ss.get("_sources_kind") == source
            if (not has_data and st.button("Run Analysis", type="primary", width="stretch")) or refreshing:
                if not url:
                    st.error("Paste a Google Sheets link (or configure one in secrets).")
                    return
                progress_bar(progress_container, 2, "Fetching Sheet", "Google Sheets · CSV export")
                r = _load_sheet(url, force=bool(ss.get("_force_next")))
                if r.value is None:
                    if refreshing:      # keep the last-good sheet; say so on the next render
                        ss["_pending_notices"] = [{"kind": "warning", "title": "Refresh failed",
                                                   "body": "The live sheet pull did not complete "
                                                           f"({r.note or 'no data'}). Showing the last-good "
                                                           "data. Try again once the source recovers."}]
                    else:
                        progress_container.empty()
                        st.error(f"Sheet load failed: {r.note or 'no data'}")
                        return
                else:
                    from data.sources import FetchResult
                    ss.update(data=r.value, run_analysis=True, _sources_kind=source,
                              _sheet_info=FetchResult(None, r.provenance, r.as_of, list(r.missing), r.note))
                if not has_data:
                    st.rerun()

    df = ss.get("data") if ss.get("run_analysis") else None

    # ─── Cold start ───────────────────────────────────────────────────────
    if df is None:
        with st.sidebar:
            _render_appearance_control()
        _render_header()
        # A session that HAD a run and lost its data is not a cold start — say so, and clear the
        # stale flags so the rail stops advertising a live session.
        if ss.get("run_analysis") and "data" not in ss:
            render_warning_box(
                "Session data expired",
                "This session had a completed run, but its data is no longer in memory — the server "
                "dropped it, which happens on hosted deployments when memory is reclaimed. Run the "
                "analysis again to rebuild it.")
            for k in ("run_analysis", "results_cache", "_result_key"):
                ss.pop(k, None)
        _render_landing_page()
        _render_footer()
        return

    # ─── Rail, loaded: Model readout · Session · Appearance ─────────────────
    with st.sidebar:
        _render_model_readout()
        st.markdown('<div class="sidebar-title">Session</div>', unsafe_allow_html=True)
        do_refresh = st.button(
            "Refresh data", type="secondary", width="stretch",
            help="Fetch the sheet and market data live now (bypassing caches), then recompute.")
        do_restart = st.button(
            "Start over", type="secondary", width="stretch",
            help="Clear this session's data and results and return to the start screen.")
        if do_restart:
            _clear_derived()
            st.rerun()
        if do_refresh:
            _clear_derived(keep_data=True)
            ss["_force_next"] = True
            st.rerun()
        _render_appearance_control()

    resolved = _resolve_inputs(df)
    if isinstance(resolved, str):
        st.error(resolved)
        return
    try:
        cfg = PipelineConfig.build(today=ss.get("_today"), target=resolved["target"], features=resolved["features"],
                                   date_col=resolved["date_col"], kappa_enabled=True)
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
            # Market data: Yahoo first, NSE archives on failure. `_sources_override` is the offline
            # injection point for tests and headless use only — the rail offers no such source.
            sources = ss.get("_sources_override")
            if sources is None:
                from data.nse_archive import AutoSources
                sources = AutoSources()
            done = [0]

            def _progress(pct, title, sub=""):
                done[0] = max(done[0], int(pct))       # monotonic, whatever the caller reports
                progress_bar(progress_container, done[0], title, sub)   # progress_bar escapes

            try:
                result = run_pipeline(df, cfg, sources, progress=_progress, force=bool(ss.pop("_force_next", False)),
                                      sheet_info=ss.get("_sheet_info"), today=ss.get("_today"))
            except InputError as e:
                progress_container.empty()
                st.error(str(e))
                return
            progress_container.empty()
            bundle = result.to_session_state()
            ss.update(bundle)
            cache[result_key] = bundle
            while len(cache) > _RESULTS_CACHE_MAX:
                cache.pop(next(iter(cache)))
            ss["_result_key"] = result_key
            st.rerun()

    _render_pages()


# ─── The page shell ──────────────────────────────────────────────────────────
def _shell_context() -> dict:
    """Everything the shell prints, read once from the run (built in the pipeline)."""
    ss = st.session_state
    rr = _d(ss.get("run_report"))
    v = read_verdict()
    notices = [n for n in _list(rr.get("notices")) if isinstance(n, dict)]
    notices += [n for n in _list(ss.get("_pending_notices")) if isinstance(n, dict)]
    return {"tape": _list(rr.get("tape")), "quote": _d(rr.get("quote")), "fresh": _d(rr.get("freshness")),
            "notices": notices, "verdict": v}


def _top_bar(ctx: dict, *, windowed: bool = False) -> None:
    """The page shell, identical on every page: the TAPE (the stocks) above the COMMAND BAR
    (the index and its read) above the NOTICE RAIL (the caveats on it). The bar names the chart
    window only on pages whose charts it frames."""
    v, q, f = ctx["verdict"], ctx["quote"], ctx["fresh"]
    render_ticker(ctx["tape"])
    read = _str(v.get("signal"), "NO READ")
    vt = _str(v.get("value_text"))
    meta = [("Read", f"{read} · {vt}" if vt and vt != "—" else read),
            ("Horizon", f"{int(num(v.get('horizon')) or 10)}D"),
            ("As of", asof_date(f.get("as_of") or q.get("as_of")))]
    if windowed:
        tf = st.session_state.get("tf_selected")
        meta.insert(1, ("Window", tf if tf in (*TIMEFRAME_TRADING_DAYS, "ALL") else "6M"))
    render_top_bar(target=INSTRUMENT, price=num(q.get("last")), change_pct=num(q.get("chg_pct")),
                   status_label=_str(f.get("label")), status_tone=_str(f.get("tone"), "neutral"),
                   meta_items=meta)
    render_notice_rail(ctx["notices"])


def _safe_render(name: str, fn) -> None:
    """Render a page's content; a failure says which page and why, and the app stays up."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — one page, not the app
        render_warning_box(f"{name} could not be drawn", f"{type(e).__name__}: {e}")


def _page_fn(title: str, render, windowed: bool):
    def _page() -> None:
        _top_bar(_shell_context(), windowed=windowed)
        _safe_render(title, render)
    _page.__name__ = f"page_{title.lower()}"
    return _page


def _render_pages() -> None:
    if st.session_state.get("tf_selected") not in (*TIMEFRAME_TRADING_DAYS, "ALL"):
        st.session_state["tf_selected"] = "6M"
    sections: dict[str, list] = {}
    for section, title, icon, url, render, windowed in PAGES:
        sections.setdefault(section, []).append(
            st.Page(_page_fn(title, render, windowed), title=title, icon=icon, url_path=url,
                    default=(title == PAGES[0][1])))
    nav = st.navigation(sections, position="sidebar")
    nav.run()
    st.session_state.pop("_pending_notices", None)     # shown on this render
    _render_footer()


if __name__ == "__main__":
    main()
