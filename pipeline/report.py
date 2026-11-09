"""Run report ("nishkarsh.run_report/1") and the console SYSTEM DIAGNOSTICS rendered FROM it.

The report is the single source of every verdict line: the console, the CLI ``--json`` and
the app's provenance/staleness banners all read it. It is JSON-safe (``json.dumps(...,
allow_nan=False)`` holds — NaN/inf become null). Volatile fields (``run_id``,
``generated_at``, ``timings``) are the only ones that differ between two identical runs.
Edge words are never composed here: the verdict lines quote the conclusion's own fields
(``trust.tier``/``trust.text``, ``nirnay.text``, ``edge.significant`` …), per ruling R4.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import core.config as C

SCHEMA = "nishkarsh.run_report/1"
VOLATILE_KEYS = ("run_id", "generated_at", "timings")
_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "numba", "threadpoolctl",
             "pyarrow", "streamlit", "yfinance")
_REPO = Path(__file__).resolve().parent.parent


def json_safe(o):
    """Recursively convert to JSON-native types; non-finite floats → None."""
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        ts = pd.Timestamp(o)
        return None if pd.isna(ts) else (str(ts.date()) if ts == ts.normalize() else ts.isoformat())
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(json_safe(v) for v in o)
    if isinstance(o, (list, tuple, np.ndarray, pd.Index)):
        return [json_safe(v) for v in list(o)]
    if isinstance(o, pd.Series):
        return {str(k): json_safe(v) for k, v in o.items()}
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _packages() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in _PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _aarambh(result) -> dict:
    eng, ts = result.engine, result.aarambh_ts
    sig = eng.get_current_signal() if len(ts) else {}
    keep = ("signal", "strength", "conviction_score", "forecast", "forecast_z", "edge_assessment", "edge_source",
            "tradeable", "has_forward_edge", "inverted", "hurst", "ou_half_life", "half_life_meaningful",
            "oversold_breadth", "overbought_breadth", "regime", "forward_signal", "forecast_skill_vs_rw")
    stats = {k: v for k, v in (eng.get_model_stats() or {}).items()
             if isinstance(v, (int, float, bool, str, np.integer, np.floating, np.bool_))}
    valid = ts["Valid"].astype(bool) if "Valid" in ts.columns else pd.Series(False, index=ts.index)
    return {"rows": int(len(ts)), "valid": int(valid.sum()),
            "first_valid": (ts.index[valid.to_numpy()][0] if valid.any() else None),
            "feature_names": list(eng.feature_names), "wf_meta": dict(eng.wf_meta or {}),
            "model_stats": stats, "signal": {k: sig.get(k) for k in keep},
            "changepoints_heuristic": len(sig.get("changepoints_heuristic") or [])}


def _nirnay(result) -> dict:
    dfs, daily = result.nirnay_constituent_dfs, result.nirnay_daily
    lasts = {}
    for sym, df in dfs.items():
        live = df[~df["Warmup"].astype(bool)] if "Warmup" in df.columns else df
        if len(live):
            lasts[sym] = live.iloc[-1]

    def _med(col):
        v = [float(r[col]) for r in lasts.values() if col in r.index and pd.notna(r[col])]
        return float(np.median(v)) if v else None

    mmr = [float(r["MMR_Quality"]) for r in lasts.values() if "MMR_Quality" in r.index and pd.notna(r["MMR_Quality"])]
    latest = {}
    if len(daily):
        cols = ("n_universe", "n_live", "coverage", "Oversold_Pct", "Overbought_Pct", "Avg_Signal",
                "Regime_Bull_Pct", "Regime_Bear_Pct", "Regime_Neutral_Pct", "Regime_Transition_Pct",
                "Dispersion", "Buy_Signals", "Sell_Signals", "Change_Points")
        lr = daily.iloc[-1]
        latest = {"date": daily.index[-1], **{c: lr.get(c) for c in cols if c in daily.columns}}
    ts = result.aarambh_ts
    valid_dates = ts.index[ts["Valid"].astype(bool).to_numpy()] if "Valid" in ts.columns else ts.index
    overlap = int(pd.DatetimeIndex(valid_dates).isin(pd.DatetimeIndex(daily.index)).sum()) if len(daily) else 0
    return {"stocks_requested": int(len(dfs) + len(result.nirnay_failures)), "stocks_analyzed": int(len(dfs)),
            "failed": dict(result.nirnay_failures), "trading_days": int(len(daily)),
            "span": ([daily.index[0], daily.index[-1]] if len(daily) else None),
            "overlap_valid_dates": overlap, "aarambh_valid_dates": int(len(valid_dates)),
            "latest": latest, "os_cut_median_live": _med("OS_Cut"), "ob_cut_median_live": _med("OB_Cut"),
            "mmr_quality": ({"label": "√skill, 20d level-fit vs RW (MSE)", "mean": float(np.mean(mmr)),
                             "min": float(np.min(mmr)), "max": float(np.max(mmr))} if mmr else None),
            "regime_latest": dict(Counter(str(r.get("Regime")) for r in lasts.values())),
            "vol_regime_latest": dict(Counter(str(r.get("Vol_Regime")) for r in lasts.values()
                                              if "Vol_Regime" in r.index))}


def _anatomy(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": 0, "note": "Agreement anatomy (descriptive, not used in the conclusion): no warm Nirnay dates."}
    return {"rows": int(len(df)), "span": [df.index[0], df.index[-1]], "latest": df.iloc[-1].to_dict(),
            "mean": df.mean(numeric_only=True).to_dict(),
            "note": "Agreement anatomy (descriptive, not used in the conclusion)."}


def build_run_report(result) -> dict:
    cfg, prep = result.config, result.prepared
    prov = {k: (v.to_dict() if hasattr(v, "to_dict") else None) for k, v in result.provenance.items()}
    prec = result.precedent or {}
    rep = {
        "schema": SCHEMA,
        "run_id": result.run_id,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"version": C.VERSION, "python": sys.version.split()[0], "platform": platform.platform(),
                       "packages": _packages(), "git_sha": _git_sha()},
        "config": cfg.to_dict(),
        "fingerprint": cfg.fingerprint(),
        "inputs": prep.summary(),
        "data": {**prov, "synthetic": bool(result.synthetic), "today": result.today,
                 "macro_panel": result.macro.summary()},
        "timings": dict(result.timings),
        "aarambh": _aarambh(result),
        "nirnay": _nirnay(result),
        "label_src": result.label_src,
        "label_alignment": getattr(result, "label_alignment", None),
        "conclusion": result.conclusion.to_dict(),
        "anatomy": _anatomy(result.anatomy_df),
        "precedent": {"hero": prec.get("hero"), "summary": prec.get("summary") or {},
                      "n_analogs": len(prec.get("analogs") or [])},
        "warnings": list(result.warnings),
        # The page shell's data (pipeline.shell): the tape, ONE freshness reading, the notices.
        "tape": (getattr(result, "shell", None) or {}).get("tape") or [],
        "freshness": (getattr(result, "shell", None) or {}).get("freshness") or {},
        "quote": (getattr(result, "shell", None) or {}).get("quote") or {},
        "notices": (getattr(result, "shell", None) or {}).get("notices") or [],
    }
    rep = json_safe(rep)
    json.dumps(rep, allow_nan=False)  # contract: strictly JSON-safe
    return rep


def strip_volatile(report: dict) -> dict:
    """Report minus the fields that legitimately differ between identical runs."""
    return {k: v for k, v in report.items() if k not in VOLATILE_KEYS}


def _f(x, spec="+.3f", none="n/a"):
    return none if x is None else format(x, spec)


def verdict_lines(report: dict) -> dict:
    """The verdict strings (console + app banners) — every word from a report field."""
    con = report.get("conclusion") or {}
    lat, trust, nir = con.get("latest") or {}, con.get("trust") or {}, con.get("nirnay") or {}
    head = (f"{lat.get('direction', 'NEUTRAL')} · {lat.get('tier', 'NO LEAN')} · C {_f(lat.get('C'), '+.1f')}"
            f" (as of {lat.get('date', 'n/a')})")
    return {"headline": head, "trust": trust.get("text") or str(trust.get("tier", "INSUFFICIENT_DATA")),
            "trust_tier": str(trust.get("tier", "INSUFFICIENT_DATA")),
            "nirnay": f"{nir.get('state', '')} — {nir.get('text', '')}",
            "synthetic": "SYNTHETIC DEMO DATA — not market data" if (report.get("data") or {}).get("synthetic") else ""}


def render_diagnostics(report: dict, log) -> None:
    """Console SYSTEM DIAGNOSTICS block, rendered ONLY from ``report``."""
    inp, data, a, n = report["inputs"], report["data"], report["aarambh"], report["nirnay"]
    con, cfg = report["conclusion"], report["config"]
    v = verdict_lines(report)
    log.section("SYSTEM DIAGNOSTICS", "DIAGNOSTICS")
    if v["synthetic"]:
        log.warning(v["synthetic"])

    log.section("Data & Provenance")
    log.item("Sheet", f"{inp['rows_used']} rows used / {inp['rows_raw']} · {inp['sheet_span'][0]} → {inp['sheet_span'][1]}"
             f" · hash {inp['content_hash'][:12]}")
    log.item("Effective start", f"{inp['effective_start']}" + (f" (bounded by {inp['bounding_predictor']})"
                                                               if inp.get("bounding_predictor") else ""))
    if inp.get("dropped_features"):
        log.item("Dropped predictors", ", ".join(inp["dropped_features"]))
    if inp.get("lagged_columns"):
        log.item("Lagged one session", ", ".join(inp["lagged_columns"]) + " (non-IST sheet columns)")
    for k in ("sheet", "constituents", "macro", "ohlcv", "index"):
        p = data.get(k)
        if p:
            log.item(f"  {k}", f"{p['provenance']}" + (f" · as of {p['as_of']}" if p.get("as_of") else "")
                     + (f" · {len(p['missing'])} missing" if p.get("missing") else "")
                     + (f" · {p['note'][:90]}" if p.get("note") else ""))
    mp = data["macro_panel"]
    log.item("Macro panel", f"{mp['raw_columns']} raw → {mp['mmr_columns']} MMR inputs"
             + (" (causal PCA factors)" if mp["factors"] else "") + f" · {mp['lagged_non_ist']} non-IST lagged")

    log.section("Aarambh — walk-forward")
    wf = a.get("wf_meta") or {}
    log.item("Mode", "PREDICTIVE (forward)" if cfg["forward_mode"] else "LEVELS — valuation context, falsified as a forecaster")
    log.item("Valid forecasts", f"{a['valid']}/{a['rows']} · first {a.get('first_valid')}")
    log.item("Walk-forward", f"train {wf.get('min_train')}→{wf.get('max_train')} · purge {wf.get('purge')} · refit "
             f"{wf.get('refit')} · chunks {wf.get('chunks_total')} (starved {wf.get('chunks_skipped_starved')}, "
             f"fallback {wf.get('chunks_fallback')})")
    ms = a.get("model_stats") or {}
    log.item("R² vs RW (magnitude)", _f(ms.get("r2_vs_rw")))
    s = a.get("signal") or {}
    log.item("Engine conviction", f"{_f(s.get('conviction_score'), '+.1f')} (− = bullish) · forecast z {_f(s.get('forecast_z'), '+.2f')}")
    log.item("Engine edge field", f"{s.get('edge_assessment')} [{s.get('edge_source')}]")

    log.section("Nirnay — constituents")
    log.item("Analyzed", f"{n['stocks_analyzed']}/{n['stocks_requested']} (failed {len(n['failed'])}) · "
             f"{n['trading_days']} days · overlap {n['overlap_valid_dates']}/{n['aarambh_valid_dates']} valid dates")
    lt = n.get("latest") or {}
    if lt:
        log.item("Latest day", f"{lt.get('date')} · live {_f(lt.get('n_live'), '.0f')}/{_f(lt.get('n_universe'), '.0f')} · OS "
                 f"{_f(lt.get('Oversold_Pct'), '.0f')}% · OB {_f(lt.get('Overbought_Pct'), '.0f')}% · "
                 f"avg osc {_f(lt.get('Avg_Signal'), '+.2f')}")
    log.item("Oversold / Overbought cuts", f"{_f(n.get('os_cut_median_live'), '+.2f')} / "
             f"{_f(n.get('ob_cut_median_live'), '+.2f')} (median live adaptive cut)")
    mq = n.get("mmr_quality")
    if mq:
        log.item(f"MMR Quality ({mq['label']})", f"mean {mq['mean']:.3f} · min {mq['min']:.3f} · max {mq['max']:.3f}")
    if n.get("regime_latest"):
        log.item("Regime (latest)", " · ".join(f"{k}:{c}" for k, c in sorted(n["regime_latest"].items())))

    log.section("Conclusion — Edge Ledger")
    log.item("Return label", report.get("label_src") or "none")
    for key, name in (("edge", "Aarambh head (full history)"), ("edge_C", "C (out-of-sample)")):
        e = con.get(key) or {}
        if e.get("available"):
            ci = e.get("boot_ci")
            log.item(name, f"IC {_f(e.get('ic'))} · n {e.get('n')} · HAC t {_f(e.get('hac_t'), '+.2f')} · "
                     f"p {_f(e.get('p'), '.3f')} · CI {('[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']') if ci else 'n/a'}"
                     f" · significant={e.get('significant')}")
        else:
            log.item(name, "not available")
    lf = con.get("lift") or {}
    log.item("Convergence lift", f"available={lf.get('available')} · lift {_f(lf.get('lift'))} · p "
             f"{_f(lf.get('p_one_sided'), '.3f')} · MDE {_f(lf.get('mde'))} · passed={lf.get('passed')} · "
             f"κ {_f(lf.get('kappa'), '.2f')} · κ enabled={con.get('kappa_enabled')}")
    oos = con.get("oos") or {}
    log.item("OOS C vs A", f"IC_C {_f(oos.get('ic_C'))} · IC_A {_f(oos.get('ic_A'))} · n {oos.get('n')} · "
             f"adds_value={oos.get('adds_value')}")
    bt = con.get("backtest") or {}
    ref = (bt.get("by_cost") or {}).get(str(float(bt.get("reference_bps", C.BACKTEST_REFERENCE_BPS)))) or {}
    log.item(f"Backtest @{bt.get('reference_bps')}bps", f"net Sharpe {_f(ref.get('net_sharpe'), '+.2f')} · t "
             f"{_f(ref.get('t'), '+.2f')} · breakeven {_f(ref.get('breakeven_bps'), '.1f')}bps · n {ref.get('n')}")
    log.item("Nirnay", v["nirnay"])
    hero = (report.get("precedent") or {}).get("hero")
    if hero:
        log.item("Precedent", f"{hero['n']} analogs · {hero['horizon']}d median {_f(hero['median'], '+.2f')}% · "
                 f"{_f(hero['positive_pct'], '.0f')}% up vs {_f(hero.get('unconditional_positive_pct'), '.0f')}% "
                 f"unconditional · p {_f(hero.get('binomial_p'), '.2f')}")

    log.section("Nishkarsh — Final Verdict")
    log.item("Conclusion", v["headline"])
    log.item("Trust", v["trust"])
    for w in report.get("warnings") or []:
        log.warning(w)
    log.summary("RUN SUMMARY", {"Run ID": report.get("run_id"), "Fingerprint": report["fingerprint"][:12],
                                **{f"t_{k}": f"{t:.1f}s" for k, t in (report.get("timings") or {}).items()}})
