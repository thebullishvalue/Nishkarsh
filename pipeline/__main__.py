"""CLI: ``python -m pipeline run --csv PATH [...]`` · ``python -m pipeline synth [...]``.

Environment pins are set BEFORE numpy / numba are imported: one BLAS thread (A5-18; the
pipeline also wraps the run in ``threadpool_limits(1)``) and a numba cache outside the
source tree.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nishkarsh", "numba"))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--target", default="NIFTY50_PE")
    ap.add_argument("--no-kappa", action="store_true", help="Intelligence OFF: κ ≡ 1 (pure Aarambh)")
    ap.add_argument("--json", metavar="OUT", help="write the run report (nishkarsh.run_report/1) here")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline", description="Nishkarsh headless pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run on a sheet CSV (live data unless --snapshot)")
    r.add_argument("--csv", required=True, type=Path)
    r.add_argument("--date-col", default="DATE")
    r.add_argument("--snapshot", type=Path, metavar="DIR", help="replay market data from a snapshot dir (offline)")
    r.add_argument("--force", action="store_true", help="bypass cache TTLs (live sources)")
    _common(r)
    s = sub.add_parser("synth", help="run on the deterministic SYNTHETIC world (offline)")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--rows", type=int, default=4000)
    s.add_argument("--planted-ic", type=float, default=0.0)
    _common(s)
    a = ap.parse_args(argv)

    import pandas as pd
    from data.sources import FetchResult
    from pipeline.config import PipelineConfig
    from pipeline.inputs import InputError
    from pipeline.run import run_pipeline

    t0 = time.perf_counter()
    try:
        if a.cmd == "synth":
            from data.synthetic import synthetic_world
            sheet, sources = synthetic_world(a.seed, a.rows, planted_ic=a.planted_ic)
            today = pd.Timestamp(sheet["DATE"].iloc[-1])          # the world's own "today"
            cfg = PipelineConfig.build(today=today, target=a.target, date_col="DATE", kappa_enabled=not a.no_kappa)
            info = sources.sheet()
            info = FetchResult(None, info.provenance, info.as_of, [], info.note)
            res = run_pipeline(sheet, cfg, sources, sheet_info=info, today=today)
        else:
            from data.sources import LiveSources, SnapshotSources
            sheet = pd.read_csv(a.csv)
            sources = SnapshotSources(a.snapshot) if a.snapshot else LiveSources()
            cfg = PipelineConfig.build(target=a.target, date_col=a.date_col, kappa_enabled=not a.no_kappa)
            info = FetchResult(None, "live", None, [], f"CSV {a.csv.name}")
            res = run_pipeline(sheet, cfg, sources, sheet_info=info, force=a.force)
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if a.json:
        Path(a.json).write_text(json.dumps(res.run_report, indent=2, allow_nan=False))
    latest = (res.run_report.get("conclusion") or {}).get("latest") or {}
    print(json.dumps({"fingerprint": res.run_report["fingerprint"][:12], "latest": latest,
                      "trust": (res.run_report["conclusion"].get("trust") or {}).get("tier"),
                      "runtime_s": round(time.perf_counter() - t0, 1)}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
