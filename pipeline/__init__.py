"""Nishkarsh headless pipeline (contract C-PIPE).

The whole analysis runs without Streamlit::

    from pipeline import PipelineConfig, run_pipeline
    from data.sources import LiveSources
    result = run_pipeline(sheet_df, PipelineConfig.build(target="NIFTY50_PE"), LiveSources())
    result.run_report            # JSON-safe "nishkarsh.run_report/1"
    result.to_session_state()    # exactly the C-PIPE keys the UI reads

CLI: ``python -m pipeline run --csv PATH`` / ``python -m pipeline synth``.
Submodules are imported lazily so ``pipeline.config`` stays import-light.
"""

from __future__ import annotations

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline", "prepare_sheet",
           "build_run_report", "render_diagnostics"]


def __getattr__(name):  # PEP 562 lazy exports
    if name == "PipelineConfig":
        from pipeline.config import PipelineConfig
        return PipelineConfig
    if name in ("PipelineResult", "run_pipeline"):
        from pipeline import run as _run
        return getattr(_run, name)
    if name == "prepare_sheet":
        from pipeline.inputs import prepare_sheet
        return prepare_sheet
    if name in ("build_run_report", "render_diagnostics"):
        from pipeline import report as _report
        return getattr(_report, name)
    raise AttributeError(name)
