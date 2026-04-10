"""
Data tab — Merged data table + CSV export.
Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime

from ui.components import render_section_header


def render_data_tab(ts_filtered, ts, active_target):
    """Data Table: Time series data with export functionality."""

    render_section_header(
        f"Time Series Data ({len(ts_filtered)} observations)",
        icon="database",
        accent="cyan",
    )

    display_cols = [
        "Date", "Actual", "FairValue", "Residual", "ModelSpread", "AvgZ",
        "OversoldBreadth", "OverboughtBreadth", "ConvictionScore", "Regime",
        "BullishDiv", "BearishDiv",
    ]
    display_cols = [c for c in display_cols if c in ts_filtered.columns]
    display_df = ts_filtered[display_cols].copy()

    rounding = {
        "AvgZ": 3, "ModelSpread": 3, "FairValue": 2,
        "Residual": 1, "ConvictionScore": 1, "OversoldBreadth": 1, "OverboughtBreadth": 1,
    }
    for col, decimals in rounding.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].round(decimals)

    if "BullishDiv" in display_df.columns:
        display_df["BullishDiv"] = display_df["BullishDiv"].apply(lambda x: "\u25cf" if x else "\u25cb")
    if "BearishDiv" in display_df.columns:
        display_df["BearishDiv"] = display_df["BearishDiv"].apply(lambda x: "\u25cf" if x else "\u25cb")

    st.dataframe(display_df, width='stretch', height=520)

    # ── Export section ──────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:var(--sp-4);"></div>',
        unsafe_allow_html=True,
    )
    csv_data = ts.to_csv(index=False).encode("utf-8")
    st.download_button(
        "\u2913  Download Full CSV",
        csv_data,
        f"nishkarsh_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
