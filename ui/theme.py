"""
Nishkarsh v2 — Shared CSS, chart theming, and color constants for the UI layer.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

UI — "Obsidian Quant" Institutional Research Terminal design language.

Aesthetic: "Obsidian Quant" — Institutional Research Terminal
-------------------------------------------------------------
Precision-instrument design language for quantitative finance.
- Display/UI:  Syne (geometric, authoritative, distinctive)
- Body/Data:   JetBrains Mono (refined monospace, tabular precision)
- Palette:     Obsidian (#0A0E17 -> #050810), Amber Gold (#D4A853),
               Cyan (#22D3EE), Emerald (#34D399), Rose (#FB7185)
- Surfaces:    Frameless glass panels with thin border strokes.
- Details:     Subtle gradient mesh, calibrated luminescence on signals,
               precision instrument aesthetic.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from core.config import COLOR_AMBER, COLOR_CYAN, COLOR_GREEN

# Path to external CSS files: the Obsidian base, then the product layer (typography, cards, Overview)
CSS_PATH = Path(__file__).parent / "theme.css"
PRODUCT_CSS_PATH = Path(__file__).parent / "product.css"

# ── Shared Plotly layout config ─────────────────────────────────────────────
# Eliminates massive duplication across all tab files.

PLOTLY_FONT = dict(family="Inter, -apple-system, sans-serif", color="#94A3B8", size=11)
PLOTLY_HOVERLABEL = dict(
    bgcolor="rgba(13, 18, 32, 0.96)",
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#F1F5F9"),
    bordercolor="rgba(255,255,255,0.08)",
    align="left",
)
PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
    font=dict(size=11, family="Inter, -apple-system, sans-serif"),
    bgcolor="rgba(0,0,0,0)",
)
PLOTLY_MARGIN = dict(t=20, l=50, r=20, b=40)
PLOTLY_GRID = "rgba(255,255,255,0.035)"
PLOTLY_GRID_ZERO = "rgba(255,255,255,0.06)"

# Interactive chart config — click + zoom + pan
def chart_layout(
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
    responsive: bool = False,
) -> dict:
    """Return a base Plotly layout dict for the Obsidian Quant theme.

    Args:
        height: Fixed pixel height for the chart.
        show_legend: Whether to show the legend.
        margin: Custom margin dict.
        responsive: If True, adds CSS-based responsive sizing via autosize.
    """
    base = dict(
        height=height,
        showlegend=show_legend,
        legend=PLOTLY_LEGEND if show_legend else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        hovermode="x unified",
        hoverlabel=PLOTLY_HOVERLABEL,
        margin=margin or PLOTLY_MARGIN,
        spikedistance=-1,
    )
    if responsive:
        base["autosize"] = True
    return base


def style_axes(fig, y_title: str = "", x_title: str = "", y_range=None, row=None, col=None) -> None:
    """Apply consistent axis styling to a Plotly figure."""
    kw = {}
    if row is not None:
        kw["row"] = row
    if col is not None:
        kw["col"] = col

    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        gridwidth=0.5,
        zeroline=False,
        linecolor="rgba(255,255,255,0.04)",
        title_text=x_title,
        tickfont=dict(size=10, family="Inter, -apple-system, sans-serif", color="#7C879A"),
        # Vertical crosshair — dashed dim grey
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=0.5,
        spikedash="dash",
        spikecolor="rgba(148,163,184,0.18)",
        **kw,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor=PLOTLY_GRID_ZERO,
        zerolinewidth=0.5,
        linecolor="rgba(255,255,255,0.04)",
        title_text=y_title,
        range=y_range,
        tickfont=dict(size=10, family="Inter, -apple-system, sans-serif", color="#7C879A"),
        hoverformat=".2f",
        **kw,
    )


def inject_css() -> None:
    """Inject the Obsidian Quant Terminal CSS into the Streamlit app.

    Loads from external theme.css file for maintainability.
    Injects on every render — Streamlit deduplicates identical <style> blocks.
    """
    # Explicit UTF-8: theme.css embeds Devanagari strings, and Path.read_text() with no
    # encoding falls back to the OS locale (commonly cp1252 on Windows) → UnicodeDecodeError.
    css = "\n".join(p.read_text(encoding="utf-8") if p.exists() else f"/* {p.name} not found */"
                    for p in (CSS_PATH, PRODUCT_CSS_PATH))

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """Render a themed progress card into an ``st.empty()`` slot. ``label``/``sub`` are plain
    text (escaped here — progress text can carry remote symbol names, A4-06)."""
    label, sub = html.escape(str(label), quote=True), html.escape(str(sub or ""), quote=True)
    pct = int(max(0, min(100, int(pct))))
    is_complete = pct >= 100
    bar_color = COLOR_GREEN if is_complete else COLOR_AMBER if pct > 50 else COLOR_CYAN
    dot_class = "pulse-dot complete" if is_complete else "pulse-dot"
    slot.markdown(
        f"""
    <div class="progress-card">
        <div class="progress-label">
            <span class="{dot_class}"></span>{label}
        </div>
        {f'<div class="progress-sub">{sub}</div>' if sub else ''}
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%;background:{bar_color};box-shadow:0 0 10px {bar_color};"></div>
        </div>
        <div class="progress-pct">{pct}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
