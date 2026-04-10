"""
Shared CSS, chart theming, and color constants for the UI layer.

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

import streamlit as st

from core.config import (
    CHART_BG,
    CHART_GRID,
    CHART_ZEROLINE,
    CHART_FONT_COLOR,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_GOLD,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_PURPLE,
    COLOR_MUTED,
)

VERSION = "3.1.0"
PRODUCT_NAME = "Nishkarsh"
COMPANY = "@thebullishvalue"

# ── Shared Plotly layout config ─────────────────────────────────────────────
# Eliminates massive duplication across all tab files.

PLOTLY_FONT = dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10)
PLOTLY_HOVERLABEL = dict(
    bgcolor="rgba(10, 14, 23, 0.95)",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
    bordercolor="rgba(255,255,255,0.08)",
)
PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
    font=dict(size=10, family="JetBrains Mono, monospace"),
    bgcolor="rgba(0,0,0,0)",
)
PLOTLY_MARGIN = dict(t=20, l=50, r=20, b=40)
PLOTLY_GRID = "rgba(255,255,255,0.035)"
PLOTLY_GRID_ZERO = "rgba(255,255,255,0.06)"


def chart_layout(
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
) -> dict:
    """Return a base Plotly layout dict for the Obsidian Quant theme."""
    return dict(
        height=height,
        showlegend=show_legend,
        legend=PLOTLY_LEGEND if show_legend else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        hovermode="x unified",
        hoverlabel=PLOTLY_HOVERLABEL,
        margin=margin or PLOTLY_MARGIN,
        # Spike lines — dim dashed grey crosshair
        spikedistance=-1,
    )


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
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
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
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        hoverformat=".2f",
        **kw,
    )


def inject_css() -> None:
    """Inject the Obsidian Quant Terminal CSS into the Streamlit app."""
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

    /* ================================================================
       DESIGN TOKENS — Obsidian Quant
       ================================================================ */
    :root {
        /* ---- Spacing scale (4px base) ---- */
        --sp-1: 0.25rem;   /* 4 */
        --sp-2: 0.5rem;    /* 8 */
        --sp-3: 0.75rem;   /* 12 */
        --sp-4: 1rem;      /* 16 */
        --sp-5: 1.25rem;   /* 20 */
        --sp-6: 1.5rem;    /* 24 */
        --sp-8: 2rem;      /* 32 */
        --sp-10: 2.5rem;   /* 40 */
        --sp-12: 3rem;     /* 48 */

        /* ---- Surfaces ---- */
        --bg-deep:      #050810;
        --bg-base:      #0A0E17;
        --bg-elevated:  #111827;
        --bg-surface:   #151C2C;
        --glass:        rgba(17, 24, 39, 0.45);
        --glass-hover:  rgba(21, 28, 44, 0.65);
        --border:       rgba(255, 255, 255, 0.05);
        --border-active: rgba(212, 168, 83, 0.35);
        --border-subtle: rgba(255, 255, 255, 0.03);

        /* ---- Ink ---- */
        --ink-primary:  #F1F5F9;
        --ink-secondary:#94A3B8;
        --ink-tertiary: #4B5563;
        --ink-inverse:  #0A0E17;

        /* ---- Accent palette ---- */
        --amber:        #D4A853;
        --amber-dim:    rgba(212, 168, 83, 0.6);
        --amber-glow:   rgba(212, 168, 83, 0.25);
        --amber-bright: #E8C478;
        --cyan:         #22D3EE;
        --cyan-glow:    rgba(34, 211, 238, 0.2);
        --cyan-dim:     rgba(34, 211, 238, 0.5);
        --emerald:      #34D399;
        --emerald-glow: rgba(52, 211, 153, 0.2);
        --emerald-bright: #6EE7B7;
        --rose:         #FB7185;
        --rose-glow:    rgba(251, 113, 133, 0.2);
        --rose-bright:  #FDA4AF;
        --violet:       #A78BFA;
        --violet-glow:  rgba(167, 139, 250, 0.2);

        /* ---- Type ---- */
        --display:      'Syne', -apple-system, sans-serif;
        --data:         'JetBrains Mono', 'Fira Code', monospace;

        /* ---- Radius ---- */
        --r-sm: 6px;
        --r-md: 10px;
        --r-lg: 14px;
    }

    /* ================================================================
       Globals & App Overrides
       ================================================================ */
    * {
        scrollbar-width: thin;
        scrollbar-color: var(--ink-tertiary) transparent;
    }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--ink-tertiary); border-radius: 3px; }

    html, body, [class*="css"] {
        font-family: var(--data);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stApp {
        background: var(--bg-deep) !important;
        color: var(--ink-primary);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 80% 60% at 20% 10%, rgba(212, 168, 83, 0.02) 0%, transparent 70%),
            radial-gradient(ellipse 60% 80% at 80% 90%, rgba(34, 211, 238, 0.015) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    footer { visibility: hidden !important; }

    .block-container {
        padding-top: var(--sp-8) !important;
        padding-bottom: var(--sp-6) !important;
        max-width: 98%;
    }

    /* ================================================================
       Masthead
       ================================================================ */
    .premium-header {
        padding: var(--sp-5) 0 var(--sp-6) 0;
        margin-bottom: var(--sp-10);
        position: relative;
    }
    .premium-header::after {
        content: "";
        display: block;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, var(--amber) 0%, var(--amber-glow) 40%, transparent 80%);
        margin-top: var(--sp-5);
    }
    .premium-header h1 {
        font-family: var(--display);
        font-weight: 800;
        font-size: 2.75rem;
        line-height: 1;
        letter-spacing: -0.02em;
        color: var(--ink-primary);
        margin: 0 0 0.5rem 0;
    }
    .premium-header .tagline {
        font-family: var(--data);
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--amber-dim);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin: 0;
    }

    /* ================================================================
       Section Headers — replaces raw ##### markdown
       ================================================================ */
    .section-hdr {
        display: flex;
        align-items: center;
        gap: var(--sp-3);
        margin: var(--sp-10) 0 var(--sp-2) 0;
        padding-bottom: var(--sp-3);
        border-bottom: 1px solid var(--border);
    }
    .section-hdr:first-child { margin-top: 0; }
    .section-hdr .icon {
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: var(--r-sm);
        background: rgba(212, 168, 83, 0.06);
        border: 1px solid rgba(212, 168, 83, 0.12);
        flex-shrink: 0;
    }
    .section-hdr .icon svg {
        width: 14px;
        height: 14px;
        color: var(--amber);
        stroke: var(--amber);
    }
    .section-hdr .icon.cyan { background: rgba(34, 211, 238, 0.06); border-color: rgba(34, 211, 238, 0.12); }
    .section-hdr .icon.cyan svg { color: var(--cyan); stroke: var(--cyan); }
    .section-hdr .icon.emerald { background: rgba(52, 211, 153, 0.06); border-color: rgba(52, 211, 153, 0.12); }
    .section-hdr .icon.emerald svg { color: var(--emerald); stroke: var(--emerald); }
    .section-hdr .icon.violet { background: rgba(167, 139, 250, 0.06); border-color: rgba(167, 139, 250, 0.12); }
    .section-hdr .icon.violet svg { color: var(--violet); stroke: var(--violet); }
    .section-hdr .icon.rose { background: rgba(251, 113, 133, 0.06); border-color: rgba(251, 113, 133, 0.12); }
    .section-hdr .icon.rose svg { color: var(--rose); stroke: var(--rose); }
    .section-hdr .text h3 {
        font-family: var(--display);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-primary);
        margin: 0;
        line-height: 1;
    }
    .section-hdr .text .desc {
        font-family: var(--data);
        font-size: 0.68rem;
        color: var(--ink-tertiary);
        margin-top: 0.2rem;
        line-height: 1.4;
    }

    /* Section spacing — used between major blocks */
    .section-gap {
        height: var(--sp-8);
    }

    /* ================================================================
       System cards (landing page)
       ================================================================ */
    .system-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--r-lg);
        padding: var(--sp-6) var(--sp-6) var(--sp-5);
        position: relative;
        transition: all 300ms cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        cursor: default;
        height: 100%;
    }
    .system-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%);
    }
    .system-card::after {
        content: "";
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 2px;
        background: var(--ink-tertiary);
        transition: background 300ms ease, box-shadow 300ms ease;
    }
    .system-card:hover {
        background: var(--glass-hover);
        border-color: var(--border-active);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    .system-card h3 {
        font-family: var(--display);
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0 0 var(--sp-4) 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        text-transform: uppercase;
    }
    .system-card h3 svg { flex-shrink: 0; }
    .system-card p {
        font-family: var(--data);
        font-size: 0.78rem;
        color: var(--ink-secondary);
        line-height: 1.7;
        margin: 0 0 var(--sp-5) 0;
    }
    .system-card .spec {
        font-family: var(--data);
        font-size: 0.68rem;
        color: var(--ink-tertiary);
        line-height: 1.9;
        border-top: 1px solid var(--border);
        padding-top: var(--sp-4);
        margin-top: auto;
    }
    .system-card .spec span { font-weight: 600; }

    /* System card variants */
    .system-card.aarambh { border-left: none; }
    .system-card.aarambh::after { background: var(--violet); box-shadow: 0 0 14px var(--violet-glow); }
    .system-card.aarambh h3 { color: var(--violet); }
    .system-card.aarambh h3 svg { color: var(--violet); filter: drop-shadow(0 0 8px var(--violet-glow)); }
    .system-card.aarambh .spec span { color: var(--violet); }

    .system-card.nirnay { border-left: none; }
    .system-card.nirnay::after { background: var(--cyan); box-shadow: 0 0 14px var(--cyan-glow); }
    .system-card.nirnay h3 { color: var(--cyan); }
    .system-card.nirnay h3 svg { color: var(--cyan); filter: drop-shadow(0 0 8px var(--cyan-glow)); }
    .system-card.nirnay .spec span { color: var(--cyan); }

    .system-card.convergence { border-left: none; }
    .system-card.convergence::after { background: var(--amber); box-shadow: 0 0 14px var(--amber-glow); }
    .system-card.convergence h3 { color: var(--amber); }
    .system-card.convergence h3 svg { color: var(--amber); filter: drop-shadow(0 0 8px var(--amber-glow)); }
    .system-card.convergence .spec span { color: var(--amber); }

    /* ================================================================
       Metric cards
       ================================================================ */
    .metric-card {
        background: var(--glass);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-radius: var(--r-md);
        padding: var(--sp-5) var(--sp-5);
        position: relative;
        transition: all 250ms cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
        cursor: default;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 2px;
        background: var(--ink-tertiary);
        transition: background 250ms ease, box-shadow 250ms ease;
    }
    .metric-card:hover {
        background: var(--glass-hover);
        border-color: rgba(255,255,255,0.08);
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    .metric-card h4 {
        font-family: var(--data);
        color: var(--ink-tertiary);
        font-size: 0.62rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 0 0 var(--sp-3) 0;
    }
    .metric-card h2 {
        font-family: var(--display);
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin: 0 0 var(--sp-2) 0;
        color: var(--ink-primary);
        letter-spacing: -0.02em;
    }
    .metric-card .sub-metric {
        font-family: var(--data);
        font-size: 0.68rem;
        color: var(--ink-secondary);
        margin: 0;
        line-height: 1.4;
    }

    /* Color variants */
    .metric-card.success { border-left: none; }
    .metric-card.success::before { background: var(--emerald); box-shadow: 0 0 12px var(--emerald-glow); }
    .metric-card.success h2 { color: var(--emerald); }

    .metric-card.danger { border-left: none; }
    .metric-card.danger::before { background: var(--rose); box-shadow: 0 0 12px var(--rose-glow); }
    .metric-card.danger h2 { color: var(--rose); }

    .metric-card.warning { border-left: none; }
    .metric-card.warning::before { background: var(--amber); box-shadow: 0 0 12px var(--amber-glow); }
    .metric-card.warning h2 { color: var(--amber); }

    .metric-card.info { border-left: none; }
    .metric-card.info::before { background: var(--cyan); box-shadow: 0 0 12px var(--cyan-glow); }
    .metric-card.info h2 { color: var(--cyan); }

    .metric-card.violet { border-left: none; }
    .metric-card.violet::before { background: var(--violet); box-shadow: 0 0 12px var(--violet-glow); }
    .metric-card.violet h2 { color: var(--violet); }

    .metric-card.neutral { border-left: none; }
    .metric-card.neutral::before { background: var(--ink-tertiary); }

    /* ================================================================
       Signal cards
       ================================================================ */
    .signal-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--r-lg);
        padding: var(--sp-8) var(--sp-10);
        position: relative;
        overflow: hidden;
        transition: border-color 300ms ease, box-shadow 300ms ease;
    }
    .signal-card:hover {
        border-color: rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .signal-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 3px;
        background: var(--ink-tertiary);
        transition: background 300ms ease, box-shadow 300ms ease;
    }
    .signal-card.undervalued::before { background: var(--emerald); box-shadow: 0 0 20px var(--emerald-glow); }
    .signal-card.overvalued::before  { background: var(--rose); box-shadow: 0 0 20px var(--rose-glow); }
    .signal-card.fair::before        { background: var(--amber); box-shadow: 0 0 16px var(--amber-glow); }

    /* Subtle top-edge glow based on signal */
    .signal-card.undervalued { border-top: 1px solid rgba(52, 211, 153, 0.15); }
    .signal-card.overvalued  { border-top: 1px solid rgba(251, 113, 133, 0.15); }
    .signal-card.fair        { border-top: 1px solid rgba(212, 168, 83, 0.15); }

    .signal-card .label {
        font-family: var(--data);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--ink-tertiary);
        font-weight: 600;
        margin-bottom: var(--sp-6);
    }
    .signal-card .value {
        font-family: var(--display);
        font-size: 3.25rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: var(--sp-3);
        display: flex;
        align-items: baseline;
        gap: var(--sp-6);
        letter-spacing: -0.03em;
    }
    .signal-card .subtext {
        font-family: var(--data);
        font-size: 0.8rem;
        color: var(--ink-secondary);
        line-height: 1.6;
    }
    .signal-card.undervalued .value { color: var(--emerald); }
    .signal-card.overvalued  .value { color: var(--rose); }
    .signal-card.fair        .value { color: var(--amber); }

    .signal-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
        position: relative;
    }
    .signal-card.undervalued .signal-dot { background: var(--emerald); box-shadow: 0 0 12px var(--emerald), 0 0 4px var(--emerald); }
    .signal-card.overvalued  .signal-dot { background: var(--rose); box-shadow: 0 0 12px var(--rose), 0 0 4px var(--rose); }
    .signal-card.fair        .signal-dot { background: var(--amber); box-shadow: 0 0 12px var(--amber), 0 0 4px var(--amber); }

    /* ================================================================
       Status badges
       ================================================================ */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: var(--r-sm);
        font-family: var(--data);
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .status-badge.buy {
        color: var(--emerald);
        background: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .status-badge.sell {
        color: var(--rose);
        background: rgba(251, 113, 133, 0.08);
        border: 1px solid rgba(251, 113, 133, 0.25);
    }
    .status-badge.neutral {
        color: var(--ink-secondary);
        border: 1px solid var(--border);
    }
    .status-badge.divergence {
        color: var(--amber);
        background: rgba(212, 168, 83, 0.08);
        border: 1px solid rgba(212, 168, 83, 0.25);
    }

    /* ================================================================
       Info / warning boxes
       ================================================================ */
    .info-box {
        background: var(--glass);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-left: 2px solid var(--cyan);
        border-radius: var(--r-md);
        padding: var(--sp-5) var(--sp-6);
        margin: var(--sp-4) 0;
    }
    .info-box h4 {
        font-family: var(--display);
        color: var(--cyan);
        margin: 0 0 var(--sp-2) 0;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .info-box p {
        color: var(--ink-secondary);
        margin: 0;
        font-size: 0.8rem;
        line-height: 1.7;
        font-family: var(--data);
    }

    .warning-box {
        background: rgba(212, 168, 83, 0.04);
        border: 1px solid rgba(212, 168, 83, 0.12);
        border-left: 2px solid var(--amber);
        border-radius: var(--r-md);
        padding: var(--sp-4) var(--sp-5);
        margin-bottom: var(--sp-5);
        display: flex;
        align-items: flex-start;
        gap: var(--sp-3);
    }
    .warning-box .icon {
        width: 8px; height: 8px;
        background: var(--amber);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--amber-glow);
        margin-top: 5px;
        flex-shrink: 0;
    }
    .warning-box .title {
        color: var(--amber);
        font-weight: 700;
        text-transform: uppercase;
        font-family: var(--display);
        font-size: 0.72rem;
        letter-spacing: 0.08em;
    }
    .warning-box .content {
        color: var(--ink-secondary);
        font-size: 0.8rem;
        line-height: 1.6;
        font-family: var(--data);
        margin-top: var(--sp-1);
    }

    /* ================================================================
       Buttons
       ================================================================ */
    .stButton > button {
        border: 1px solid var(--border) !important;
        background: transparent !important;
        color: var(--ink-primary) !important;
        font-family: var(--display) !important;
        font-weight: 500;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        border-radius: 8px;
        padding: 0.6rem 1.75rem;
        transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
    }
    .stButton > button:hover {
        border-color: var(--amber) !important;
        color: var(--amber) !important;
        background: rgba(212, 168, 83, 0.05) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--amber) !important;
        border-color: var(--amber) !important;
        color: var(--ink-inverse) !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--amber-bright) !important;
        border-color: var(--amber-bright) !important;
        box-shadow: 0 0 20px var(--amber-glow);
    }
    .stButton > button:disabled {
        opacity: 0.35 !important;
        cursor: not-allowed !important;
    }

    /* ================================================================
       Tabs
       ================================================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 36px;
        background: transparent;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--ink-tertiary);
        background: transparent;
        font-family: var(--display);
        font-weight: 600;
        font-size: 0.76rem;
        padding: 14px 0;
        border-bottom: 2px solid transparent;
        transition: all 200ms;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--ink-secondary);
    }
    .stTabs [aria-selected="true"] {
        color: var(--amber) !important;
        border-bottom: 2px solid var(--amber) !important;
        background: transparent !important;
    }

    /* ================================================================
       Plotly charts — Frameless glass containers
       ================================================================ */
    .stPlotlyChart {
        border-radius: var(--r-md) !important;
        border: 1px solid var(--border);
        overflow: hidden;
        background: var(--glass);
        margin: var(--sp-2) 0 var(--sp-4) 0;
    }

    /* ================================================================
       Section divider
       ================================================================ */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
        margin: var(--sp-6) 0;
        border: none;
    }

    /* ================================================================
       Inputs & Sidebar
       ================================================================ */
    [data-testid="stSidebar"] {
        background: rgba(5, 8, 16, 0.92) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid var(--border);
    }
    .sidebar-title {
        font-family: var(--display);
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--ink-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: var(--sp-6) 0 var(--sp-3) 0;
    }
    .stTextInput input, .stNumberInput input, .stDateInput input,
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: rgba(10, 14, 23, 0.6) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--ink-primary) !important;
        font-family: var(--data) !important;
        font-size: 0.8rem !important;
        transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTextInput input:focus, .stSelectbox > div > div:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 1px var(--amber) !important;
    }

    /* ================================================================
       Dataframes / Tables
       ================================================================ */
    [data-testid="stDataFrame"] {
        border-radius: var(--r-md);
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--glass);
    }
    [data-testid="stDataFrame"] table {
        font-family: var(--data) !important;
        font-size: 0.75rem !important;
    }
    [data-testid="stDataFrame"] th {
        background: rgba(10, 14, 23, 0.9) !important;
        color: var(--ink-tertiary) !important;
        font-family: var(--data) !important;
        font-size: 0.62rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        padding: var(--sp-3) var(--sp-3) !important;
    }
    [data-testid="stDataFrame"] td {
        color: var(--ink-primary) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        padding: var(--sp-2) var(--sp-3) !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: rgba(212, 168, 83, 0.03) !important;
    }

    /* ================================================================
       Progress / loading
       ================================================================ */
    .progress-card {
        background: var(--glass);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-radius: var(--r-md);
        padding: var(--sp-5) var(--sp-6);
        margin: var(--sp-4) 0;
        transition: border-color 250ms ease;
    }
    .progress-card:hover { border-color: rgba(255,255,255,0.08); }
    .progress-label {
        font-family: var(--display);
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--ink-primary);
        display: flex;
        align-items: center;
        gap: 0.7rem;
        letter-spacing: 0.02em;
    }
    .progress-sub {
        font-family: var(--data);
        font-size: 0.68rem;
        color: var(--ink-tertiary);
        margin-top: 0.35rem;
    }
    .progress-track {
        margin-top: var(--sp-4);
        height: 3px;
        background: rgba(255,255,255,0.03);
        border-radius: 2px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 300ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    .progress-pct {
        text-align: right;
        font-size: 0.65rem;
        color: var(--ink-tertiary);
        margin-top: 0.35rem;
        font-family: var(--data);
    }
    .pulse-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: var(--amber);
        border-radius: 50%;
        animation: pulse 1.5s ease-in-out infinite;
    }
    .pulse-dot.complete { background: var(--emerald); box-shadow: 0 0 8px var(--emerald-glow); }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.3; transform: scale(0.85); }
    }

    /* ================================================================
       Landing page prompt
       ================================================================ */
    .landing-prompt {
        background: var(--glass);
        border: 1px solid var(--border);
        border-left: 2px solid var(--amber);
        border-radius: var(--r-md);
        padding: var(--sp-6) var(--sp-6);
        margin: var(--sp-4) 0;
    }
    .landing-prompt h4 {
        font-family: var(--display);
        color: var(--amber);
        margin: 0 0 var(--sp-3) 0;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .landing-prompt p {
        color: var(--ink-secondary);
        margin: 0;
        font-size: 0.8rem;
        line-height: 1.7;
        font-family: var(--data);
    }

    /* ================================================================
       Timeframe buttons
       ================================================================ */
    .timeframe-btn {
        font-family: var(--data) !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em;
        padding: 0.45rem 0 !important;
        border-radius: var(--r-sm) !important;
        border: 1px solid var(--border) !important;
        background: transparent !important;
        color: var(--ink-tertiary) !important;
        transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: center;
    }
    .timeframe-btn:hover {
        color: var(--ink-primary) !important;
        border-color: var(--ink-secondary) !important;
        background: rgba(255,255,255,0.02) !important;
    }

    /* ================================================================
       Expanders
       ================================================================ */
    .streamlit-expanderHeader {
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ================================================================
       Lookback state rows
       ================================================================ */
    .lookback-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--sp-3) var(--sp-3);
        border-bottom: 1px solid var(--border-subtle);
        transition: background 150ms ease;
    }
    .lookback-row:hover {
        background: rgba(255,255,255,0.015);
    }
    .lookback-row:last-child {
        border-bottom: none;
    }
    .lookback-row .label {
        font-family: var(--data);
        color: var(--ink-secondary);
        font-size: 0.76rem;
    }
    .lookback-row .value {
        font-weight: 600;
        font-family: var(--data);
        font-size: 0.76rem;
    }

    /* ================================================================
       Footer
       ================================================================ */
    .app-footer {
        margin-top: var(--sp-12);
        padding: var(--sp-6) 0 var(--sp-4) 0;
        border-top: 1px solid var(--border);
        text-align: center;
    }
    .app-footer .content {
        font-family: var(--data);
        font-size: 0.65rem;
        color: var(--ink-tertiary);
        letter-spacing: 0.04em;
    }
    .app-footer .content strong {
        color: var(--ink-secondary);
    }

    /* ================================================================
       Download button
       ================================================================ */
    .stDownloadButton > button {
        border: 1px solid var(--border) !important;
        background: var(--glass) !important;
        color: var(--ink-secondary) !important;
        font-family: var(--data) !important;
        font-size: 0.75rem !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 200ms ease !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--amber) !important;
        color: var(--amber) !important;
        background: rgba(212, 168, 83, 0.05) !important;
    }

    /* ================================================================
       Plotly unified hover tooltip — hide thick white vertical line
       ================================================================ */
    .js-plotly-plot .plotly .hoverlayer .spikeline {
        stroke: rgba(148, 163, 184, 0.25) !important;
        stroke-width: 1 !important;
        stroke-dasharray: 4, 3 !important;
    }

    /* ================================================================
       z-index layering
       ================================================================ */
    .stMarkdown, .stPlotlyChart, .stDataFrame, .stButton, .stSelectbox,
    .stMultiSelect, .stTextInput, .stNumberInput, .stRadio, .stCheckbox,
    .stDownloadButton {
        position: relative;
        z-index: 1;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """Render a themed progress card into an ``st.empty()`` slot."""
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


def apply_chart_theme(fig) -> None:
    """Apply the Obsidian Quant Terminal theme to a Plotly figure (mutates in place)."""
    fig.update_layout(**chart_layout())
    style_axes(fig)
