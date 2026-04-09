"""
Shared CSS, chart theming, and color constants for the UI layer.

Aesthetic: "Obsidian Quant" — Institutional Research Terminal
─────────────────────────────────────────────────────────────
Precision-instrument design language for quantitative finance.
- Display/UI:  Syne (geometric, authoritative, distinctive)
- Body/Data:   JetBrains Mono (refined monospace, tabular precision)
- Palette:     Obsidian (#0A0E17 → #050810), Amber Gold (#D4A853),
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


def inject_css() -> None:
    """Inject the Obsidian Quant Terminal CSS into the Streamlit app."""
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

    /* ════════════════════════════════════════════════════════════════════
       DESIGN TOKENS — Obsidian Quant
       ════════════════════════════════════════════════════════════════════ */
    :root {
        /* Surfaces */
        --bg-deep:      #050810;
        --bg-base:      #0A0E17;
        --bg-elevated:  #111827;
        --bg-surface:   #151C2C;
        --glass:        rgba(17, 24, 39, 0.4);
        --glass-hover:  rgba(21, 28, 44, 0.6);
        --border:       rgba(255, 255, 255, 0.05);
        --border-active: rgba(212, 168, 83, 0.35);

        /* Ink */
        --ink-primary:  #F1F5F9;
        --ink-secondary:#94A3B8;
        --ink-tertiary: #4B5563;
        --ink-inverse:#0A0E17;

        /* Accent palette */
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

        /* Type */
        --display:      'Syne', -apple-system, sans-serif;
        --data:         'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ════════════════════════════════════════════════════════════════════
       Globals & App Overrides
       ════════════════════════════════════════════════════════════════════ */
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
    /* Subtle gradient mesh background */
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
        padding-top: 2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 98%;
    }

    /* ════════════════════════════════════════════════════════════════════
       Masthead
       ════════════════════════════════════════════════════════════════════ */
    .premium-header {
        padding: 1.25rem 0 1.5rem 0;
        margin-bottom: 2.5rem;
        position: relative;
    }
    .premium-header::after {
        content: "";
        display: block;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, var(--amber) 0%, transparent 60%);
        margin-top: 1.25rem;
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

    /* ════════════════════════════════════════════════════════════════════
       System cards (landing page)
       ════════════════════════════════════════════════════════════════════ */
    .system-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.75rem 1.75rem 1.5rem;
        position: relative;
        transition: all 250ms cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        cursor: default;
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
        transition: background 250ms ease, box-shadow 250ms ease;
    }
    .system-card:hover {
        background: var(--glass-hover);
        border-color: var(--border-active);
        transform: translateY(-2px);
    }
    .system-card h3 {
        font-family: var(--display);
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        text-transform: uppercase;
    }
    .system-card h3 svg {
        flex-shrink: 0;
    }
    .system-card p {
        font-family: var(--data);
        font-size: 0.8rem;
        color: var(--ink-secondary);
        line-height: 1.65;
        margin: 0 0 1.25rem 0;
    }
    .system-card .spec {
        font-family: var(--data);
        font-size: 0.68rem;
        color: var(--ink-tertiary);
        line-height: 1.8;
        border-top: 1px solid var(--border);
        padding-top: 1rem;
        margin-top: auto;
    }
    .system-card .spec span {
        font-weight: 600;
    }

    /* System card variants */
    .system-card.aarambh { border-left: 2px solid var(--violet); }
    .system-card.aarambh h3 { color: var(--violet); }
    .system-card.aarambh h3 svg { color: var(--violet); filter: drop-shadow(0 0 8px var(--violet-glow)); }
    .system-card.aarambh::after { background: var(--violet); box-shadow: 0 0 12px var(--violet-glow); }
    .system-card.aarambh .spec span { color: var(--violet); }

    .system-card.nirnay { border-left: 2px solid var(--cyan); }
    .system-card.nirnay h3 { color: var(--cyan); }
    .system-card.nirnay h3 svg { color: var(--cyan); filter: drop-shadow(0 0 8px var(--cyan-glow)); }
    .system-card.nirnay::after { background: var(--cyan); box-shadow: 0 0 12px var(--cyan-glow); }
    .system-card.nirnay .spec span { color: var(--cyan); }

    .system-card.convergence { border-left: 2px solid var(--amber); }
    .system-card.convergence h3 { color: var(--amber); }
    .system-card.convergence h3 svg { color: var(--amber); filter: drop-shadow(0 0 8px var(--amber-glow)); }
    .system-card.convergence::after { background: var(--amber); box-shadow: 0 0 12px var(--amber-glow); }
    .system-card.convergence .spec span { color: var(--amber); }

    /* ════════════════════════════════════════════════════════════════════
       Metric cards
       ════════════════════════════════════════════════════════════════════ */
    .metric-card {
        background: var(--glass);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        position: relative;
        transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 115px;
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
        transition: background 200ms ease, box-shadow 200ms ease;
    }
    .metric-card:hover {
        background: var(--glass-hover);
        border-color: rgba(255,255,255,0.08);
        transform: translateY(-1px);
    }
    .metric-card h4 {
        font-family: var(--data);
        color: var(--ink-tertiary);
        font-size: 0.62rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 0 0 0.6rem 0;
    }
    .metric-card h2 {
        font-family: var(--display);
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1;
        margin: 0 0 0.4rem 0;
        color: var(--ink-primary);
        letter-spacing: -0.02em;
    }
    .metric-card .sub-metric {
        font-family: var(--data);
        font-size: 0.7rem;
        color: var(--ink-secondary);
        margin: 0;
    }

    /* Color variants */
    .metric-card.success { border-left: 2px solid var(--emerald); }
    .metric-card.success::before { background: var(--emerald); box-shadow: 0 0 12px var(--emerald-glow); }
    .metric-card.success h2 { color: var(--emerald); }

    .metric-card.danger  { border-left: 2px solid var(--rose); }
    .metric-card.danger::before { background: var(--rose); box-shadow: 0 0 12px var(--rose-glow); }
    .metric-card.danger h2  { color: var(--rose); }

    .metric-card.warning   { border-left: 2px solid var(--amber); }
    .metric-card.warning::before { background: var(--amber); box-shadow: 0 0 12px var(--amber-glow); }
    .metric-card.warning h2   { color: var(--amber); }

    .metric-card.info      { border-left: 2px solid var(--cyan); }
    .metric-card.info::before { background: var(--cyan); box-shadow: 0 0 12px var(--cyan-glow); }
    .metric-card.info h2      { color: var(--cyan); }

    .metric-card.violet    { border-left: 2px solid var(--violet); }
    .metric-card.violet::before { background: var(--violet); box-shadow: 0 0 12px var(--violet-glow); }
    .metric-card.violet h2    { color: var(--violet); }

    .metric-card.neutral { border-left: 2px solid var(--ink-tertiary); }

    /* ════════════════════════════════════════════════════════════════════
       Signal cards
       ════════════════════════════════════════════════════════════════════ */
    .signal-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2.25rem 2.5rem;
        position: relative;
        overflow: hidden;
        transition: border-color 250ms ease;
    }
    .signal-card:hover { border-color: rgba(255,255,255,0.1); }

    .signal-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 3px;
        background: var(--ink-tertiary);
        transition: background 250ms ease, box-shadow 250ms ease;
    }
    .signal-card.undervalued::before {
        background: var(--emerald);
        box-shadow: 0 0 16px var(--emerald-glow);
    }
    .signal-card.overvalued::before {
        background: var(--rose);
        box-shadow: 0 0 16px var(--rose-glow);
    }
    .signal-card.fair::before {
        background: var(--amber);
        box-shadow: 0 0 12px var(--amber-glow);
    }

    .signal-card .label {
        font-family: var(--data);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--ink-tertiary);
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .signal-card .value {
        font-family: var(--display);
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: baseline;
        gap: 1.5rem;
        letter-spacing: -0.03em;
    }
    .signal-card .subtext {
        font-family: var(--data);
        font-size: 0.82rem;
        color: var(--ink-secondary);
        line-height: 1.6;
    }
    .signal-card.undervalued .value { color: var(--emerald); }
    .signal-card.overvalued  .value { color: var(--rose); }
    .signal-card.fair        .value { color: var(--amber); }

    .signal-dot {
        width: 14px; height: 14px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .signal-card.undervalued .signal-dot { background: var(--emerald); box-shadow: 0 0 10px var(--emerald); }
    .signal-card.overvalued  .signal-dot { background: var(--rose); box-shadow: 0 0 10px var(--rose); }
    .signal-card.fair        .signal-dot { background: var(--amber); box-shadow: 0 0 10px var(--amber); }

    /* ════════════════════════════════════════════════════════════════════
       Status badges
       ════════════════════════════════════════════════════════════════════ */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
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

    /* ════════════════════════════════════════════════════════════════════
       Info / warning boxes
       ════════════════════════════════════════════════════════════════════ */
    .info-box {
        background: var(--glass);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-left: 2px solid var(--cyan);
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin: 1.5rem 0;
    }
    .info-box h4 {
        font-family: var(--display);
        color: var(--cyan);
        margin: 0 0 0.5rem 0;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .info-box p {
        color: var(--ink-secondary);
        margin: 0;
        font-size: 0.82rem;
        line-height: 1.7;
        font-family: var(--data);
    }

    .warning-box {
        background: rgba(212, 168, 83, 0.04);
        border: 1px solid rgba(212, 168, 83, 0.15);
        border-left: 2px solid var(--amber);
        border-radius: 10px;
        padding: 1.15rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-start;
        gap: 14px;
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
        font-size: 0.75rem;
        letter-spacing: 0.08em;
    }
    .warning-box .content {
        color: var(--ink-secondary);
        font-size: 0.82rem;
        line-height: 1.6;
        font-family: var(--data);
        margin-top: 0.25rem;
    }

    /* ════════════════════════════════════════════════════════════════════
       Buttons
       ════════════════════════════════════════════════════════════════════ */
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
        opacity: 0.4 !important;
        cursor: not-allowed !important;
    }

    /* ════════════════════════════════════════════════════════════════════
       Tabs
       ════════════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        background: transparent;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--ink-tertiary);
        background: transparent;
        font-family: var(--display);
        font-weight: 600;
        font-size: 0.78rem;
        padding: 12px 0;
        border-bottom: 2px solid transparent;
        transition: all 200ms;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--ink-secondary); }
    .stTabs [aria-selected="true"] {
        color: var(--amber) !important;
        border-bottom: 2px solid var(--amber) !important;
        background: transparent !important;
    }

    /* ════════════════════════════════════════════════════════════════════
       Plotly charts — Frameless
       ════════════════════════════════════════════════════════════════════ */
    .stPlotlyChart {
        border-radius: 10px !important;
        border: 1px solid var(--border);
        overflow: hidden;
        background: var(--glass);
    }

    /* ════════════════════════════════════════════════════════════════════
       Section divider
       ════════════════════════════════════════════════════════════════════ */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
        margin: 2rem 0;
        border: none;
    }

    /* ════════════════════════════════════════════════════════════════════
       Inputs & Sidebar
       ════════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: rgba(5, 8, 16, 0.9) !important;
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
        margin: 1.5rem 0 0.85rem 0;
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

    /* ════════════════════════════════════════════════════════════════════
       Dataframes / Tables
       ════════════════════════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--glass);
    }
    [data-testid="stDataFrame"] table {
        font-family: var(--data) !important;
        font-size: 0.75rem !important;
    }
    [data-testid="stDataFrame"] th {
        background: rgba(10, 14, 23, 0.85) !important;
        color: var(--ink-tertiary) !important;
        font-family: var(--data) !important;
        font-size: 0.65rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        border-bottom: 1px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] td {
        color: var(--ink-primary) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    /* ════════════════════════════════════════════════════════════════════
       Progress / loading
       ════════════════════════════════════════════════════════════════════ */
    .progress-card {
        background: var(--glass);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem 1.75rem;
        margin: 1rem 0;
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
        margin-top: 1rem;
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

    /* ════════════════════════════════════════════════════════════════════
       Landing page prompt
       ════════════════════════════════════════════════════════════════════ */
    .landing-prompt {
        background: var(--glass);
        border: 1px solid var(--border);
        border-left: 2px solid var(--amber);
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
    }
    .landing-prompt h4 {
        font-family: var(--display);
        color: var(--amber);
        margin: 0 0 0.6rem 0;
        font-size: 0.75rem;
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
        font-size: 0.82rem;
        line-height: 1.7;
        font-family: var(--data);
    }

    /* ════════════════════════════════════════════════════════════════════
       Timeframe buttons
       ════════════════════════════════════════════════════════════════════ */
    .timeframe-btn {
        font-family: var(--data) !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em;
        padding: 0.45rem 0 !important;
        border-radius: 6px !important;
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

    /* ════════════════════════════════════════════════════════════════════
       Expanders
       ════════════════════════════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ════════════════════════════════════════════════════════════════════
       Streamlit element overrides — ensure proper z-index
       ════════════════════════════════════════════════════════════════════ */
    .stMarkdown, .stPlotlyChart, .stDataFrame, .stButton, .stSelectbox,
    .stMultiSelect, .stTextInput, .stNumberInput, .stRadio, .stCheckbox {
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
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10),
        margin=dict(t=10, l=10, r=10, b=35),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10, 14, 23, 0.9)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
            bordercolor="rgba(255,255,255,0.1)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, family="JetBrains Mono, monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.03)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.03)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.06)",
        linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
    )
