"""
Shared CSS, chart theming, and color constants for the UI layer.
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

VERSION = "1.1.0"
PRODUCT_NAME = "NISHKARSH"
COMPANY = "@thebullishvalue"


def inject_css() -> None:
    """Inject the shared dark theme CSS into the Streamlit app."""
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --primary-color: {COLOR_GOLD};
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: {COLOR_GREEN};
        --danger-red: {COLOR_RED};
        --warning-amber: {COLOR_AMBER};
        --info-cyan: {COLOR_CYAN};
        --purple: {COLOR_PURPLE};
        --neutral: {COLOR_MUTED};
    }}

    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .main {{ background-color: var(--background-color); color: var(--text-primary); }}
    .stApp > header {{ background-color: transparent; }}
    #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}}
    .block-container {{ padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }}

    .premium-header {{
        background: var(--secondary-background-color); padding: 1.25rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color); position: relative; overflow: hidden; margin-top: 1rem;
    }}
    .premium-header::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }}
    .premium-header h1 {{ margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }}
    .premium-header .tagline {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }}

    .metric-card {{
        background-color: var(--bg-card); padding: 1.25rem; border-radius: 12px;
        border: 1px solid var(--border-color); box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden; min-height: 120px;
        display: flex; flex-direction: column; justify-content: center;
    }}
    .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }}
    .metric-card h4 {{ color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
    .metric-card h3 {{ color: var(--text-primary); font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }}
    .metric-card h2 {{ color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }}
    .metric-card p {{ color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin: 0; }}
    .metric-card .sub-metric {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }}
    .metric-card.primary h2 {{ color: var(--primary-color); }}
    .metric-card.success h2 {{ color: var(--success-green); }}
    .metric-card.danger h2 {{ color: var(--danger-red); }}
    .metric-card.info h2 {{ color: var(--info-cyan); }}
    .metric-card.warning h2 {{ color: var(--warning-amber); }}
    .metric-card.purple h2 {{ color: var(--purple); }}
    .metric-card.neutral h2 {{ color: var(--neutral); }}

    .signal-card {{ background: var(--bg-card); border-radius: 16px; border: 2px solid var(--border-color); padding: 1.5rem; position: relative; overflow: hidden; }}
    .signal-card.overvalued {{ border-color: var(--danger-red); box-shadow: 0 0 30px rgba(239, 68, 68, 0.15); }}
    .signal-card.undervalued {{ border-color: var(--success-green); box-shadow: 0 0 30px rgba(16, 185, 129, 0.15); }}
    .signal-card.fair {{ border-color: var(--primary-color); box-shadow: 0 0 30px rgba(255, 195, 0, 0.15); }}
    .signal-card .label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }}
    .signal-card .value {{ font-size: 2.5rem; font-weight: 700; line-height: 1; }}
    .signal-card .subtext {{ font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.75rem; }}
    .signal-card.overvalued .value {{ color: var(--danger-red); }}
    .signal-card.undervalued .value {{ color: var(--success-green); }}
    .signal-card.fair .value {{ color: var(--primary-color); }}

    .status-badge {{ display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}
    .status-badge.buy {{ background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }}
    .status-badge.sell {{ background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }}
    .status-badge.neutral {{ background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }}
    .status-badge.divergence {{ background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); border: 1px solid rgba(var(--primary-rgb), 0.3); }}

    .info-box {{ background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }}
    .info-box h4 {{ color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }}
    .info-box p {{ color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }}

    .stButton>button {{ border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }}
    .stButton>button:hover {{ box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{ color: var(--text-muted); border-bottom: 2px solid transparent; background: transparent; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }}

    .stPlotlyChart {{ border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }}

    .section-divider {{ height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }}

    .sidebar-title {{ font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }}
    [data-testid="stSidebar"] {{ background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }}

    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--background-color); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px; }}

    /* ── Themed loading state (from Arthagati) ─────────────────── */
    @keyframes pulse-glow {{
        0%, 100% {{ opacity: 0.6; }}
        50%       {{ opacity: 1.0; }}
    }}
    .loading-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        position: relative;
        overflow: hidden;
    }}
    .loading-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 0% 50%, rgba(var(--primary-rgb), 0.06) 0%, transparent 60%);
        pointer-events: none;
    }}
    .loading-label {{
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.3px;
        position: relative;
    }}
    .loading-sub {{
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
        font-weight: 400;
        position: relative;
        letter-spacing: 0.2px;
    }}
    .loading-dot {{
        display: inline-block;
        width: 5px; height: 5px;
        border-radius: 50%;
        background: var(--primary-color);
        animation: pulse-glow 1.2s ease-in-out infinite;
        margin-right: 0.6rem;
        vertical-align: middle;
        position: relative;
        top: -1px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """Render a themed progress card into an ``st.empty()`` slot.

    Parameters
    ----------
    slot : streamlit.delta_generator.DeltaGenerator
        The ``st.empty()`` container to render into.
    pct : int
        Progress percentage (0-100).
    label : str
        Primary label text.
    sub : str
        Optional subtitle / detail text.

    Usage
    -----
    >>> prog = st.empty()
    >>> progress_bar(prog, 5, "Fetching data", "Google Sheets")
    >>> ...
    >>> progress_bar(prog, 100, "Complete")
    >>> time.sleep(0.25)
    >>> prog.empty()
    """
    bar_color = COLOR_GREEN if pct == 100 else COLOR_GOLD
    slot.markdown(
        f"""
    <div class="loading-card">
        <div class="loading-label">
            <span class="loading-dot"></span>{label}
        </div>
        {"" if not sub else f'<div class="loading-sub">{sub}</div>'}
        <div style="margin-top: 0.65rem; height: 3px; background: #2A2A2A; border-radius: 2px; overflow: hidden;">
            <div style="width: {pct}%; height: 100%;
                        background: linear-gradient(90deg, {bar_color}, {COLOR_AMBER});
                        border-radius: 2px; transition: width 0.3s ease;">
            </div>
        </div>
        <div style="text-align: right; font-size: 0.65rem; color: #555; margin-top: 0.25rem; font-family: monospace;">
            {pct}%
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig) -> None:
    """Apply the dark theme to a Plotly figure (mutates in place).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to theme.
    """
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(family="Inter", color=CHART_FONT_COLOR),
        margin=dict(t=40, l=20, r=20, b=20),
        hoverlabel=dict(bgcolor=CHART_BG, font_size=12),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)
