"""
Reusable UI components — metric cards, signal badges, headers.
Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import html as html_mod

import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
) -> None:
    """Render a terminal-styled metric card."""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f"<h4>{html_mod.escape(label)}</h4>"
        f"<h2>{html_mod.escape(value)}</h2>"
        f'{f"<div class=\"sub-metric\">{html_mod.escape(subtext)}</div>" if subtext else ""}'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_signal_card(
    signal: str,
    strength: str,
    confidence: str,
    detail: str = "",
) -> None:
    """Render the primary walk-forward signal card."""
    signal_class = (
        "undervalued" if signal == "BUY"
        else "overvalued" if signal == "SELL"
        else "fair"
    )

    st.markdown(
        f'<div class="signal-card {html_mod.escape(signal_class)}">'
        f'<div class="label">WALK-FORWARD SIGNAL</div>'
        f'<div class="value"><div class="signal-dot"></div> {html_mod.escape(signal)}</div>'
        f'<div class="subtext">'
        f"<strong>{html_mod.escape(strength)}</strong> Strength &bull; "
        f"<strong>{html_mod.escape(confidence)}</strong> Confidence"
        f'{f"<br>{html_mod.escape(detail)}" if detail else ""}'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_header(title: str, tagline: str) -> None:
    """Render the terminal masthead."""
    st.markdown(
        f'<div class="premium-header">'
        f"<h1>{html_mod.escape(title)}</h1>"
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str, color: str = "cyan") -> None:
    """Render an info box."""
    st.markdown(
        f'<div class="info-box">'
        f"<h4>{html_mod.escape(title)}</h4>"
        f"<p>{html_mod.escape(content)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_nishkarsh_signal_card(
    signal: str,
    conviction: float,
    agreement: float,
    explanation: str,
) -> None:
    """Render the primary Nishkarsh convergence signal card."""
    if "BUY" in signal:
        signal_class = "undervalued"
    elif "SELL" in signal:
        signal_class = "overvalued"
    else:
        signal_class = "fair"
    agreement_text = "Strong" if agreement > 0.7 else "Moderate" if agreement > 0.5 else "Weak"

    st.markdown(
        f"""
        <div class="signal-card {html_mod.escape(signal_class)}">
            <div class="label">NISHKARSH CONVERGENCE SIGNAL &#40;&#x0928;&#x093F;&#x0937;&#x094D;&#x0915;&#x0930;&#x094D;&#x0937;&#41;</div>
            <div class="value"><div class="signal-dot"></div> {html_mod.escape(signal)}</div>
            <div class="subtext">
                Score: <strong style="color:var(--ink-primary)">{conviction:+.0f}</strong> &bull;
                Agreement: <strong style="color:var(--ink-primary)">{agreement:.0%}</strong> ({html_mod.escape(agreement_text)})
            </div>
            <div style="margin-top:1.5rem;padding-top:1.5rem;border-top:1px solid var(--border);font-size:0.82rem;line-height:1.7;color:var(--ink-secondary);font-family:var(--data);">
                <strong style="color:var(--amber);font-family:var(--display);font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;">INTERPRETATION</strong><br>
                {html_mod.escape(explanation)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_box(title: str, content: str) -> None:
    """Render a themed alert/warning box."""
    st.markdown(
        f"""
        <div class="warning-box">
            <div class="icon"></div>
            <div>
                <div class="title">{html_mod.escape(title)}</div>
                <div class="content">{html_mod.escape(content)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
