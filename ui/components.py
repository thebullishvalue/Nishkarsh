"""
Reusable UI components — metric cards, signal badges, headers.
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
    """Render a styled metric card.

    Parameters
    ----------
    label : str
        Card title (uppercase).
    value : str
        Primary metric value.
    subtext : str
        Optional subtitle / context.
    color_class : str
        One of ``primary``, ``success``, ``danger``, ``info``,
        ``warning``, ``purple``, ``neutral``.
    """
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f"<h4>{html_mod.escape(label)}</h4>"
        f"<h2>{html_mod.escape(value)}</h2>"
        f'{"" if not subtext else f"<div class=\"sub-metric\">{html_mod.escape(subtext)}</div>"}'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_signal_card(
    signal: str,
    strength: str,
    confidence: str,
    detail: str = "",
) -> None:
    """Render the primary signal card.

    Parameters
    ----------
    signal : str
        Signal direction (``BUY``, ``SELL``, ``HOLD``).
    strength : str
        Signal strength (``STRONG``, ``MODERATE``, ``WEAK``, ``NEUTRAL``).
    confidence : str
        Signal confidence (``HIGH``, ``MEDIUM``, ``LOW``).
    detail : str
        Optional additional context.
    """
    signal_class = (
        "undervalued" if signal == "BUY"
        else "overvalued" if signal == "SELL"
        else "fair"
    )
    signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"

    st.markdown(
        f'<div class="signal-card {html_mod.escape(signal_class)}" style="padding: 1.5rem;">'
        f'<div class="label">WALK-FORWARD SIGNAL</div>'
        f'<div class="value">{signal_emoji} {html_mod.escape(signal)}</div>'
        f'<div class="subtext">'
        f"<strong>{html_mod.escape(strength)}</strong> Strength • "
        f"<strong>{html_mod.escape(confidence)}</strong> Confidence"
        f'{"" if not detail else f"<br>{html_mod.escape(detail)}"}'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_header(title: str, tagline: str) -> None:
    """Render the premium header.

    Parameters
    ----------
    title : str
        Primary heading text.
    tagline : str
        Subtitle / descriptive tagline.
    """
    st.markdown(
        f'<div class="premium-header">'
        f"<h1>{html_mod.escape(title)}</h1>"
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str, color: str = "gold") -> None:
    """Render an info box.

    Parameters
    ----------
    title : str
        Box title.
    content : str
        Box body text.
    color : str
        Unused placeholder for future color variants.
    """
    st.markdown(
        f'<div class="info-box">'
        f"<h4>{html_mod.escape(title)}</h4>"
        f"<p>{html_mod.escape(content)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
