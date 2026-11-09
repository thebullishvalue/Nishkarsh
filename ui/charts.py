"""
Nishkarsh — shared Plotly figures. Display only: every value drawn is a session-state column,
unchanged (clipping to the plot range and choosing colours by sign are presentation).
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.config import CONV2_TIER_MODERATE, CONV2_TIER_STRONG, CONV2_TIER_WEAK
from ui.components import signed
from ui.theme import chart_layout

INK = "rgba(226,232,240,0.78)"
TICK = dict(family="Inter, sans-serif", size=11, color="#7C879A")
GRID = "rgba(255,255,255,0.05)"
ZERO = "rgba(148,163,184,0.28)"
BULL_FILL, BEAR_FILL = "rgba(52,211,153,0.28)", "rgba(251,113,133,0.28)"
SIDE_RGBA = {"bull": "rgba(52,211,153,0.55)", "bear": "rgba(251,113,133,0.55)", "flat": "rgba(148,163,184,0.45)"}
SINGLE_FILL = "rgba(148,163,184,0.14)"
AMBER = "#D4A853"
CONFIG = {"displayModeBar": False}


def since(obj, start):
    """De-duplicated, sorted rows on/after ``start`` (None = all)."""
    if obj is None or len(obj) == 0:
        return obj.iloc[0:0] if obj is not None else pd.DataFrame()
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    return obj if start is None else obj[pd.DatetimeIndex(obj.index) >= pd.Timestamp(start)]


def single_system_runs(fr: pd.DataFrame) -> list[tuple]:
    """Contiguous stretches where the read exists but only the top-down pillar counted (two_system False)."""
    if fr is None or not len(fr) or "NKO" not in fr.columns or "two_system" not in fr.columns:
        return []
    mask = pd.to_numeric(fr["NKO"], errors="coerce").notna() & ~fr["two_system"].astype(bool)
    out, s0, prev = [], None, None
    for d, m in mask.items():
        if m and s0 is None:
            s0 = d
        elif not m and s0 is not None:
            out.append((s0, prev))
            s0 = None
        prev = d
    if s0 is not None:
        out.append((s0, prev))
    return out


def _bar_widths(xs: pd.DatetimeIndex) -> list[float]:
    gaps = pd.Series(xs[1:] - xs[:-1]).dt.total_seconds().mul(1000).tolist() if len(xs) > 1 else []
    return gaps + [float(pd.Series(gaps).median()) if gaps else 86_400_000.0]


def x_axes(fig, **kw) -> None:
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, tickfont=TICK, ticks="", showspikes=True,
                     spikemode="across", spikethickness=0.5, spikedash="dot", spikecolor="rgba(148,163,184,0.35)",
                     **kw)


def nko_figure(fr: pd.DataFrame, price: pd.Series | None, index_name: str, height: int = 320,
               marks: list[tuple] | None = None, today: tuple | None = None, line: bool = False) -> go.Figure:
    """The Nishkarsh read over time as side-coloured contiguous bars (muted inside the no-lean band),
    the index on the right axis, single-system stretches shaded. ``marks`` = [(date, value, label)]
    precedent matches; ``today`` = (date, value)."""
    nko = pd.to_numeric(fr["NKO"], errors="coerce")
    xs = pd.DatetimeIndex(fr.index)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for x0, x1 in single_system_runs(fr):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=SINGLE_FILL, line_width=0, layer="below")
    fig.add_hrect(y0=-CONV2_TIER_WEAK, y1=CONV2_TIER_WEAK, fillcolor="rgba(148,163,184,0.06)", line_width=0,
                  layer="below")
    if line:   # long histories: a quiet line, so the marked matches stand out
        fig.add_trace(go.Scatter(x=xs, y=nko, mode="lines", name="Read", showlegend=False, connectgaps=False,
                                 line=dict(color="rgba(148,163,184,0.55)", width=0.9),
                                 hovertemplate="Read %{y:+.0f}<extra></extra>"), secondary_y=False)
    else:      # contiguous side-coloured bars: a stepped area that breaks on days without a reading
        vals = [None if pd.isna(v) else float(v) for v in nko]
        colors = [SIDE_RGBA["bull" if (v or 0) > CONV2_TIER_WEAK else "bear" if (v or 0) < -CONV2_TIER_WEAK
                            else "flat"] for v in vals]
        fig.add_trace(go.Bar(x=list(xs), y=vals, width=_bar_widths(xs), offset=0,
                             marker=dict(color=colors, line=dict(width=0)), name="Nishkarsh read", showlegend=False,
                             hovertemplate="Read %{y:+.0f}<extra></extra>"), secondary_y=False)
    if price is not None and len(price):
        fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name=index_name, showlegend=False,
                                 line=dict(color="rgba(226,232,240,0.4)" if line else INK, width=1.2 if line else 1.4),
                                 hovertemplate=f"{index_name} " + "%{y:,.0f}<extra></extra>"), secondary_y=True)
    if marks:
        fig.add_trace(go.Scatter(x=[m[0] for m in marks], y=[m[1] for m in marks], mode="markers+text",
                                 text=[m[2] for m in marks], textposition="top center",
                                 textfont=dict(size=10, color=AMBER),
                                 marker=dict(size=10, color=AMBER, line=dict(color="#0B1020", width=2)),
                                 name="Past moments", showlegend=False,
                                 hovertemplate="Match %{text} · read then %{y:+.0f}<extra></extra>"), secondary_y=False)
    if today:
        fig.add_trace(go.Scatter(x=[today[0]], y=[today[1]], mode="markers", name="Today", showlegend=False,
                                 marker=dict(size=13, color="#E2E8F0", symbol="diamond",
                                             line=dict(color="#0B1020", width=2)),
                                 hovertemplate="Today %{y:+.0f}<extra></extra>"), secondary_y=False)
    fig.update_layout(**chart_layout(height=height, show_legend=False, margin=dict(t=8, l=8, r=8, b=8)), bargap=0)
    x_axes(fig)
    ticks = [-100, -CONV2_TIER_STRONG, -CONV2_TIER_MODERATE, 0, CONV2_TIER_MODERATE, CONV2_TIER_STRONG, 100]
    fig.update_yaxes(range=[-105, 105], tickvals=ticks, ticktext=[signed(t, "+.0f") for t in ticks],
                     showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=ZERO, zerolinewidth=1,
                     tickfont=TICK, secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickformat=",.0f", tickfont=TICK, secondary_y=True)
    return fig


def _diverging(fig, x, y: pd.Series, row=None, col=None, name: str = "", line: str = INK, hover: str = "+.2f") -> None:
    """A z series as a thin line over green (above 0) / red (below 0) fills."""
    kw = dict(row=row, col=col) if row is not None else {}
    y = pd.to_numeric(y, errors="coerce")
    for part, color in ((y.clip(lower=0), BULL_FILL), (y.clip(upper=0), BEAR_FILL)):
        fig.add_trace(go.Scatter(x=x, y=part, mode="lines", line=dict(width=0), fill="tozeroy", fillcolor=color,
                                 hoverinfo="skip", showlegend=False, connectgaps=False), **kw)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=line, width=1.2), name=name, showlegend=False,
                             connectgaps=False, hovertemplate=(name + " " if name else "") + "%{y:" + hover + "}<extra></extra>"),
                  **kw)


def _z_axis(y: pd.Series) -> dict:
    """Symmetric range around 0 with integer ticks written with a typographic minus."""
    v = pd.to_numeric(y, errors="coerce").abs().max()
    m = 3.0 if v is None or not math.isfinite(v) else max(1.5, min(4.2, float(v) * 1.1))
    step = 1 if m < 2 else 2
    ticks = list(range(-int(m) // step * step, int(m) + 1, step))
    return dict(range=[-m, m], tickvals=ticks, ticktext=[signed(t, "+.0f") for t in ticks])


def z_figure(y: pd.Series, name: str = "", height: int = 170) -> go.Figure:
    """One diverging z line (a pillar over time)."""
    fig = go.Figure()
    _diverging(fig, y.index, y, name=name)
    fig.update_layout(**chart_layout(height=height, show_legend=False, margin=dict(t=6, l=8, r=8, b=6)))
    x_axes(fig)
    fig.update_yaxes(**_z_axis(y), showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=ZERO, tickfont=TICK)
    return fig


def small_multiples(tiles: list[dict], height_per_row: int = 165) -> go.Figure:
    """A 2-column grid of compact charts. Each tile: {"title", "kind", "series"}. ``kind`` "z": one
    series drawn diverging around 0; "mirror": two shares in % — the first drawn up (green), the
    second down (red) — so the bullish side is above the line as everywhere else."""
    n = max(1, len(tiles))
    rows = math.ceil(n / 2)
    fig = make_subplots(rows=rows, cols=2, shared_xaxes=False, vertical_spacing=min(0.62 / rows, 0.2),
                        horizontal_spacing=0.07, subplot_titles=[t["title"] for t in tiles])
    for i, t in enumerate(tiles):
        r, c = i // 2 + 1, i % 2 + 1
        if t.get("kind") == "mirror":
            (up, up_name), (down, down_name) = t["series"]
            up, down = pd.to_numeric(up, errors="coerce"), pd.to_numeric(down, errors="coerce")
            for y, name, fill, line, sign in ((up, up_name, BULL_FILL, "#34D399", 1), (down, down_name, BEAR_FILL,
                                                                                         "#FB7185", -1)):
                fig.add_trace(go.Scatter(x=y.index, y=sign * y, mode="lines", line=dict(color=line, width=1),
                                         fill="tozeroy", fillcolor=fill, name=name, showlegend=False,
                                         connectgaps=False, customdata=y.values,
                                         hovertemplate=f"{name} " + "%{customdata:.0f}%<extra></extra>"),
                              row=r, col=c)
            fig.update_yaxes(range=[-105, 105], tickvals=[-100, 0, 100], ticktext=["100%", "0", "100%"],
                             zeroline=True, zerolinecolor=ZERO,
                             row=r, col=c)
        else:
            y, name = t["series"][0]
            _diverging(fig, y.index, y, row=r, col=c, name=name)
            fig.update_yaxes(**_z_axis(y), zeroline=True, zerolinecolor=ZERO, row=r, col=c)
    fig.update_layout(**chart_layout(height=max(190, rows * height_per_row + 30), show_legend=False,
                                     margin=dict(t=26, l=8, r=8, b=6)))
    x_axes(fig, nticks=5)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, tickfont=TICK)
    fig.update_annotations(font=dict(family="Inter, sans-serif", size=12, color="#AEB6C4"), xanchor="left")
    for i, a in enumerate(fig.layout.annotations):
        a.x = 0.0 if i % 2 == 0 else 0.535
    return fig


def lines_figure(series: list[tuple], height: int = 200, y_range=None, pct: bool = False,
                 legend: bool = True) -> go.Figure:
    """Plain multi-line chart: [(y, name, colour, width)]."""
    fig = go.Figure()
    for y, name, color, width in series:
        fig.add_trace(go.Scatter(x=y.index, y=pd.to_numeric(y, errors="coerce"), mode="lines", name=name,
                                 line=dict(color=color, width=width), connectgaps=False,
                                 hovertemplate=f"{name} " + ("%{y:.0%}" if pct else "%{y:+.2f}") + "<extra></extra>"))
    fig.update_layout(**chart_layout(height=height, show_legend=legend, margin=dict(t=28 if legend else 6, l=8, r=8, b=6)))
    x_axes(fig)
    fig.update_yaxes(range=y_range, showgrid=True, gridcolor=GRID, tickfont=TICK,
                     tickformat=".0%" if pct else "+.1f", nticks=4)
    return fig


def yearly_bars(yearly: dict, height: int = 200) -> go.Figure:
    ys = list(yearly)
    vs = [yearly[k] for k in ys]
    fig = go.Figure(go.Bar(x=ys, y=vs, marker_color=["rgba(52,211,153,0.7)" if v > 0 else "rgba(251,113,133,0.7)"
                                                       for v in vs],
                           hovertemplate="%{x}: accuracy %{y:+.3f}<extra></extra>"))
    fig.update_layout(**chart_layout(height=height, show_legend=False, margin=dict(t=6, l=8, r=8, b=6)))
    fig.update_xaxes(showgrid=False, tickfont=TICK, type="category")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=ZERO, tickfont=TICK,
                     tickformat="+.2f", nticks=5)
    return fig
