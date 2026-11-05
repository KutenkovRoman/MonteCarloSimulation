"""Plotly figures: the map, the warehouse fan, and the stock breakdown."""
from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go

from simviz.config import (
    PERIODS_IN_DAY,
    ScenarioConfig,
    calendar_point,
    calendar_track,
    month_names_ru,
)
from simviz.geo import (
    LON_MAX,
    LON_MIN,
    PORTS,
    coastline_rings,
    point_xy,
    position_on_leg,
    route_polyline,
)
from simviz.runner import Ensemble
from simviz.trace import POINTS, STOCK_BUCKETS, Trace

logger = logging.getLogger(__name__)

__all__ = [
    "COLORS",
    "CURSOR_SHAPE_INDEX",
    "ROUTE_COLORS",
    "date_label",
    "make_fan_figure",
    "make_map_figure",
    "make_stock_figure",
    "vessel_frame",
    "vessel_trace_index",
]

#: Both the fan and the stock chart keep the time cursor as their first shape,
#: so a frame update can patch it by index instead of rebuilding the figure.
CURSOR_SHAPE_INDEX = 0

COLORS = {
    "bg": "#0f1720",
    "panel": "#16212d",
    "land": "#1d2a37",
    "land_line": "#31465c",
    "grid": "#22303f",
    "text": "#dfe7ef",
    "muted": "#7d8fa3",
    "port": "#f2f5f8",
    "danger": "#e05d5d",
    "warn": "#e8a33d",
    "ok": "#6ac47e",
    "accent": "#4c9be8",
}

ROUTE_COLORS = {
    "A": "#4c9be8",   # Dudinka - Murmansk - China
    "B": "#6ac47e",   # Dudinka - Arkhangelsk / St Petersburg - China
    "C": "#e8a33d",   # Northern Sea Route
}

_LEG_STYLE = {
    ("S", "A"): ("A", "solid"),
    ("S", "B"): ("B", "solid"),
    ("S", "D"): ("C", "solid"),
    # No ("D", "A"): route C's backup return shares the Suez lane below, and
    # drawing it again would just lay a dotted line over a dashed one.
    ("A", "D"): ("A", "dash"),
    ("P", "D"): ("B", "dash"),
}


def _seasons() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Interflood pause and NSR window, taken from the model's own calendar.

    Month indices come from the model, where December is 0 -- hardcoding them
    here silently shifts every band by a month.
    """
    return (
        (calendar_point("interflood_start"), calendar_point("interflood_end")),
        (calendar_point("alt_route_start"), calendar_point("alt_route_end")),
    )


#: Where a port's name is drawn relative to its marker. Murmansk, Arkhangelsk
#: and St Petersburg sit within ten degrees of longitude on a map two hundred
#: wide, so they have to be pushed apart by hand.
_PORT_LABEL_SIDE = {
    "S": "top center",
    "A": "top left",
    "B": "middle right",
    "P": "bottom left",
    "D": "bottom center",
}

#: The map is coastline, one trace per sea lane, ports, then the fleet last, so
#: a frame update can patch the fleet by index. The coastline is a single
#: None-separated polygon rather than one trace per island: at eighty-odd
#: WebGL traces every animation frame took most of a second to redraw.
VESSEL_TRACE_INDEX = 1 + len(_LEG_STYLE) + 1


def vessel_trace_index() -> int:
    return VESSEL_TRACE_INDEX


def date_label(trace: Trace, period: int) -> str:
    """Calendar date of a period.

    A run that fails on its very first period records nothing at all, and the
    usual clamp ``np.clip(period, 0, n_periods - 1)`` then degenerates to -1 and
    indexes an empty array. There is still a right answer in that case: the run
    died on period 0, whose date is the start of the calendar.
    """
    names = month_names_ru()
    if trace.n_periods == 0:
        month, day = calendar_track(1)[0]
        return f"{day} {names[month]}, год 1"
    period = int(np.clip(period, 0, trace.n_periods - 1))
    year = int(trace.day_index[period]) // 365 + 1
    return f"{int(trace.day[period])} {names[int(trace.month[period])]}, год {year}"


def _base_map_layout() -> go.Layout:
    return go.Layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        dragmode="pan",
        xaxis=dict(
            range=[LON_MIN, LON_MAX], showgrid=False, zeroline=False,
            visible=False, constrain="domain",
        ),
        yaxis=dict(
            range=[-6, 82], showgrid=False, zeroline=False, visible=False,
            scaleanchor="x", scaleratio=1.45, constrain="domain",
        ),
        hoverlabel=dict(bgcolor=COLORS["panel"], font=dict(color=COLORS["text"])),
    )


def make_map_figure(trace: Trace, period: int) -> go.Figure:
    """Coastline, sea lanes, ports, and the fleet at one period."""
    fig = go.Figure(layout=_base_map_layout())

    land_x: list[float] = []
    land_y: list[float] = []
    for ring in coastline_rings():
        land_x.extend(ring[:, 0].tolist() + [None])
        land_y.extend(ring[:, 1].tolist() + [None])
    fig.add_trace(go.Scatter(
        x=land_x, y=land_y, mode="lines", fill="toself",
        fillcolor=COLORS["land"], line=dict(color=COLORS["land_line"], width=0.7),
        hoverinfo="skip", showlegend=False, name="land",
    ))

    for (origin, target), (route, dash) in _LEG_STYLE.items():
        line = route_polyline(origin, target)
        fig.add_trace(go.Scatter(
            x=line[:, 0], y=line[:, 1], mode="lines",
            line=dict(color=ROUTE_COLORS[route], width=1.4, dash=dash),
            opacity=0.55, hoverinfo="skip", showlegend=False,
            name=f"{origin}-{target}",
        ))

    port_x, port_y, port_text, port_side = [], [], [], []
    for code in POINTS:
        x, y = point_xy(code)
        port_x.append(x)
        port_y.append(y)
        port_text.append(PORTS[code].name)
        port_side.append(_PORT_LABEL_SIDE[code])
    fig.add_trace(go.Scatter(
        x=port_x, y=port_y, mode="markers+text", text=port_text,
        textposition=port_side,
        textfont=dict(color=COLORS["text"], size=11),
        marker=dict(size=9, color=COLORS["port"], line=dict(color=COLORS["bg"], width=1.5)),
        hoverinfo="text", showlegend=False, name="ports",
    ))

    x, y, size, color, hover = vessel_frame(trace, period)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=size, color=color, line=dict(color="#0b1119", width=1.2),
                    opacity=0.95),
        showlegend=False, name="vessels",
    ))
    return fig


def vessel_frame(
    trace: Trace, period: int,
) -> tuple[list[float], list[float], list[float], list[str], list[str]]:
    """Positions, sizes, colours and hover text for every vessel at one period."""
    if trace.n_periods == 0:
        return [], [], [], [], []      # the run died before anything was recorded
    period = int(np.clip(period, 0, trace.n_periods - 1))
    xs, ys, sizes, colors, hovers = [], [], [], [], []
    for i, vessel_id in enumerate(trace.vessel_ids):
        origin = POINTS[trace.v_from[period, i]]
        target = POINTS[trace.v_to[period, i]]
        lon, lat = position_on_leg(origin, target, float(trace.v_progress[period, i]))
        loaded = int(trace.v_loaded[period, i])
        capacity = int(trace.v_capacity[i])
        share = loaded / capacity if capacity else 0.0

        xs.append(lon)
        ys.append(lat)
        sizes.append(7.0 + 11.0 * share)
        colors.append(ROUTE_COLORS[vessel_id[-1]])
        if origin == target:
            where = f"в порту {PORTS[origin].name}"
        else:
            where = (f"{PORTS[origin].name} → {PORTS[target].name}, "
                     f"пройдено {trace.v_progress[period, i] * 100:.0f}%")
        hovers.append(f"<b>{vessel_id}</b><br>{where}<br>груз {loaded} / {capacity}")
    return xs, ys, sizes, colors, hovers


def _calendar_ticks(n_periods: int) -> tuple[list[int], list[str]]:
    """Quarterly axis labels for the whole horizon, taken from the calendar."""
    track = calendar_track(n_periods)
    names = month_names_ru()
    values = list(range(0, n_periods, 90 * PERIODS_IN_DAY))
    return values, [f"{track[t][1]} {names[track[t][0]]}" for t in values]


def _season_bands(n_periods: int) -> list[dict]:
    """Shaded vertical bands for the interflood pause and the NSR window.

    Driven by the calendar rather than by a trace: a run that stopped early has
    a short trace, and taking the bands from it would shrink the seasons along
    with it whenever the user flips to a failed run.
    """
    track = calendar_track(n_periods)
    month = np.array([m for m, _ in track])
    day = np.array([d for _, d in track])

    bands: list[dict] = []
    palette = ((COLORS["danger"], 0.13), (ROUTE_COLORS["C"], 0.06))
    for (start, end), (color, opacity) in zip(_seasons(), palette):
        inside = (
            ((month > start[0]) | ((month == start[0]) & (day >= start[1])))
            & ((month < end[0]) | ((month == end[0]) & (day <= end[1])))
        )
        edges = np.diff(np.concatenate([[False], inside, [False]]).astype(np.int8))
        for begin, finish in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            bands.append(dict(
                type="rect", xref="x", yref="paper", x0=int(begin), x1=int(finish),
                y0=0, y1=1, line_width=0, fillcolor=color, opacity=opacity, layer="below",
            ))
    return bands


def make_fan_figure(
    trace: Trace, ensemble: Ensemble | None, cfg: ScenarioConfig, period: int,
) -> go.Figure:
    """Empty containers in Dudinka: the fan over every run plus this one run."""
    steps = np.arange(trace.n_periods)
    tick_values, tick_text = _calendar_ticks(cfg.n_periods)
    fig = go.Figure()

    if ensemble is not None:
        # Over the ensemble's own length, never the selected run's: clipping to
        # the run would shrink the bands and lift the survival curve whenever
        # the user flips to a run that stopped early.
        horizon = ensemble.series.shape[1]
        band_x = np.arange(horizon)
        for lo, hi, alpha in ((5, 95, 0.13), (25, 75, 0.20)):
            low = ensemble.percentile(lo)[:horizon]
            high = ensemble.percentile(hi)[:horizon]
            fig.add_trace(go.Scatter(
                x=np.concatenate([band_x, band_x[::-1]]),
                y=np.concatenate([high, low[::-1]]),
                fill="toself", fillcolor=f"rgba(76,155,232,{alpha})",
                line=dict(width=0), hoverinfo="skip",
                name=f"{lo}–{hi} перцентиль",
            ))
        fig.add_trace(go.Scatter(
            x=band_x, y=ensemble.percentile(50)[:horizon], mode="lines",
            line=dict(color=COLORS["accent"], width=1.4, dash="dot"),
            name="медиана", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=band_x, y=ensemble.alive[:horizon] / ensemble.n_runs * 100.0,
            mode="lines", yaxis="y2",
            line=dict(color=COLORS["muted"], width=1.2),
            name="доля живых прогонов, %",
            hovertemplate="живых %{y:.0f}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=steps, y=trace.warehouse, mode="lines",
        line=dict(color=COLORS["text"], width=2),
        name="показанный прогон",
        hovertemplate="%{y:,d} порожних<extra></extra>",
    ))

    cursor = dict(
        type="line", xref="x", yref="paper", x0=period, x1=period, y0=0, y1=1,
        line=dict(color=COLORS["warn"], width=1.5),
    )
    shapes = [cursor] + _season_bands(cfg.n_periods)
    warmup = min(cfg.warmup_periods, cfg.n_periods)
    if warmup > 0:
        shapes.append(dict(
            type="rect", xref="x", yref="paper", x0=0, x1=warmup, y0=0, y1=1,
            line_width=0, fillcolor="#000000", opacity=0.28, layer="below",
        ))
    if trace.failed and trace.fail_period is not None:
        shapes.append(dict(
            type="line", xref="x", yref="paper",
            x0=trace.fail_period, x1=trace.fail_period, y0=0, y1=1,
            line=dict(color=COLORS["danger"], width=2, dash="dash"),
        ))

    annotations = []
    if warmup > 0:
        annotations.append(dict(
            x=warmup / 2, y=1, yref="paper", yanchor="bottom", showarrow=False,
            text="выход на режим — из показателей исключён",
            font=dict(color=COLORS["muted"], size=10),
        ))
    if trace.failed and trace.fail_period is not None:
        annotations.append(dict(
            x=trace.fail_period, y=1, yref="paper", yanchor="bottom",
            xanchor="right", showarrow=False,
            text=(f"производство остановилось "
                  f"{date_label(trace, trace.fail_period - 1)}"
                  f", не хватило {trace.fail_deficit}  "),
            font=dict(color=COLORS["danger"], size=11),
        ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        margin=dict(l=60, r=55, t=26, b=28), height=250,
        shapes=shapes, annotations=annotations,
        legend=dict(orientation="h", y=-0.18, font=dict(color=COLORS["muted"], size=10)),
        hovermode="x unified",
        xaxis=dict(
            title=None, gridcolor=COLORS["grid"], zeroline=False,
            range=[0, max(cfg.n_periods - 1, 1)], tickmode="array",
            tickvals=tick_values, ticktext=tick_text,
            tickfont=dict(color=COLORS["muted"], size=10),
        ),
        yaxis=dict(
            title=dict(text="порожние в Дудинке", font=dict(color=COLORS["muted"], size=11)),
            gridcolor=COLORS["grid"], zeroline=False, rangemode="tozero",
            tickfont=dict(color=COLORS["muted"], size=10),
        ),
        yaxis2=dict(
            overlaying="y", side="right", range=[0, 105], showgrid=False,
            tickmode="array", tickvals=[0, 25, 50, 75, 100],
            ticksuffix="%", tickfont=dict(color=COLORS["muted"], size=10),
        ),
    )
    return fig


def make_stock_figure(trace: Trace, cfg: ScenarioConfig, period: int) -> go.Figure:
    """Where every container sits, stacked over time."""
    steps = np.arange(trace.n_periods)
    # Dudinka bright because it is the thing at stake, the Russian hubs a muted
    # slate ramp, the two long-haul stages distinct. A near-white fill here
    # glares against the dark ground and swamps everything above it.
    palette = [
        "#8fd3ff", "#4f93bd",
        ROUTE_COLORS["A"], ROUTE_COLORS["B"], ROUTE_COLORS["C"],
        "#4a6b8a", "#3f5c78", "#354d66", "#2c4055",
        "#8f7bd0", "#c96f8e",
    ]
    fig = go.Figure()
    for name, color in zip(STOCK_BUCKETS, palette):
        fig.add_trace(go.Scatter(
            x=steps, y=trace.stocks[name], mode="lines", stackgroup="one",
            name=name, line=dict(width=0), fillcolor=color, opacity=0.9,
            hovertemplate="%{fullData.name}: %{y:,d}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        margin=dict(l=60, r=55, t=10, b=28), height=210,
        hovermode="x unified",
        shapes=[dict(type="line", xref="x", yref="paper", x0=period, x1=period,
                     y0=0, y1=1, line=dict(color=COLORS["warn"], width=1.5))],
        legend=dict(orientation="h", y=-0.22, font=dict(color=COLORS["muted"], size=9)),
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False, showticklabels=False,
                   range=[0, max(cfg.n_periods - 1, 1)]),
        yaxis=dict(
            title=dict(text="все контейнеры", font=dict(color=COLORS["muted"], size=11)),
            gridcolor=COLORS["grid"], zeroline=False,
            tickfont=dict(color=COLORS["muted"], size=10),
        ),
    )
    return fig
