"""Component tree for the Dash app."""
from __future__ import annotations

from dash import dcc, html

from simviz.config import ScenarioConfig
from simviz.figures import COLORS

__all__ = ["FAN_RUN_MARKS", "build_layout", "kpi_tile"]

_STEP_BUTTON = {
    "width": "28px", "height": "30px", "backgroundColor": "#0d1620",
    "color": "#dfe7ef", "border": "1px solid #22303f", "borderRadius": "6px",
    "fontSize": "12px", "cursor": "pointer", "flex": "0 0 auto",
}

#: Fan sizes offered, with the cost the user is signing up for. One run of two
#: simulated years takes about 0.1 s, spread over the machine's cores.
FAN_RUN_MARKS = {100: "100", 200: "200", 500: "500", 1000: "1000", 2000: "2000"}

_PANEL = {
    "backgroundColor": COLORS["panel"],
    "borderRadius": "10px",
    "padding": "14px 16px",
}
_LABEL = {
    "color": COLORS["muted"],
    "fontSize": "11px",
    "textTransform": "uppercase",
    "letterSpacing": "0.06em",
    "marginBottom": "4px",
    "display": "block",
}
_INPUT = {
    "width": "100%",
    "backgroundColor": "#0d1620",
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['grid']}",
    "borderRadius": "6px",
    "padding": "6px 8px",
    "fontSize": "13px",
}


def _number(component_id: str, label: str, value: float, step: float = 1,
            minimum: float = 0, maximum: float = 1_000_000) -> html.Div:
    """A bounded number field.

    `maximum` is not optional in practice. Dash 4.4.1 has a copy-paste bug in
    the -/+ stepper: the clamp reads

        null !== x.max && void 0 !== x.min && (o = Math.min(o, parseFloat(x.max)))

    where the second test should be on `x.max`. With `min` set and `max` left
    out, that runs `Math.min(o, NaN)`, writes the string "NaN" into the field
    and the browser drops it, so one click empties the input for good. Giving
    every field a max sidesteps it, and bounds are worth having anyway.
    """
    return html.Div(
        [
            html.Label(label, style=_LABEL),
            dcc.Input(id=component_id, type="number", value=value, step=step,
                      min=minimum, max=maximum, style=_INPUT),
        ],
        style={"marginBottom": "12px"},
    )


def _slider(component_id: str, label: str, value: float, minimum: float,
            maximum: float, step: float | None, marks: dict) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_LABEL),
            dcc.Slider(id=component_id, min=minimum, max=maximum, step=step,
                       value=value, marks=marks, included=False,
                       tooltip={"placement": "bottom", "always_visible": False}),
        ],
        style={"marginBottom": "18px"},
    )


def kpi_tile(title: str, value: str, note: str, accent: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={**_LABEL, "marginBottom": "6px"}),
            html.Div(value, style={"color": accent, "fontSize": "26px",
                                   "fontWeight": 600, "lineHeight": "1.1"}),
            html.Div(note, style={"color": COLORS["muted"], "fontSize": "11px",
                                  "marginTop": "5px", "minHeight": "28px"}),
        ],
        style={**_PANEL, "flex": "1 1 0", "minWidth": "170px"},
    )


def _controls(cfg: ScenarioConfig) -> html.Div:
    return html.Div(
        [
            html.Div("Сценарий", style={"color": COLORS["text"], "fontSize": "14px",
                                        "fontWeight": 600, "marginBottom": "14px"}),
            _number("in-containers", "Контейнеров в парке", cfg.n_containers, 500, 1000, 300_000),
            html.Div(
                [
                    html.Div(_number("in-feeders-a", "Фидеры A", cfg.n_feeders_A, 1, 0, 40),
                             style={"flex": 1, "marginRight": "8px"}),
                    html.Div(_number("in-feeders-b", "Фидеры B", cfg.n_feeders_B, 1, 0, 40),
                             style={"flex": 1, "marginRight": "8px"}),
                    html.Div(_number("in-feeders-c", "Фидеры C", cfg.n_feeders_C, 1, 0, 40),
                             style={"flex": 1}),
                ],
                style={"display": "flex"},
            ),
            html.Div(
                [
                    html.Div(_number("in-main-a", "Магистрали A", cfg.n_mainlands_A, 1, 0, 40),
                             style={"flex": 1, "marginRight": "8px"}),
                    html.Div(_number("in-main-b", "Магистрали B", cfg.n_mainlands_B, 1, 0, 40),
                             style={"flex": 1}),
                ],
                style={"display": "flex"},
            ),
            _number("in-main-rate", "Производство, конт./мес", cfg.main_monthly_rate, 100, 0, 20_000),
            _number("in-alt-rate", "Маршрут C, конт./год", cfg.alt_annual_total, 250, 0, 50_000),
            _slider("in-horizon", "Горизонт, лет", cfg.horizon_years, 1, 6, 0.5,
                    {i: str(i) for i in range(1, 7)}),
            _slider("in-warmup", "Выход на режим, лет", cfg.warmup_years, 0, 2, 0.5,
                    {0: "0", 1: "1", 2: "2"}),
            _slider("in-runs", "Прогонов в веере", cfg.n_fan_runs, 1, 1000, None,
                    FAN_RUN_MARKS),
            _number("in-seed", "Seed", cfg.seed, 1, 0, 1_000_000),
            html.Button("Пересчитать", id="btn-run", n_clicks=0, style={
                "width": "100%", "padding": "10px", "marginTop": "4px",
                "backgroundColor": COLORS["accent"], "color": "#08111b",
                "border": "none", "borderRadius": "7px", "fontSize": "14px",
                "fontWeight": 600, "cursor": "pointer",
            }),
            html.Div(id="run-status", style={"color": COLORS["muted"], "fontSize": "11px",
                                             "marginTop": "10px", "minHeight": "30px"}),
            html.Div(
                "Аренда контейнеров в Китае выключена: один прогон с ней стоит "
                "100–140 секунд вместо 0.1 и в интерактив не помещается.",
                style={"color": COLORS["muted"], "fontSize": "10px", "marginTop": "14px",
                       "lineHeight": "1.5", "borderTop": f"1px solid {COLORS['grid']}",
                       "paddingTop": "10px"},
            ),
        ],
        style={**_PANEL, "width": "270px", "flex": "0 0 270px",
               "maxHeight": "calc(100vh - 40px)", "overflowY": "auto"},
    )


def _run_picker() -> html.Div:
    """Which member of the ensemble is drawn. A viewing control only: every
    headline figure comes from the whole fan and does not move with it."""
    return html.Div(
        [
            html.Span("ПРОГОН", style={**_LABEL, "display": "inline", "marginBottom": 0}),
            html.Button("◀", id="btn-prev-run", n_clicks=0, style=_STEP_BUTTON),
            dcc.Input(id="run-index", type="number", value=1, min=1, max=1, step=1,
                      debounce=True,
                      style={**_INPUT, "width": "62px", "textAlign": "center",
                             "padding": "5px 4px"}),
            html.Span(id="run-total", style={"color": COLORS["muted"], "fontSize": "12px"}),
            html.Button("▶", id="btn-next-run", n_clicks=0, style=_STEP_BUTTON),
            html.Button("к упавшему", id="btn-failed-run", n_clicks=0, style={
                **_STEP_BUTTON, "width": "auto", "padding": "0 10px",
            }),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "6px",
               "flex": "0 0 auto", "paddingLeft": "14px",
               "borderLeft": f"1px solid {COLORS['grid']}"},
    )


def _playback() -> html.Div:
    return html.Div(
        [
            html.Button("▶", id="btn-play", n_clicks=0, style={
                "width": "42px", "height": "34px", "backgroundColor": COLORS["grid"],
                "color": COLORS["text"], "border": "none", "borderRadius": "6px",
                "fontSize": "14px", "cursor": "pointer", "flex": "0 0 auto",
            }),
            html.Div(id="date-label", style={
                "color": COLORS["text"], "fontSize": "14px", "fontWeight": 600,
                "width": "180px", "flex": "0 0 180px", "paddingLeft": "12px",
            }),
            html.Div(dcc.Slider(id="period", min=0, max=1, step=1, value=0, marks=None,
                                updatemode="drag", allow_direct_input=False,
                                tooltip={"placement": "top", "always_visible": False}),
                     style={"flex": "1 1 auto", "paddingLeft": "8px"}),
            dcc.Dropdown(
                id="speed", value=8, clearable=False,
                options=[{"label": "×1", "value": 1}, {"label": "×3", "value": 3},
                         {"label": "×8", "value": 8}, {"label": "×20", "value": 20}],
                # colour comes from style.css: this used to force dark text for the
                # old dropdown's light menu, and now it is dark text on dark
                style={"width": "80px", "flex": "0 0 80px"},
            ),
            _run_picker(),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "6px",
               "padding": "8px 12px", **_PANEL, "marginTop": "10px"},
    )


def build_layout(cfg: ScenarioConfig) -> html.Div:
    return html.Div(
        [
            dcc.Store(id="frames"),
            dcc.Store(id="scenario"),
            dcc.Store(id="failed-runs", data=[]),
            dcc.Store(id="playing", data=False),
            dcc.Interval(id="tick", interval=60, disabled=True),
            _controls(cfg),
            html.Div(
                [
                    html.Div(id="kpi-strip",
                             style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                    html.Div(
                        dcc.Graph(id="map", config={"displayModeBar": False,
                                                    "scrollZoom": True},
                                  style={"height": "100%"}),
                        style={**_PANEL, "padding": "6px", "marginTop": "10px",
                               "height": "clamp(340px, 52vh, 700px)"},
                    ),
                    _playback(),
                    html.Div(
                        [
                            dcc.Graph(id="fan", config={"displayModeBar": False}),
                            dcc.Graph(id="stock", config={"displayModeBar": False}),
                        ],
                        style={**_PANEL, "marginTop": "10px", "padding": "6px"},
                    ),
                ],
                style={"flex": "1 1 auto", "minWidth": 0, "marginLeft": "12px"},
            ),
        ],
        style={"display": "flex", "padding": "12px", "backgroundColor": COLORS["bg"],
               "minHeight": "100vh", "fontFamily": "system-ui, -apple-system, sans-serif"},
    )
