"""Dash application: one simulated run on the map, the ensemble underneath it.

Two server callbacks. The first runs the fan and is the expensive one; the
second draws whichever run of that fan is selected and is nearly free, because
the ensemble is cached and only the chosen trace is recorded. Playback itself
never reaches the server -- see ``assets/animate.js``.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging

from dash import (
    ClientsideFunction,
    Dash,
    Input,
    Output,
    State,
    clientside_callback,
    html,
    no_update,
)

from simviz.config import ScenarioConfig
from simviz.figures import (
    COLORS,
    date_label,
    make_fan_figure,
    make_map_figure,
    make_stock_figure,
)
from simviz.frames import build_frames
from simviz.indicators import Indicators, RunSummary, summarise_run
from simviz.layout import build_layout, kpi_tile
from simviz.state import Result, failed_run_indices, get_result, get_trace
from simviz.trace import Trace

logger = logging.getLogger(__name__)

__all__ = ["build_app", "main"]

_SCENARIO_INPUTS = (
    "in-containers", "in-feeders-a", "in-feeders-b", "in-feeders-c",
    "in-main-a", "in-main-b", "in-main-rate", "in-alt-rate",
    "in-horizon", "in-warmup", "in-runs", "in-seed",
)


def _scenario(containers, fa, fb, fc, ma, mb, rate, alt, horizon, warmup, runs, seed):
    """Build a config from the control values, tolerating blank inputs."""
    def whole(value, fallback: int, low: int = 0) -> int:
        return max(int(value), low) if value is not None else fallback

    default = ScenarioConfig()
    return ScenarioConfig(
        n_containers=whole(containers, default.n_containers, 1000),
        n_feeders_A=whole(fa, default.n_feeders_A),
        n_feeders_B=whole(fb, default.n_feeders_B),
        n_feeders_C=whole(fc, default.n_feeders_C),
        n_mainlands_A=whole(ma, default.n_mainlands_A),
        n_mainlands_B=whole(mb, default.n_mainlands_B),
        main_monthly_rate=whole(rate, default.main_monthly_rate),
        alt_annual_total=whole(alt, default.alt_annual_total),
        horizon_years=float(horizon or default.horizon_years),
        warmup_years=float(warmup if warmup is not None else default.warmup_years),
        n_fan_runs=whole(runs, default.n_fan_runs, 1),
        seed=whole(seed, default.seed),
    )


def _thousands(value: float | int) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Russian numeric agreement: 1 пауза, 2 паузы, 5 пауз."""
    tail, tens = count % 10, count % 100
    if 11 <= tens <= 14 or tail == 0 or tail >= 5:
        return many
    return one if tail == 1 else few


def _pause_tile(ind: Indicators) -> html.Div:
    if ind.pause_margin is None:
        # Five different causes, so the reason is carried, not guessed: reading
        # it off the failure rate labelled a merely short horizon as a wipe-out.
        alarming = ind.blank_reason in ("none_reached", "none_survived")
        return kpi_tile(
            "Запас на входе в паузу", "—",
            ind.blank_note or "показатель недоступен",
            COLORS["danger"] if alarming else COLORS["muted"],
        )
    sign = "+" if ind.pause_margin >= 0 else "−"
    span = ("паузу" if ind.n_pauses == 1 else
            f"{ind.n_pauses} {_plural(ind.n_pauses, 'паузу', 'паузы', 'пауз')}")
    below = (f"; ниже порога заходит {ind.share_below_threshold * 100:.0f}% "
             f"из {ind.n_runs}"
             if ind.share_below_threshold is not None else "")
    # The median is conditioned on reaching a pause; say so whenever some runs
    # did not, otherwise it reads as a median over the whole ensemble.
    among = ("" if ind.n_reached_pause == ind.n_runs
             else f" по {ind.n_reached_pause} дошедшим")
    return kpi_tile(
        f"Запас на входе в {span}",
        f"{sign}{_thousands(abs(ind.pause_margin))}",
        f"худший вход, медиана{among} {_thousands(ind.pause_entry_median)} против "
        f"{_thousands(ind.pause_consumption)}, которые съедает пауза{below}",
        COLORS["danger"] if ind.at_risk else COLORS["ok"],
    )


def _failure_tile(ind: Indicators) -> html.Div | None:
    if ind.failure_rate is None:
        return None
    failed = int(round(ind.failure_rate * ind.n_runs))
    accent = (COLORS["ok"] if failed == 0
              else COLORS["warn"] if ind.failure_rate < 0.1 else COLORS["danger"])
    note = f"{failed} прогонов из {ind.n_runs}"
    if ind.failure_ci is not None:
        low, high = ind.failure_ci
        note += f"; 95% интервал {low * 100:.1f}–{high * 100:.1f}%"
    return kpi_tile(
        "Остановки производства", f"{ind.failure_rate * 100:.1f}%", note, accent
    )


def _run_tile(run: RunSummary, trace: Trace, ind: Indicators) -> html.Div:
    # fail_period 0 is a real period, not a missing value: a truthiness test
    # here would silently relabel the worst case as the second-worst.
    outcome = "дожил до конца горизонта"
    if run.failed:
        when = date_label(trace, (trace.fail_period or 0) - 1)
        outcome = (f"остановился в первый же период, {when}, "
                   f"не хватило {trace.fail_deficit}"
                   if trace.n_periods == 0 else
                   f"остановился {when}, не хватило {trace.fail_deficit}")
    entry = (f"; вошёл в паузу с {_thousands(run.entry_stock)}"
             if run.entry_stock is not None else "")
    return kpi_tile(
        "Прогон на карте", f"№{run.run_index + 1} из {ind.n_runs}",
        f"{outcome}{entry}",
        COLORS["danger"] if run.failed else COLORS["muted"],
    )


def _kpi_strip(result: Result, trace: Trace, run: RunSummary) -> list[html.Div]:
    ind = result.indicators
    tiles: list[html.Div] = [_pause_tile(ind)]

    if ind.pipeline_at_entry is not None:
        tiles.append(kpi_tile(
            "В обороте на 9 мая", _thousands(ind.pipeline_at_entry),
            "контейнеров вне склада Дудинки; это число почти не зависит от "
            "размера парка, поэтому запас = парк минус оно",
            COLORS["text"],
        ))

    if ind.min_stock_p5 is not None:
        tiles.append(kpi_tile(
            "Минимум склада, 5-й перцентиль", _thousands(ind.min_stock_p5),
            f"по всем {ind.n_runs} прогонам, остановившиеся учтены нулём",
            COLORS["danger"] if ind.min_stock_p5 <= 0 else COLORS["text"],
        ))

    failure = _failure_tile(ind)
    if failure is not None:
        tiles.append(failure)

    if run.delivered_per_year is not None:
        window = ("контейнеров в год, показанный прогон"
                  if run.measured_days >= 350 else
                  f"контейнеров в год в пересчёте с {run.measured_days} "
                  f"{_plural(run.measured_days, 'дня', 'дней', 'дней')}, "
                  f"показанный прогон")
        tiles.append(kpi_tile("Доставлено в Китай",
                              _thousands(run.delivered_per_year),
                              window, COLORS["text"]))

    tiles.append(_run_tile(run, trace, ind))
    return tiles


def _register(app: Dash) -> None:

    @app.callback(
        Output("scenario", "data"),
        Output("failed-runs", "data"),
        Output("run-index", "value"),
        Output("run-index", "max"),
        Output("run-total", "children"),
        Output("run-status", "children"),
        Input("btn-run", "n_clicks"),
        *[State(name, "value") for name in _SCENARIO_INPUTS],
        prevent_initial_call="initial_duplicate",
    )
    def recompute(_clicks, *values):
        """The expensive half: run the fan for a scenario.

        Anything that raises here used to 500 the callback, and with debug off
        the screen simply kept the previous scenario while the sidebar claimed
        the new one. Failures now land in the status line instead.
        """
        cfg = _scenario(*values)
        try:
            result = get_result(cfg)
        except Exception as failure:                     # noqa: BLE001 - surfaced to the user
            logger.exception("scenario rejected")
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                f"сценарий не посчитан: {failure}",
            )
        failed = failed_run_indices(result)
        status = (
            f"{cfg.n_periods} периодов, {cfg.n_fan_runs} прогонов, {result.seconds:.1f} с"
        )
        return (
            dataclasses.asdict(cfg),
            failed,
            1,  # always land on the first run of a new fan
            cfg.n_fan_runs,
            f"из {cfg.n_fan_runs}",
            status,
        )

    @app.callback(
        Output("frames", "data"),
        Output("kpi-strip", "children"),
        Output("map", "figure"),
        Output("fan", "figure"),
        Output("stock", "figure"),
        Output("period", "max"),
        Output("period", "value"),
        Output("playing", "data", allow_duplicate=True),
        Input("scenario", "data"),
        Input("run-index", "value"),
        prevent_initial_call=True,
    )
    def show_run(scenario, run_number):
        """The cheap half: draw one member of the cached fan."""
        if not scenario:
            # recompute never succeeded, so there is nothing to draw; the run
            # picker must not turn that into a second traceback.
            return (no_update,) * 8
        cfg = ScenarioConfig(**scenario)
        index = max(int(run_number or 1), 1) - 1
        result = get_result(cfg)
        trace = get_trace(cfg, index)
        run = summarise_run(trace, cfg)
        return (
            build_frames(trace),
            _kpi_strip(result, trace, run),
            make_map_figure(trace, 0),
            make_fan_figure(trace, result.ensemble, cfg, 0),
            make_stock_figure(trace, cfg, 0),
            trace.n_periods - 1,
            0,
            False,  # a newly drawn run starts paused at day one
        )


def _register_playback() -> None:
    """Playback and run stepping are clientside: no round trip per frame."""
    clientside_callback(
        ClientsideFunction(namespace="simviz", function_name="render"),
        Output("date-label", "children"),
        Input("period", "value"),
        # frames is an Input, not State: on the first load the period is already
        # 0 and never "changes", so the frame would never be drawn.
        Input("frames", "data"),
    )
    clientside_callback(
        ClientsideFunction(namespace="simviz", function_name="toggle"),
        Output("playing", "data"),
        Input("btn-play", "n_clicks"),
        State("playing", "data"),
        prevent_initial_call=True,
    )
    clientside_callback(
        ClientsideFunction(namespace="simviz", function_name="applyPlaying"),
        Output("tick", "disabled"),
        Output("tick", "interval"),
        Output("btn-play", "children"),
        Input("playing", "data"),
    )
    clientside_callback(
        ClientsideFunction(namespace="simviz", function_name="advance"),
        Output("period", "value", allow_duplicate=True),
        Output("playing", "data", allow_duplicate=True),
        Input("tick", "n_intervals"),
        State("period", "value"),
        State("playing", "data"),
        State("speed", "value"),
        State("frames", "data"),
        prevent_initial_call=True,
    )
    clientside_callback(
        ClientsideFunction(namespace="simviz", function_name="stepRun"),
        Output("run-index", "value", allow_duplicate=True),
        Input("btn-prev-run", "n_clicks"),
        Input("btn-next-run", "n_clicks"),
        Input("btn-failed-run", "n_clicks"),
        State("run-index", "value"),
        State("run-index", "max"),
        State("failed-runs", "data"),
        prevent_initial_call=True,
    )


def build_app() -> Dash:
    app = Dash(__name__, title="Дудинка — Китай: оборот контейнеров",
               update_title=None, suppress_callback_exceptions=True)
    app.layout = build_layout(ScenarioConfig())
    _register(app)
    _register_playback()
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
