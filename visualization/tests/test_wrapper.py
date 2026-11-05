"""Guards the contract this project relies on in the unmodified model.

If upstream renames an attribute of ``SimulationEnvironment`` or adds a new
place a container can sit, these tests fail here rather than silently drawing a
wrong picture.
"""
from __future__ import annotations

import numpy as np
import pytest

from simviz.config import ScenarioConfig
from simviz.geo import ROUTES, position_on_leg, route_polyline
from simviz.model_loader import load_model
from simviz.runner import run_ensemble, run_trace, spawn_seeds
from simviz.trace import POINTS, STOCK_BUCKETS, leg_endpoints, read_stocks

SHORT = ScenarioConfig(horizon_years=0.75, warmup_years=0.0, n_fan_runs=6)


@pytest.fixture(scope="module")
def trace():
    return run_trace(SHORT, spawn_seeds(SHORT.seed, 1)[0])


def test_environment_exposes_everything_read_stocks_needs():
    pm = load_model()
    env = pm.SimulationEnvironment(SHORT.to_model_config())
    pm.setup_fleet(env)
    buckets = read_stocks(env)
    assert set(buckets) == set(STOCK_BUCKETS)
    assert sum(buckets.values()) == SHORT.n_containers


def test_simulate_period_is_steppable_from_outside():
    pm = load_model()
    env = pm.SimulationEnvironment(SHORT.to_model_config())
    pm.setup_fleet(env)
    env.rng = np.random.default_rng(0)
    assert env.simulate_period() == 0


def test_every_vessel_state_maps_to_a_leg():
    pm = load_model()
    for vessel_id in ("feeder_1:A", "feeder_1:B", "feeder_1:C",
                      "mainland_1:A", "mainland_1:B"):
        for state in pm.VesselState:
            origin, target = leg_endpoints(vessel_id, state.name)
            assert origin in POINTS and target in POINTS


def test_containers_are_conserved(trace):
    assert trace.conservation_error == 0
    total = sum(values for values in trace.stocks.values())
    assert np.all(total == SHORT.n_containers)


def test_leg_progress_is_known_whenever_a_vessel_is_at_sea(trace):
    at_sea = trace.v_from != trace.v_to
    assert at_sea.any()
    progress = trace.v_progress[at_sea]
    assert np.all((progress >= 0.0) & (progress <= 1.0))
    # a leg that never advances would mean the total length was mis-detected
    assert progress.max() > 0.9


def test_delivered_is_monotone(trace):
    assert np.all(np.diff(trace.delivered) >= 0)


def test_route_c_returns_through_murmansk():
    assert leg_endpoints("feeder_1:C", "SHIPPING_FROM_DEST_TO_INTERM") == ("D", "A")
    assert leg_endpoints("feeder_1:C", "UNLOADING_AT_INTERM") == ("A", "A")


def test_china_to_murmansk_is_one_lane_read_both_ways():
    """Route C's backup return and every mainland A return share this leg.

    The model puts both in ``SHIPPING_FROM_DEST_TO_INTERM`` and credits the same
    counter, so ``leg_endpoints`` cannot tell them apart. Giving ("D", "A") its
    own Arctic polyline therefore caught the mainlands too and sailed them over
    the pole, outbound through Suez and home across the Bering Strait.

    The key assertions carry the weight. Comparing the two polylines alone would
    also pass with no lane at all, because reversing a straight line gives a
    straight line -- see the next test.
    """
    assert ("A", "D") in ROUTES, "the Suez lane both vessels share must stay keyed"
    assert ("D", "A") not in ROUTES, "a second key here is what sailed them over the pole"
    out = route_polyline("A", "D")
    back = route_polyline("D", "A")
    assert np.array_equal(back, out[::-1])


def test_every_leg_the_fleet_sails_has_a_drawn_lane(trace):
    """A missing key does not raise: ``route_polyline`` logs and joins the ports.

    That fallback is invisible to a symmetry check and drags the leg across
    Karelia, the Urals and Mongolia, so assert instead that every leg the state
    machine actually produces resolves to a hand-drawn polyline.
    """
    at_sea = np.nonzero(trace.v_from != trace.v_to)
    legs = {
        (POINTS[trace.v_from[period, i]], POINTS[trace.v_to[period, i]])
        for period, i in zip(*at_sea)
    }
    assert len(legs) >= 8, f"only {len(legs)} legs sailed; the fixture is too short"
    for origin, target in sorted(legs):
        assert len(route_polyline(origin, target)) > 2, (
            f"{origin}->{target} fell back to a straight line: no lane is keyed for it"
        )


def test_mainland_vessels_never_take_the_arctic_lane(trace):
    """The threshold sits in the gap between the two lanes.

    The Suez lane peaks at 70.5 off the North Cape, and any Arctic routing has
    to cross the Kara Sea at 72.8 or higher, so 72 separates them with room for
    a future correction to the North Cape approach.
    """
    seen_going_home = False
    worst, culprit = -90.0, ""
    for i, vessel_id in enumerate(trace.vessel_ids):
        if not vessel_id.startswith("mainland"):
            continue
        for period in range(trace.n_periods):
            origin = POINTS[trace.v_from[period, i]]
            target = POINTS[trace.v_to[period, i]]
            if (origin, target) == ("D", "A"):
                seen_going_home = True
            _, lat = position_on_leg(
                origin, target, float(trace.v_progress[period, i])
            )
            if lat > worst:
                worst, culprit = lat, f"{vessel_id} on period {period}"
    assert seen_going_home, "no mainland sailed China -> Murmansk; nothing was tested"
    assert worst <= 72.0, f"{culprit} is drawn at latitude {worst:.1f}"


def test_starved_fleet_fails_and_reports_a_deficit():
    starved = ScenarioConfig(n_containers=9_000, horizon_years=0.75, warmup_years=0.0)
    result = run_trace(starved, spawn_seeds(starved.seed, 1)[0])
    assert result.failed
    assert result.fail_period is not None
    assert result.n_periods == result.fail_period


def test_ensemble_marks_dead_runs_and_counts_survivors():
    starved = ScenarioConfig(n_containers=9_000, horizon_years=0.75, warmup_years=0.0, n_fan_runs=6)
    ens = run_ensemble(starved)
    assert ens.n_runs == 6
    assert ens.n_failed > 0
    assert ens.alive[0] == 6
    assert ens.alive[-1] == 6 - ens.n_failed
    dead = ens.fail_period >= 0
    for row, stop in zip(ens.series[dead], ens.fail_period[dead]):
        assert np.all(row[stop:] == 0)


def test_calendar_track_matches_the_model(trace):
    """The indicators date events without a trace; that clock must not drift."""
    from simviz.config import calendar_track

    track = calendar_track(trace.n_periods)
    got = [(int(m), int(d)) for m, d in zip(trace.month, trace.day)]
    assert track == got


def test_pause_window_is_found_even_when_the_run_died_inside_it():
    from simviz.indicators import pause_windows

    cfg = ScenarioConfig(horizon_years=2.0, warmup_years=1.0)
    windows = pause_windows(cfg.n_periods, after=cfg.warmup_periods)
    assert windows, "the second year's pause must be located from the calendar alone"
    start, end = windows[0]
    assert cfg.warmup_periods <= start < end < cfg.n_periods


def test_selected_run_matches_its_row_in_the_ensemble():
    """Flipping runs must show a member of the fan, not a fresh draw."""
    from simviz.state import get_result, get_trace

    cfg = ScenarioConfig(horizon_years=1.0, warmup_years=0.0, n_fan_runs=6)
    ens = get_result(cfg).ensemble
    for index in (0, 3, 5):
        trace = get_trace(cfg, index)
        assert trace.run_index == index
        n = min(trace.n_periods, ens.series.shape[1])
        assert np.array_equal(trace.warehouse[:n], ens.series[index, :n])


def test_headline_indicators_do_not_depend_on_the_selected_run():
    """The whole point of the split: the fan's figures must hold still."""
    from simviz.indicators import compute_indicators, summarise_run
    from simviz.state import get_result, get_trace

    cfg = ScenarioConfig(horizon_years=2.0, warmup_years=1.0, n_fan_runs=12)
    result = get_result(cfg)
    reference = compute_indicators(result.ensemble, cfg)
    summaries = [summarise_run(get_trace(cfg, i), cfg) for i in range(4)]

    assert compute_indicators(result.ensemble, cfg) == reference
    assert {s.run_index for s in summaries} == {0, 1, 2, 3}
    # ...while the per-run figures are allowed to, and in practice do, differ
    assert len({s.entry_stock for s in summaries}) > 1


def test_run_index_is_clamped_to_the_ensemble():
    from simviz.state import get_trace

    cfg = ScenarioConfig(horizon_years=1.0, warmup_years=0.0, n_fan_runs=4)
    assert get_trace(cfg, 99).run_index == 3
    assert get_trace(cfg, -5).run_index == 0


def test_failed_run_indices_point_at_runs_that_actually_died():
    from simviz.state import failed_run_indices, get_result, get_trace

    cfg = ScenarioConfig(n_containers=24_000, horizon_years=2.0,
                         warmup_years=1.0, n_fan_runs=6)
    result = get_result(cfg)
    indices = failed_run_indices(result)
    assert indices, "this configuration is meant to fail"
    for index in indices:
        assert get_trace(cfg, index).failed


def test_wilson_interval_brackets_the_rate_without_going_negative():
    from simviz.indicators import wilson_interval

    low, high = wilson_interval(3, 200)
    assert 0.0 <= low <= 0.015 <= high <= 0.06
    assert wilson_interval(0, 200) == (0.0, wilson_interval(0, 200)[1])
    assert wilson_interval(0, 0) == (0.0, 0.0)


# --- defect A: a run that dies before anything is recorded --------------------

#: Warehouse starts at 500 (1000 minus the scaled initial load of 250 per
#: route) and February's daily production is 15000/28 = 536, so the very first
#: period fails. Both values pass the sidebar's own min/step validation.
DEAD_ON_ARRIVAL = ScenarioConfig(n_containers=1000, main_monthly_rate=15_000,
                                 n_fan_runs=8)


def test_a_run_can_die_before_recording_a_single_period():
    """The warehouse starts at n_containers - 1200 and the first day's
    production is subtracted on period 0, so this is reachable from the UI."""
    from simviz.state import get_trace

    trace = get_trace(DEAD_ON_ARRIVAL, 0)
    assert trace.failed and trace.fail_period == 0
    assert trace.n_periods == 0
    assert trace.fail_deficit > 0


def test_every_renderer_survives_an_empty_trace():
    """This crashed show_run with an IndexError and, with debug off, left the
    screen silently showing the previous scenario."""
    from simviz.figures import (date_label, make_fan_figure, make_map_figure,
                                make_stock_figure, vessel_frame)
    from simviz.frames import build_frames
    from simviz.indicators import summarise_run
    from simviz.state import get_result, get_trace

    cfg = DEAD_ON_ARRIVAL
    trace = get_trace(cfg, 0)
    ensemble = get_result(cfg).ensemble

    assert vessel_frame(trace, 0) == ([], [], [], [], [])
    assert date_label(trace, 0) == "2 февраля, год 1"     # the day it died
    assert date_label(trace, -1) == date_label(trace, 999)
    for build in (lambda: make_map_figure(trace, 0),
                  lambda: make_fan_figure(trace, ensemble, cfg, 0),
                  lambda: make_stock_figure(trace, cfg, 0),
                  lambda: build_frames(trace),
                  lambda: summarise_run(trace, cfg)):
        build()


def test_the_run_tile_calls_a_period_zero_failure_what_it_is():
    from simviz.app import _kpi_strip
    from simviz.indicators import summarise_run
    from simviz.state import get_result, get_trace

    cfg = DEAD_ON_ARRIVAL
    trace = get_trace(cfg, 0)
    tiles = _kpi_strip(get_result(cfg), trace, summarise_run(trace, cfg))
    rendered = str(tiles)
    assert "в первый же период" in rendered


# --- defect B: the ensemble fan must not follow the selected run -------------

def test_the_fan_does_not_shrink_when_a_failed_run_is_selected():
    """The whole point of splitting ensemble from run: the picture underneath
    the map must hold still while the user flips through runs."""
    from simviz.figures import make_fan_figure, make_stock_figure
    from simviz.state import failed_run_indices, get_result, get_trace

    cfg = ScenarioConfig(n_containers=26_000, n_fan_runs=40)
    result = get_result(cfg)
    dead = failed_run_indices(result)
    assert dead, "this configuration is meant to produce failures"

    alive = next(i for i in range(cfg.n_fan_runs) if i not in dead)
    reference = None
    for index in (alive, dead[0]):
        trace = get_trace(cfg, index)
        fan = make_fan_figure(trace, result.ensemble, cfg, 0)
        survival = next(t for t in fan.data if t.name and "живых" in t.name)
        median = next(t for t in fan.data if t.name == "медиана")
        signature = (len(survival.x), float(survival.y[-1]), len(median.x),
                     tuple(fan.layout.xaxis.range))
        if reference is None:
            reference = signature
        else:
            assert signature == reference, "the fan moved with the selected run"

    # both charts share one x domain, so the time cursor means the same thing
    trace = get_trace(cfg, dead[0])
    stock = make_stock_figure(trace, cfg, 0)
    assert tuple(stock.layout.xaxis.range) == (0, cfg.n_periods - 1)


def test_season_bands_cover_the_horizon_not_the_selected_run():
    from simviz.figures import _season_bands

    short, long = _season_bands(365 * 3), _season_bands(365 * 3 * 3)
    assert len(long) > len(short), "a longer horizon must show more pauses"


# --- defect C: the pause indicator must span the same window as the risk -----

def test_the_pause_indicator_follows_the_horizon():
    """Reading only the first pause froze this tile: at 27500 containers it
    stayed at +1818 / 1% while the failure rate went 1.5% -> 93.5%."""
    from simviz.indicators import compute_indicators
    from simviz.state import get_result

    seen = []
    for years in (2.0, 4.0):
        cfg = ScenarioConfig(n_containers=27_500, horizon_years=years,
                             warmup_years=1.0, n_fan_runs=40)
        ind = compute_indicators(get_result(cfg).ensemble, cfg)
        seen.append((ind.n_pauses, ind.pause_margin, ind.share_below_threshold,
                     ind.failure_rate))

    (short_pauses, short_margin, short_share, short_fail), \
        (long_pauses, long_margin, long_share, long_fail) = seen
    assert long_pauses > short_pauses
    assert long_fail > short_fail, "a longer horizon must accumulate more failures"
    assert long_margin < short_margin, "the margin must tighten with the horizon"
    assert long_share > short_share, "so must the share entering under-provisioned"
    # the indicator is meant to under-call, never over-call, the risk:
    # the October failure mechanism is invisible to it
    for _, _, share, fail in seen:
        assert share <= fail + 0.05


def test_plural_agreement_for_the_pause_count():
    from simviz.app import _plural

    got = [_plural(n, "пауза", "паузы", "пауз") for n in (1, 2, 4, 5, 11, 21, 22)]
    assert got == ["пауза", "паузы", "паузы", "пауз", "пауз", "пауза", "паузы"]


# --- defect D: the UI's own container floor was below the model's -----------

def test_the_whole_container_range_the_ui_allows_actually_runs():
    """layout.py offers a floor of 1000, but the model rejects any config whose
    initial load exceeds the fleet, and that load used to be a flat 600 + 600."""
    from simviz.state import get_result

    for containers in (1000, 1199, 1200, 2400):
        cfg = ScenarioConfig(n_containers=containers, horizon_years=1.0,
                             warmup_years=0.0, n_fan_runs=4)
        get_result(cfg)                       # must not raise
        model = cfg.to_model_config()
        assert model.n_loaded_A + model.n_loaded_B <= containers


def test_scaling_the_initial_load_leaves_realistic_fleets_untouched():
    for containers in (20_000, 27_500, 40_000):
        model = ScenarioConfig(n_containers=containers).to_model_config()
        assert model.n_loaded_A == 600 and model.n_loaded_B == 600


# --- defect E: say why the pause figures are missing, do not guess ----------

def test_each_blank_pause_reason_is_reported_as_itself():
    """The note used to be picked off the truthiness of failure_rate, so a
    merely short horizon was announced as "no run survived the pause"."""
    from simviz.indicators import BLANK_REASONS, compute_indicators
    from simviz.state import get_result

    # "no_ensemble" is no longer reachable through get_result -- a single-run
    # fan is still an ensemble -- but compute_indicators still accepts None.
    solo = ScenarioConfig(horizon_years=2.0, n_fan_runs=1)
    assert compute_indicators(None, solo).blank_reason == "no_ensemble"

    cases = {
        # the warm-up swallows the only pause in the span
        "no_pause": ScenarioConfig(horizon_years=1.0, warmup_years=1.0, n_fan_runs=6),
        # the pause fits, its 45-day recovery tail does not
        "short_horizon": ScenarioConfig(horizon_years=1.5, warmup_years=1.0,
                                        n_fan_runs=6),
        # runs die during year 1, before the measured pause even arrives
        "none_reached": ScenarioConfig(n_containers=20_000, n_fan_runs=6),
        # runs reach the pause, none come out the far side of its recovery tail
        "none_survived": ScenarioConfig(n_containers=24_000, n_fan_runs=20),
    }
    for expected, cfg in cases.items():
        ind = compute_indicators(get_result(cfg).ensemble, cfg)
        assert ind.pause_margin is None, expected
        assert ind.blank_reason == expected, f"{expected}: got {ind.blank_reason}"
        assert ind.blank_note == BLANK_REASONS[expected]


def test_a_healthy_scenario_carries_no_blank_reason():
    from simviz.indicators import compute_indicators
    from simviz.state import get_result

    cfg = ScenarioConfig(n_fan_runs=20)
    ind = compute_indicators(get_result(cfg).ensemble, cfg)
    assert ind.pause_margin is not None
    assert ind.blank_reason is None and ind.blank_note is None


# --- defects H, I, J ---------------------------------------------------------

def test_a_single_run_fan_is_still_an_ensemble():
    """With ensemble=None the failure tile, the "jump to a failed run" button
    and the pause tile each contradicted the run on the map."""
    from simviz.state import failed_run_indices, get_result, get_trace

    cfg = ScenarioConfig(n_containers=26_000, n_fan_runs=1)
    result = get_result(cfg)
    assert result.ensemble is not None and result.ensemble.n_runs == 1
    assert result.indicators.failure_rate is not None

    trace = get_trace(cfg, 0)
    assert (trace.failed) == (len(failed_run_indices(result)) == 1)


def test_delivered_is_blank_rather_than_zero_on_a_zero_length_window():
    """warm-up == horizon collapsed the span to a single index, and the tile
    reported 0 containers a year for a run that delivered tens of thousands."""
    from simviz.indicators import summarise_run
    from simviz.state import get_trace

    cfg = ScenarioConfig(horizon_years=1.0, warmup_years=1.0, n_fan_runs=4)
    run = summarise_run(get_trace(cfg, 0), cfg)
    assert run.measured_days == 0
    assert run.delivered_per_year is None

    # spell the park out: this case needs a run that LIVES through year two, and
    # the default moved once already when upstream rewidened its distributions
    healthy = ScenarioConfig(n_containers=33_000, horizon_years=2.0,
                             warmup_years=1.0, n_fan_runs=4)
    ok = summarise_run(get_trace(healthy, 0), healthy)
    assert ok.measured_days >= 350 and ok.delivered_per_year > 0


def test_the_share_below_threshold_counts_the_whole_ensemble():
    """It used to be divided by the survivors while the failure tile beside it
    was divided by every run."""
    from simviz.indicators import compute_indicators, pause_windows
    from simviz.state import get_result

    cfg = ScenarioConfig(n_containers=26_000, horizon_years=3.0,
                         warmup_years=2.0, n_fan_runs=60)
    result = get_result(cfg)
    ind = compute_indicators(result.ensemble, cfg)
    entry = pause_windows(cfg.n_periods, after=cfg.warmup_periods)[0][0]
    died_before = int((result.ensemble.fail_period >= 0).sum()
                      - (result.ensemble.fail_period > entry).sum())
    assert died_before > 0, "this scenario is meant to kill runs before the pause"
    # every run that never reached the pause is under-provisioned by definition
    assert ind.share_below_threshold >= died_before / cfg.n_fan_runs


# --- the warehouse trough the model actually tests ---------------------------

def test_daily_production_reproduces_the_model_exactly():
    from simviz.config import daily_production

    cfg = ScenarioConfig(horizon_years=1.0, warmup_years=0.0,
                         main_monthly_rate=3000, alt_annual_total=4000)
    loads = daily_production(cfg)
    assert int((loads > 0).sum()) == 365, "one charge a day, on the day's first period"
    assert int(loads.sum()) == 40_000, "and they must total the annual figure exactly"


def test_the_trough_is_read_where_the_model_tests_it():
    """The model subtracts the day's production and checks the sign before
    adding that period's returning empties, so the recorded series never shows
    the dip. Reading the series minimum flatters the warehouse by a day's
    returns, about 110 containers."""
    import numpy as np
    from simviz.indicators import exact_trough
    from simviz.state import get_result

    cfg = ScenarioConfig(n_containers=30_000, n_fan_runs=60)
    ens = get_result(cfg).ensemble
    start = cfg.warmup_periods

    naive = ens.series[:, start:].min(axis=1)
    exact = exact_trough(ens, cfg, start)
    assert np.all(exact <= naive), "the true trough can never be the higher one"
    gap = float(np.median(naive - exact))
    assert 80 <= gap <= 200, f"expected about a day's returns, got {gap}"


def test_a_park_that_survives_is_exactly_the_one_the_trough_predicts():
    """Validated on brain against a direct sweep with zero discrepancy; this is
    the cheap version of that check."""
    import numpy as np
    from simviz.indicators import exact_trough
    from simviz.state import get_result

    reference = 200_000
    common = dict(horizon_years=2.0, warmup_years=1.0, n_fan_runs=40)
    big_cfg = ScenarioConfig(n_containers=reference, **common)
    big = get_result(big_cfg).ensemble
    assert big.n_failed == 0
    need = reference - exact_trough(big, big_cfg, big_cfg.warmup_periods)

    for park in (26_000, 27_500, 30_000):
        actual = get_result(ScenarioConfig(n_containers=park, **common)).ensemble
        assert int((need > park).sum()) == actual.n_failed, park
