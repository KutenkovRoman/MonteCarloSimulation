"""
Numbers the screen is allowed to claim.

Three rules come out of the diagnostic work on this model and are enforced here:

* everything is measured from the end of the warm-up, because a run started
  cold spends its first year filling the pipeline and its warehouse minimum
  says nothing about the steady state;
* the failure risk is read off the whole ensemble with dead runs pinned at
  zero, never off the survivors;
* the headline numbers come from the ensemble, never from whichever single run
  happens to be on the map. One run's stock entering the pause swings by 2600
  containers between root seeds; the ensemble median does not.

The one predictive statement the model supports is the margin entering the
interflood pause. Runs that go on to fail enter it about a thousand empties
poorer than survivors while being indistinguishable by leg and by vessel idle
time: there is no guilty bottleneck. The share of runs entering below what the
pause itself consumes tracks the failure rate closely -- 1% against 1.5% at
27500 containers, 36% against 47% at 26000. The gap is the separate October
mechanism, which this indicator is blind to.
"""
from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np

from simviz.config import (
    PERIODS_IN_DAY,
    ScenarioConfig,
    calendar_point,
    calendar_track,
    daily_production,
)
from simviz.runner import Ensemble
from simviz.trace import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "BLANK_REASONS",
    "Indicators",
    "RunSummary",
    "compute_indicators",
    "pause_windows",
    "summarise_run",
    "wilson_interval",
]

#: The warehouse keeps falling for a few weeks after the pause lifts: feeders
#: still have to sail back and reload, and the observed minima land in the first
#: half of July. Measuring the drop only to 30 June understates it threefold.
RECOVERY_DAYS = 45

#: Why the pause figures came out empty. ``pause_margin is None`` has five
#: structurally different causes and they need different words on screen;
#: guessing from the failure rate labelled a short horizon as a wipe-out.
BLANK_REASONS: dict[str, str] = {
    "no_ensemble": "нужно больше одного прогона в веере",
    "no_pause": "в измеряемом окне нет паузы: увеличьте горизонт или "
                "уменьшите выход на режим",
    "none_reached": "ни один прогон не дожил до паузы",
    "short_horizon": "горизонта не хватает на 45 дней после паузы, "
                     "где склад и достигает минимума",
    "none_survived": "паузу не пережил ни один прогон, сравнивать не с чем",
}


def pause_windows(n_periods: int, after: int = 0) -> list[tuple[int, int]]:
    """``(period of 9 May, period of 30 June)`` for each pause after ``after``.

    Computed from the calendar, not from a trace: a run that fails in June ends
    before its own pause does, and the entry stock is still worth reporting.
    """
    track = calendar_track(n_periods)
    pause_start = calendar_point("interflood_start")
    pause_end = calendar_point("interflood_end")

    windows: list[tuple[int, int]] = []
    start: int | None = None
    for period in range(after, len(track)):
        if start is None and track[period] == pause_start:
            start = period
        elif start is not None and track[period] == pause_end:
            windows.append((max(start - 1, 0), period))
            start = None
    return windows


def exact_trough(ensemble: Ensemble, cfg: ScenarioConfig, start: int) -> np.ndarray:
    """The lowest the warehouse actually goes in each run, per the model's own test.

    ``SrcTransportHandler.handle`` subtracts the day's production and checks the
    sign right there, BEFORE adding that period's returning empties, so the
    recorded series never shows the intra-period dip. Reading the minimum off
    the series understates the trough by one period's returns -- about 110
    containers at the stock configuration -- and understates the required park
    by the same amount. The value the model tests is the previous boundary's
    stock minus that day's charge.

    Runs that stopped are reported as zero, matching the convention the fan and
    the KPI label already use.
    """
    loads = daily_production(cfg)
    charged = np.nonzero(loads[start:])[0] + start
    charged = charged[charged > 0]
    if charged.size == 0:
        return np.zeros(ensemble.n_runs, dtype=np.int64)

    tested = ensemble.series[:, charged - 1] - loads[charged]
    reached = (
        (ensemble.fail_period[:, None] < 0)
        | (charged[None, :] < ensemble.fail_period[:, None])
    )
    trough = np.where(reached, tested, np.inf).min(axis=1)
    trough[~np.isfinite(trough)] = 0
    return np.where(ensemble.fail_period >= 0, 0, trough).astype(np.int64)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion, honest at the small rates that matter.

    Five root seeds at 27500 containers gave failure rates from 1.5% to 5.5%
    over 200 runs each. That spread is ordinary binomial noise, and printing a
    bare "2%" hides it.
    """
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    spread /= denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


@dataclasses.dataclass(frozen=True)
class Indicators:
    """Ensemble-wide figures. None of these move when you flip through runs."""

    measured_from_period: int
    n_runs: int

    n_pauses: int                        # pauses inside the measured span
    n_reached_pause: int                 # runs that lived to see one
    pause_entry_median: int | None       # median of each run's WORST pause entry
    pause_consumption: int | None        # what a pause eats, this configuration
    pause_margin: int | None             # median worst entry minus consumption
    share_below_threshold: float | None  # runs under-provisioned at some pause
    pipeline_at_entry: int | None        # containers outside the warehouse, first pause

    min_stock_p5: int | None             # 5th percentile over all runs, dead at zero
    failure_rate: float | None
    failure_ci: tuple[float, float] | None

    blank_reason: str | None             # set when the pause figures are empty

    @property
    def blank_note(self) -> str | None:
        return BLANK_REASONS.get(self.blank_reason or "")

    @property
    def at_risk(self) -> bool:
        return self.pause_margin is not None and self.pause_margin < 0


@dataclasses.dataclass(frozen=True)
class RunSummary:
    """The one run drawn on the map."""

    run_index: int
    entry_stock: int | None
    min_stock: int | None
    measured_days: int                   # span the annualised figure covers
    delivered_per_year: float | None
    failed: bool
    fail_period: int | None
    fail_deficit: int


def _pause_consumption(
    ensemble: Ensemble, start: int, end: int,
) -> tuple[int | None, str | None]:
    """How far the warehouse falls once the pause begins, down to its trough.

    The trough sits weeks past the end of the pause, so the window runs to
    ``end + RECOVERY_DAYS``. Only runs that survived the whole window count: a
    run that died inside it has been pinned at zero and would report its entire
    entry stock as consumption -- a number that looks measured but is not.
    """
    stop = end + RECOVERY_DAYS * PERIODS_IN_DAY
    if stop >= ensemble.series.shape[1]:
        return None, "short_horizon"
    survived = (ensemble.fail_period < 0) | (ensemble.fail_period > stop)
    if not survived.any():
        return None, "none_survived"
    window = ensemble.series[survived][:, start:stop + 1]
    return int(np.median(window[:, 0] - window.min(axis=1))), None


def compute_indicators(ensemble: Ensemble | None, cfg: ScenarioConfig) -> Indicators:
    start = min(cfg.warmup_periods, max(cfg.n_periods - 1, 0))
    windows = pause_windows(cfg.n_periods, after=start)

    entry = consumption = margin = share = pipeline = None
    min_p5 = failure_rate = failure_ci = None
    reached_pause = 0
    n_runs = ensemble.n_runs if ensemble is not None else 1
    reason = None if ensemble is not None else "no_ensemble"

    if ensemble is not None:
        if not windows:
            reason = "no_pause"
        if windows:
            # Every pause in the span, not just the first. Reading only the
            # first froze this tile against the horizon slider: at 27500
            # containers it stayed at "+1818, 1% below threshold" while the
            # failure rate beside it went from 1.5% at two years to 93.5% at
            # five.
            entry_periods = np.array([w[0] for w in windows])
            stock = ensemble.series[:, entry_periods]
            reached = (
                (ensemble.fail_period[:, None] < 0)
                | (entry_periods[None, :] < ensemble.fail_period[:, None])
            )
            worst = np.where(reached, stock, np.inf).min(axis=1)
            got_there = np.isfinite(worst)
            reached_pause = int(got_there.sum())
            if got_there.any():
                entry = int(np.median(worst[got_there]))
            else:
                reason = "none_reached"
            # The pipeline figure is about one date, so it stays on the first
            # pause; taking a minimum across pauses would not be a pipeline.
            if reached[:, 0].any():
                pipeline = cfg.n_containers - int(np.median(stock[reached[:, 0], 0]))

            consumption, blocked = _pause_consumption(ensemble, *windows[0])
            if blocked is not None and reason is None:
                reason = blocked
            if entry is not None and consumption is not None:
                margin = entry - consumption
                # A run that died before reaching a pause was under-provisioned
                # by any reading, so it counts, and the denominator is the
                # whole ensemble to match the failure tile beside it.
                share = float(((~got_there) | (worst < consumption)).mean())
            elif blocked == "none_survived":
                # Nobody lived through the pause window, so consumption cannot be
                # measured and the margin stays blank. The share is still known:
                # every run either never reached the pause or died inside it, and
                # both readings are under-provisioned. Leaving it None made the
                # tile go quiet in exactly the scenario it exists to shout about.
                share = 1.0

        if start < ensemble.series.shape[1]:
            min_p5 = int(np.percentile(exact_trough(ensemble, cfg, start), 5))
        failed = int((ensemble.fail_period >= 0).sum())
        failure_rate = failed / n_runs
        failure_ci = wilson_interval(failed, n_runs)

    return Indicators(
        measured_from_period=start,
        n_runs=n_runs,
        n_pauses=len(windows),
        n_reached_pause=reached_pause,
        pause_entry_median=entry,
        pause_consumption=consumption,
        pause_margin=margin,
        share_below_threshold=share,
        pipeline_at_entry=pipeline,
        min_stock_p5=min_p5,
        failure_rate=failure_rate,
        failure_ci=failure_ci,
        blank_reason=reason if margin is None else None,
    )


def summarise_run(trace: Trace, cfg: ScenarioConfig) -> RunSummary:
    """Figures for the run on the map, kept separate from the ensemble ones."""
    start = min(cfg.warmup_periods, max(trace.n_periods - 1, 0))
    windows = pause_windows(cfg.n_periods, after=start)

    entry = None
    if windows and windows[0][0] < trace.n_periods:
        entry = int(trace.warehouse[windows[0][0]])

    tail = trace.warehouse[start:]

    # The old guard, `trace.n_periods > start`, admitted a span of one period:
    # with the warm-up reaching the end of a trace both ends landed on the same
    # index, the fallback turned a zero span into one day, and the tile reported
    # 0 containers a year for a run that delivered tens of thousands.
    span_days = 0
    if trace.n_periods > start:
        span_days = int(trace.day_index[-1] - trace.day_index[start])
    delivered = None
    if span_days >= 1:
        delivered = (
            float(trace.delivered[-1] - trace.delivered[start]) / span_days * 365.0
        )

    return RunSummary(
        run_index=trace.run_index,
        entry_stock=entry,
        min_stock=int(tail.min()) if tail.size else None,
        measured_days=span_days,
        delivered_per_year=delivered,
        failed=trace.failed,
        fail_period=trace.fail_period,
        fail_deficit=trace.fail_deficit,
    )
