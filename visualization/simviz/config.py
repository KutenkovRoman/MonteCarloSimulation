"""Scenario configuration exposed to the interface.

Only the fields the customer may safely change are here. Deliberately absent:

* ``periods_in_day`` -- voyage lengths and loading rates are hardwired in
  8-hour periods and do not rescale (problem 13 of the code review).
* the start date -- a start later than the end of the interflood pause or the
  opening of the NSR window makes ``setup_fleet`` produce negative waiting
  times and the whole fleet departs at once (problem 14).

Both are pinned to the values the model was calibrated with.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

from simviz.model_loader import load_model

logger = logging.getLogger(__name__)

__all__ = [
    "ALT_PRODUCTION_SHAPE",
    "FIXED_CALENDAR",
    "FIXED_POLICY",
    "PERIODS_IN_DAY",
    "ScenarioConfig",
    "calendar_point",
    "calendar_track",
    "daily_production",
    "month_index",
    "month_names_ru",
]

PERIODS_IN_DAY = 3
DAYS_PER_YEAR = 365

FIXED_CALENDAR: dict[str, Any] = {
    "start_month": "February",
    "start_day": 2,
    "interflood_start": ("May", 10),
    "interflood_end": ("June", 20),
    "alt_route_start": ("July", 1),
    "alt_route_end": ("November", 10),
    "resample_interflood_period": False,
    "resample_alt_route_period": False,
}

# Departure policy the model was calibrated with. With disable_underloaded_I2S
# left at True everything collapses once the interflood pause ends.
FIXED_POLICY: dict[str, Any] = {
    "disable_underloaded_I2S": False,
    "underload_share_I2D": 0.85,
    "underload_share_S2I": 0.50,
    "underload_share_S2D": 0.85,
}

# Seasonal shape of route C production, in the model's own month order, which
# starts at DECEMBER: these ones are May through October. Do not re-sort.
ALT_PRODUCTION_SHAPE: tuple[int, ...] = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0)

#: Containers already loaded at Dudinka when a run starts, per mainland route.
#: Scaled down for a tiny fleet: the model rejects any config where the initial
#: load exceeds the fleet, and the UI's own floor of 1000 is below 600 + 600.
#: At any realistic fleet size this is exactly 600 and changes nothing.
_INITIAL_LOADED_PER_ROUTE = 600


def _initial_loaded(n_containers: int) -> int:
    return min(_INITIAL_LOADED_PER_ROUTE, max(n_containers // 4, 0))

_MONTHS_RU = {
    "December": "декабря", "January": "января", "February": "февраля",
    "March": "марта", "April": "апреля", "May": "мая", "June": "июня",
    "July": "июля", "August": "августа", "September": "сентября",
    "October": "октября", "November": "ноября",
}


def month_index(name: str) -> int:
    """Index of a month in the model's calendar, which begins at December."""
    return load_model().MONTH_NAMES.index(name)


def month_names_ru() -> tuple[str, ...]:
    """Genitive Russian month names in the model's own order."""
    return tuple(_MONTHS_RU[name] for name in load_model().MONTH_NAMES)


def calendar_point(key: str) -> tuple[int, int]:
    """A ``FIXED_CALENDAR`` boundary as ``(month index, day)``."""
    name, day = FIXED_CALENDAR[key]
    return month_index(name), day


def calendar_track(n_periods: int) -> list[tuple[int, int]]:
    """``(month index, day)`` for every period of the horizon.

    Derived from the calendar rather than from a trace: a run that fails stops
    early, and the indicators still need to know when its pause began.
    """
    pm = load_model()
    month = month_index(FIXED_CALENDAR["start_month"])
    day = FIXED_CALENDAR["start_day"]
    period = 0
    track: list[tuple[int, int]] = []
    for _ in range(n_periods):
        period += 1
        if period == PERIODS_IN_DAY:
            period = 0
            day += 1
            if day > pm.MONTH_DAYS[month]:
                month = (month + 1) % 12
                day = 1
        track.append((month, day))
    return track


def daily_production(cfg: "ScenarioConfig") -> np.ndarray:
    """Containers charged to the Dudinka warehouse at each period, 0 elsewhere.

    Reproduces ``SrcTransportHandler.handle`` exactly, including its integer
    split of a monthly figure across the days of the month. Production is
    deterministic, so this is not an estimate: it lets an outside observer
    reconstruct the value the model actually tests for failure, which is the
    stock at the previous period boundary minus that day's charge, before the
    period's returning empties are added back.
    """
    pm = load_model()
    model = cfg.to_model_config()
    track = calendar_track(cfg.n_periods)
    start = (month_index(FIXED_CALENDAR["start_month"]), FIXED_CALENDAR["start_day"])

    loads = np.zeros(cfg.n_periods, dtype=np.int64)
    for period in range(0, cfg.n_periods, PERIODS_IN_DAY):
        month, day = start if period == 0 else track[period - 1]
        days_in_month = pm.MONTH_DAYS[month]
        charge = 0
        for monthly in (model.main_monthly_prod_rate[month],
                        model.alt_monthly_prod_rate[month]):
            daily = monthly // days_in_month
            if day + (monthly - days_in_month * daily) > days_in_month:
                daily += 1
            charge += daily
        loads[period] = charge
    return loads


@dataclasses.dataclass(frozen=True)
class ScenarioConfig:
    """What the interface lets a user change, in the units they think in."""

    n_containers: int = 29_500

    n_feeders_A: int = 3
    n_feeders_B: int = 2
    n_feeders_C: int = 4
    n_mainlands_A: int = 6
    n_mainlands_B: int = 4

    main_monthly_rate: int = 3_000     # containers per month, mainland routes
    alt_annual_total: int = 4_000      # containers per year, route C (NSR)

    horizon_years: float = 4.0
    warmup_years: float = 1.0          # shaded in the UI, excluded from the KPIs

    seed: int = 42
    n_fan_runs: int = 200

    #: Nested sub-simulations per rental decision. 0 turns triangulation off,
    #: which is the default because one run then costs 0.08 s instead of 3.4 and
    #: the screens have to stay interactive. Only raise it to 1: measured, 1, 3
    #: and 10 give byte-identical results because the probe's horizon sits in
    #: the model's deterministic zone.
    triangulation_n_sims: int = 0

    def __post_init__(self) -> None:
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive")
        if self.warmup_years >= self.horizon_years:
            logger.warning(
                "warm-up (%.1f y) covers the whole horizon (%.1f y); KPIs will be empty",
                self.warmup_years, self.horizon_years,
            )

    @property
    def n_periods(self) -> int:
        return int(round(self.horizon_years * DAYS_PER_YEAR * PERIODS_IN_DAY))

    @property
    def warmup_periods(self) -> int:
        return int(round(self.warmup_years * DAYS_PER_YEAR * PERIODS_IN_DAY))

    @property
    def alt_monthly_rate(self) -> list[int]:
        """Spread the annual route-C total over its season, keeping the total exact."""
        active = sum(ALT_PRODUCTION_SHAPE)
        if active == 0:
            return [0] * 12
        base, remainder = divmod(self.alt_annual_total, active)
        rates, given = [], 0
        for flag in ALT_PRODUCTION_SHAPE:
            if not flag:
                rates.append(0)
                continue
            extra = 1 if given < remainder else 0
            rates.append(base + extra)
            given += extra
        return rates

    def to_model_config(self) -> Any:
        """Build the model's own ``SimulationConfig``."""
        pm = load_model()
        return pm.SimulationConfig(
            simulation_period=self.n_periods,
            periods_in_day=PERIODS_IN_DAY,
            n_containers=self.n_containers,
            n_loaded_A=_initial_loaded(self.n_containers),
            n_loaded_B=_initial_loaded(self.n_containers),
            main_monthly_prod_rate=[self.main_monthly_rate] * 12,
            alt_monthly_prod_rate=self.alt_monthly_rate,
            # upstream added this on 22-09-2026 and gave it no default: a list of
            # (days from now, how many) containers arriving from outside the loop.
            # None keeps the closed park this app is built around.
            replenishments=None,
            n_feeders_A=self.n_feeders_A,
            n_feeders_B=self.n_feeders_B,
            n_feeders_C=self.n_feeders_C,
            n_mainlands_A=self.n_mainlands_A,
            n_mainlands_B=self.n_mainlands_B,
            triangulation_n_sims=self.triangulation_n_sims,
            **FIXED_CALENDAR,
            **FIXED_POLICY,
        )
