"""Per-period recording of the simulation state.

The model is never modified. Everything here is read off the public attributes
of ``SimulationEnvironment`` after each ``simulate_period()`` call:

* twelve stock buckets that together account for every container -- the sum is
  asserted against ``n_containers`` on every period, which is the conservation
  test the model itself does not have;
* one row per vessel per period with its leg endpoints and progress along the
  leg, reconstructed from ``transition_time`` alone (when the counter goes up a
  new leg started, and its value at that moment is the leg's full length).

``n_delivered_A/B/C`` are cumulative flow counters, not stocks: delivered
containers come back as empties in China and would be counted twice. They are
tracked separately from the balance.
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

from simviz.config import ScenarioConfig

logger = logging.getLogger(__name__)

__all__ = [
    "POINTS",
    "POINT_INDEX",
    "STOCK_BUCKETS",
    "Trace",
    "leg_endpoints",
    "read_stocks",
]

POINTS: tuple[str, ...] = ("S", "A", "B", "P", "D")
POINT_INDEX: dict[str, int] = {code: i for i, code in enumerate(POINTS)}

#: Display order of the stock buckets, from Dudinka outward.
STOCK_BUCKETS: tuple[str, ...] = (
    "склад Дудинка",
    "порт Дудинка",
    "фидеры A",
    "фидеры B",
    "фидеры C",
    "Мурманск",
    "Архангельск",
    "Петербург",
    "суша РФ",
    "магистрали",
    "Китай",
    # containers out on rent sit in neither a port nor a vessel: without this
    # bucket the balance is short by up to a third of the fleet once
    # triangulation is on
    "у арендаторов",
)


def read_stocks(env) -> dict[str, int]:
    """Split every container in the system into the eleven display buckets."""
    src, dest, interm = (
        env.src_transport_handler,
        env.dest_transport_handler,
        env.interm_transport_handler,
    )
    by_route = {"A": 0, "B": 0, "C": 0}
    for vessel in env.feeder_vessels:
        by_route[vessel.id[-1]] += vessel.n_loaded

    return {
        "склад Дудинка": env.n_unloaded_at_src,
        "порт Дудинка": (
            env.n_unloaded_at_src_port
            + env.n_loaded_at_src_port_A
            + env.n_loaded_at_src_port_B
            + env.n_loaded_at_src_port_C
            + src.n_unloaded_at_src_port
            + sum(src.schedule_A) + sum(src.schedule_B)
            + sum(src.schedule_C) + sum(src.schedule_unloaded)
        ),
        "фидеры A": by_route["A"],
        "фидеры B": by_route["B"],
        "фидеры C": by_route["C"],
        "Мурманск": (
            env.n_unloaded_at_A + env.n_loaded_at_A
            + interm.n_loaded_at_A + interm.n_unloaded_at_A
        ),
        "Архангельск": env.n_unloaded_at_B + env.n_loaded_at_B + interm.n_loaded_at_B,
        "Петербург": env.n_unloaded_at_P + env.n_loaded_at_P + interm.n_unloaded_at_P,
        "суша РФ": sum(interm.schedule_loaded) + sum(interm.schedule_unloaded),
        "магистрали": sum(v.n_loaded for v in env.mainland_vessels),
        "Китай": (
            env.n_unloaded_at_dest + env.n_unloaded_at_dest_port
            + env.n_loaded_at_dest_port_A + env.n_loaded_at_dest_port_B
            + env.n_loaded_at_dest_port_C
            + dest.loaded_A + dest.loaded_B + dest.loaded_C
            + sum(dest.schedule_A) + sum(dest.schedule_B)
            + sum(dest.schedule_C) + sum(dest.schedule_unloaded)
        ),
        "у арендаторов": sum(n for _, _, n in env.delayed_returns),
    }


def leg_endpoints(vessel_id: str, state_name: str) -> tuple[str, str]:
    """Where a vessel is, as ``(from, to)``.

    Both entries are the same point when the vessel is in port. Route C's
    backup return runs through Murmansk: the vessel writes into
    ``interm_transport_handler.n_unloaded_at_A`` even though its own log
    string labels that port 'B' (problem 17 of the code review).
    """
    route = vessel_id[-1]
    is_feeder = vessel_id.startswith("feeder")

    if is_feeder:
        interm = "A" if route == "C" else route
    else:
        interm = "A" if route == "A" else "P"

    if state_name.startswith("SHIPPING_FROM_"):
        origin, _, target = state_name[len("SHIPPING_FROM_"):].partition("_TO_")
        resolve = {"SRC": "S", "INTERM": interm, "DEST": "D"}
        return resolve[origin], resolve[target]

    for suffix, point in (("_AT_SRC", "S"), ("_AT_INTERM", interm), ("_AT_DEST", "D")):
        if state_name.endswith(suffix):
            return point, point

    raise ValueError(f"unhandled vessel state {state_name!r}")


@dataclasses.dataclass(frozen=True)
class Trace:
    """One recorded run."""

    config: ScenarioConfig
    run_index: int             # position in the ensemble, NOT the seed value

    month: np.ndarray          # (T,) int8, in the model's order: 0 = December
    day: np.ndarray            # (T,) int8
    day_index: np.ndarray      # (T,) int32, days since the start

    stocks: dict[str, np.ndarray]     # bucket -> (T,) int32
    delivered: np.ndarray             # (T,) int32, cumulative, all routes

    vessel_ids: tuple[str, ...]
    v_state: np.ndarray        # (T, V) int8, VesselState.value
    v_loaded: np.ndarray       # (T, V) int32
    v_capacity: np.ndarray     # (V,)   int32
    v_from: np.ndarray         # (T, V) int8, index into POINTS
    v_to: np.ndarray           # (T, V) int8, index into POINTS
    v_progress: np.ndarray     # (T, V) float32, 0..1 along the leg

    failed: bool
    fail_period: int | None
    fail_deficit: int
    conservation_error: int

    @property
    def n_periods(self) -> int:
        return int(self.month.shape[0])

    @property
    def warehouse(self) -> np.ndarray:
        return self.stocks["склад Дудинка"]

    def period_of(self, month: int, day: int, after_period: int = 0) -> int | None:
        """First period on the given calendar date at or after ``after_period``."""
        hits = np.nonzero(
            (self.month[after_period:] == month) & (self.day[after_period:] == day)
        )[0]
        return int(hits[0]) + after_period if hits.size else None
