"""Drive the simulation from outside and collect traces.

``SimulationEnvironment.simulate_period()`` is public and ``simulate()`` only
loops over it after assigning ``env.rng``, so the whole run can be stepped from
here without patching the model.
"""
from __future__ import annotations

import atexit
import dataclasses
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from simviz.config import PERIODS_IN_DAY, ScenarioConfig
from simviz.model_loader import load_model
from simviz.trace import POINT_INDEX, STOCK_BUCKETS, Trace, leg_endpoints, read_stocks

logger = logging.getLogger(__name__)

__all__ = [
    "Ensemble",
    "WORKERS_ENV_VAR",
    "run_ensemble",
    "run_trace",
    "shutdown_pool",
    "spawn_seeds",
]

#: Cap the worker pool from outside. Needed on a shared machine: the default is
#: one worker per core, which on a box already running at three times its core
#: count makes every run slower for everyone, this one included.
WORKERS_ENV_VAR = "SIMVIZ_WORKERS"

_pool: ProcessPoolExecutor | None = None


def spawn_seeds(seed: int, count: int) -> list[np.random.SeedSequence]:
    """Independent seeds; ``SeedSequence.spawn`` avoids the duplicates that
    ``rng.integers`` produced upstream (53 collisions in 10000 draws)."""
    return np.random.SeedSequence(seed).spawn(count)


def _new_environment(cfg: ScenarioConfig, seed: np.random.SeedSequence):
    pm = load_model()
    env = pm.SimulationEnvironment(cfg.to_model_config())
    pm.setup_fleet(env)
    # seed_rngs splits the physical stream from the triangulation controller's;
    # assigning env.rng by hand leaves ctrl_rng unset and the model says so loudly
    env.seed_rngs(seed)                        # what simulate() does internally
    return env


def run_trace(cfg: ScenarioConfig, seed: np.random.SeedSequence, run_index: int = 0) -> Trace:
    """Step one run to the end of the horizon (or to its failure) and record it."""
    env = _new_environment(cfg, seed)
    vessels = list(env.feeder_vessels) + list(env.mainland_vessels)
    n_v, horizon = len(vessels), cfg.n_periods

    month = np.zeros(horizon, np.int8)
    day = np.zeros(horizon, np.int8)
    stocks = {name: np.zeros(horizon, np.int32) for name in STOCK_BUCKETS}
    delivered = np.zeros(horizon, np.int32)
    v_state = np.zeros((horizon, n_v), np.int8)
    v_loaded = np.zeros((horizon, n_v), np.int32)
    v_from = np.zeros((horizon, n_v), np.int8)
    v_to = np.zeros((horizon, n_v), np.int8)
    v_progress = np.zeros((horizon, n_v), np.float32)

    prev_time = [v.transition_time for v in vessels]
    leg_total = [max(t, 1) for t in prev_time]

    failed, fail_period, fail_deficit, worst_error = False, None, 0, 0
    recorded = 0

    for t in range(horizon):
        if env.simulate_period() != 0:
            failed, fail_period = True, t
            fail_deficit = max(-int(env.n_unloaded_at_src), 0)
            break

        month[t], day[t] = env.month, env.day
        total = 0
        for name, value in read_stocks(env).items():
            stocks[name][t] = value
            total += value
        worst_error = max(worst_error, abs(total - cfg.n_containers))
        delivered[t] = env.n_delivered_A + env.n_delivered_B + env.n_delivered_C

        for i, vessel in enumerate(vessels):
            left = vessel.transition_time
            if left > prev_time[i] - 1:        # counter went up: a new leg began
                leg_total[i] = max(left, 1)
            prev_time[i] = left

            origin, target = leg_endpoints(vessel.id, vessel.state.name)
            v_state[t, i] = vessel.state.value
            v_loaded[t, i] = vessel.n_loaded
            v_from[t, i] = POINT_INDEX[origin]
            v_to[t, i] = POINT_INDEX[target]
            v_progress[t, i] = (
                0.0 if origin == target
                else float(np.clip(1.0 - left / leg_total[i], 0.0, 1.0))
            )
        recorded = t + 1

    if worst_error:
        logger.warning(
            "container balance off by %d: the bucket split in read_stocks() is "
            "incomplete, most likely because the model gained a new holding place",
            worst_error,
        )

    cut = slice(0, recorded)
    return Trace(
        config=cfg,
        run_index=run_index,
        month=month[cut], day=day[cut],
        day_index=np.arange(recorded, dtype=np.int32) // PERIODS_IN_DAY,
        stocks={name: values[cut] for name, values in stocks.items()},
        delivered=delivered[cut],
        vessel_ids=tuple(v.id for v in vessels),
        v_state=v_state[cut], v_loaded=v_loaded[cut],
        v_capacity=np.array([v.max_capacity for v in vessels], np.int32),
        v_from=v_from[cut], v_to=v_to[cut], v_progress=v_progress[cut],
        failed=failed, fail_period=fail_period, fail_deficit=fail_deficit,
        conservation_error=int(worst_error),
    )


def _warehouse_series(
    args: tuple[ScenarioConfig, np.random.SeedSequence],
) -> tuple[np.ndarray, int]:
    """Warehouse stock per period, and the period the run died on (-1 if it lived).

    After a failure the series is left at zero: the warehouse is empty and
    production has stopped, which is the honest floor. Percentiles taken over
    survivors only sit up to 1800 containers too high once runs drop out.
    """
    cfg, seed = args
    env = _new_environment(cfg, seed)
    out = np.zeros(cfg.n_periods, np.int32)
    for t in range(cfg.n_periods):
        if env.simulate_period() != 0:
            return out, t
        out[t] = env.n_unloaded_at_src
    return out, -1


@dataclasses.dataclass(frozen=True)
class Ensemble:
    """Warehouse trajectories for a whole batch of seeds."""

    series: np.ndarray        # (N, T) int32, zero from the failure period on
    fail_period: np.ndarray   # (N,)   int32, -1 for runs that finished
    alive: np.ndarray         # (T,)   int32, runs still going at that period

    @property
    def n_runs(self) -> int:
        return int(self.series.shape[0])

    @property
    def n_failed(self) -> int:
        return int((self.fail_period >= 0).sum())

    def percentile(self, q: float) -> np.ndarray:
        """Percentile over *all* runs, dead ones pinned at zero."""
        return np.percentile(self.series, q, axis=0)


def _get_pool() -> ProcessPoolExecutor:
    """Reused pool of workers.

    ``forkserver`` where it exists: ``spawn`` re-imports the parent's
    ``__main__``, which breaks whenever the caller is a REPL, a notebook or a
    heredoc, and it pays the cost of importing the model in every worker.

    Windows has no ``forkserver`` at all -- ``get_all_start_methods()`` there
    returns ``['spawn']`` -- so fall back rather than raise. Under spawn the
    pool is still a module-level singleton, so the import cost is paid once,
    and anything driving ``run_ensemble`` needs an ``if __name__ ==
    "__main__"`` guard.
    """
    global _pool
    if _pool is None:
        requested = os.environ.get(WORKERS_ENV_VAR)
        workers = max(int(requested), 1) if requested else None
        available = multiprocessing.get_all_start_methods()
        method = "forkserver" if "forkserver" in available else "spawn"
        context = multiprocessing.get_context(method)
        if method == "forkserver":
            # Preload the package, not the submodule: importing "simviz.runner"
            # directly re-enters simviz/__init__.py mid-import and the worker
            # dies at startup with a partially initialised module.
            context.set_forkserver_preload(["simviz"])
        _pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        logger.info(
            "started worker pool (%s workers, %d cpus, %s)", workers or "default", os.cpu_count(), method
        )
    return _pool


def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


atexit.register(shutdown_pool)


def run_ensemble(cfg: ScenarioConfig) -> Ensemble:
    """Run ``cfg.n_fan_runs`` independent seeds and collect the warehouse fan."""
    payload = [(cfg, s) for s in spawn_seeds(cfg.seed, cfg.n_fan_runs)]
    if cfg.n_fan_runs <= 4:
        results = [_warehouse_series(a) for a in payload]
    else:
        results = list(_get_pool().map(_warehouse_series, payload, chunksize=4))

    series = np.vstack([row for row, _ in results])
    fail_period = np.array([fp for _, fp in results], np.int32)

    horizon = series.shape[1]
    steps = np.arange(horizon)
    still_running = (fail_period[:, None] < 0) | (steps[None, :] < fail_period[:, None])
    alive = still_running.sum(axis=0).astype(np.int32)

    return Ensemble(series=series, fail_period=fail_period, alive=alive)
