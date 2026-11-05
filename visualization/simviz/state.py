"""Server-side cache of computed scenarios.

Traces are numpy arrays of a few megabytes; round-tripping them through a
browser store would cost more than recomputing them. The app is a single-user
local demo, so results live here and only a key travels to the client.

The ensemble and the individual traces are cached separately on purpose:
flipping through runs on the map must not re-run the fan, which costs seconds.
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import time

from simviz.config import ScenarioConfig
from simviz.indicators import Indicators, compute_indicators
from simviz.runner import Ensemble, run_ensemble, run_trace, spawn_seeds
from simviz.trace import Trace

logger = logging.getLogger(__name__)

__all__ = ["Result", "failed_run_indices", "get_result", "get_trace"]

_RESULT_CACHE_SIZE = 8
_TRACE_CACHE_SIZE = 24

_results: "collections.OrderedDict[ScenarioConfig, Result]" = collections.OrderedDict()
_traces: "collections.OrderedDict[tuple[ScenarioConfig, int], Trace]" = (
    collections.OrderedDict()
)


@dataclasses.dataclass(frozen=True)
class Result:
    """One evaluated scenario: the fan and the figures that come off it."""

    config: ScenarioConfig
    ensemble: Ensemble | None
    indicators: Indicators
    seconds: float


def _remember(cache: collections.OrderedDict, key, value, limit: int):
    cache[key] = value
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def get_result(cfg: ScenarioConfig) -> Result:
    """Run the ensemble for a scenario, reusing the cached one when unchanged."""
    cached = _results.get(cfg)
    if cached is not None:
        _results.move_to_end(cfg)
        return cached

    started = time.perf_counter()
    # Always, even for a single run. Skipping it left `ensemble is None`, and
    # then the failure tile, the "jump to a failed run" button and the pause
    # tile each contradicted the run drawn on the map.
    ensemble = run_ensemble(cfg)
    result = Result(
        config=cfg,
        ensemble=ensemble,
        indicators=compute_indicators(ensemble, cfg),
        seconds=time.perf_counter() - started,
    )
    logger.info(
        "ensemble computed in %.2f s: %d containers, %d runs, %s остановок",
        result.seconds, cfg.n_containers, cfg.n_fan_runs,
        "—" if ensemble is None else ensemble.n_failed,
    )
    return _remember(_results, cfg, result, _RESULT_CACHE_SIZE)


def get_trace(cfg: ScenarioConfig, run_index: int) -> Trace:
    """Record one member of the ensemble in full, for the map.

    ``SeedSequence(seed).spawn(n)`` numbers its children 0, 1, ... regardless of
    ``n``, so run ``k`` here is the same run as row ``k`` of the ensemble.
    """
    total = max(cfg.n_fan_runs, 1)
    index = min(max(int(run_index), 0), total - 1)

    key = (cfg, index)
    cached = _traces.get(key)
    if cached is not None:
        _traces.move_to_end(key)
        return cached

    trace = run_trace(cfg, spawn_seeds(cfg.seed, total)[index], run_index=index)
    return _remember(_traces, key, trace, _TRACE_CACHE_SIZE)


def failed_run_indices(result: Result) -> list[int]:
    """Indices of the runs where production stopped, in order."""
    if result.ensemble is None:
        return []
    return [int(i) for i in (result.ensemble.fail_period >= 0).nonzero()[0]]
