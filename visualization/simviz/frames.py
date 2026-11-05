"""Flatten a trace into the payload the browser animates from.

Playback used to cost two HTTP round trips per frame, which is a poor way to
drive an animation even on localhost. Instead the whole fleet track is shipped
to the client once and every frame is drawn there. For a two-year run that is
about 2200 x 19 positions, roughly a megabyte of JSON.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simviz.figures import ROUTE_COLORS, date_label, vessel_trace_index
from simviz.geo import PORTS, position_on_leg
from simviz.trace import POINTS, Trace

logger = logging.getLogger(__name__)

__all__ = ["build_frames"]


def build_frames(trace: Trace) -> dict[str, Any]:
    """Positions, loads and labels for every period, as flat JSON-able arrays."""
    n_periods, n_vessels = trace.n_periods, len(trace.vessel_ids)

    # Interpolating every vessel-period in Python would be slow; the distinct
    # (leg, rounded progress) pairs are far fewer, so memoise on those.
    cache: dict[tuple[int, int, int], tuple[float, float]] = {}
    xs = np.empty(n_periods * n_vessels, dtype=np.float64)
    ys = np.empty(n_periods * n_vessels, dtype=np.float64)

    for t in range(n_periods):
        base = t * n_vessels
        for i in range(n_vessels):
            src, dst = int(trace.v_from[t, i]), int(trace.v_to[t, i])
            step = int(round(float(trace.v_progress[t, i]) * 500))
            key = (src, dst, step)
            spot = cache.get(key)
            if spot is None:
                spot = position_on_leg(POINTS[src], POINTS[dst], step / 500.0)
                cache[key] = spot
            xs[base + i], ys[base + i] = spot

    return {
        "n": n_periods,
        "vesselTrace": vessel_trace_index(),
        "ids": list(trace.vessel_ids),
        "capacity": trace.v_capacity.astype(int).tolist(),
        "colors": [ROUTE_COLORS[vid[-1]] for vid in trace.vessel_ids],
        "points": [PORTS[code].name for code in POINTS],
        # rounded in float64: rounding a float32 leaves 86.18000030517578
        # in the JSON and triples the payload
        "x": np.round(xs, 2).tolist(),
        "y": np.round(ys, 2).tolist(),
        "loaded": trace.v_loaded.reshape(-1).astype(int).tolist(),
        "src": trace.v_from.reshape(-1).astype(int).tolist(),
        "dst": trace.v_to.reshape(-1).astype(int).tolist(),
        "labels": [date_label(trace, t) for t in range(n_periods)],
        "failPeriod": -1 if trace.fail_period is None else int(trace.fail_period),
    }
