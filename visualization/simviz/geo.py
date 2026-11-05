"""Map geometry: ports, sea routes, and where a vessel sits along its leg.

The model knows nothing about geography -- only about voyage durations. The
routes here are hand-drawn sea lanes consistent with those durations: 41 days
Murmansk to China and 40 from St Petersburg put the mainland vessels through
Suez, and 25 days Dudinka to China puts route C on the Northern Sea Route.
Route C's backup return goes through Suez as well: at 42 to 49 days it is slower
than the mainlands' own Suez run, and far too slow for the 23-to-28-day NSR.
The model reaches for it once the rest of the window is shorter than an NSR
voyage home, which can be four weeks before the window's 10 November close.

Longitudes are carried in an extended frame where the Bering Strait and
Chukotka sit past 180 rather than jumping to -180, so a polyline crossing the
antimeridian stays continuous.
"""
from __future__ import annotations

import functools
import json
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "LON_MAX",
    "LON_MIN",
    "PORTS",
    "ROUTES",
    "coastline_rings",
    "point_xy",
    "position_on_leg",
    "route_polyline",
    "unwrap_lon",
]

#: Where the map is cut. Anything west of LON_MIN is drawn shifted by +360.
LON_MIN = -22.0
LON_MAX = 205.0

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "land_north.json"


class Port:
    """A named point on the map."""

    __slots__ = ("code", "name", "lon", "lat")

    def __init__(self, code: str, name: str, lon: float, lat: float) -> None:
        self.code, self.name, self.lon, self.lat = code, name, lon, lat


PORTS: dict[str, Port] = {
    "S": Port("S", "Дудинка", 86.18, 69.41),
    "A": Port("A", "Мурманск", 33.08, 68.97),
    "B": Port("B", "Архангельск", 40.54, 64.54),
    "P": Port("P", "Санкт-Петербург", 30.24, 59.89),
    # The model says only "Китай"; the customer named Lianyungang, so the marker
    # sits there. Moving it moves the marker and the lane endpoints below, never
    # the simulated durations, which the model keeps to itself.
    "D": Port("D", "Китай (Льянюнган)", 119.6, 34.8),
}

# --- shared segments -------------------------------------------------------

_KARA_GATE = [                       # Dudinka -> Yenisei gulf -> Kara Gate
    (86.18, 69.41), (83.5, 71.5), (79.0, 72.8), (68.0, 72.5),
    (58.0, 70.8), (57.5, 70.4), (52.0, 70.0),
]

_NORTH_SEA_TO_CHINA = [              # North Sea -> Suez -> Malacca -> Lianyungang
    (2.0, 57.0), (2.5, 51.5), (-1.5, 50.0), (-5.0, 48.5), (-9.5, 44.0),
    (-9.5, 38.0), (-6.5, 36.0), (-2.0, 36.0), (5.0, 37.5), (13.0, 37.0),
    (22.0, 34.0), (30.0, 32.0), (32.5, 30.0), (33.5, 27.5), (38.0, 21.0),
    (42.0, 15.0), (43.5, 12.6), (48.0, 12.0), (55.0, 11.0), (65.0, 8.0),
    (76.0, 6.0), (85.0, 5.5), (95.0, 5.5), (99.0, 3.5), (103.5, 1.3),
    (106.0, 3.0), (110.0, 10.0), (114.0, 18.0), (118.0, 23.0), (120.0, 26.0),
    (122.0, 30.0), (122.5, 33.0), (122.5, 34.4), (119.6, 34.8),
]

_NSR_TO_CHINA = [                    # Kara Sea -> Vilkitsky -> Bering -> Yellow Sea
    (82.0, 74.5), (90.0, 77.0), (100.0, 77.8), (110.0, 76.5), (125.0, 76.0),
    (140.0, 75.5), (155.0, 73.0), (170.0, 70.5), (185.0, 67.5), (191.0, 65.8),
    (188.0, 60.0), (180.0, 54.0), (168.0, 48.0), (155.0, 42.0), (145.0, 38.0),
    # south of Kyushu, then up the East China Sea: the old straight run from
    # (139, 33) to (132, 32) clipped the island
    (139.0, 33.0), (135.0, 32.0), (131.5, 30.5), (128.5, 31.5),
    (125.5, 33.0), (122.5, 34.4), (119.6, 34.8),
]

#: Sea lanes, keyed by ``(from, to)``. The reverse direction reuses the same
#: line read backwards.
ROUTES: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("S", "A"): _KARA_GATE + [(42.0, 69.8), (34.5, 69.4), (33.08, 68.97)],
    ("S", "B"): _KARA_GATE + [
        (44.0, 69.3), (40.5, 68.3), (39.5, 66.5), (40.0, 65.2), (40.54, 64.54),
    ],
    # Northern Sea Route: Vilkitsky Strait, Laptev, East Siberian, Chukchi,
    # Bering, then down the Pacific side of Japan into the Yellow Sea.
    ("S", "D"): [(86.18, 69.41), (83.5, 71.5)] + _NSR_TO_CHINA,
    # China -> Murmansk carries two different vessels over one lane: every
    # mainland A vessel on its way home, and a route C feeder left with too
    # little NSR window to sail home through it (the model credits
    # ``interm_transport_handler.n_unloaded_at_A``, whatever its log string
    # says). Deliberately no ("D", "A") key -- ``route_polyline`` reverses this
    # one. Giving the feeder its own Arctic lane also caught the mainlands,
    # which the model puts in the same state on the same leg, and sailed them
    # home over the pole from the first return of year 1 onward -- late April
    # at the default seed, though the date is sampled, not fixed.
    ("A", "D"): [
        (33.08, 68.97), (28.0, 70.5), (18.0, 70.0), (8.0, 65.0), (4.0, 61.0),
    ] + _NORTH_SEA_TO_CHINA,
    ("P", "D"): [
        (30.24, 59.89), (27.0, 59.9), (22.5, 59.5), (19.0, 58.5), (17.5, 56.0),
        (14.5, 55.2), (11.0, 55.3), (10.5, 56.5), (10.0, 57.6), (7.0, 57.8),
        (4.0, 57.0),
    ] + _NORTH_SEA_TO_CHINA,
}


def unwrap_lon(lon: float) -> float:
    """Move a western-hemisphere longitude past 180 so lines stay continuous."""
    return lon + 360.0 if lon < LON_MIN else lon


def point_xy(code: str) -> tuple[float, float]:
    port = PORTS[code]
    return unwrap_lon(port.lon), port.lat


@functools.lru_cache(maxsize=None)
def route_polyline(origin: str, target: str) -> np.ndarray:
    """The sea lane between two ports as an ``(n, 2)`` array of ``(lon, lat)``."""
    if (origin, target) in ROUTES:
        pts = ROUTES[(origin, target)]
    elif (target, origin) in ROUTES:
        pts = list(reversed(ROUTES[(target, origin)]))
    else:
        pts = [
            (PORTS[origin].lon, PORTS[origin].lat),
            (PORTS[target].lon, PORTS[target].lat),
        ]
        logger.warning("no drawn route for %s->%s, using a straight line", origin, target)
    return np.array([(unwrap_lon(lon), lat) for lon, lat in pts], dtype=float)


@functools.lru_cache(maxsize=None)
def _cumulative(origin: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Polyline plus the fraction of total length reached at each vertex."""
    line = route_polyline(origin, target)
    mean_lat = np.radians((line[:-1, 1] + line[1:, 1]) / 2.0)
    dx = np.diff(line[:, 0]) * np.cos(mean_lat)
    dy = np.diff(line[:, 1])
    steps = np.hypot(dx, dy)
    total = steps.sum()
    fractions = np.concatenate([[0.0], np.cumsum(steps) / (total or 1.0)])
    return line, fractions


def position_on_leg(origin: str, target: str, progress: float) -> tuple[float, float]:
    """Interpolate along the sea lane by great-circle-ish distance, not by vertex."""
    if origin == target:
        return point_xy(origin)
    line, fractions = _cumulative(origin, target)
    t = min(max(progress, 0.0), 1.0)
    i = int(np.searchsorted(fractions, t, side="right")) - 1
    i = min(max(i, 0), len(fractions) - 2)
    span = fractions[i + 1] - fractions[i]
    local = 0.0 if span <= 0 else (t - fractions[i]) / span
    lon = line[i, 0] + local * (line[i + 1, 0] - line[i, 0])
    lat = line[i, 1] + local * (line[i + 1, 1] - line[i, 1])
    return float(lon), float(lat)


@functools.lru_cache(maxsize=1)
def coastline_rings() -> list[np.ndarray]:
    """Vendored Natural Earth land, duplicated across the antimeridian."""
    payload = json.loads(_DATA.read_text())
    rings: list[np.ndarray] = []
    for ring in payload["rings"]:
        arr = np.array(ring, dtype=float)
        rings.append(arr)
        if arr[:, 0].min() < LON_MIN:                 # e.g. Chukotka, Alaska
            shifted = arr.copy()
            shifted[:, 0] += 360.0
            rings.append(shifted)
    return rings
