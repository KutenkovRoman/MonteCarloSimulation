"""One-off: clip Natural Earth 110m land to the region the routes cross and vendor it.

Run once with network access; the result is committed so the app never needs the
internet. Source: https://github.com/nvkelso/natural-earth-vector (public domain).

    uv run python tools/vendor_coastline.py /tmp/ne110.json
"""
from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import Sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_LAT = -12.0
RDP_TOLERANCE = 0.35   # degrees; coarse on purpose, the map is continental scale
MIN_RING_POINTS = 6

Point = tuple[float, float]


def _perpendicular_distance(pt: Point, start: Point, end: Point) -> float:
    (x, y), (x1, y1), (x2, y2) = pt, start, end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / (dx * dx + dy * dy) ** 0.5


def rdp(points: Sequence[Point], tolerance: float) -> list[Point]:
    """Ramer-Douglas-Peucker simplification."""
    if len(points) < 3:
        return list(points)
    worst, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _perpendicular_distance(points[i], points[0], points[-1])
        if d > worst:
            worst, index = d, i
    if worst <= tolerance:
        return [points[0], points[-1]]
    left = rdp(points[: index + 1], tolerance)
    right = rdp(points[index:], tolerance)
    return left[:-1] + right


def _clip_ring(ring: Sequence[Sequence[float]]) -> list[Point] | None:
    """Keep a ring only if part of it lies north of MIN_LAT; clamp the rest."""
    if not any(lat >= MIN_LAT for _, lat in ring):
        return None
    clamped = [(float(lon), max(float(lat), MIN_LAT)) for lon, lat in ring]
    simplified = rdp(clamped, RDP_TOLERANCE)
    if len(simplified) < MIN_RING_POINTS:
        return None
    return simplified


def main(src: pathlib.Path, dst: pathlib.Path) -> None:
    data = json.loads(src.read_text())
    rings: list[list[Point]] = []
    for feature in data["features"]:
        geom = feature["geometry"]
        polygons = (
            [geom["coordinates"]]
            if geom["type"] == "Polygon"
            else geom["coordinates"]
        )
        for polygon in polygons:
            outer = _clip_ring(polygon[0])
            if outer is not None:
                rings.append(outer)

    rings.sort(key=len, reverse=True)
    payload = {
        "source": "Natural Earth 110m land (public domain), clipped and simplified",
        "min_lat": MIN_LAT,
        "tolerance_deg": RDP_TOLERANCE,
        "rings": [[[round(lon, 3), round(lat, 3)] for lon, lat in r] for r in rings],
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, separators=(",", ":")))
    total = sum(len(r) for r in rings)
    logger.info(
        "wrote %s: %d rings, %d points, %.0f KB",
        dst, len(rings), total, dst.stat().st_size / 1024,
    )


if __name__ == "__main__":
    source = pathlib.Path(sys.argv[1])
    target = pathlib.Path(__file__).resolve().parent.parent / "simviz" / "data" / "land_north.json"
    main(source, target)
