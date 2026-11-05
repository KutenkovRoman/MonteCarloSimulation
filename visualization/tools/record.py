"""Build a self-contained animation of two runs: one that lives, one that dies."""
import io, json, sys
sys.path.insert(0, "/Users/vladimiraletov/Projects/ports_containers/simviz")
import numpy as np
from simviz.config import ScenarioConfig, month_names_ru
from simviz.runner import run_trace, spawn_seeds
from simviz.frames import build_frames
from simviz.figures import COLORS, ROUTE_COLORS, _LEG_STYLE, _PORT_LABEL_SIDE
from simviz.geo import PORTS, LON_MIN, LON_MAX, coastline_rings, route_polyline, point_xy
from simviz.trace import POINTS

PARK, YEARS, RUNS = 28_500, 2.0, (0, 2)
cfg = ScenarioConfig(n_containers=PARK, horizon_years=YEARS, warmup_years=0.0, n_fan_runs=40)
seeds = spawn_seeds(cfg.seed, cfg.n_fan_runs)
ru = month_names_ru()

payload = []
for idx in RUNS:
    tr = run_trace(cfg, seeds[idx], run_index=idx)
    fr = build_frames(tr)
    wh = tr.warehouse.astype(int).tolist()
    payload.append({
        "index": int(idx),
        "failed": bool(tr.failed),
        "failPeriod": None if tr.fail_period is None else int(tr.fail_period),
        "failDeficit": int(tr.fail_deficit),
        "n": int(fr["n"]),
        "x": fr["x"], "y": fr["y"], "loaded": fr["loaded"],
        "src": fr["src"], "dst": fr["dst"],
        "labels": fr["labels"], "ids": fr["ids"],
        "capacity": fr["capacity"], "colors": fr["colors"],
        "warehouse": wh,
        "minStock": int(tr.warehouse.min()),
        "delivered": int(tr.delivered[-1]),
        "conservation": int(tr.conservation_error),
    })
    print(f"  прогон #{idx}: периодов {fr['n']}, "
          f"{'ПРОВАЛ на ' + tr.__class__.__name__ if False else ''}"
          f"{'провал, период ' + str(tr.fail_period) if tr.failed else 'дошёл до конца'}, "
          f"мин. склада {tr.warehouse.min()}, доставлено {tr.delivered[-1]}, "
          f"баланс {tr.conservation_error}")

# --- static layers, identical for both maps -------------------------------
land_x, land_y = [], []
for ring in coastline_rings():
    land_x += ring[:, 0].tolist() + [None]
    land_y += ring[:, 1].tolist() + [None]

lanes = []
for (o, t), (route, dash) in _LEG_STYLE.items():
    line = route_polyline(o, t)
    lanes.append({"x": np.round(line[:, 0], 2).tolist(),
                  "y": np.round(line[:, 1], 2).tolist(),
                  "color": ROUTE_COLORS[route], "dash": dash})

ports = {"x": [], "y": [], "text": [], "pos": []}
for code in POINTS:
    x, y = point_xy(code)
    ports["x"].append(x); ports["y"].append(y)
    ports["text"].append(PORTS[code].name); ports["pos"].append(_PORT_LABEL_SIDE[code])

DATA = {
    "park": PARK, "years": YEARS,
    "runs": payload,
    # the coastline is one trace with None between rings, so round past them
    "land": {"x": [None if v is None else round(v, 2) for v in land_x],
             "y": [None if v is None else round(v, 2) for v in land_y]},
    "lanes": lanes, "ports": ports,
    "points": [PORTS[c].name for c in POINTS],
    "lon": [LON_MIN, LON_MAX],
    "colors": COLORS,
    "periodsInDay": 3,
}
# None survives json as null, which is what the JS wants for line breaks
blob = json.dumps(DATA, separators=(",", ":"))
io.open(f"{sys.argv[1]}/sim_data.json", "w", encoding="utf-8").write(blob)
print(f"\nданных: {len(blob)/1e6:.2f} МБ")
