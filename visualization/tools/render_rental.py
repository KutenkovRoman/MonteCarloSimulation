"""Same seed, rental off against rental on, rendered to an MP4 for a deck."""
import sys, time
sys.path.insert(0, "/Users/vladimiraletov/Projects/ports_containers/simviz")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import imageio.v2 as imageio

from simviz.config import ScenarioConfig, month_names_ru
from simviz.runner import run_trace, spawn_seeds
from simviz.figures import COLORS as C, ROUTE_COLORS, _LEG_STYLE
from simviz.geo import PORTS, LON_MIN, LON_MAX, coastline_rings, route_polyline, point_xy, position_on_leg
from simviz.trace import POINTS

PARK, YEARS, SEED_IDX = 31_300, 2.0, 5
FPS, HOLD_S, P = 25, 3.0, 3
RENT_XY = (131.0, 27.5)          # offshore, clear of the port marker
OUT = sys.argv[1]

print("считаю трассы (с арендой это медленно)...")
traces, tags = [], ["без аренды", "с арендой"]
for ns in (0, 1):
    cfg = ScenarioConfig(n_containers=PARK, horizon_years=YEARS, warmup_years=0.0,
                         n_fan_runs=1, triangulation_n_sims=ns)
    t0 = time.time()
    tr = run_trace(cfg, spawn_seeds(cfg.seed, 8)[SEED_IDX])
    traces.append(tr)
    print(f"  n_sims={ns}: {tr.n_periods} периодов, "
          f"{'провал ' + str(tr.fail_period) if tr.failed else 'дошёл'}, "
          f"мин {tr.warehouse.min()}, у арендаторов макс {int(tr.stocks['у арендаторов'].max())}, "
          f"баланс {tr.conservation_error}, {time.time()-t0:.0f}с")

ru = month_names_ru()
N_PER = max(t.n_periods for t in traces)
DAYS = N_PER // P
V = len(traces[0].vessel_ids)
V_COLORS = [ROUTE_COLORS[v[-1]] for v in traces[0].vessel_ids]
Y_TOP = max(t.warehouse.max() for t in traces) * 1.06

print("считаю позиции...")
pos, rented = [], []
for tr in traces:
    xs = np.zeros((DAYS, V)); ys = np.zeros((DAYS, V)); sz = np.zeros((DAYS, V))
    cache = {}
    for d in range(DAYS):
        t = min(d * P, tr.n_periods - 1)
        for i in range(V):
            key = (int(tr.v_from[t, i]), int(tr.v_to[t, i]), int(round(float(tr.v_progress[t, i]) * 400)))
            spot = cache.get(key) or position_on_leg(POINTS[key[0]], POINTS[key[1]], key[2] / 400.0)
            cache[key] = spot
            xs[d, i], ys[d, i] = spot
            sz[d, i] = 28 + 300 * min(tr.v_loaded[t, i] / tr.v_capacity[i], 1.0)
    pos.append((xs, ys, sz))
    rented.append(tr.stocks["у арендаторов"])

def label(tr, t):
    t = min(t, tr.n_periods - 1)
    return f"{int(tr.day[t])} {ru[int(tr.month[t])]}, год {int(tr.day_index[t]) // 365 + 1}"

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=C["bg"])
gs = fig.add_gridspec(3, 2, height_ratios=[0.16, 1.0, 0.34],
                      left=0.064, right=0.988, top=0.965, bottom=0.085, hspace=0.18, wspace=0.11)

head = fig.add_subplot(gs[0, :]); head.axis("off")
head.text(0, 0.62, "Что меняет аренда: один прогон, с ней и без неё",
          color=C["text"], fontsize=24, fontweight="bold", va="center")
head.text(0, 0.05, f"парк {PARK:,} контейнеров  ·  горизонт 2 года  ·  один и тот же seed  ·  "
                   f"до первого решения об аренде обе картины совпадают бит в бит".replace(",", " "),
          color=C["muted"], fontsize=15, va="center")
date_txt = head.text(1.0, 0.45, "", color=C["text"], fontsize=23, fontweight="bold", ha="right", va="center")
season_txt = head.text(1.0, 0.0, "", color=C["warn"], fontsize=15, ha="right", va="center")

rings = coastline_rings()
fleets, rent_dots, rent_lbls, dead_txt = [], [], [], []
for k, tr in enumerate(traces):
    ax = fig.add_subplot(gs[1, k], facecolor=C["bg"])
    for r in rings:
        ax.fill(r[:, 0], r[:, 1], facecolor=C["land"], edgecolor=C["land_line"], linewidth=0.6, zorder=1)
    for (o, t), (route, dash) in _LEG_STYLE.items():
        line = route_polyline(o, t)
        ax.plot(line[:, 0], line[:, 1], color=ROUTE_COLORS[route], linewidth=1.5, alpha=0.55,
                linestyle="--" if dash == "dash" else "-", zorder=2)
    for code in POINTS:
        x, y = point_xy(code)
        ax.plot([x], [y], "o", color=C["port"], markersize=7,
                markeredgecolor=C["bg"], markeredgewidth=1.4, zorder=4)
        ax.annotate(PORTS[code].name, (x, y), textcoords="offset points", xytext=(9, 7),
                    color=C["text"], fontsize=12, zorder=5)
    # the containers that are in neither a port nor a vessel
    rd = ax.scatter([RENT_XY[0]], [RENT_XY[1]], s=[0], c=[C["danger"]], alpha=0.75,
                    edgecolors=C["danger"], linewidths=1.6, zorder=6)
    rl = ax.annotate("", RENT_XY, textcoords="offset points", xytext=(0, -34),
                     ha="center", color=C["danger"], fontsize=13, fontweight="bold", zorder=7)
    sc = ax.scatter(pos[k][0][0], pos[k][1][0], s=pos[k][2][0], c=V_COLORS,
                    edgecolors="#0b1119", linewidths=1.0, zorder=8)
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(-6, 82); ax.set_aspect(1.45)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.set_title(tags[k], color=C["ok"] if k == 0 else C["danger"],
                 fontsize=19, fontweight="bold", loc="left", pad=10)
    dead_txt.append(ax.text(0.5, 0.085, "", transform=ax.transAxes, ha="center", va="center",
                            color=C["danger"], fontsize=24, fontweight="bold", zorder=9,
                            bbox=dict(facecolor=C["bg"], edgecolor=C["danger"], pad=9, alpha=0.9)))
    fleets.append(sc); rent_dots.append(rd); rent_lbls.append(rl)

cursors = []
for k, tr in enumerate(traces):
    ax = fig.add_subplot(gs[2, k], facecolor=C["bg"])
    ax.plot(np.arange(tr.n_periods), tr.warehouse,
            color=C["ok"] if not tr.failed else C["danger"], linewidth=1.8, zorder=3)
    m, d = tr.month, tr.day
    pause = ((m == 5) & (d >= 10)) | (m == 6)
    nsr = np.isin(m, [7, 8, 9, 10]) | ((m == 11) & (d <= 10))
    for mask, col in ((pause, "#7d8fa3"), (nsr, C["warn"])):
        on = None
        for i in range(len(mask)):
            if mask[i] and on is None: on = i
            elif not mask[i] and on is not None:
                ax.axvspan(on, i, color=col, alpha=0.16, zorder=1); on = None
        if on is not None: ax.axvspan(on, len(mask), color=col, alpha=0.16, zorder=1)
    if tr.failed:
        ax.plot([tr.n_periods - 1], [tr.warehouse[-1]], "x", color=C["danger"],
                markersize=14, markeredgewidth=3, zorder=5)
    cursors.append(ax.axvline(0, color=C["text"], linewidth=1.2, linestyle=":", zorder=6))
    ax.set_xlim(0, N_PER - 1); ax.set_ylim(0, Y_TOP); ax.set_xticks([])
    ax.tick_params(colors=C["muted"], labelsize=12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v/1000:.0f} тыс." if v else "0"))
    ax.grid(axis="y", color=C["grid"], linewidth=0.7)
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    if k == 0:
        ax.text(0.0, 1.10, "пустых контейнеров на складе в Дудинке, шт",
                transform=ax.transAxes, color=C["muted"], fontsize=12.5, va="bottom")

fig.text(0.064, 0.045, "маршрут A — Мурманск        маршрут B — Архангельск и Петербург        "
                       "маршрут C — Севморпуть        размер точки — загрузка судна",
         color=C["muted"], fontsize=12.5)
fig.text(0.064, 0.015, "Красный круг у Китая — контейнеры у арендаторов: они ни в порту, ни на судне, "
                       "и возвращаются в хаб на 38 суток позже. Серая полоса — межпаводковая пауза, "
                       "жёлтая — окно Севморпути.", color=C["muted"], fontsize=11.5)

def frame(day):
    t = day * P
    ref = traces[0]
    tt = min(t, ref.n_periods - 1)
    date_txt.set_text(label(ref, t))
    mm, dd = int(ref.month[tt]), int(ref.day[tt])
    season_txt.set_text("межпаводковая пауза" if ((mm == 5 and dd >= 10) or mm == 6)
                        else ("Севморпуть открыт" if (mm in (7, 8, 9, 10) or (mm == 11 and dd <= 10)) else ""))
    for k, tr in enumerate(traces):
        xs, ys, sz = pos[k]
        gone = tr.failed and t >= tr.n_periods
        fleets[k].set_offsets(np.c_[xs[day], ys[day]])
        fleets[k].set_sizes(sz[day])
        fleets[k].set_alpha(0.28 if gone else 0.95)
        n = int(rented[k][min(t, tr.n_periods - 1)])
        rent_dots[k].set_sizes([0 if n == 0 else 60 + 900 * n / 12000])
        rent_lbls[k].set_text("" if n == 0 else f"у арендаторов\n{n:,}".replace(",", " "))
        dead_txt[k].set_text(f"производство встало\n{label(tr, tr.n_periods - 1)}" if gone else "")
        cursors[k].set_xdata([min(t, tr.n_periods - 1)] * 2)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

print(f"рендерю {DAYS} кадров...")
t0 = time.time()
with imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                        ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
    for day in range(DAYS):
        img = frame(day)
        w.append_data(img)
        if day % 150 == 0: print(f"  {day}/{DAYS}  {time.time()-t0:.0f}с")
    for _ in range(int(FPS * HOLD_S)):
        w.append_data(img)
print(f"готово за {time.time()-t0:.0f}с: {OUT}")
