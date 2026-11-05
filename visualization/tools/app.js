(function () {
  "use strict";
  var D = window.SIM, C = D.colors, P = D.periodsInDay;
  var LONGEST = Math.max.apply(null, D.runs.map(function (r) { return r.n; }));
  var VESSELS = D.runs[0].ids.length;
  var FLEET_TRACE = 1 + D.lanes.length + 1;   // land, lanes, ports, fleet

  function mapTraces(run) {
    var t = [{
      x: D.land.x, y: D.land.y, mode: "lines", fill: "toself",
      fillcolor: C.land, line: { color: C.land_line, width: 0.7 },
      hoverinfo: "skip", showlegend: false
    }];
    D.lanes.forEach(function (l) {
      t.push({ x: l.x, y: l.y, mode: "lines", opacity: 0.5, hoverinfo: "skip",
               showlegend: false, line: { color: l.color, width: 1.3, dash: l.dash } });
    });
    t.push({ x: D.ports.x, y: D.ports.y, mode: "markers+text", text: D.ports.text,
             textposition: D.ports.pos, hoverinfo: "text", showlegend: false,
             textfont: { color: C.text, size: 10 },
             marker: { size: 8, color: C.port, line: { color: C.bg, width: 1.4 } } });
    t.push({ x: [], y: [], mode: "markers", hoverinfo: "text", showlegend: false,
             marker: { size: [], color: run.colors, opacity: 0.95,
                       line: { color: "#0b1119", width: 1.1 } } });
    return t;
  }

  // a function, not an object: Plotly mutates the layout it is handed, so
  // sharing one between the two maps left the second with the first's
  // computed ranges and squeezed its geography into a corner
  function mapLayout() {
    return {
      paper_bgcolor: C.bg, plot_bgcolor: C.bg, showlegend: false, dragmode: "pan",
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: { range: D.lon.slice(), visible: false, constrain: "domain" },
      yaxis: { range: [-6, 82], visible: false, scaleanchor: "x", scaleratio: 1.45,
               constrain: "domain" },
      hoverlabel: { bgcolor: C.panel, font: { color: C.text } }
    };
  }

  // seasons as shaded bands over the warehouse chart, taken off the date labels
  function bands(run) {
    var out = [], open = null, kind = null;
    for (var t = 0; t < run.n; t += P) {
      var lab = run.labels[t], k = null;
      if (/ма[йя]|мая/.test(lab) || /июня/.test(lab)) {
        var d = parseInt(lab, 10);
        if (/мая/.test(lab) && d >= 10) k = "pause";
        else if (/июня/.test(lab)) k = "pause";
      }
      if (!k) {
        if (/июля|августа|сентября|октября/.test(lab)) k = "nsr";
        else if (/ноября/.test(lab) && parseInt(lab, 10) <= 10) k = "nsr";
      }
      if (k !== kind) {
        if (open !== null) out.push({ from: open, to: t, kind: kind });
        open = k ? t : null; kind = k;
      }
    }
    if (open !== null) out.push({ from: open, to: run.n, kind: kind });
    return out.filter(function (b) { return b.kind; });
  }

  function chartFigure(run) {
    var shapes = bands(run).map(function (b) {
      return { type: "rect", xref: "x", yref: "paper", x0: b.from, x1: b.to,
               y0: 0, y1: 1, layer: "below", line: { width: 0 },
               fillcolor: b.kind === "pause" ? "rgba(125,143,163,.20)"
                                             : "rgba(232,163,61,.13)" };
    });
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: 0, x1: 0,
                  y0: 0, y1: 1, line: { color: C.text, width: 1, dash: "dot" } });
    var data = [{
      x: run.warehouse.map(function (_, i) { return i; }),
      y: run.warehouse, mode: "lines", hoverinfo: "skip",
      line: { color: run.failed ? C.danger : C.ok, width: 1.6 }
    }];
    if (run.failed) {
      data.push({ x: [run.n - 1], y: [run.warehouse[run.n - 1]], mode: "markers+text",
                  text: ["производство встало"], textposition: "top left",
                  textfont: { color: C.danger, size: 11 }, hoverinfo: "skip",
                  marker: { color: C.danger, size: 11, symbol: "x" } });
    }
    return {
      data: data,
      layout: {
        paper_bgcolor: C.bg, plot_bgcolor: C.bg, showlegend: false, shapes: shapes,
        margin: { l: 48, r: 8, t: 6, b: 22 },
        xaxis: { range: [0, LONGEST - 1], showgrid: false, zeroline: false,
                 tickvals: [], color: C.muted },
        yaxis: { title: { text: "склад, шт", font: { color: C.muted, size: 10 } },
                 gridcolor: C.grid, zeroline: false, color: C.muted,
                 rangemode: "tozero" }
      }
    };
  }

  D.runs.forEach(function (run, k) {
    Plotly.newPlot("map" + k, mapTraces(run), mapLayout(),
                   { displayModeBar: false, responsive: true });
    var fig = chartFigure(run);
    Plotly.newPlot("chart" + k, fig.data, fig.layout,
                   { displayModeBar: false, responsive: true });
    run._cursor = fig.layout.shapes.length - 1;
    document.getElementById("stats" + k).innerHTML =
      "<span>минимум склада <b>" + run.minStock.toLocaleString("ru") + "</b></span>" +
      "<span>доставлено в Китай <b>" + run.delivered.toLocaleString("ru") + "</b></span>" +
      (run.failed
        ? "<span class='bad'>дефицит <b>" + run.failDeficit + "</b> на " +
          run.labels[run.n - 1] + "</span>"
        : "<span class='ok'>дошёл до " + run.labels[run.n - 1] + "</span>") +
      "<span class='q'>баланс контейнеров расходится на " + run.conservation + "</span>";
  });

  var period = 0, playing = false, clock = null;
  var scrub = document.getElementById("scrub");
  scrub.max = LONGEST - 1;

  function draw() {
    D.runs.forEach(function (run, k) {
      var t = Math.min(period, run.n - 1), off = t * VESSELS;
      var x = [], y = [], size = [], hover = [];
      for (var i = 0; i < VESSELS; i++) {
        var loaded = run.loaded[off + i], cap = run.capacity[i] || 1;
        var from = run.src[off + i], to = run.dst[off + i];
        x.push(run.x[off + i]); y.push(run.y[off + i]);
        size.push(6 + 11 * Math.min(loaded / cap, 1));
        hover.push("<b>" + run.ids[i] + "</b><br>" +
                   (from === to ? "в порту " + D.points[from]
                                : D.points[from] + " → " + D.points[to]) +
                   "<br>груз " + loaded + " / " + cap);
      }
      var dead = run.failed && period >= run.n;
      Plotly.restyle("map" + k, {
        x: [x], y: [y], text: [hover], "marker.size": [size],
        "marker.opacity": dead ? 0.28 : 0.95
      }, [FLEET_TRACE]);
      var rl = {}; rl["shapes[" + run._cursor + "].x0"] = t;
      rl["shapes[" + run._cursor + "].x1"] = t;
      Plotly.relayout("chart" + k, rl);
      document.querySelector('[data-run="' + k + '"]').classList.toggle("dead", dead);
    });
    var ref = D.runs[0];
    document.getElementById("date").textContent = ref.labels[Math.min(period, ref.n - 1)];
    var lab = ref.labels[Math.min(period, ref.n - 1)];
    var s = "";
    if (/мая/.test(lab) && parseInt(lab, 10) >= 10) s = "межпаводковая пауза";
    else if (/июня/.test(lab)) s = "межпаводковая пауза";
    else if (/июля|августа|сентября|октября/.test(lab)) s = "Севморпуть открыт";
    else if (/ноября/.test(lab) && parseInt(lab, 10) <= 10) s = "Севморпуть открыт";
    document.getElementById("season").textContent = s;
    scrub.value = period;
  }

  function tick(now) {
    if (!playing) { clock = null; return; }
    if (clock === null) { clock = now; requestAnimationFrame(tick); return; }
    var per = parseInt(document.getElementById("speed").value, 10);
    var step = Math.max(1, Math.round(Math.min(now - clock, 900) / 1000 * per));
    clock = now;
    period += step;
    if (period >= LONGEST - 1) { period = LONGEST - 1; toggle(false); }
    draw();
    requestAnimationFrame(tick);
  }

  function toggle(on) {
    playing = on === undefined ? !playing : on;
    if (playing && period >= LONGEST - 1) period = 0;
    document.getElementById("play").textContent = playing ? "⏸" : "▶";
    clock = null;
    if (playing) requestAnimationFrame(tick);
  }

  document.getElementById("play").onclick = function () { toggle(); };
  scrub.oninput = function () { toggle(false); period = +scrub.value; draw(); };
  document.onkeydown = function (e) {
    if (e.code === "Space") { e.preventDefault(); toggle(); }
    if (e.code === "ArrowRight") { toggle(false); period = Math.min(period + P, LONGEST - 1); draw(); }
    if (e.code === "ArrowLeft") { toggle(false); period = Math.max(period - P, 0); draw(); }
  };
  draw();
})();
