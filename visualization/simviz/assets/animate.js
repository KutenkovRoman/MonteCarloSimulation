/* Playback runs entirely in the browser: the fleet track is shipped once and
   every frame is drawn with Plotly.restyle, so scrubbing and playing cost no
   network round trips. */

(function () {
  "use strict";

  function plot(id) {
    var host = document.getElementById(id);
    return host ? host.querySelector(".js-plotly-plot") : null;
  }

  function moveCursor(id, period) {
    var gd = plot(id);
    if (gd && window.Plotly) {
      window.Plotly.relayout(gd, { "shapes[0].x0": period, "shapes[0].x1": period });
    }
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    simviz: {

      /* Draw one frame: fleet on the map, time cursor on both charts. */
      render: function (period, frames) {
        var nu = window.dash_clientside.no_update;
        if (!frames || period === null || period === undefined) {
          return nu;
        }
        var count = frames.ids.length;
        if (frames.n === 0) {
          /* The run died before a single period was recorded: clear the fleet
             rather than indexing frames.labels[-1]. */
          var empty = plot("map");
          if (empty && window.Plotly) {
            window.Plotly.restyle(
              empty,
              { x: [[]], y: [[]], text: [[]], "marker.size": [[]] },
              [frames.vesselTrace]
            );
          }
          return "—";
        }
        var t = Math.max(0, Math.min(period, frames.n - 1));
        var offset = t * count;

        var x = [], y = [], size = [], text = [];
        for (var i = 0; i < count; i++) {
          var loaded = frames.loaded[offset + i];
          var capacity = frames.capacity[i] || 1;
          var from = frames.src[offset + i];
          var to = frames.dst[offset + i];
          var where = from === to
            ? "в порту " + frames.points[from]
            : frames.points[from] + " → " + frames.points[to];

          x.push(frames.x[offset + i]);
          y.push(frames.y[offset + i]);
          size.push(7 + 11 * Math.min(loaded / capacity, 1));
          text.push("<b>" + frames.ids[i] + "</b><br>" + where +
                    "<br>груз " + loaded + " / " + frames.capacity[i]);
        }

        var map = plot("map");
        if (map && window.Plotly) {
          window.Plotly.restyle(
            map,
            { x: [x], y: [y], text: [text], "marker.size": [size] },
            [frames.vesselTrace]
          );
        }
        moveCursor("fan", t);
        moveCursor("stock", t);
        return frames.labels[t];
      },

      /* Advance by wall-clock time rather than by tick count, so a throttled
         timer takes bigger steps instead of running in slow motion. One second
         of real time is `speed` days of simulation. */
      advance: function (_ticks, period, playing, speed, frames) {
        var nu = window.dash_clientside.no_update;
        if (!playing || !frames) {
          window.__simvizClock = null;
          return [nu, nu];
        }
        var now = window.performance.now();
        var previous = window.__simvizClock;
        window.__simvizClock = now;
        if (previous === null || previous === undefined) {
          return [nu, nu];
        }
        /* Cap only guards against a long stall (tab restored, GC pause);
           a background tab throttled to 1 Hz must still advance a full second. */
        var elapsed = Math.min(now - previous, 1200);
        var step = Math.max(1, Math.round((elapsed / 1000) * (speed || 1) * 3));
        var next = (period || 0) + step;
        if (next >= frames.n - 1) {
          return [frames.n - 1, false];
        }
        return [next, nu];
      },

      /* The play button flips a flag; the timer and the icon follow it. */
      toggle: function (_clicks, playing) {
        window.__simvizClock = null;
        return !playing;
      },

      /* Step through the ensemble: previous, next, or the next run where
         production stopped. Pure viewing -- no headline figure moves. */
      stepRun: function (_prev, _next, _failed, current, total, failedRuns) {
        var dash = window.dash_clientside;
        var trigger = (dash.callback_context.triggered[0] || {}).prop_id || "";
        var now = Math.max(parseInt(current, 10) || 1, 1);
        var last = Math.max(parseInt(total, 10) || 1, 1);

        if (trigger.indexOf("btn-prev-run") === 0) {
          return now > 1 ? now - 1 : last;
        }
        if (trigger.indexOf("btn-next-run") === 0) {
          return now < last ? now + 1 : 1;
        }
        if (trigger.indexOf("btn-failed-run") === 0) {
          var list = (failedRuns || []).map(function (i) { return i + 1; });
          if (!list.length) {
            return dash.no_update;
          }
          for (var i = 0; i < list.length; i++) {
            if (list[i] > now) {
              return list[i];
            }
          }
          return list[0];
        }
        return dash.no_update;
      },

      applyPlaying: function (playing) {
        /* Smoothness is set here; how fast time moves is set by the step. */
        return [!playing, 60, playing ? "⏸" : "▶"];
      }
    }
  });
})();
