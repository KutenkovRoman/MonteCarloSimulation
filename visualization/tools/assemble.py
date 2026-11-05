import io, sys
SP = sys.argv[1]
PLOTLY = "/Users/vladimiraletov/Projects/ports_containers/simviz/.venv/lib/python3.12/site-packages/plotly/package_data/plotly.min.js"

css = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0f1720;color:#dfe7ef;
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1680px;margin:0 auto;padding:18px 20px 28px}
header h1{font-size:20px;margin:0 0 4px;font-weight:600}
.sub{color:#7d8fa3;margin:0 0 14px}
.sub b{color:#dfe7ef;font-weight:600}
code{background:#16212d;padding:1px 5px;border-radius:4px;font-size:12px}
.bar{display:flex;align-items:center;gap:14px;background:#16212d;border:1px solid #22303f;
     border-radius:10px;padding:10px 14px;margin-bottom:16px;position:sticky;top:0;z-index:5}
.bar button{width:40px;height:32px;font-size:15px;cursor:pointer;background:#22303f;
     color:#dfe7ef;border:1px solid #31465c;border-radius:7px}
.bar button:hover{background:#2c3d4f}
#scrub{flex:1;accent-color:#4c9be8;cursor:pointer}
.date{font-variant-numeric:tabular-nums;min-width:170px;font-weight:600}
.spd{color:#7d8fa3;font-size:12px;display:flex;align-items:center;gap:6px}
.spd select{background:#22303f;color:#dfe7ef;border:1px solid #31465c;border-radius:6px;
     padding:3px 6px;font-size:12px}
.season{color:#e8a33d;font-size:12px;min-width:170px;text-align:right}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:1180px){.cols{grid-template-columns:1fr}}
.col{background:#16212d;border:1px solid #22303f;border-radius:10px;padding:12px;
     transition:opacity .25s}
.col.dead{opacity:.72}
.col h2{font-size:14px;margin:0 0 10px;font-weight:600;display:flex;align-items:center;gap:8px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dot.ok{background:#6ac47e}.dot.bad{background:#e05d5d}
.map{height:396px}
.chart{height:132px;margin-top:6px}
.stats{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:8px;font-size:12px;color:#7d8fa3}
.stats b{color:#dfe7ef}
.stats .ok{color:#6ac47e}.stats .bad{color:#e05d5d}
.stats .q{opacity:.6}
footer{margin-top:18px}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:12px;color:#7d8fa3;
     align-items:center}
.legend i{width:16px;height:3px;display:inline-block;margin-right:6px;vertical-align:middle;
     border-radius:2px}
.legend .hint{opacity:.7;font-style:italic}
.note{color:#7d8fa3;font-size:12px;margin:8px 0 0;max-width:110ch}
"""

plotly = io.open(PLOTLY, encoding="utf-8").read()
body = io.open(f"{SP}/body.html", encoding="utf-8").read()
app = io.open(f"{SP}/app.js", encoding="utf-8").read()
data = io.open(f"{SP}/sim_data.json", encoding="utf-8").read()

html = (
    "<!doctype html>\n<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
    "<title>Оборот парка контейнеров: два прогона</title>\n"
    f"<style>{css}</style>\n</head>\n<body>\n{body}\n"
    f"<script>{plotly}</script>\n"
    f"<script>window.SIM={data};</script>\n"
    f"<script>{app}</script>\n</body>\n</html>\n"
)
out = f"{SP}/simulation-two-runs.html"
io.open(out, "w", encoding="utf-8").write(html)
print(f"{out}\n{len(html)/1e6:.2f} МБ, plotly вложен ({len(plotly)/1e6:.2f} МБ), данных {len(data)/1e6:.2f} МБ")
