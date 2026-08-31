"""
Generate a standalone HTML training report for a Sat2Map run.

Reads ``history.json`` / ``config.json`` (written by ``train.py``) plus any
``*.png`` previews in the run directory and ``eval/*.png`` triptychs (written by
``evaluate.py``), and bakes everything - data and images - into a single
self-contained ``report.html`` that opens straight from ``file://`` with no
server and no external assets.

This is a training artefact, not part of the FastAPI app.

Usage:
    python -m training.sat2map.report --run ./weights/sat2map
    # -> ./weights/sat2map/report.html
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _collect_images(run: Path) -> list[dict]:
    images: list[dict] = []
    for p in sorted(run.glob("*.png")):
        images.append({"name": p.stem, "group": "previews", "uri": _data_uri(p)})
    for p in sorted((run / "eval").glob("*.png")):
        images.append({"name": p.stem, "group": "eval", "uri": _data_uri(p)})
    return images


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sat2Map training report</title>
<style>
  :root {{ color-scheme: light dark; --bg:#faf9f7; --card:#fff; --ink:#1c1c1c;
    --muted:#6b6b6b; --line:#e6e3de; --accent:#c0392b; --ok:#2e7d32; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --card:#1f2024;
    --ink:#e9e9e9; --muted:#9a9a9a; --line:#33353a; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:32px 0 12px; }}
  .sub {{ color:var(--muted); margin:0 0 24px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:18px; margin-bottom:16px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  tr.best td {{ color:var(--ok); font-weight:600; }}
  .grid {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); }}
  figure {{ margin:0; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:8px; display:block; }}
  figcaption {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:20px; margin:0; padding:0; list-style:none; }}
  .kpis li {{ min-width:120px; }}
  .kpis b {{ display:block; font-size:22px; font-variant-numeric:tabular-nums; }}
  .kpis span {{ color:var(--muted); font-size:12px; }}
  svg {{ width:100%; height:auto; display:block; }}
  .legend {{ display:flex; gap:16px; font-size:12px; color:var(--muted); margin-top:8px; }}
  .legend i {{ width:18px; height:3px; display:inline-block; vertical-align:middle;
    margin-right:5px; border-radius:2px; }}
  code {{ background:var(--bg); padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Sat2Map training report</h1>
  <p class="sub" id="sub"></p>

  <div class="card"><ul class="kpis" id="kpis"></ul></div>

  <h2>Loss curves</h2>
  <div class="card">
    <div id="chart"></div>
    <div class="legend">
      <span><i style="background:var(--accent)"></i>val total</span>
      <span><i style="background:#888"></i>train total</span>
      <span><i style="background:#4a90d9"></i>val L1</span>
      <span><i style="background:#e0a030"></i>val SSIM</span>
      <span><i style="background:#7aa06a"></i>val edge</span>
    </div>
  </div>

  <h2>Config</h2>
  <div class="card"><table id="config"></table></div>

  <h2>Per-epoch</h2>
  <div class="card" style="overflow-x:auto"><table id="epochs"></table></div>

  <h2 id="img-h" hidden>Previews &amp; evaluation</h2>
  <div class="grid" id="images"></div>
</div>

<script id="data" type="application/json">{payload}</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const H = D.history, C = D.config;

const bestIdx = H.reduce((b,r,i)=> r.val.total < H[b].val.total ? i : b, 0);
document.getElementById('sub').textContent =
  `${{H.length}} epochs · best val ${{H[bestIdx].val.total.toFixed(4)}} at epoch ${{H[bestIdx].epoch}}`
  + ` · device ${{C.device||'?'}}`;

const kpi = (v,l)=>`<li><b>${{v}}</b><span>${{l}}</span></li>`;
const totalSecs = H.reduce((s,r)=>s+(r.seconds||0),0);
document.getElementById('kpis').innerHTML =
  kpi(H[bestIdx].val.total.toFixed(4),'best val loss') +
  kpi('#'+H[bestIdx].epoch,'best epoch') +
  kpi(H[H.length-1].train.total.toFixed(4),'last train loss') +
  kpi((totalSecs/60).toFixed(1)+' min','total train time') +
  kpi((C.size||'?')+'px','resolution') +
  kpi((C.base_channels||'?'),'U-Net base ch');

// config table
document.getElementById('config').innerHTML = Object.entries(C)
  .map(([k,v])=>`<tr><th>${{k}}</th><td><code>${{v}}</code></td></tr>`).join('');

// epoch table
const rows = H.map((r,i)=>`<tr class="${{i===bestIdx?'best':''}}">
  <td class="num">${{r.epoch}}</td>
  <td class="num">${{r.train.total.toFixed(4)}}</td>
  <td class="num">${{r.val.total.toFixed(4)}}</td>
  <td class="num">${{(r.val.l1??0).toFixed(4)}}</td>
  <td class="num">${{(r.val.ssim_loss??0).toFixed(4)}}</td>
  <td class="num">${{(r.val.edge_loss??0).toFixed(4)}}</td>
  <td class="num">${{Math.round(r.seconds||0)}}s</td></tr>`).join('');
document.getElementById('epochs').innerHTML =
  `<tr><th class="num">epoch</th><th class="num">train</th><th class="num">val</th>
   <th class="num">val L1</th><th class="num">val SSIM</th><th class="num">val edge</th>
   <th class="num">time</th></tr>` + rows;

// chart (inline SVG)
(function(){{
  const W=880,Hh=300,m={{t:12,r:12,b:28,l:44}};
  const xs=H.map(r=>r.epoch);
  const series=[
    {{k:r=>r.val.total, c:'var(--accent)', w:2.5}},
    {{k:r=>r.train.total, c:'#888', w:1.5}},
    {{k:r=>r.val.l1??null, c:'#4a90d9', w:1}},
    {{k:r=>r.val.ssim_loss??null, c:'#e0a030', w:1}},
    {{k:r=>r.val.edge_loss??null, c:'#7aa06a', w:1}},
  ];
  const all=series.flatMap(s=>H.map(s.k)).filter(v=>v!=null && isFinite(v));
  const ymax=Math.max(...all)*1.05, ymin=Math.min(0,...all);
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  const X=e=> m.l + (xmax===xmin?0:(e-xmin)/(xmax-xmin))*(W-m.l-m.r);
  const Y=v=> m.t + (1-(v-ymin)/(ymax-ymin))*(Hh-m.t-m.b);
  let g=`<svg viewBox="0 0 ${{W}} ${{Hh}}" role="img" aria-label="loss curves">`;
  for(let i=0;i<=4;i++){{ const v=ymin+(ymax-ymin)*i/4, y=Y(v);
    g+=`<line x1="${{m.l}}" y1="${{y}}" x2="${{W-m.r}}" y2="${{y}}" stroke="var(--line)"/>`;
    g+=`<text x="${{m.l-6}}" y="${{y+3}}" text-anchor="end" font-size="10" fill="var(--muted)">${{v.toFixed(2)}}</text>`; }}
  for(const e of xs){{ if(xs.length>20 && e%2) continue;
    g+=`<text x="${{X(e)}}" y="${{Hh-8}}" text-anchor="middle" font-size="10" fill="var(--muted)">${{e}}</text>`; }}
  for(const s of series){{
    let d='',started=false;
    H.forEach(r=>{{ const v=s.k(r); if(v==null||!isFinite(v)){{return;}}
      d+=(started?'L':'M')+X(r.epoch)+' '+Y(v)+' '; started=true; }});
    g+=`<path d="${{d}}" fill="none" stroke="${{s.c}}" stroke-width="${{s.w}}"/>`;
  }}
  g+=`<circle cx="${{X(H[bestIdx].epoch)}}" cy="${{Y(H[bestIdx].val.total)}}" r="4"
      fill="var(--accent)"/></svg>`;
  document.getElementById('chart').innerHTML=g;
}})();

// images
if(D.images.length){{
  document.getElementById('img-h').hidden=false;
  document.getElementById('images').innerHTML = D.images.map(im=>`<figure>
    <img src="${{im.uri}}" alt="${{im.name}}" loading="lazy">
    <figcaption>${{im.group}} / ${{im.name}}</figcaption></figure>`).join('');
}}
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Sat2Map training report.")
    ap.add_argument("--run", type=Path, default=Path("./weights/sat2map"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run
    history = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    if not history:
        raise SystemExit(f"No history.json in {run} - run training first.")

    payload = json.dumps({"history": history, "config": config,
                          "images": _collect_images(run)})
    out = args.out or (run / "report.html")
    out.write_text(PAGE.format(payload=payload), encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size/1024:.0f} KB, {len(history)} epochs)")


if __name__ == "__main__":
    main()
