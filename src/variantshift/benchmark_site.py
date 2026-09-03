"""Dependency-free static explorer for committed VariantShift result tables."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd

from .provenance import sha256_file


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.where(pd.notna(frame), None).to_json(orient="records"))


def build_benchmark_site(config_path: Path, output_dir: Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    downloads_dir = output_dir / "downloads"
    data_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, object]] = []
    for table in config["tables"]:
        source = Path(str(table["path"]))
        if not source.is_absolute():
            source = config_path.parent / source
        if not source.is_file():
            raise ValueError(f"Site input table is unavailable: {source}")
        frame = pd.read_csv(source)
        identifier = str(table["id"])
        data_path = data_dir / f"{identifier}.json"
        data_path.write_text(
            json.dumps(_json_records(frame), separators=(",", ":")), encoding="utf-8"
        )
        download_path = downloads_dir / f"{identifier}.csv"
        shutil.copy2(source, download_path)
        catalog.append(
            {
                "id": identifier,
                "title": str(table["title"]),
                "description": str(table.get("description", "")),
                "kind": str(table.get("kind", "table")),
                "rows": len(frame),
                "columns": list(frame.columns),
                "data": f"data/{identifier}.json",
                "download": f"downloads/{identifier}.csv",
                "sha256": sha256_file(download_path),
            }
        )
    catalog_path = data_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    title = html.escape(str(config.get("title", "VariantShift")))
    subtitle = html.escape(
        str(config.get("subtitle", "Audited model transport under biological shift"))
    )
    index = output_dir / "index.html"
    styles = output_dir / "styles.css"
    script = output_dir / "app.js"
    index.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="VariantShift benchmark and reliability explorer">
  <title>{title}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <p class="eyebrow">Outcome-aware evaluation without outcome-aware deployment</p>
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
  </header>
  <main>
    <section id="overview" class="overview" aria-label="Benchmark summary"></section>
    <nav id="tabs" aria-label="Result tables"></nav>
    <section class="panel">
      <div class="toolbar">
        <label>Filter <input id="filter" type="search" placeholder="model, assay, family…"></label>
        <a id="download" class="download" href="#">Download CSV</a>
      </div>
      <p id="description"></p>
      <div id="chart" class="chart" aria-label="Risk coverage chart"></div>
      <div class="table-wrap"><table id="results"></table></div>
      <p id="empty" hidden>No rows match this filter.</p>
    </section>
  </main>
  <footer>Every displayed table is downloadable and checksum-addressed.</footer>
  <script src="app.js"></script>
</body>
</html>
""",
        encoding="utf-8",
    )
    styles.write_text(
        """:root{color-scheme:light;--ink:#10231f;--muted:#5a6d67;--paper:#f4f1e8;--panel:#fffdf7;--accent:#0b6b55;--line:#d8d2c3}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header,main,footer{width:min(1180px,calc(100% - 32px));margin:auto}header{padding:64px 0 34px;border-bottom:1px solid var(--line)}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;color:var(--accent);font-weight:750}h1{font-family:Georgia,serif;font-size:clamp(42px,8vw,86px);line-height:.95;letter-spacing:-.045em;margin:.2em 0}.subtitle{font-size:19px;color:var(--muted);max-width:720px}.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;padding:24px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 30px #10231f0a}.card{padding:18px}.card strong{display:block;font:32px/1 Georgia,serif;margin-bottom:8px}.card span{color:var(--muted)}nav{display:flex;gap:8px;overflow:auto;padding:4px 0 14px}button{border:1px solid var(--line);background:transparent;padding:9px 14px;border-radius:999px;color:var(--ink);white-space:nowrap;cursor:pointer}button.active{background:var(--ink);color:white;border-color:var(--ink)}.panel{padding:18px;margin-bottom:40px}.toolbar{display:flex;gap:16px;justify-content:space-between;align-items:end;flex-wrap:wrap}.toolbar label{font-weight:650}.toolbar input{display:block;width:min(360px,75vw);margin-top:5px;padding:10px;border:1px solid var(--line);border-radius:8px;background:white}.download{color:var(--accent);font-weight:700}.table-wrap{overflow:auto;max-height:620px;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #e9e4d9;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);z-index:1}tr:hover td{background:#eef7f2}.chart{min-height:0;margin:14px 0}.chart svg{width:100%;height:auto;background:#fbfaf5;border:1px solid var(--line);border-radius:10px}.chart text{font-size:11px;fill:var(--muted)}footer{padding:20px 0 50px;color:var(--muted)}@media(max-width:600px){header{padding-top:36px}.panel{padding:12px}}""",
        encoding="utf-8",
    )
    script.write_text(
        """const state={catalog:[],active:null,rows:[]};const $=s=>document.querySelector(s);function value(v){if(v===null||v===undefined)return '';if(typeof v==='number')return Number.isInteger(v)?String(v):v.toFixed(4);return String(v)}function renderOverview(){const rows=state.catalog.reduce((n,t)=>n+t.rows,0);$('#overview').innerHTML=`<div class="card"><strong>${state.catalog.length}</strong><span>audited result tables</span></div><div class="card"><strong>${rows.toLocaleString()}</strong><span>displayed records</span></div><div class="card"><strong>100%</strong><span>downloadable tables</span></div>`}function renderTabs(){const tabs=$('#tabs');tabs.innerHTML='';state.catalog.forEach(t=>{const b=document.createElement('button');b.textContent=t.title;b.className=t.id===state.active?.id?'active':'';b.onclick=()=>loadTable(t);tabs.appendChild(b)})}function drawRisk(rows){const chart=$('#chart');if(!rows.length||!('coverage'in rows[0])||!('failure_rate'in rows[0])){chart.innerHTML='';return}const policies=[...new Set(rows.map(r=>r.policy||'result'))];const colors=['#0b6b55','#bd5b2a','#4f62a8','#8b4d90','#6c7335','#333'];const w=900,h=300,p=48;let svg=`<svg viewBox="0 0 ${w} ${h}" role="img"><line x1="${p}" y1="${h-p}" x2="${w-p}" y2="${h-p}" stroke="#aaa"/><line x1="${p}" y1="${p}" x2="${p}" y2="${h-p}" stroke="#aaa"/><text x="${w/2}" y="${h-8}">Task coverage</text><text x="8" y="${h/2}">Failure risk</text>`;policies.forEach((policy,i)=>{const pts=rows.filter(r=>(r.policy||'result')===policy).sort((a,b)=>a.coverage-b.coverage).map(r=>`${p+r.coverage*(w-2*p)},${h-p-r.failure_rate*(h-2*p)}`).join(' ');svg+=`<polyline fill="none" stroke="${colors[i%colors.length]}" stroke-width="3" points="${pts}"/><text x="${p+8}" y="${p+15*i+12}" fill="${colors[i%colors.length]}">${policy}</text>`});chart.innerHTML=svg+'</svg>'}function renderRows(){const q=$('#filter').value.toLowerCase();const rows=state.rows.filter(r=>Object.values(r).some(v=>value(v).toLowerCase().includes(q)));const columns=state.active.columns;$('#results').innerHTML=`<thead><tr>${columns.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,1000).map(r=>`<tr>${columns.map(c=>`<td>${value(r[c])}</td>`).join('')}</tr>`).join('')}</tbody>`;$('#empty').hidden=rows.length>0;drawRisk(rows)}async function loadTable(table){state.active=table;state.rows=await fetch(table.data).then(r=>r.json());$('#description').textContent=`${table.description} ${table.rows.toLocaleString()} rows · SHA-256 ${table.sha256.slice(0,12)}…`;$('#download').href=table.download;renderTabs();renderRows()}async function start(){state.catalog=await fetch('data/catalog.json').then(r=>r.json());renderOverview();if(state.catalog.length)await loadTable(state.catalog[0])}$('#filter').addEventListener('input',renderRows);start();""",
        encoding="utf-8",
    )
    return {"index": index, "styles": styles, "script": script, "catalog": catalog_path}
