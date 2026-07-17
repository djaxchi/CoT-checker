#!/usr/bin/env python3
"""cot_causal_graph_v0 Stage 4: build the self-contained HTML causal-graph explorer.

Reads stage3/graphs/*.json + crosstab.json and writes one offline explorer.html
(S4/S5 explorer convention: single file, payload embedded, no external assets).

The explorer is the analysis interface, not the claim: each trace renders as a
vertical step chain with the probe readout per node, the belief (answer-margin)
curve, the solve-from-here curve when rollouts exist, and the two causal edge
families as arcs (teacher-forced right, free-generation left). Plain-language
summaries translate every number; a "how to read this" intro opens on first view.

Usage:
  python scripts/causal_graph/cg_explorer.py --run_dir runs/causal_graph \
      [--max_traces 250] [--out explorer.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def select_traces(graph_dir: Path, max_traces: int) -> list[dict]:
    """Prioritize: fork traces with a classified site (all four cells evenly),
    then repaired sites, then on-policy traces, then the rest."""
    graphs = [json.loads(p.read_text()) for p in sorted(graph_dir.glob("*.json"))]
    forks = [g for g in graphs if g.get("site")]
    onpol = [g for g in graphs if g["arm"] == "onpolicy"]
    rest = [g for g in graphs if g not in forks and g not in onpol]
    by_cell: dict[str, list[dict]] = {}
    for g in forks:
        by_cell.setdefault(g["site"]["taxonomy"], []).append(g)
    for cell in by_cell.values():
        cell.sort(key=lambda g: -(abs(g["site"].get("d_margin_final") or 0)))
    picked: list[dict] = []
    while len(picked) < max_traces * 2 // 3 and any(by_cell.values()):
        for cell in list(by_cell):
            if by_cell[cell]:
                picked.append(by_cell[cell].pop(0))
    for g in onpol + rest:
        if len(picked) >= max_traces:
            break
        picked.append(g)
    return picked[:max_traces]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max_traces", type=int, default=250)
    args = ap.parse_args()

    stage3 = args.run_dir / "stage3"
    traces = select_traces(stage3 / "graphs", args.max_traces)
    crosstab = {}
    if (stage3 / "crosstab.json").exists():
        crosstab = json.loads((stage3 / "crosstab.json").read_text())
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crosstab": crosstab,
        "traces": traces,
    }
    out = args.out or (args.run_dir / "explorer.html")
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    out.write_text(html, encoding="utf-8")
    print(f"[explorer] {len(traces)} traces, {out} ({out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoT causal graph explorer</title>
<style>
.viz-root{
  --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --edge-harm:#e34948; --edge-help:#2a78d6; --edge-null:#c3c2b7;
  --probe:#1baf7a; --margin-line:#2a78d6; --solve-line:#1baf7a;
  --tax-di:#2a78d6; --tax-dn:#1baf7a; --tax-ui:#eda100; --tax-un:#008300;
}
@media (prefers-color-scheme: dark){ .viz-root{
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --edge-harm:#e66767; --edge-help:#3987e5; --edge-null:#383835;
  --probe:#199e70; --margin-line:#3987e5; --solve-line:#199e70;
  --tax-di:#3987e5; --tax-dn:#199e70; --tax-ui:#c98500; --tax-un:#008300;
}}
:root[data-theme="dark"] .viz-root{
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --edge-harm:#e66767; --edge-help:#3987e5; --edge-null:#383835;
  --probe:#199e70; --margin-line:#3987e5; --solve-line:#199e70;
  --tax-di:#3987e5; --tax-dn:#199e70; --tax-ui:#c98500; --tax-un:#008300;
}
:root[data-theme="light"] .viz-root{
  --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --edge-harm:#e34948; --edge-help:#2a78d6; --edge-null:#c3c2b7;
  --probe:#1baf7a; --margin-line:#2a78d6; --solve-line:#1baf7a;
  --tax-di:#2a78d6; --tax-dn:#1baf7a; --tax-ui:#eda100; --tax-un:#008300;
}
*{box-sizing:border-box;margin:0}
body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{background:var(--plane);color:var(--ink);min-height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px 10px;border-bottom:1px solid var(--grid)}
header h1{font-size:17px;font-weight:650}
header .sub{color:var(--ink-2);font-size:12.5px;margin-top:2px}
.layout{display:flex;flex:1;min-height:0}
.sidebar{width:290px;min-width:220px;border-right:1px solid var(--grid);display:flex;flex-direction:column;max-height:calc(100vh - 62px)}
.filters{padding:10px 12px;border-bottom:1px solid var(--grid);display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--axis);border-radius:12px;padding:2px 10px;font-size:12px;cursor:pointer;color:var(--ink-2);background:none}
.chip.on{background:var(--ink);color:var(--plane);border-color:var(--ink)}
.tracelist{overflow-y:auto;flex:1}
.trow{padding:8px 12px;border-bottom:1px solid var(--grid);cursor:pointer}
.trow:hover{background:var(--surface-1)}
.trow.sel{background:var(--surface-1);box-shadow:inset 3px 0 0 var(--ink)}
.trow .tid{font-size:11px;color:var(--muted);font-family:ui-monospace,monospace}
.trow .tsum{font-size:12.5px;margin-top:1px}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;border:1px solid var(--ring);
  border-radius:9px;padding:1px 8px;color:var(--ink-2);background:var(--surface-1);margin:1px 3px 1px 0}
.badge .dot{width:8px;height:8px;border-radius:50%;flex:none}
main{flex:1;overflow-y:auto;padding:16px 22px;max-height:calc(100vh - 62px)}
.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.card h2{font-size:13px;font-weight:650;color:var(--ink-2);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px}
.tiles{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
.tile{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:10px 14px;min-width:130px}
.tile .v{font-size:22px;font-weight:650}
.tile .l{font-size:11.5px;color:var(--ink-2);margin-top:2px;display:flex;align-items:center;gap:5px}
.howto{font-size:13px;color:var(--ink-2);display:none;margin-top:8px;max-width:74ch}
.howto.open{display:block}
.howto li{margin:4px 0 4px 16px}
button.link{background:none;border:none;color:var(--ink);text-decoration:underline;cursor:pointer;font-size:12.5px;padding:0}
.summary{font-size:14px;max-width:80ch}
.summary b.harm{color:var(--edge-harm)} .summary b.help{color:var(--edge-help)}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:6px 0 10px}
.controls select{font:inherit;background:var(--surface-1);color:var(--ink);border:1px solid var(--axis);border-radius:6px;padding:3px 6px}
.controls label{font-size:12.5px;color:var(--ink-2);display:flex;gap:5px;align-items:center}
.chainwrap{position:relative}
svg.edges{position:absolute;inset:0;pointer-events:none;overflow:visible}
svg.edges path{pointer-events:stroke}
.chain{position:relative;display:flex;flex-direction:column;gap:8px;margin:0 150px 0 120px}
.node{background:var(--surface-1);border:1px solid var(--ring);border-radius:8px;padding:8px 12px 8px 14px;position:relative}
.node .pb{position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--probe)}
.node.qnode,.node.anode{border-style:dashed;color:var(--ink-2)}
.node .stepno{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.node .txt{font-size:13px;margin-top:2px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer}
.node .txt.open{-webkit-line-clamp:unset}
.node.errsite{border-color:var(--ink)}
.node .meta{font-size:11px;color:var(--muted);margin-top:3px}
.curves{display:flex;gap:18px;flex-wrap:wrap}
.curvebox{flex:1;min-width:280px}
.curvebox .cap{font-size:11.5px;color:var(--ink-2);margin-bottom:3px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink-2);margin-top:8px}
.legend .sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.tooltip{position:fixed;background:var(--surface-1);border:1px solid var(--ring);border-radius:8px;
  padding:7px 10px;font-size:12px;pointer-events:none;box-shadow:0 4px 14px rgba(0,0,0,.15);
  z-index:50;display:none;max-width:320px;color:var(--ink)}
.etable{width:100%;border-collapse:collapse;font-size:12.5px}
.etable th{color:var(--ink-2);font-weight:600;text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}
.etable td{padding:4px 8px;border-bottom:1px solid var(--grid)}
.etable td.num{font-variant-numeric:tabular-nums;text-align:right}
.overflow{overflow-x:auto}
@media (max-width:900px){.chain{margin:0 60px 0 40px}.sidebar{width:210px}}
</style></head>
<body><div class="viz-root">
<header>
  <h1>CoT causal graph explorer <span style="color:var(--muted);font-weight:400">cot_causal_graph_v0</span></h1>
  <div class="sub">Detection (probe) vs influence (interventions) vs repair, per reasoning step.
    <button class="link" onclick="document.getElementById('howto').classList.toggle('open')">How do I read this?</button>
    <span id="gen" style="float:right"></span></div>
  <div id="howto" class="howto">
    <ul>
      <li><b>Each row is one reasoning step</b> of a real chain-of-thought. The <span style="color:var(--probe)">green left bar</span> is the internal correctness probe: taller/darker fill = the probe thinks this step is wrong ("detected").</li>
      <li><b>Right-side arcs = teacher-forced edges.</b> We replace one step (with the human-labeled wrong sibling, a paraphrase, or an off-topic step), keep everything after it fixed, and measure how the model's belief in the correct answer changes. <b style="color:var(--edge-harm)">Red arcs harm</b> the answer, <b style="color:var(--edge-help)">blue arcs help</b>; width = effect size.</li>
      <li><b>Left-side arcs = free-generation edges.</b> Same intervention, but the model regenerates everything after it; we measure the drop in how often it still reaches the correct answer over K rollouts.</li>
      <li><b>Curves:</b> the belief curve tracks the answer margin after each step; the solve curve tracks P(correct) when regenerating from each prefix. A dip followed by recovery is a candidate <b>repair</b>.</li>
      <li><b>Taxonomy badges</b> classify the ground-truth error site: detected/undetected × influential/inert. "Undetected + influential" is the dangerous cell.</li>
      <li>Probe-delta edges are shown only as a <i>diagnostic</i> (the probe is a readout, not a lever — S3 Stage 5).</li>
    </ul>
  </div>
</header>
<div class="layout">
  <div class="sidebar">
    <div class="filters" id="filters"></div>
    <div class="tracelist" id="tracelist"></div>
  </div>
  <main id="main"></main>
</div>
<div class="tooltip" id="tooltip"></div>
</div>
<script>
const PAYLOAD = __PAYLOAD__;
const TAX = {
  detected_influential:{v:"--tax-di",label:"detected + influential",desc:"genuine propagated error"},
  detected_inert:{v:"--tax-dn",label:"detected + inert",desc:"flagged, never affects the answer"},
  undetected_influential:{v:"--tax-ui",label:"undetected + influential",desc:"dangerous hidden failure"},
  undetected_inert:{v:"--tax-un",label:"undetected + inert",desc:"benign noise"}};
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const fmt = (x,d=2) => (x===null||x===undefined||Number.isNaN(x)) ? "–" : (+x).toFixed(d);
const pct = x => (x===null||x===undefined||Number.isNaN(x)) ? "–" : Math.round(100*x)+"%";
let state = {arm:"all", tax:"all", sel:null};

document.getElementById('gen').textContent = "generated "+PAYLOAD.generated;

function traceTitle(g){
  if(g.arm==="forks") return "fork trace · error at step "+(g.site?g.site.t+1:"?")+" / "+g.n_steps;
  return "on-policy · "+(g.traj_correct?"ended correct":"ended wrong")+" · "+g.n_steps+" steps";
}
function badge(txt, varName, title){
  return `<span class="badge" title="${title||''}"><span class="dot" style="background:var(${varName})"></span>${txt}</span>`;
}
function taxBadge(t){ const x=TAX[t]; return x?badge(x.label,x.v,x.desc):""; }

function renderFilters(){
  const f=document.getElementById('filters');
  const armChips=["all","forks","onpolicy"].map(a=>
    `<button class="chip ${state.arm===a?'on':''}" onclick="state.arm='${a}';refresh()">${a}</button>`).join("");
  const taxChips=["all",...Object.keys(TAX)].map(t=>
    `<button class="chip ${state.tax===t?'on':''}" title="${t==='all'?'':TAX[t].desc}"
      onclick="state.tax='${t}';refresh()">${t==="all"?"all cells":TAX[t].label}</button>`).join("");
  f.innerHTML=armChips+taxChips;
}
function visibleTraces(){
  return PAYLOAD.traces.filter(g=>
    (state.arm==="all"||g.arm===state.arm) &&
    (state.tax==="all"||(g.site&&g.site.taxonomy===state.tax)));
}
function renderList(){
  const el=document.getElementById('tracelist');
  const vis=visibleTraces();
  el.innerHTML=vis.map(g=>{
    const sel=state.sel===g.trace_id?"sel":"";
    return `<div class="trow ${sel}" onclick="state.sel='${g.trace_id}';refresh()">
      <div class="tid">${g.trace_id.slice(0,26)}</div>
      <div class="tsum">${traceTitle(g)}</div>
      <div>${g.site?taxBadge(g.site.taxonomy):""}${g.site&&g.site.repaired?badge("repaired","--tax-dn","model recovers despite the error"):""}</div>
    </div>`;}).join("") || `<div style="padding:14px;color:var(--muted)">no traces match</div>`;
  if(vis.length && !vis.find(g=>g.trace_id===state.sel)) state.sel=vis[0].trace_id;
}

function overview(){
  const ct=PAYLOAD.crosstab||{}; const cells=ct.cells||{};
  const tiles=Object.keys(TAX).map(t=>
    `<div class="tile"><div class="v">${cells[t]||0}</div>
     <div class="l"><span class="dot" style="width:8px;height:8px;border-radius:50%;background:var(${TAX[t].v})"></span>${TAX[t].label}</div></div>`).join("");
  const extra=[];
  if(ct.p_inert_errors!==undefined) extra.push(`<div class="tile"><div class="v">${pct(ct.p_inert_errors)}</div><div class="l">of labeled errors are causally inert</div></div>`);
  const rep=(ct.repair_inventory||{}).n_repaired_sites;
  if(rep!==undefined) extra.push(`<div class="tile"><div class="v">${rep}</div><div class="l">verified repair candidates</div></div>`);
  return `<div class="card"><h2>Detection × influence, ${ct.n_sites_test||0} ground-truth error sites (test split)</h2>
    <div class="tiles">${tiles}${extra.join("")}</div></div>`;
}

function summaryText(g){
  if(!g.site) {
    return g.arm==="onpolicy"
      ? `The model's own trajectory, final answer ${g.traj_correct?"<b class='help'>correct</b>":"<b class='harm'>wrong</b>"}. Explore where the solve-from-here curve drops and whether the probe flags those steps.`
      : "No classified error site for this trace.";
  }
  const s=g.site, step=s.t+1;
  const det=s.detected?`the probe <b>flagged it</b> (logit ${fmt(s.probe_at_error)} > threshold)`:`the probe <b>missed it</b> (logit ${fmt(s.probe_at_error)})`;
  let inf;
  if(s.influential_fg!==null&&s.influential_fg!==undefined){
    inf=s.influential_fg?`regenerating after the swap, the solve rate <b class="harm">drops by ${pct(-s.fg_delta)}</b> (recovery ${pct(s.recovery_rate)})`
      :`the solve rate barely moves (Δ ${fmt(s.fg_delta,2)}, recovery ${pct(s.recovery_rate)})`;
  } else {
    inf=s.influential_tf?`belief in the correct answer <b class="harm">moves by ${fmt(s.d_margin_final)}</b> (beyond the control null)`
      :`belief in the correct answer barely moves (Δ ${fmt(s.d_margin_final)}, within control noise)`;
  }
  const rep=s.repaired?` The model <b class="help">often recovers anyway</b> — a repair candidate.`:"";
  return `Human annotators labeled <b>step ${step}</b> wrong. Internally, ${det}. Causally, when we swap in that wrong step, ${inf}.${rep}
    <div style="margin-top:6px">${taxBadge(s.taxonomy)}${s.repaired?badge("repaired","--tax-dn","recovery rate ≥ 50%"):""}</div>`;
}

function probeScale(g){
  const vals=g.nodes.map(n=>n.probe_l28).filter(v=>v!==null);
  const lo=Math.min(...vals), hi=Math.max(...vals);
  return v=>v===null?0:(hi>lo?(v-lo)/(hi-lo):0.5);
}

function renderTrace(g){
  const intervs=[...new Set(g.edges.filter(e=>e.family==="tf").map(e=>e.interv))];
  const sel=window._interv&&intervs.includes(window._interv)?window._interv:(intervs.includes("swap_wrong")?"swap_wrong":intervs[0]);
  window._interv=sel;
  const ps=probeScale(g);
  const nodes=g.nodes.map(n=>{
    const flag=n.detected?badge("probe flag","--probe","probe logit above the val-selected threshold"):"";
    const err=n.is_error_site?badge("labeled error","--edge-harm","PRM800K rating −1 sibling swapped here"):"";
    return `<div class="node ${n.is_error_site?'errsite':''}" id="nd${n.idx}"
        data-tip="step ${n.idx+1} · probe ${fmt(n.probe_l28)} · margin ${fmt(n.margin)} · logp ${fmt(n.step_logp)} · entropy ${fmt(n.entropy)}">
      <div class="pb" style="opacity:${0.25+0.75*ps(n.probe_l28)}"></div>
      <div class="stepno">step ${n.idx+1} ${err}${flag}</div>
      <div class="txt" onclick="this.classList.toggle('open')">${escapeHtml(n.text||"")}</div>
      <div class="meta">probe ${fmt(n.probe_l28)} · belief margin ${fmt(n.margin)}</div>
    </div>`;}).join("");
  const chain=`<div class="chainwrap"><svg class="edges" id="edgesvg"></svg>
    <div class="chain" id="chain">
      <div class="node qnode" id="ndq"><div class="stepno">question</div><div class="txt" onclick="this.classList.toggle('open')">${escapeHtml(g.question||g.trace_id)}</div></div>
      ${nodes}
      <div class="node anode" id="nda"><div class="stepno">final answer</div>
        <div class="txt">answer margin ${fmt(g.margin_curve[g.margin_curve.length-1])} at the last boundary${g.traj_correct!==undefined?" · trajectory "+(g.traj_correct?"correct":"wrong"):""}</div></div>
    </div></div>`;
  const controls=`<div class="controls">
      <label>teacher-forced intervention
        <select onchange="window._interv=this.value;refresh()">
          ${intervs.map(i=>`<option ${i===sel?"selected":""}>${i}</option>`).join("")}
        </select></label>
      <label><input type="checkbox" id="showdiag" ${window._diag?"checked":""}
        onchange="window._diag=this.checked;refresh()"> show probe-delta edges (diagnostic)</label>
    </div>`;
  const legend=`<div class="legend">
    <span><span class="sw" style="background:var(--edge-harm)"></span>harms the answer</span>
    <span><span class="sw" style="background:var(--edge-help)"></span>helps the answer</span>
    <span><span class="sw" style="background:var(--edge-null);height:2px"></span>within control noise</span>
    <span>right arcs: teacher-forced · left arcs: free-generation</span></div>`;
  const curves=`<div class="card"><h2>Curves</h2><div class="curves">
      <div class="curvebox"><div class="cap">Belief curve: answer margin after each step (teacher-forced)</div>${sparkline(g.margin_curve,"--margin-line",g.site?g.site.t:null)}</div>
      ${g.fg_curve?`<div class="curvebox"><div class="cap">Solve curve: P(correct) regenerating from each prefix (K rollouts)</div>${sparkline(g.fg_curve,"--solve-line",null,0,1)}</div>`:""}
    </div></div>`;
  return `<div class="card"><h2>What happened here</h2><div class="summary">${summaryText(g)}</div></div>
    ${controls}${chain}${legend}${curves}
    <div class="card overflow"><h2>All edges</h2>${edgeTable(g)}</div>`;
}

function escapeHtml(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}

function sparkline(vals,colorVar,mark,fixMin,fixMax){
  const W=340,H=70,P=6;
  const vs=vals.map(v=>v===null?NaN:v), ok=vs.filter(v=>!Number.isNaN(v));
  if(!ok.length) return "<div style='color:var(--muted)'>no data</div>";
  const lo=fixMin!==undefined&&fixMin!==null?fixMin:Math.min(...ok), hi=fixMax!==undefined&&fixMax!==null?fixMax:Math.max(...ok);
  const x=i=>P+(W-2*P)*(vals.length===1?0.5:i/(vals.length-1));
  const y=v=>H-P-(H-2*P)*((v-lo)/((hi-lo)||1));
  let d="",pen=false;
  vs.forEach((v,i)=>{if(Number.isNaN(v)){pen=false;return;} d+=(pen?"L":"M")+x(i).toFixed(1)+","+y(v).toFixed(1);pen=true;});
  const zero=(lo<0&&hi>0)?`<line x1="${P}" x2="${W-P}" y1="${y(0)}" y2="${y(0)}" stroke="var(--axis)" stroke-dasharray="3 3"/>`:"";
  const dot=(mark!==null&&mark!==undefined&&!Number.isNaN(vs[mark]))?`<circle cx="${x(mark)}" cy="${y(vs[mark])}" r="4.5" fill="var(--edge-harm)" stroke="var(--surface-1)" stroke-width="2"/>`:"";
  const dots=vs.map((v,i)=>Number.isNaN(v)?"":`<circle cx="${x(i)}" cy="${y(v)}" r="7" fill="transparent" data-tip="step ${i+1}: ${fmt(v)}"/>`).join("");
  return `<svg width="${W}" height="${H}" style="max-width:100%;background:var(--surface-1);border:1px solid var(--ring);border-radius:8px">
    ${zero}<path d="${d}" fill="none" stroke="var(${colorVar})" stroke-width="2"/>${dot}${dots}</svg>`;
}

function edgeTable(g){
  const rows=g.edges.filter(e=>e.kind!=="probe_diag"||window._diag).map(e=>{
    const dst=e.dst==="answer"?"final answer":"step "+(e.dst+1);
    const kind={margin:"Δ answer margin (TF)",logp:"Δ step log-prob (TF)",
      solve_rate:"Δ solve rate (FG)",probe_diag:"Δ probe logit (diagnostic)"}[e.kind]||e.kind;
    const sig=e.significant===true?"yes":e.significant===false?"no":"–";
    return `<tr><td>${e.interv}</td><td>step ${e.src+1} → ${dst}</td><td>${kind}</td>
      <td class="num">${fmt(e.delta,3)}</td><td>${sig}</td></tr>`;}).join("");
  return `<table class="etable"><thead><tr><th>intervention</th><th>edge</th><th>measure</th><th>Δ</th><th>beyond null?</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function drawEdges(g){
  const svg=document.getElementById('edgesvg'); if(!svg) return;
  const wrap=svg.parentElement, wr=wrap.getBoundingClientRect();
  svg.setAttribute("width",wr.width); svg.setAttribute("height",wr.height);
  const pos=id=>{const el=document.getElementById(id); if(!el) return null;
    const r=el.getBoundingClientRect();
    return {l:r.left-wr.left,r:r.right-wr.left,y:r.top-wr.top+r.height/2};};
  const cap=v=>Math.min(Math.abs(v),3);
  let paths="";
  const arc=(a,b,side,w,color,tip,dash)=>{
    const x0=side>0?a.r:a.l, x1=side>0?b.r:b.l;
    const bow=side*(46+18*w);
    paths+=`<path d="M${x0},${a.y} C${x0+bow},${a.y} ${x1+bow},${b.y} ${x1},${b.y}"
      fill="none" stroke="${color}" stroke-width="${(1+2.4*w).toFixed(1)}" ${dash?'stroke-dasharray="4 4"':''}
      stroke-linecap="round" opacity="0.85" data-tip="${tip}"><\/path>`;
  };
  const selInterv=window._interv;
  const tf=g.edges.filter(e=>e.family==="tf"&&e.interv===selInterv);
  const fg=g.edges.filter(e=>e.family==="fg"&&(e.interv===selInterv||e.interv==="swap_xtrace"));
  for(const e of tf){
    if(e.kind==="probe_diag"&&!window._diag) continue;
    const a=pos("nd"+e.src), b=pos(e.dst==="answer"?"nda":"nd"+e.dst);
    if(!a||!b) continue;
    const w=cap(e.delta)/3;
    const color=e.kind==="probe_diag"?css('--muted')
      :(e.significant===false&&e.kind==="margin")?css('--edge-null')
      :(e.delta<0?css('--edge-harm'):css('--edge-help'));
    const what={margin:"Δ answer margin",logp:"Δ step log-prob",probe_diag:"Δ probe (diagnostic)"}[e.kind];
    arc(a,b,+1,w,color,`${e.interv}: step ${e.src+1} → ${e.dst==="answer"?"answer":"step "+(e.dst+1)} · ${what} = ${fmt(e.delta,3)}`,e.kind==="probe_diag");
  }
  for(const e of fg){
    const a=pos("nd"+e.src), b=pos("nda");
    if(!a||!b) continue;
    const w=Math.min(Math.abs(e.delta),1);
    const color=e.significant?(e.delta<0?css('--edge-harm'):css('--edge-help')):css('--edge-null');
    arc(a,b,-1,w,color,`${e.interv} (free generation): solve rate Δ ${pct(e.delta)} · recovery ${pct(e.recovery_rate)}`);
  }
  svg.innerHTML=paths;
  attachTips(svg);
}

function attachTips(root){
  const tip=document.getElementById('tooltip');
  root.querySelectorAll('[data-tip]').forEach(el=>{
    el.addEventListener('mousemove',ev=>{
      tip.style.display='block'; tip.textContent=el.getAttribute('data-tip');
      tip.style.left=Math.min(ev.clientX+14,window.innerWidth-330)+'px';
      tip.style.top=(ev.clientY+12)+'px';});
    el.addEventListener('mouseleave',()=>tip.style.display='none');
  });
}

function refresh(){
  renderFilters(); renderList();
  const g=PAYLOAD.traces.find(t=>t.trace_id===state.sel);
  const main=document.getElementById('main');
  main.innerHTML=overview()+(g?renderTrace(g):"<div class='card'>select a trace</div>");
  if(g){ requestAnimationFrame(()=>{drawEdges(g); attachTips(document.getElementById('main'));}); }
  else attachTips(main);
}
window.addEventListener('resize',()=>{const g=PAYLOAD.traces.find(t=>t.trace_id===state.sel); if(g) drawEdges(g);});
if(!PAYLOAD.traces.length){document.getElementById('main').innerHTML="<div class='card'>payload empty — run stage 3 first</div>";}
else refresh();
</script></body></html>
"""


if __name__ == "__main__":
    main()
