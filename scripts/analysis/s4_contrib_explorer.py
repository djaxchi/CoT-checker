"""S4 contrib-cluster (post-hoc): build a single self-contained HTML explorer
over the step representations.

Computes one UMAP 2D embedding per (representation, layer) — the same
normalize -> PCA-50 -> UMAP(cosine, seed) recipe as the pipeline plots — and
embeds everything (coords, cluster ids, tags, surface features, optional
correctness ratings, step text snippets) as JSON in one HTML file with a
canvas scatter: switch repr/layer, filter by step index, color by cluster /
tag / step index / length / tokens / correctness, hover for step text,
click a legend entry to isolate it.

UMAP here is navigation, not evidence; every quantitative claim goes through
the clustering tables.

Output: runs/contrib_cluster/explorer.html

Usage:
  python scripts/analysis/s4_contrib_explorer.py --run_dir runs/contrib_cluster \
    --reprs state qres contribution --layers 20 28
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.contrib_cluster import REPR_NAMES, TAG_NAMES, l2_normalize  # noqa: E402

RATING_ORDER = ["1", "0", "-1", "human", "ambiguous", "unmatched"]
RATING_LABELS = {"1": "correct (+1)", "0": "neutral (0)", "-1": "incorrect (-1)",
                 "human": "human-written", "ambiguous": "ambiguous", "unmatched": "unmatched"}


def compute_embedding(reprs_dir: Path, name: str, layer: int, pca_dim: int,
                      seed: int) -> np.ndarray:
    cache = reprs_dir / f"umap_{name}_layer_{layer}_pca{pca_dim}_seed{seed}.npy"
    if cache.exists():
        return np.load(cache)
    from umap import UMAP
    norm_path = reprs_dir / f"repr_{name}_norm_layer_{layer}.npy"
    if norm_path.exists():
        X = np.load(norm_path).astype(np.float32)
    else:
        X = l2_normalize(np.load(reprs_dir / f"repr_{name}_layer_{layer}.npy")
                         .astype(np.float32))
    Xp = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed).fit_transform(X)
    emb = UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(Xp)
    # normalize to [0,1] for the canvas; quantized later
    emb = emb - emb.min(axis=0)
    emb = emb / np.maximum(emb.max(axis=0), 1e-9)
    np.save(cache, emb.astype(np.float32))
    return emb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--reprs", type=str, nargs="+", default=list(REPR_NAMES))
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snippet_chars", type=int, default=220)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or (args.run_dir / "explorer.html")

    reprs_dir = args.run_dir / "reprs"
    meta = pd.read_parquet(reprs_dir / "step_metadata.parquet")
    tags = pd.read_parquet(args.run_dir / "tags.parquet")
    assert (tags["row_id"].to_numpy() == meta["row_id"].to_numpy()).all()
    n = len(meta)

    ratings_path = args.run_dir / "step_ratings.parquet"
    if ratings_path.exists():
        rat = pd.read_parquet(ratings_path)
        assert (rat["row_id"].to_numpy() == meta["row_id"].to_numpy()).all()
        rating_idx = [RATING_ORDER.index(r) for r in rat["rating"]]
        has_ratings = True
    else:
        rating_idx = [RATING_ORDER.index("unmatched")] * n
        has_ratings = False
        print("[explorer] no step_ratings.parquet; correctness mode will be flat")

    combos, emb_x, emb_y, clusters = [], {}, {}, {}
    for name in args.reprs:
        for li in args.layers:
            key = f"{name}_L{li}"
            combos.append(key)
            print(f"[explorer] UMAP {key} ...", flush=True)
            emb = compute_embedding(reprs_dir, name, li, args.pca, args.seed)
            emb_x[key] = np.round(emb[:, 0], 4).tolist()
            emb_y[key] = np.round(emb[:, 1], 4).tolist()
            cl_path = args.run_dir / "clusters" / f"clusters_{name}_layer_{li}.parquet"
            if cl_path.exists():
                cl = pd.read_parquet(cl_path)
                assert (cl["row_id"].to_numpy() == meta["row_id"].to_numpy()).all()
                clusters[key] = cl["cluster"].astype(int).tolist()
            else:
                clusters[key] = [-1] * n

    snippets = [" ".join(t.split())[: args.snippet_chars] for t in meta["step_text"]]
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "combos": combos,
        "reprs": args.reprs,
        "layers": args.layers,
        "hasRatings": has_ratings,
        "tagNames": list(TAG_NAMES) + ["NONE"],
        "ratingNames": [RATING_LABELS[r] for r in RATING_ORDER],
        "x": emb_x,
        "y": emb_y,
        "cluster": clusters,
        "stepIndex": meta["step_index"].astype(int).tolist(),
        "numSteps": meta["num_steps_in_trajectory"].astype(int).tolist(),
        "charLen": tags["char_len"].astype(int).tolist(),
        "tokenCount": meta["token_count"].astype(int).tolist(),
        "topTag": [(list(TAG_NAMES) + ["NONE"]).index(t) for t in tags["top_tag"]],
        "rating": rating_idx,
        "traj": meta["trajectory_id"].tolist(),
        "text": snippets,
    }

    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    out_path.write_text(html)
    print(f"[explorer] wrote {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB, {n} steps x {len(combos)} views)")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S4 step-representation explorer</title>
<style>
  :root {
    --surface: #fcfcfb; --panel: #f4f4f2; --border: #dddcd8;
    --text: #0b0b0b; --text-2: #52514e; --accent: #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #1a1a19; --panel: #242423; --border: #3a3a38;
      --text: #ffffff; --text-2: #c3c2b7; --accent: #3987e5;
    }
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--surface); color: var(--text);
    font: 14px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }
  header {
    padding: 10px 16px 8px; border-bottom: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
  }
  header h1 { font-size: 15px; font-weight: 600; margin-right: 6px; }
  .ctrl { display: flex; align-items: center; gap: 6px; }
  .ctrl label { color: var(--text-2); font-size: 12px; }
  select {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 3px 6px; font-size: 13px;
  }
  #main { display: flex; flex: 1; min-height: 0; }
  #plotwrap { flex: 1; position: relative; }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  #side {
    width: 260px; border-left: 1px solid var(--border); padding: 10px 12px;
    overflow-y: auto; font-size: 12.5px;
  }
  #side h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
             color: var(--text-2); margin: 4px 0 6px; }
  .leg { display: flex; align-items: center; gap: 7px; padding: 2px 4px;
         border-radius: 5px; cursor: pointer; user-select: none; }
  .leg:hover { background: var(--panel); }
  .leg.off { opacity: .32; }
  .leg .sw { width: 11px; height: 11px; border-radius: 3px; flex: none; }
  .leg .nm { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .leg .ct { color: var(--text-2); font-variant-numeric: tabular-nums; }
  #ramp { height: 10px; border-radius: 5px; margin: 4px 2px 2px; }
  #rampLab { display: flex; justify-content: space-between; color: var(--text-2);
             font-size: 11px; padding: 0 2px; }
  #tip {
    position: absolute; pointer-events: none; max-width: 340px; z-index: 5;
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 12px; display: none;
    box-shadow: 0 4px 14px rgba(0,0,0,.18);
  }
  #tip .meta { color: var(--text-2); margin-bottom: 4px; }
  #stat { color: var(--text-2); font-size: 12px; margin-left: auto; }
  footer { padding: 5px 16px; border-top: 1px solid var(--border);
           color: var(--text-2); font-size: 11.5px; }
</style>
</head>
<body>
<header>
  <h1>S4 step representations</h1>
  <div class="ctrl"><label>repr</label><select id="selRepr"></select></div>
  <div class="ctrl"><label>layer</label><select id="selLayer"></select></div>
  <div class="ctrl"><label>step</label><select id="selStep"></select></div>
  <div class="ctrl"><label>color</label><select id="selColor">
    <option value="cluster">cluster</option>
    <option value="topTag">top regex tag</option>
    <option value="stepIndex">step index</option>
    <option value="charLen">step length (chars)</option>
    <option value="tokenCount">prefix tokens</option>
    <option value="rating">correctness (post-hoc)</option>
  </select></div>
  <div class="ctrl"><label>size</label><select id="selSize">
    <option value="1.6">s</option><option value="2.4" selected>m</option>
    <option value="3.4">l</option></select></div>
  <span id="stat"></span>
</header>
<div id="main">
  <div id="plotwrap"><canvas id="cv"></canvas><div id="tip"></div></div>
  <div id="side">
    <h2 id="legTitle">legend</h2>
    <div id="legend"></div>
    <div id="rampBox" style="display:none">
      <div id="ramp"></div><div id="rampLab"><span id="rlo"></span><span id="rhi"></span></div>
    </div>
    <h2 style="margin-top:12px">notes</h2>
    <div style="color:var(--text-2)">
      UMAP(cosine) of PCA-50 on L2-normalized vectors, seed 42 — navigation only,
      not evidence of cluster structure. Click a legend entry to isolate it;
      click again to restore. Correctness ratings are matched post-hoc and were
      never used by the pipeline.
    </div>
  </div>
</div>
<footer id="foot"></footer>
<script>
const D = __PAYLOAD__;
// palette (dataviz reference categorical, light/dark selected)
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const CAT = dark
  ? ['#3987e5','#199e70','#c98500','#008300','#9085e9','#e66767','#d55181','#d95926']
  : ['#2a78d6','#1baf7a','#eda100','#008300','#4a3aa7','#e34948','#e87ba4','#eb6834'];
const GRAY = dark ? '#6f6e69' : '#b0afab';
const SEQ = dark
  ? ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b']
  : ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];
const RATE_COLOR = {  // status semantics: good / neutral / serious / other
  0: dark ? '#199e70' : '#1baf7a',   // correct
  1: dark ? '#c98500' : '#eda100',   // neutral
  2: dark ? '#e66767' : '#e34948',   // incorrect
  3: dark ? '#3987e5' : '#2a78d6',   // human-written
  4: dark ? '#9085e9' : '#4a3aa7',   // ambiguous
  5: GRAY,                            // unmatched
};
const n = D.n;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), wrap = document.getElementById('plotwrap');
const sels = ['selRepr','selLayer','selStep','selColor','selSize']
  .map(id => document.getElementById(id));
const [selRepr, selLayer, selStep, selColor, selSize] = sels;
D.reprs.forEach(r => selRepr.add(new Option(r, r)));
D.layers.forEach(l => selLayer.add(new Option('L' + l, l)));
selRepr.value = D.reprs.includes('contribution') ? 'contribution' : D.reprs[0];
const maxStep = Math.max(...D.stepIndex);
selStep.add(new Option('all', 'all'));
for (let s = 1; s <= maxStep; s++) selStep.add(new Option('step ' + s, s));
let hidden = new Set();   // hidden category keys (legend isolate)
let colorMode = 'cluster';

function comboKey() { return selRepr.value + '_L' + selLayer.value; }
function isCat(m) { return m === 'cluster' || m === 'topTag' || m === 'rating'; }

function catInfo(m) {
  const key = comboKey();
  if (m === 'cluster') {
    const cl = D.cluster[key];
    const ids = [...new Set(cl)].sort((a, b) => a - b);
    return {
      vals: cl,
      cats: ids,
      name: c => c < 0 ? 'noise' : 'cluster ' + c,
      color: (c, i) => c < 0 ? GRAY : CAT[ids.filter(x => x >= 0).indexOf(c) % 8],
    };
  }
  if (m === 'topTag') {
    const counts = {};
    D.topTag.forEach(t => counts[t] = (counts[t] || 0) + 1);
    const ids = Object.keys(counts).map(Number).sort((a, b) => counts[b] - counts[a]);
    const slot = {}; let k = 0;
    ids.forEach(t => {
      const nm = D.tagNames[t];
      slot[t] = (nm === 'NONE' || k >= 8) ? -1 : k++;
    });
    return {
      vals: D.topTag, cats: ids,
      name: t => D.tagNames[t],
      color: t => slot[t] < 0 ? GRAY : CAT[slot[t]],
    };
  }
  // rating
  return {
    vals: D.rating, cats: [0, 1, 2, 3, 4, 5],
    name: r => D.ratingNames[r],
    color: r => RATE_COLOR[r],
  };
}

function seqColor(v, lo, hi) {
  const t = Math.max(0, Math.min(1, (v - lo) / Math.max(hi - lo, 1e-9)));
  const i = Math.min(SEQ.length - 1, Math.floor(t * (SEQ.length - 1)));
  const j = Math.min(SEQ.length - 1, i + 1), f = t * (SEQ.length - 1) - i;
  const a = SEQ[i], b = SEQ[j];
  const ch = (s, k) => parseInt(s.slice(k, k + 2), 16);
  return 'rgb(' + [1, 3, 5].map(k =>
    Math.round(ch(a, k) + f * (ch(b, k) - ch(a, k)))).join(',') + ')';
}

let colors = new Array(n), visible = new Uint8Array(n);
let xs, ys, W, H, dpr;

function recompute() {
  const key = comboKey();
  xs = D.x[key]; ys = D.y[key];
  colorMode = selColor.value;
  const stepF = selStep.value;
  const legend = document.getElementById('legend');
  const rampBox = document.getElementById('rampBox');
  document.getElementById('legTitle').textContent = selColor.selectedOptions[0].text;
  let nVis = 0;
  if (isCat(colorMode)) {
    const info = catInfo(colorMode);
    rampBox.style.display = 'none';
    legend.innerHTML = '';
    const counts = {};
    for (let i = 0; i < n; i++) counts[info.vals[i]] = (counts[info.vals[i]] || 0) + 1;
    info.cats.forEach(c => {
      if (!(c in counts)) return;
      const div = document.createElement('div');
      div.className = 'leg' + (hidden.has(String(c)) ? ' off' : '');
      div.innerHTML = '<span class="sw" style="background:' + info.color(c) +
        '"></span><span class="nm">' + info.name(c) + '</span><span class="ct">' +
        counts[c] + '</span>';
      div.onclick = () => {
        const k = String(c);
        if (hidden.has(k)) hidden.delete(k);
        else if (hidden.size === 0) {           // isolate: hide all others
          info.cats.forEach(o => { if (o !== c) hidden.add(String(o)); });
        } else hidden.add(k);
        if (hidden.size >= Object.keys(counts).length) hidden.clear();
        recompute();
      };
      legend.appendChild(div);
    });
    for (let i = 0; i < n; i++) {
      const v = info.vals[i];
      visible[i] = (stepF === 'all' || D.stepIndex[i] == stepF) &&
                   !hidden.has(String(v)) ? 1 : 0;
      if (visible[i]) { colors[i] = info.color(v); nVis++; }
    }
  } else {
    legend.innerHTML = ''; rampBox.style.display = 'block';
    const raw = D[colorMode];
    const log = colorMode === 'charLen' || colorMode === 'tokenCount';
    const tr = v => log ? Math.log10(Math.max(v, 1)) : v;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { const t = tr(raw[i]); if (t < lo) lo = t; if (t > hi) hi = t; }
    document.getElementById('ramp').style.background =
      'linear-gradient(90deg,' + SEQ.join(',') + ')';
    document.getElementById('rlo').textContent = log ? Math.round(Math.pow(10, lo)) : lo;
    document.getElementById('rhi').textContent = log ? Math.round(Math.pow(10, hi)) : hi;
    for (let i = 0; i < n; i++) {
      visible[i] = (stepF === 'all' || D.stepIndex[i] == stepF) ? 1 : 0;
      if (visible[i]) { colors[i] = seqColor(tr(raw[i]), lo, hi); nVis++; }
    }
  }
  document.getElementById('stat').textContent =
    nVis.toLocaleString() + ' / ' + n.toLocaleString() + ' steps';
  draw();
}

function resize() {
  dpr = devicePixelRatio || 1;
  W = wrap.clientWidth; H = wrap.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

const PAD = 26;
function px(i) { return PAD + xs[i] * (W - 2 * PAD); }
function py(i) { return H - PAD - ys[i] * (H - 2 * PAD); }

function draw() {
  if (!xs) return;
  ctx.clearRect(0, 0, W, H);
  const r = parseFloat(selSize.value);
  ctx.globalAlpha = 0.62;
  for (let i = 0; i < n; i++) {
    if (!visible[i]) continue;
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.arc(px(i), py(i), r, 0, 6.2832);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

cv.addEventListener('mousemove', e => {
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = -1, bd = 90;                      // <= ~9.5px hit radius
  for (let i = 0; i < n; i++) {
    if (!visible[i]) continue;
    const dx = px(i) - mx, dy = py(i) - my, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  if (best < 0) { tip.style.display = 'none'; return; }
  tip.style.display = 'block';
  const ratingStr = D.hasRatings ? ' · ' + D.ratingNames[D.rating[best]] : '';
  tip.innerHTML = '<div class="meta">' + D.traj[best] + ' · step ' +
    D.stepIndex[best] + '/' + D.numSteps[best] + ' · ' +
    D.tagNames[D.topTag[best]] + ratingStr + '</div>' +
    D.text[best].replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const tw = 340;
  tip.style.left = Math.min(mx + 14, W - tw - 8) + 'px';
  tip.style.top = Math.min(my + 14, H - 120) + 'px';
});
cv.addEventListener('mouseleave', () => tip.style.display = 'none');

sels.forEach(s => s.onchange = () => {
  if (s === selColor || s === selRepr || s === selLayer) hidden.clear();
  if (s === selSize) { draw(); return; }
  recompute();
});
addEventListener('resize', resize);
document.getElementById('foot').textContent =
  'Generated ' + D.generated + ' · ' + n.toLocaleString() +
  ' PRM800K golden-path steps · views: ' + D.combos.join(', ') +
  (D.hasRatings ? '' : ' · correctness unavailable (run s4_contrib_ratings.py)');
resize(); recompute();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
