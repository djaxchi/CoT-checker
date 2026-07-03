"""S4 contrib-cluster (post-hoc): build a single self-contained HTML explorer
over the step representations.

Two datasets in one file, switched from the UI:
  - trajectory steps: the 18,691 golden-path steps (cluster / tag / position /
    length / post-hoc correctness coloring);
  - fork pairs (if runs/contrib_cluster/forks_hidden/ exists): the matched
    correct/incorrect continuations of identical prefixes, colored by label,
    with an optional line connecting the two sides of each pair.

Each (repr, layer[, dataset]) view is its own UMAP 2D embedding — the same
normalize -> PCA-50 -> UMAP(cosine, seed) recipe as the pipeline plots — with
coords cached as .npy next to the reprs so regeneration is cheap. UMAP here is
navigation, not evidence; every quantitative claim goes through the clustering
tables. The output overwrites runs/contrib_cluster/explorer.html in place.

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

from src.analysis.contrib_cluster import (  # noqa: E402
    REPR_NAMES,
    TAG_NAMES,
    l2_normalize,
    tag_step,
)

RATING_ORDER = ["1", "0", "-1", "human", "ambiguous", "unmatched"]
RATING_LABELS = {"1": "correct (+1)", "0": "neutral (0)", "-1": "incorrect (-1)",
                 "human": "human-written", "ambiguous": "ambiguous", "unmatched": "unmatched"}
ALL_TAGS = list(TAG_NAMES) + ["NONE"]


def umap_2d(X: np.ndarray, pca_dim: int, seed: int, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    from umap import UMAP
    Xp = PCA(n_components=min(pca_dim, X.shape[1], X.shape[0] - 1),
             random_state=seed).fit_transform(X)
    emb = UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(Xp)
    emb = emb - emb.min(axis=0)
    emb = emb / np.maximum(emb.max(axis=0), 1e-9)
    emb = emb.astype(np.float32)
    np.save(cache, emb)
    return emb


def top_tag_indices(texts: list[str]) -> list[int]:
    """Rarest-matching-tag index per text (corpus-relative), NONE if untagged."""
    M = np.array([[tag_step(t)[n] for n in TAG_NAMES] for t in texts], dtype=bool)
    order = np.argsort(M.sum(axis=0).astype(float))
    out = []
    for row in M:
        top = len(ALL_TAGS) - 1  # NONE
        for j in order:
            if row[j]:
                top = int(j)
                break
        out.append(top)
    return out


def snip(text: str, n: int) -> str:
    return " ".join(text.split())[:n]


def build_steps_dataset(args, reprs_dir: Path) -> dict:
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

    x, y, cluster = {}, {}, {}
    for name in args.reprs:
        for li in args.layers:
            key = f"{name}_L{li}"
            print(f"[explorer] steps UMAP {key} ...", flush=True)
            norm = reprs_dir / f"repr_{name}_norm_layer_{li}.npy"
            X = (np.load(norm).astype(np.float32) if norm.exists() else
                 l2_normalize(np.load(reprs_dir / f"repr_{name}_layer_{li}.npy")
                              .astype(np.float32)))
            cache = reprs_dir / f"umap_{name}_layer_{li}_pca{args.pca}_seed{args.seed}.npy"
            emb = umap_2d(X, args.pca, args.seed, cache)
            x[key] = np.round(emb[:, 0], 4).tolist()
            y[key] = np.round(emb[:, 1], 4).tolist()
            cl_path = args.run_dir / "clusters" / f"clusters_{name}_layer_{li}.parquet"
            if cl_path.exists():
                cl = pd.read_parquet(cl_path)
                assert (cl["row_id"].to_numpy() == meta["row_id"].to_numpy()).all()
                cluster[key] = cl["cluster"].astype(int).tolist()
            else:
                cluster[key] = [-1] * n

    return {
        "n": n,
        "hasRatings": has_ratings,
        "x": x, "y": y, "cluster": cluster,
        "stepIndex": meta["step_index"].astype(int).tolist(),
        "numSteps": meta["num_steps_in_trajectory"].astype(int).tolist(),
        "charLen": tags["char_len"].astype(int).tolist(),
        "tokenCount": meta["token_count"].astype(int).tolist(),
        "topTag": [ALL_TAGS.index(t) for t in tags["top_tag"]],
        "rating": rating_idx,
        "traj": meta["trajectory_id"].tolist(),
        "text": [snip(t, args.snippet_chars) for t in meta["step_text"]],
    }


def build_pairs_dataset(args) -> dict | None:
    fdir = args.run_dir / "forks_hidden"
    meta_path = fdir / "metadata.parquet"
    if not meta_path.exists():
        print("[explorer] no forks_hidden/metadata.parquet; pairs view skipped")
        return None
    meta = pd.read_parquet(meta_path)
    # rows are sorted (fork_id, role p0<prefix<correct<wrong) by the merge
    roles = meta["role"].to_numpy()
    assert (roles.reshape(-1, 4) == np.array(["p0", "prefix", "correct", "wrong"])).all(), \
        "forks metadata is not in (p0,prefix,correct,wrong) blocks"
    fk = meta.iloc[0::4].reset_index(drop=True)
    n_forks = len(fk)
    cor_meta = meta.iloc[2::4].reset_index(drop=True)
    wr_meta = meta.iloc[3::4].reset_index(drop=True)

    x, y = {}, {}
    for name in args.reprs:
        for li in args.layers:
            key = f"{name}_L{li}"
            print(f"[explorer] pairs UMAP {key} ...", flush=True)
            H = np.load(fdir / f"h_layer_{li}.npy").astype(np.float32)
            p0, pre = H[0::4], H[1::4]
            sides = []
            for h in (H[2::4], H[3::4]):        # correct, wrong
                if name == "state":
                    sides.append(h)
                elif name == "qres":
                    sides.append(h - p0)
                else:                            # contribution
                    sides.append(h - pre)
            X = l2_normalize(np.concatenate(sides, axis=0))
            cache = fdir / f"umap_pairs_{name}_layer_{li}_pca{args.pca}_seed{args.seed}.npy"
            emb = umap_2d(X, args.pca, args.seed, cache)
            x[key] = np.round(emb[:, 0], 4).tolist()
            y[key] = np.round(emb[:, 1], 4).tolist()

    texts = cor_meta["step_text"].tolist() + wr_meta["step_text"].tolist()
    return {
        "n": 2 * n_forks,
        "nForks": n_forks,
        "x": x, "y": y,
        # order: all correct sides, then all wrong sides
        "label": [0] * n_forks + [1] * n_forks,
        "pairId": list(range(n_forks)) * 2,
        "stepIndex": fk["step_index"].astype(int).tolist() * 2,
        "charLen": [len(t) for t in texts],
        "tokenCount": cor_meta["token_count"].astype(int).tolist()
                      + wr_meta["token_count"].astype(int).tolist(),
        "topTag": top_tag_indices(texts),
        "fork": fk["fork_id"].tolist() * 2,
        "text": [snip(t, args.snippet_chars) for t in texts],
    }


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

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "reprs": args.reprs,
        "layers": args.layers,
        "combos": [f"{n}_L{li}" for n in args.reprs for li in args.layers],
        "tagNames": ALL_TAGS,
        "ratingNames": [RATING_LABELS[r] for r in RATING_ORDER],
        "steps": build_steps_dataset(args, args.run_dir / "reprs"),
        "pairs": build_pairs_dataset(args),
    }

    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    out_path.write_text(html)
    n_pairs = payload["pairs"]["nForks"] if payload["pairs"] else 0
    print(f"[explorer] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB; "
          f"{payload['steps']['n']} steps, {n_pairs} fork pairs)")


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
  #lineWrap { display: none; align-items: center; gap: 5px; color: var(--text-2);
              font-size: 12px; }
  #main { display: flex; flex: 1; min-height: 0; }
  #plotwrap { flex: 1; position: relative; }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%;
           cursor: grab; touch-action: none; }
  canvas.panning { cursor: grabbing; }
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
  <div class="ctrl"><label>data</label><select id="selData"></select></div>
  <div class="ctrl"><label>repr</label><select id="selRepr"></select></div>
  <div class="ctrl"><label>layer</label><select id="selLayer"></select></div>
  <div class="ctrl"><label>step</label><select id="selStep"></select></div>
  <div class="ctrl"><label>color</label><select id="selColor"></select></div>
  <div class="ctrl"><label>size</label><select id="selSize">
    <option value="1.6">s</option><option value="2.4" selected>m</option>
    <option value="3.4">l</option></select></div>
  <div class="ctrl" id="lineWrap">
    <input type="checkbox" id="chkLines"><label for="chkLines">pair lines</label>
  </div>
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
    <div style="color:var(--text-2)" id="notes"></div>
  </div>
</div>
<footer id="foot"></footer>
<script>
const D = __PAYLOAD__;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const CAT = dark
  ? ['#3987e5','#199e70','#c98500','#008300','#9085e9','#e66767','#d55181','#d95926']
  : ['#2a78d6','#1baf7a','#eda100','#008300','#4a3aa7','#e34948','#e87ba4','#eb6834'];
const GRAY = dark ? '#6f6e69' : '#b0afab';
const SEQ = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];
const GOOD = dark ? '#199e70' : '#1baf7a', BAD = dark ? '#e66767' : '#e34948';
const RATE_COLOR = {0: GOOD, 1: dark ? '#c98500' : '#eda100', 2: BAD,
                    3: dark ? '#3987e5' : '#2a78d6',
                    4: dark ? '#9085e9' : '#4a3aa7', 5: GRAY};

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), wrap = document.getElementById('plotwrap');
const el = id => document.getElementById(id);
const selData = el('selData'), selRepr = el('selRepr'), selLayer = el('selLayer'),
      selStep = el('selStep'), selColor = el('selColor'), selSize = el('selSize'),
      chkLines = el('chkLines');

selData.add(new Option('trajectory steps', 'steps'));
if (D.pairs) selData.add(new Option('fork pairs (±1)', 'pairs'));
D.reprs.forEach(r => selRepr.add(new Option(r, r)));
D.layers.forEach(l => selLayer.add(new Option('L' + l, l)));
selRepr.value = D.reprs.includes('contribution') ? 'contribution' : D.reprs[0];

let highlight = null;   // {idxs:[...], path:bool} — traced trajectory or fork pair
const COLOR_MODES = {
  steps: [['cluster','cluster'],['topTag','top regex tag'],['stepIndex','step index'],
          ['charLen','step length (chars)'],['tokenCount','prefix tokens'],
          ['rating','correctness (post-hoc)']],
  pairs: [['label','correct vs incorrect'],['topTag','top regex tag'],
          ['stepIndex','fork step index'],['charLen','step length (chars)'],
          ['tokenCount','prefix tokens']],
};
let hidden = new Set();
let ds, xs, ys, W, H, dpr, colors, visible;

function dataset() { return D[selData.value]; }
function comboKey() { return selRepr.value + '_L' + selLayer.value; }
function isCat(m) { return m === 'cluster' || m === 'topTag' || m === 'rating' || m === 'label'; }

function rebuildControls() {
  ds = dataset();
  const prev = selColor.value;
  selColor.innerHTML = '';
  COLOR_MODES[selData.value].forEach(([v, t]) => selColor.add(new Option(t, v)));
  selColor.value = [...selColor.options].some(o => o.value === prev) ? prev
    : COLOR_MODES[selData.value][0][0];
  const prevStep = selStep.value;
  selStep.innerHTML = '';
  selStep.add(new Option('all', 'all'));
  const mx = Math.max(...ds.stepIndex);
  for (let s = 1; s <= mx; s++) selStep.add(new Option('step ' + s, s));
  selStep.value = [...selStep.options].some(o => o.value === prevStep) ? prevStep : 'all';
  el('lineWrap').style.display = selData.value === 'pairs' ? 'flex' : 'none';
  el('notes').textContent = selData.value === 'pairs'
    ? 'Matched forks: one correct (+1) and one incorrect (-1) continuation of the ' +
      'IDENTICAL prefix, so position/length confounds cancel within a pair. Joint ' +
      'UMAP per repr/layer, seed 42 — navigation only. Click a legend entry to ' +
      'isolate; click a point to link it to its pair partner (Esc clears).'
    : 'UMAP(cosine) of PCA-50 on L2-normalized vectors, seed 42 — navigation only, ' +
      'not evidence of cluster structure. Click a legend entry to isolate it. ' +
      'Click a point to trace its trajectory step 1 → T (Esc clears). Correctness ' +
      'ratings are matched post-hoc and were never used by the pipeline.';
}

function catInfo(m) {
  const key = comboKey();
  if (m === 'cluster') {
    const cl = ds.cluster[key];
    const ids = [...new Set(cl)].sort((a, b) => a - b);
    const pos = ids.filter(x => x >= 0);
    return {vals: cl, cats: ids,
            name: c => c < 0 ? 'noise' : 'cluster ' + c,
            color: c => c < 0 ? GRAY : CAT[pos.indexOf(c) % 8]};
  }
  if (m === 'topTag') {
    const counts = {};
    ds.topTag.forEach(t => counts[t] = (counts[t] || 0) + 1);
    const ids = Object.keys(counts).map(Number).sort((a, b) => counts[b] - counts[a]);
    const slot = {}; let k = 0;
    ids.forEach(t => slot[t] = (D.tagNames[t] === 'NONE' || k >= 8) ? -1 : k++);
    return {vals: ds.topTag, cats: ids, name: t => D.tagNames[t],
            color: t => slot[t] < 0 ? GRAY : CAT[slot[t]]};
  }
  if (m === 'label')
    return {vals: ds.label, cats: [0, 1],
            name: v => v ? 'incorrect (-1)' : 'correct (+1)',
            color: v => v ? BAD : GOOD};
  return {vals: ds.rating, cats: [0,1,2,3,4,5],
          name: r => D.ratingNames[r], color: r => RATE_COLOR[r]};
}

function seqColor(v, lo, hi) {
  const t = Math.max(0, Math.min(1, (v - lo) / Math.max(hi - lo, 1e-9)));
  const i = Math.min(SEQ.length - 1, Math.floor(t * (SEQ.length - 1)));
  const j = Math.min(SEQ.length - 1, i + 1), f = t * (SEQ.length - 1) - i;
  const ch = (s, k) => parseInt(s.slice(k, k + 2), 16);
  return 'rgb(' + [1, 3, 5].map(k =>
    Math.round(ch(SEQ[i], k) + f * (ch(SEQ[j], k) - ch(SEQ[i], k)))).join(',') + ')';
}

function recompute() {
  ds = dataset();
  const key = comboKey();
  xs = ds.x[key]; ys = ds.y[key];
  const n = ds.n;
  colors = new Array(n); visible = new Uint8Array(n);
  const m = selColor.value, stepF = selStep.value;
  const legend = el('legend'), rampBox = el('rampBox');
  el('legTitle').textContent = selColor.selectedOptions[0].text;
  let nVis = 0;
  if (isCat(m)) {
    const info = catInfo(m);
    rampBox.style.display = 'none'; legend.innerHTML = '';
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
        else if (hidden.size === 0)
          info.cats.forEach(o => { if (o !== c) hidden.add(String(o)); });
        else hidden.add(k);
        if (hidden.size >= Object.keys(counts).length) hidden.clear();
        recompute();
      };
      legend.appendChild(div);
    });
    for (let i = 0; i < n; i++) {
      const v = info.vals[i];
      visible[i] = (stepF === 'all' || ds.stepIndex[i] == stepF) &&
                   !hidden.has(String(v)) ? 1 : 0;
      if (visible[i]) { colors[i] = info.color(v); nVis++; }
    }
  } else {
    legend.innerHTML = ''; rampBox.style.display = 'block';
    const raw = ds[m];
    const log = m === 'charLen' || m === 'tokenCount';
    const tr = v => log ? Math.log10(Math.max(v, 1)) : v;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { const t = tr(raw[i]); if (t < lo) lo = t; if (t > hi) hi = t; }
    el('ramp').style.background = 'linear-gradient(90deg,' + SEQ.join(',') + ')';
    el('rlo').textContent = log ? Math.round(Math.pow(10, lo)) : lo;
    el('rhi').textContent = log ? Math.round(Math.pow(10, hi)) : hi;
    for (let i = 0; i < n; i++) {
      visible[i] = (stepF === 'all' || ds.stepIndex[i] == stepF) ? 1 : 0;
      if (visible[i]) { colors[i] = seqColor(tr(raw[i]), lo, hi); nVis++; }
    }
  }
  el('stat').textContent = nVis.toLocaleString() + ' / ' + n.toLocaleString() +
    (selData.value === 'pairs' ? ' fork sides' : ' steps');
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
let view = {k: 1, tx: 0, ty: 0};          // zoom scale + pan offset (screen px)
function px(i) { return (PAD + xs[i] * (W - 2 * PAD)) * view.k + view.tx; }
function py(i) { return (H - PAD - ys[i] * (H - 2 * PAD)) * view.k + view.ty; }

function draw() {
  if (!xs) return;
  ctx.clearRect(0, 0, W, H);
  const n = ds.n, r = parseFloat(selSize.value);
  if (selData.value === 'pairs' && chkLines.checked) {
    const half = ds.nForks;
    ctx.globalAlpha = 0.14; ctx.strokeStyle = GRAY; ctx.lineWidth = 1;
    for (let i = 0; i < half; i++) {
      if (!visible[i] || !visible[i + half]) continue;
      ctx.beginPath();
      ctx.moveTo(px(i), py(i));
      ctx.lineTo(px(i + half), py(i + half));
      ctx.stroke();
    }
  }
  ctx.globalAlpha = highlight ? 0.18 : 0.62;
  for (let i = 0; i < n; i++) {
    if (!visible[i]) continue;
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.arc(px(i), py(i), r, 0, 6.2832);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  if (highlight) {
    const acc = dark ? '#3987e5' : '#2a78d6';
    const ring = dark ? '#ffffff' : '#0b0b0b';
    if (highlight.path && highlight.idxs.length > 1) {
      ctx.strokeStyle = acc; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.moveTo(px(highlight.idxs[0]), py(highlight.idxs[0]));
      for (let k = 1; k < highlight.idxs.length; k++)
        ctx.lineTo(px(highlight.idxs[k]), py(highlight.idxs[k]));
      ctx.stroke();
    }
    highlight.idxs.forEach((i, k) => {
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors[i] || GRAY;
      ctx.strokeStyle = ring; ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(px(i), py(i), r + 2.2, 0, 6.2832);
      ctx.fill(); ctx.stroke();
      if (highlight.path) {   // step number above each node of the traced path
        ctx.fillStyle = ring;
        ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(String(ds.stepIndex[i]), px(i), py(i) - r - 5);
      }
    });
    ctx.globalAlpha = 1;
  }
}

// ---- zoom & pan -----------------------------------------------------------
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const f = Math.exp(-e.deltaY * 0.0022);
  const k = Math.min(80, Math.max(1, view.k * f));
  const g = k / view.k;
  view.tx = mx - (mx - view.tx) * g;
  view.ty = my - (my - view.ty) * g;
  view.k = k;
  if (view.k === 1) { view.tx = 0; view.ty = 0; }
  tip.style.display = 'none';
  draw();
}, {passive: false});
let pan = null;
cv.addEventListener('mousedown', e => {
  pan = {x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false};
  cv.classList.add('panning');
});
addEventListener('mouseup', () => { pan = null; cv.classList.remove('panning'); });
cv.addEventListener('dblclick', () => { view = {k: 1, tx: 0, ty: 0}; draw(); });

function nearestVisible(mx, my, maxD2) {
  let best = -1, bd = maxD2;
  for (let i = 0; i < ds.n; i++) {
    if (!visible[i]) continue;
    const dx = px(i) - mx, dy = py(i) - my, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
cv.addEventListener('click', e => {
  if (pan && pan.moved) return;                       // drag, not a click
  const rect = cv.getBoundingClientRect();
  const best = nearestVisible(e.clientX - rect.left, e.clientY - rect.top, 120);
  if (best < 0) { highlight = null; draw(); return; }
  if (selData.value === 'pairs') {
    const half = ds.nForks;
    const j = best < half ? best + half : best - half;
    highlight = {idxs: [best, j], path: true};        // partner + connecting line
  } else {
    const tid = ds.traj[best];
    const idxs = [];
    for (let i = 0; i < ds.n; i++) if (ds.traj[i] === tid) idxs.push(i);
    idxs.sort((a, b) => ds.stepIndex[a] - ds.stepIndex[b]);
    highlight = {idxs, path: true};
  }
  draw();
});
addEventListener('keydown', e => {
  if (e.key === 'Escape') { highlight = null; draw(); }
});

cv.addEventListener('mousemove', e => {
  if (pan) {
    view.tx = pan.tx + (e.clientX - pan.x);
    view.ty = pan.ty + (e.clientY - pan.y);
    if (Math.abs(e.clientX - pan.x) + Math.abs(e.clientY - pan.y) > 2) pan.moved = true;
    tip.style.display = 'none';
    draw();
    return;
  }
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = -1, bd = 90;
  for (let i = 0; i < ds.n; i++) {
    if (!visible[i]) continue;
    const dx = px(i) - mx, dy = py(i) - my, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  if (best < 0) { tip.style.display = 'none'; return; }
  tip.style.display = 'block';
  let head;
  if (selData.value === 'pairs') {
    head = 'fork ' + ds.fork[best].slice(0, 8) + ' · step ' + ds.stepIndex[best] +
      ' · <b style="color:' + (ds.label[best] ? BAD : GOOD) + '">' +
      (ds.label[best] ? 'incorrect' : 'correct') + '</b> · ' +
      D.tagNames[ds.topTag[best]];
  } else {
    head = ds.traj[best] + ' · step ' + ds.stepIndex[best] + '/' + ds.numSteps[best] +
      ' · ' + D.tagNames[ds.topTag[best]] +
      (ds.hasRatings ? ' · ' + D.ratingNames[ds.rating[best]] : '');
  }
  tip.innerHTML = '<div class="meta">' + head + '</div>' +
    ds.text[best].replace(/&/g, '&amp;').replace(/</g, '&lt;');
  tip.style.left = Math.min(mx + 14, W - 348) + 'px';
  tip.style.top = Math.min(my + 14, H - 120) + 'px';
});
cv.addEventListener('mouseleave', () => tip.style.display = 'none');

[selData, selRepr, selLayer, selStep, selColor].forEach(s => s.onchange = () => {
  hidden.clear();
  if (s === selData || s === selRepr || s === selLayer) {
    view = {k: 1, tx: 0, ty: 0};   // new embedding -> reset zoom
    highlight = null;
  }
  if (s === selData) rebuildControls();
  recompute();
});
selSize.onchange = draw;
chkLines.onchange = draw;
addEventListener('resize', resize);
el('foot').textContent = 'Generated ' + D.generated + ' · ' +
  D.steps.n.toLocaleString() + ' golden-path steps' +
  (D.pairs ? ' · ' + D.pairs.nForks.toLocaleString() + ' matched fork pairs' : '') +
  ' · views: ' + D.combos.join(', ') +
  ' · scroll = zoom, drag = pan, double-click = reset';
// URL presets, e.g. explorer.html?data=pairs&repr=contribution&layer=28&color=label
const q = new URLSearchParams(location.search);
if (q.get('data') && D[q.get('data')]) selData.value = q.get('data');
if (q.get('repr') && D.reprs.includes(q.get('repr'))) selRepr.value = q.get('repr');
if (q.get('layer') && D.layers.includes(+q.get('layer'))) selLayer.value = q.get('layer');
rebuildControls();
if (q.get('color') && [...selColor.options].some(o => o.value === q.get('color')))
  selColor.value = q.get('color');
resize(); recompute();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
