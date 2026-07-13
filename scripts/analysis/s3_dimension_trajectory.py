"""Trajectory through DIMENSION space: track which hidden dims carry the top activations
as we move from one vector to the next, and ask whether correct and incorrect reasoning
visit different dimensions.

Earlier scripts put a SUMMARY VALUE (max / norm / probe margin) on the y-axis. Here the
y-axis is the DIMENSION INDEX: at each position we take the top-k activations of the
residual-stream vector and plot their indices, linking the argmax across positions into a
path. Because every layer's hidden state lives in the same residual basis (and the step
case is all one layer), a dimension index is comparable across the sequence.

Two sequence axes, mirroring the previous experiments:
  --source prm_layers : per step, the path across the 6 stored LAYERS (PRM800K heldout)
  --source pb_steps   : per trace, the path across STEPS at the last layer (ProcessBench)

For each source we draw, side by side, a sample of correct vs incorrect/erroring sequences
(top-1 argmax as a connected line, top-2/3 as faint markers), plus a population panel: how
often each dimension is the argmax at each position, split by label -- so a shared rogue
dimension shows up as a flat band and any label-dependent dimension shows up as a split.

Pure numpy + matplotlib; reuses existing loaders. Activations are SIGNED ("highest
activation"); pass --abs to rank by magnitude instead.

Outputs (results/dimension_trajectory/):
  - dim_trajectory_{source}.png
  - dim_trajectory_{source}.json   (top-dim frequencies per position, per label)

Usage:
    python scripts/analysis/s3_dimension_trajectory.py --source prm_layers
    python scripts/analysis/s3_dimension_trajectory.py --source pb_steps --subset math
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.prm800k_val_data import load_prm800k_multitoken

PRM_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
PB_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/processbench_eval_shards")
ROOT = Path("results/dimension_trajectory")
SEED = 42
GREEN, RED = "#3cb44b", "#e6194B"


def topk_dims(V: np.ndarray, k: int, use_abs: bool) -> np.ndarray:
    """(L, d) -> (L, k) indices of the k largest (signed or |.|) activations per row."""
    score = np.abs(V) if use_abs else V
    return np.argsort(-score, axis=1)[:, :k]


def draw_seq_panel(ax, seqs, positions, k, use_abs, color, label):
    """Draw a group's dimension trajectories. `seqs` is a list of (L, d) arrays."""
    for V in seqs:
        td = topk_dims(V, k, use_abs)                 # (L, k)
        x = positions[: len(td)]
        ax.plot(x, td[:, 0], "-o", color=color, alpha=0.55, lw=1.2, ms=4)   # argmax line
        for j in range(1, k):                          # runners-up as faint dots
            ax.scatter(x, td[:, j], color=color, alpha=0.25, s=10)
    ax.set_xlabel("position"); ax.set_ylabel("hidden dimension index")
    ax.set_title(label)


def population_freq(seqs, positions, use_abs):
    """At each position, Counter of argmax dimension over the group (-> top dims/freq)."""
    per_pos = [Counter() for _ in positions]
    for V in seqs:
        am = topk_dims(V, 1, use_abs)[:, 0]
        for p, d in enumerate(am):
            per_pos[p][int(d)] += 1
    return per_pos


def draw_population(ax, freq_c, freq_i, positions):
    """Scatter the top argmax dims per position; size ~ frequency; green=correct red=incorrect."""
    for freq, color, dx in [(freq_c, GREEN, -0.12), (freq_i, RED, +0.12)]:
        for p, c in enumerate(freq):
            if not c:
                continue
            tot = sum(c.values())
            for dim, n in c.most_common(5):
                ax.scatter(positions[p] + dx, dim, s=12 + 240 * n / tot,
                           color=color, alpha=0.5, edgecolors="none")
    ax.set_xlabel("position"); ax.set_ylabel("argmax dimension index")
    ax.set_title("population argmax dim per position (size ~ freq; green=correct red=incorrect)")


def load_prm_layers():
    manifest = json.loads((PRM_DIR / "prm800k_heldout_test_manifest.json").read_text())
    layers = list(manifest["layer_indices"])
    d0 = load_prm800k_multitoken(PRM_DIR, "prm800k_heldout_test", layers[0], "last")
    y = d0.label.astype(int)
    stack = np.stack([load_prm800k_multitoken(PRM_DIR, "prm800k_heldout_test", L, "last").hidden
                      for L in layers], axis=1)   # (n, L, d)
    # each example is its own sequence across layers
    seqs = [stack[i] for i in range(stack.shape[0])]
    return seqs, y, np.array(layers), "layer"


def load_pb_steps(subset):
    shard = PB_DIR / subset
    H = np.load(shard / "pb_step_h.npy").astype(np.float32)
    meta = [json.loads(l) for l in (shard / "pb_step_meta.jsonl").read_text().splitlines() if l]
    step_idx = np.array([m["step_idx"] for m in meta])
    gold = np.array([m["label"] for m in meta])
    skipped = np.array([bool(m.get("skipped", False)) for m in meta])
    tid = np.array([m["id"] for m in meta])
    seqs, ys = [], []
    for t in dict.fromkeys(tid):
        m = (tid == t) & (~skipped)
        if m.sum() < 2:
            continue
        order = np.argsort(step_idx[m])
        seqs.append(H[m][order])                  # (n_steps, d)
        ys.append(int(gold[m][0] != -1))          # 1 = trace has an error
    maxlen = max(len(s) for s in seqs)
    return seqs, np.array(ys), np.arange(maxlen), "step"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["prm_layers", "pb_steps"], default="pb_steps")
    ap.add_argument("--subset", default="math", help="ProcessBench subset (pb_steps only)")
    ap.add_argument("--k", type=int, default=3, help="top-k dims to mark per position")
    ap.add_argument("--n_per_group", type=int, default=8)
    ap.add_argument("--abs", action="store_true", help="rank by |activation| instead of signed")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    if args.source == "prm_layers":
        seqs, y, positions, axis = load_prm_layers()
        tag = "prm_layers"
    else:
        seqs, y, positions, axis = load_pb_steps(args.subset)
        tag = f"pb_steps_{args.subset}"

    ci = np.where(y == 0)[0]; ii = np.where(y == 1)[0]
    print(f"[dim-traj] source={args.source} axis={axis} n_seqs={len(seqs)} "
          f"correct={len(ci)} incorrect/erroring={len(ii)} k={args.k} abs={args.abs}")

    pick_c = rng.choice(ci, min(args.n_per_group, len(ci)), replace=False)
    pick_i = rng.choice(ii, min(args.n_per_group, len(ii)), replace=False)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    draw_seq_panel(ax[0, 0], [seqs[i] for i in pick_c], positions, args.k, args.abs,
                   GREEN, f"correct: top-{args.k} dim trajectory ({len(pick_c)} seqs)")
    draw_seq_panel(ax[0, 1], [seqs[i] for i in pick_i], positions, args.k, args.abs,
                   RED, f"incorrect/erroring: top-{args.k} dim trajectory ({len(pick_i)} seqs)")
    # shared y-limits for fair comparison
    ylim = (min(ax[0, 0].get_ylim()[0], ax[0, 1].get_ylim()[0]),
            max(ax[0, 0].get_ylim()[1], ax[0, 1].get_ylim()[1]))
    ax[0, 0].set_ylim(ylim); ax[0, 1].set_ylim(ylim)

    freq_c = population_freq([seqs[i] for i in ci], positions, args.abs)
    freq_i = population_freq([seqs[i] for i in ii], positions, args.abs)
    draw_population(ax[1, 0], freq_c, freq_i, positions)

    # overlay both groups' argmax lines on one axis for direct visual comparison
    draw_seq_panel(ax[1, 1], [seqs[i] for i in pick_c], positions, 1, args.abs,
                   GREEN, "argmax dim overlay (green=correct, red=incorrect)")
    draw_seq_panel(ax[1, 1], [seqs[i] for i in pick_i], positions, 1, args.abs, RED, "")
    ax[1, 1].set_title("argmax dim overlay (green=correct, red=incorrect)")

    xt = "layer" if axis == "layer" else "step"
    for a in ax.flat:
        a.set_xlabel(xt)
        if axis == "layer":
            a.set_xticks(positions)
    fig.suptitle(f"Dimension trajectory - {tag}  "
                 f"({'signed' if not args.abs else 'abs'} top activations)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = args.out_dir / f"dim_trajectory_{tag}.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    # dump per-position top dims per group
    def freq_json(freq):
        return [dict(c.most_common(5)) for c in freq]
    (args.out_dir / f"dim_trajectory_{tag}.json").write_text(json.dumps(
        {"source": args.source, "axis": axis, "positions": positions.tolist(),
         "k": args.k, "abs": args.abs, "n_correct": int(len(ci)), "n_incorrect": int(len(ii)),
         "argmax_freq_correct": freq_json(freq_c),
         "argmax_freq_incorrect": freq_json(freq_i)}, indent=2))
    print(f"[done] -> {png}")


if __name__ == "__main__":
    main()
