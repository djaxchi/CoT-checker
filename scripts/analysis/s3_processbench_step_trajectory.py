"""Last-layer trajectories across the reasoning: does the path deflect where it errs?

Companion to s3_prm800k_layer_trajectory.py, but the axis is flipped. There we fixed the
token and walked across LAYERS. Here we fix the LAYER (the last one, L28 = `layer:"last"`
in the encoding) and walk across the STEPS generated throughout one reasoning trace.

Substrate: the S1 7B DenseLinear ProcessBench eval shards. Unlike the balanced PRM800K
heldout sample, these encode EVERY step of each trace contiguously (step_idx 0..n_steps-1),
last token, last layer -- so a per-trace curve is a real reasoning timeline. No re-encode.

ProcessBench labels each trace with the index of its FIRST erroneous step (-1 = fully
correct). So per step we know: correct-prefix (before the error or in a correct trace),
the error step itself, and post-error steps.

Each step -> one summary number per the chosen metric (same as the layer-trajectory script):
  - raw_max : max over hidden dims (exposed to massive-activation dims)
  - l2_norm : L2 norm of the last-layer representation

Plots, per metric:
  (1) a readable sample of traces: summary vs step_idx, green=correct trace,
      red=erroring trace with a star at its first-error step
  (2) error-aligned mean trajectory: x = step_idx - first_error, mean +/- 95% CI over
      erroring traces -- the honest test of whether the curve deflects AT the error (x=0)
  (3) per-step discriminability: AUC and Cohen's d of the summary separating the
      first-error step from correct-prefix steps

Pure numpy + matplotlib; reads npy + meta directly (no probe, no HF text).

Outputs (results/processbench_trajectory/):
  - step_trajectory_{subset}.png
  - step_trajectory_{subset}.json

Usage:
    python scripts/analysis/s3_processbench_step_trajectory.py                 # all subsets
    python scripts/analysis/s3_processbench_step_trajectory.py --subset math
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.processbench_probe_data import load_probe

RUN_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b")
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
ROOT = Path("results/processbench_trajectory")
SEED = 42
GREEN, RED = "#3cb44b", "#e6194B"


def summarize(H: np.ndarray, metric: str, probe=None) -> np.ndarray:
    """One scalar per step at the last layer. `probe`=(w,b) needed for probe_margin."""
    if metric == "raw_max":
        return H.max(axis=1)
    if metric == "l2_norm":
        return np.linalg.norm(H, axis=1)
    if metric == "probe_margin":
        if probe is None:
            raise ValueError("probe_margin needs the trained probe (w, b)")
        w, b = probe
        return H.astype(np.float32) @ w.astype(np.float32) + b   # signed logit margin
    raise ValueError(metric)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def directed_auc(score, y):
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


def load_steps(subset: str):
    shard = RUN_DIR / "processbench_eval_shards" / subset
    H = np.load(shard / "pb_step_h.npy").astype(np.float32)
    meta = [json.loads(l) for l in (shard / "pb_step_meta.jsonl").read_text().splitlines() if l]
    if len(meta) != H.shape[0]:
        raise ValueError(f"{subset}: meta {len(meta)} != hidden {H.shape[0]}")
    step_idx = np.array([m["step_idx"] for m in meta], dtype=np.int64)
    gold = np.array([m["label"] for m in meta], dtype=np.int64)        # first-error idx, -1=ok
    skipped = np.array([bool(m.get("skipped", False)) for m in meta])
    tid = np.array([m["id"] for m in meta])
    return H, step_idx, gold, skipped, tid


def plot_subset_traces(ax, traces, metric_vals, gold_by_trace, rng, n_per_group, center):
    """Draw a readable sample of per-trace curves. `traces` maps tid -> (step_idx, summary)."""
    correct = [t for t in traces if gold_by_trace[t] == -1]
    erroring = [t for t in traces if gold_by_trace[t] != -1]
    pick = (list(rng.choice(correct, min(n_per_group, len(correct)), replace=False))
            + list(rng.choice(erroring, min(n_per_group, len(erroring)), replace=False)))
    for t in pick:
        si, sm = traces[t]
        order = np.argsort(si)
        si, sm = si[order], sm[order]
        if center:
            sm = sm - sm.mean()
        g = gold_by_trace[t]
        col = GREEN if g == -1 else RED
        ax.plot(si, sm, "-o", color=col, alpha=0.5, lw=1.0, ms=3)
        if g != -1 and g in si:
            ax.plot(g, sm[list(si).index(g)], "*", color="#8a0d2c", ms=14, zorder=5)
    ax.set_xlabel("step index in trace")
    ax.set_ylabel(f"{metric_vals}" + (" (trace-centered)" if center else ""))


def error_aligned(ax, traces, gold_by_trace, win=6, center=True):
    """Mean +/- 95% CI of summary vs (step_idx - first_error). When `center`, each trace
    is offset to its own mean (reveals deflection for magnitude metrics); when not, the
    absolute level is kept (so probe_margin can be read against the margin=0 boundary)."""
    buckets = defaultdict(list)
    for t, (si, sm) in traces.items():
        g = gold_by_trace[t]
        if g == -1:
            continue
        if center:
            sm = sm - sm.mean()                  # remove trace-level offset
        for s, v in zip(si, sm):
            r = int(s - g)
            if -win <= r <= win:
                buckets[r].append(v)
    xs = sorted(buckets)
    mu = np.array([np.mean(buckets[r]) for r in xs])
    ci = np.array([1.96 * np.std(buckets[r], ddof=1) / np.sqrt(len(buckets[r]))
                   if len(buckets[r]) > 1 else 0.0 for r in xs])
    n_at = {r: len(buckets[r]) for r in xs}
    ax.axvline(0, color="#888", ls=":", lw=1)
    ax.axhline(0, color="#ccc", ls="-", lw=0.6)
    ax.plot(xs, mu, "-o", color=RED, lw=2)
    ax.fill_between(xs, mu - ci, mu + ci, color=RED, alpha=0.2)
    ax.set_xlabel("step index - first_error  (0 = the error step)")
    ax.set_ylabel("trace-centered summary" if center else "summary (absolute)")
    ax.set_title("error-aligned mean (deflection at 0?)")
    return {int(r): {"mean": float(m), "n": n_at[r]} for r, m in zip(xs, mu)}


def run_subset(subset, metrics, n_per_group, out_dir, rng):
    H, step_idx, gold, skipped, tid = load_steps(subset)
    keep = ~skipped
    H, step_idx, gold, tid = H[keep], step_idx[keep], gold[keep], tid[keep]
    n = len(tid)
    gold_by_trace = {}
    for t, g in zip(tid, gold):
        gold_by_trace[t] = int(g)
    n_traces = len(gold_by_trace)
    n_err = sum(v != -1 for v in gold_by_trace.values())
    print(f"\n=== {subset}: {n} steps, {n_traces} traces ({n_err} erroring) ===")

    # per-step group masks for discriminability
    is_error = (gold != -1) & (step_idx == gold)
    is_prefix = (gold == -1) | ((gold != -1) & (step_idx < gold))   # genuinely-correct steps

    R = {"subset": subset, "n_steps": int(n), "n_traces": n_traces,
         "n_erroring_traces": int(n_err), "metrics": {}}

    probe = load_probe(RUN_DIR) if "probe_margin" in metrics else None

    fig, axes = plt.subplots(len(metrics), 2, figsize=(13, 4.6 * len(metrics)),
                             squeeze=False)
    for r, metric in enumerate(metrics):
        summ = summarize(H, metric, probe)
        center = metric != "probe_margin"        # keep the margin on its absolute scale
        traces = {}
        for t in gold_by_trace:
            m = tid == t
            traces[t] = (step_idx[m], summ[m])

        plot_subset_traces(axes[r, 0], traces, metric, gold_by_trace, rng,
                           n_per_group, center=center)
        axes[r, 0].set_title(f"{metric}: sample traces (green=correct, red=erroring, *=error)")
        rel = error_aligned(axes[r, 1], traces, gold_by_trace, center=center)

        a = summ[is_error]; b = summ[is_prefix]
        disc = {"auc_error_vs_prefix": round(directed_auc(
                    np.concatenate([a, b]),
                    np.concatenate([np.ones(len(a)), np.zeros(len(b))])), 4),
                "cohens_d_error_minus_prefix": round(cohens_d(a, b), 4),
                "mean_error_step": float(a.mean()), "mean_correct_prefix": float(b.mean()),
                "n_error_steps": int(len(a)), "n_prefix_steps": int(len(b))}
        R["metrics"][metric] = {"discriminability": disc, "error_aligned": rel}
        print(f"[{metric}] error vs correct-prefix step: AUC={disc['auc_error_vs_prefix']:.3f} "
              f"d={disc['cohens_d_error_minus_prefix']:+.3f}  "
              f"(err {disc['mean_error_step']:.3g} vs prefix {disc['mean_correct_prefix']:.3g})")

    fig.suptitle(f"Last-layer step trajectories - ProcessBench/{subset}  "
                 f"({n_traces} traces, {n_err} erroring)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = out_dir / f"step_trajectory_{subset}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    (out_dir / f"step_trajectory_{subset}.json").write_text(json.dumps(R, indent=2))
    print(f"[done] -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", nargs="+", default=list(SUBSETS),
                    help=f"one or more of {SUBSETS}")
    ap.add_argument("--metrics", nargs="+", default=["raw_max", "l2_norm"])
    ap.add_argument("--n_per_group", type=int, default=10,
                    help="traces per group (correct/erroring) drawn in the sample panel")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for subset in args.subset:
        run_subset(subset, args.metrics, args.n_per_group, args.out_dir, rng)


if __name__ == "__main__":
    main()
