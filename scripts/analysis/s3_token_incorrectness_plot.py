#!/usr/bin/env python3
"""Plots for the per-token incorrectness trajectory (consumes s3_token_incorrectness_extract).

Reads {stem}_tokens.jsonl (+ {stem}_steps.jsonl) and answers, for the PRM800K 6k
held-out set, the question the last-token-only §15 study could not:

  1. TOKEN HEATMAP  For example incorrect steps, the step's tokens in reading order
     coloured by P(incorrect)=sigmoid(probe score), with a per-token certainty track.
     Shows *which* token fires the L28 detector.
  2. SPIKE          Is the incorrectness signal a localized spike or a plateau? Compares
     the per-step peakiness / prominence of incorrect vs correct steps, and where the
     peak sits within the step (mid-step vs the step-end the probe was trained on).
  3. COINCIDENCE    Does the firing token coincide with a certainty dip? Pooled
     probe-score vs certainty scatter, within-step argmax distance, and the
     probe<->certainty correlation.

Usage:
  python scripts/analysis/s3_token_incorrectness_plot.py \
    --tokens runs/s3_token_traj/qwen2_5_7b/prm800k_heldout_test_tokens.jsonl \
    --out runs/s3_token_traj/qwen2_5_7b/plots --probe_layer 28 --n_examples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.token_trajectory import coincidence, spike_stats  # noqa: E402

GREEN, RED = "#2c7fb8", "#e6550d"
CMAP = "coolwarm"


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def prettify(token: str) -> str:
    """Byte-level BPE token -> printable glyph (Qwen/GPT2 markers)."""
    t = token.replace("Ġ", "·").replace("▁", "·").replace("Ċ", "\\n").replace("ĉ", "\\t")
    return t if len(t) <= 12 else t[:11] + "…"


def group_by_step(rows: list[dict]) -> dict:
    steps: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        steps[r["uid"]].append(r)
    for uid in steps:
        steps[uid].sort(key=lambda r: r["tok_pos"])
    return steps


def _score_key(layer: int) -> str:
    return f"probe_score_l{layer}"


# --------------------------------------------------------------------------- #
# 1. token heatmap
# --------------------------------------------------------------------------- #

def plot_step_heatmap(step_rows, layer, out_path, cols=14):
    scores = np.array([r[_score_key(layer)] for r in step_rows])
    probs = sigmoid(scores)
    toks = [prettify(r["token"]) for r in step_rows]
    nll = np.array([r["nll"] for r in step_rows])
    T = len(step_rows)
    rows_n = int(np.ceil(T / cols))

    grid = np.full((rows_n, cols), np.nan)
    for k in range(T):
        grid[k // cols, k % cols] = probs[k]

    fig, (axh, axc) = plt.subplots(
        2, 1, figsize=(0.95 * cols, 0.62 * rows_n + 2.4),
        gridspec_kw={"height_ratios": [rows_n + 0.5, 2.2]})
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    axh.imshow(grid, cmap=CMAP, norm=norm, aspect="auto")
    for k in range(T):
        r, c = k // cols, k % cols
        axh.text(c, r, toks[k], ha="center", va="center", fontsize=8,
                 family="monospace",
                 color="white" if abs(probs[k] - 0.5) > 0.32 else "black")
    axh.set_xticks([]); axh.set_yticks([])
    lab = "INCORRECT" if step_rows[0]["label"] == 1 else "correct"
    peak = int(np.argmax(scores))
    axh.set_title(f"{step_rows[0]['uid']}  [{lab}]  L{layer} P(incorrect) per token "
                  f"(reading order; peak tok #{peak})", fontsize=9)

    x = np.arange(T)
    axc.plot(x, probs, "-o", ms=3, color="#333", label="P(incorrect)")
    axc.axhline(0.5, color="grey", lw=0.7, ls=":")
    axc.axvline(peak, color=RED, lw=1.0, ls="--")
    axc.set_ylim(0, 1); axc.set_ylabel("P(inc)", fontsize=8)
    axt = axc.twinx()
    axt.plot(x, nll, "-", color="#7a0177", lw=1.0, alpha=0.7, label="nll (surprise)")
    axt.set_ylabel("nll", fontsize=8, color="#7a0177")
    axc.set_xlabel("token position in step", fontsize=8)
    axc.set_xlim(-0.5, T - 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def select_examples(steps, layer, n):
    """Most-localized incorrect firings, plateau incorrect, and top false-fire correct."""
    def prom(rows):
        return spike_stats([r[_score_key(layer)] for r in rows])["prominence"]
    def peaky(rows):
        return spike_stats([r[_score_key(layer)] for r in rows])["peakiness"]

    inc = [u for u, rw in steps.items() if rw[0]["label"] == 1 and rw[0]["n_step_tokens"] >= 4]
    cor = [u for u, rw in steps.items() if rw[0]["label"] == 0 and rw[0]["n_step_tokens"] >= 4]
    inc_spiky = sorted(inc, key=lambda u: peaky(steps[u]), reverse=True)
    inc_flat = sorted(inc, key=lambda u: peaky(steps[u]))
    cor_hot = sorted(cor, key=lambda u: prom(steps[u]), reverse=True)
    k = max(1, n // 3)
    picks = []
    for grp, tag in ((inc_spiky, "inc_spiky"), (inc_flat, "inc_flat"), (cor_hot, "cor_hot")):
        for u in grp[:k]:
            if u not in [p[0] for p in picks]:
                picks.append((u, tag))
    return picks[:n]


# --------------------------------------------------------------------------- #
# 2. spike distributions
# --------------------------------------------------------------------------- #

def plot_spike(steps, layer, out_path):
    def feats(rows):
        return spike_stats([r[_score_key(layer)] for r in rows])
    inc = [feats(rw) for rw in steps.values() if rw[0]["label"] == 1 and rw[0]["n_step_tokens"] >= 4]
    cor = [feats(rw) for rw in steps.values() if rw[0]["label"] == 0 and rw[0]["n_step_tokens"] >= 4]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title in (
        (axes[0], "peakiness", "peakiness  (peak-mean)/std\nhigh = one token spikes"),
        (axes[1], "prominence", "prominence  peak - median (logits)"),
        (axes[2], "argmax_frac", "position of peak token in step\n(1.0 = last token)")):
        vi = np.array([f[key] for f in inc]); vc = np.array([f[key] for f in cor])
        lo, hi = np.percentile(np.concatenate([vi, vc]), [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(vc, bins=bins, color=GREEN, alpha=0.55, density=True, label="correct")
        ax.hist(vi, bins=bins, color=RED, alpha=0.55, density=True, label="incorrect")
        ax.axvline(np.median(vc), color=GREEN, lw=1.5, ls="--")
        ax.axvline(np.median(vi), color=RED, lw=1.5, ls="--")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.suptitle(f"L{layer} within-step spike structure (incorrect vs correct)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "peakiness_median_incorrect": float(np.median([f["peakiness"] for f in inc])),
        "peakiness_median_correct": float(np.median([f["peakiness"] for f in cor])),
        "argmax_at_last_token_frac_incorrect":
            float(np.mean([f["argmax_frac"] == 1.0 for f in inc])),
        "argmax_at_last_token_frac_correct":
            float(np.mean([f["argmax_frac"] == 1.0 for f in cor])),
        "n_incorrect": len(inc), "n_correct": len(cor),
    }


# --------------------------------------------------------------------------- #
# 3. coincidence
# --------------------------------------------------------------------------- #

def plot_coincidence(rows, steps, layer, out_path):
    sk = _score_key(layer)
    score = np.array([r[sk] for r in rows])
    label = np.array([r["label"] for r in rows])
    nll = np.array([r["nll"] for r in rows])
    entropy = np.array([r["entropy"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) pooled scatter probe score vs nll
    ax = axes[0]
    for lb, col, name in ((0, GREEN, "correct"), (1, RED, "incorrect")):
        m = label == lb
        ax.scatter(nll[m], score[m], s=4, c=col, alpha=0.25, linewidths=0, label=name)
    r_all = np.corrcoef(nll, score)[0, 1]
    ax.set_xlabel("per-token nll (surprise)"); ax.set_ylabel(f"L{layer} probe score")
    ax.set_title(f"probe score vs surprise (r={r_all:.2f})", fontsize=9)
    ax.set_xlim(np.percentile(nll, [0.5, 99]))
    ax.legend(fontsize=8)

    # (b) within-step argmax distance (incorrect steps)
    ax = axes[1]
    dist, corr_nll = [], []
    for rw in steps.values():
        if rw[0]["label"] != 1 or rw[0]["n_step_tokens"] < 4:
            continue
        s = [r[sk] for r in rw]; u = [r["nll"] for r in rw]
        c = coincidence(s, u)
        dist.append(c["argmax_distance_frac"])
        if not np.isnan(c["within_step_corr"]):
            corr_nll.append(c["within_step_corr"])
    ax.hist(dist, bins=np.linspace(0, 1, 21), color=RED, alpha=0.7)
    ax.axvline(np.median(dist), color="k", lw=1.5, ls="--",
               label=f"median={np.median(dist):.2f}")
    ax.set_xlabel("|argmax(probe) - argmax(nll)| / (T-1)")
    ax.set_title("firing token vs most-surprising token\n(incorrect steps; 0 = same token)",
                 fontsize=9)
    ax.legend(fontsize=8)

    # (c) within-step corr distribution
    ax = axes[2]
    ax.hist(corr_nll, bins=np.linspace(-1, 1, 31), color="#7a0177", alpha=0.7)
    ax.axvline(np.median(corr_nll), color="k", lw=1.5, ls="--",
               label=f"median r={np.median(corr_nll):.2f}")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("within-step corr(probe score, nll)")
    ax.set_title("does probe track surprise inside a step?", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    inc_m = label == 1
    return {
        "pooled_corr_probe_nll_all": float(r_all),
        "pooled_corr_probe_nll_incorrect": float(np.corrcoef(nll[inc_m], score[inc_m])[0, 1]),
        "pooled_corr_probe_entropy_all": float(np.corrcoef(entropy, score)[0, 1]),
        "median_argmax_distance_incorrect": float(np.median(dist)),
        "median_within_step_corr_probe_nll_incorrect": float(np.median(corr_nll)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=Path, required=True, help="{stem}_tokens.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe_layer", type=int, default=28)
    ap.add_argument("--n_examples", type=int, default=9)
    ap.add_argument("--cols", type=int, default=14, help="tokens per heatmap row")
    args = ap.parse_args()
    (args.out / "examples").mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.tokens)
    if _score_key(args.probe_layer) not in rows[0]:
        sys.exit(f"{_score_key(args.probe_layer)} not in tokens file; "
                 f"available: {[k for k in rows[0] if k.startswith('probe_score')]}")
    steps = group_by_step(rows)
    print(f"[plot] {len(rows)} tokens over {len(steps)} steps", flush=True)

    made = []
    for uid, tag in select_examples(steps, args.probe_layer, args.n_examples):
        safe = uid.replace("::", "__").replace("/", "_")
        p = args.out / "examples" / f"{tag}__{safe}.png"
        plot_step_heatmap(steps[uid], args.probe_layer, p, cols=args.cols)
        made.append(str(p))

    spike_metrics = plot_spike(steps, args.probe_layer, args.out / "spike_structure.png")
    coin_metrics = plot_coincidence(rows, steps, args.probe_layer,
                                    args.out / "coincidence.png")
    made += [str(args.out / "spike_structure.png"), str(args.out / "coincidence.png")]

    metrics = {"probe_layer": args.probe_layer, "n_steps": len(steps),
               "n_tokens": len(rows), "spike": spike_metrics, "coincidence": coin_metrics}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    verdict = ("localized spike" if spike_metrics["peakiness_median_incorrect"]
               > spike_metrics["peakiness_median_correct"] + 0.1 else "diffuse / plateau")
    summary = [
        "# Per-token incorrectness trajectory (PRM800K 6k held-out)\n",
        f"- probe layer L{args.probe_layer}; {len(steps)} steps, {len(rows)} tokens\n",
        "## Spike structure",
        f"- peakiness median: incorrect {spike_metrics['peakiness_median_incorrect']:.2f} "
        f"vs correct {spike_metrics['peakiness_median_correct']:.2f} -> **{verdict}**",
        f"- peak token is the LAST token in "
        f"{spike_metrics['argmax_at_last_token_frac_incorrect']*100:.0f}% of incorrect steps "
        f"(vs {spike_metrics['argmax_at_last_token_frac_correct']*100:.0f}% correct) "
        "-- how much of the firing is the step-end position the probe was trained on\n",
        "## Coincidence with certainty",
        f"- pooled corr(probe, nll) = {coin_metrics['pooled_corr_probe_nll_all']:.2f} "
        f"(incorrect-only {coin_metrics['pooled_corr_probe_nll_incorrect']:.2f})",
        f"- within incorrect steps: median |argmax(probe)-argmax(nll)| = "
        f"{coin_metrics['median_argmax_distance_incorrect']:.2f} of step length; "
        f"median within-step corr = "
        f"{coin_metrics['median_within_step_corr_probe_nll_incorrect']:.2f}",
        "  (near 0 distance + positive corr => the firing token IS the surprising token;",
        "   large distance + ~0 corr => the detector is not just per-token surprise)\n",
        "## Plots",
        "- `spike_structure.png`, `coincidence.png`, `examples/*.png`",
    ]
    (args.out / "token_incorrectness_report.md").write_text("\n".join(summary))

    print("[plot] wrote:")
    for p in made + [str(args.out / "metrics.json"),
                     str(args.out / "token_incorrectness_report.md")]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
