"""parametric_retrieval_component_v1 figures (local, no GPU).

Reads expE/{depth_curve.csv, results.csv, capture.parquet} and writes one
multi-panel PNG:

  A  depth curves: matched d_margin vs decoder layer for full / attn / mlp
     (which layers, and which component, carry the same-fact rescue).
  B  representation space (the headline): final-layer readout of every
     captured test pair, projected onto a 2D map whose x-axis is the LDA
     "retrieval axis" fit on non-retrieved (fail) vs retrieved (success)
     states. Shows the fail cloud, the success cloud, and where the SAME
     failed prompts land once patched with the matched donor vs mismatched /
     random controls. Centroid arrow makes the movement explicit.
  C  test control battery: matched vs the six controls, per component.

  python scripts/parametric_retrieval/prgc_viz.py \
      --run_dir runs/parametric_retrieval_access_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval_causal import lda_direction  # noqa: E402

MODE_COLOR = {"full": "#1f77b4", "attn": "#d62728", "mlp": "#2ca02c"}
COND_STYLE = {
    "fail":            dict(c="#c0392b", marker="o", label="non-retrieved (fail)"),
    "success":         dict(c="#27ae60", marker="o", label="retrieved (success)"),
    "matched":         dict(c="#2471a3", marker="^", label="fail + matched patch"),
    "mismatched_type": dict(c="#7f8c8d", marker="s", label="fail + mismatched patch"),
    "random_noise":    dict(c="#000000", marker="x", label="fail + random patch"),
}


def _stack(cap, mode, cond):
    sub = cap[(cap["mode"] == mode) & (cap["condition"] == cond)]
    if sub.empty:
        return np.empty((0, 0)), sub
    Z = np.vstack([np.asarray(z, dtype=np.float32) for z in sub.z])
    return Z, sub


def panel_depth(ax, run_dir):
    dc = pd.read_csv(run_dir / "expE" / "depth_curve.csv")
    for mode in ["full", "attn", "mlp"]:
        g = dc[dc["mode"] == mode].sort_values("layer")
        if g.empty:
            continue
        ax.plot(g.layer, g.d_margin, "-o", ms=3, color=MODE_COLOR[mode],
                label=mode)
        ax.fill_between(g.layer, g.lo, g.hi, color=MODE_COLOR[mode], alpha=0.15)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("decoder layer L (patched)")
    ax.set_ylabel("matched patch: d(gold - distractor margin)")
    ax.set_title("A. Rescue vs depth,\nby component", fontsize=10)
    ax.legend(fontsize=8)


def panel_space(ax, cap):
    Zf, _ = _stack(cap, "none", "fail")
    Zs, _ = _stack(cap, "none", "success")
    if Zf.size == 0 or Zs.size == 0:
        ax.text(0.5, 0.5, "no capture data", ha="center")
        return
    H = np.vstack([Zs, Zf]).astype(np.float64)
    y = np.concatenate([np.ones(len(Zs), bool), np.zeros(len(Zf), bool)])
    w = lda_direction(H, y)                      # retrieval axis
    Hc = H - H.mean(0)
    resid = Hc - np.outer(Hc @ w, w)
    _, _, vt = np.linalg.svd(resid, full_matrices=False)
    v2 = vt[0]
    mu = H.mean(0)

    def proj(Z):
        Zc = Z.astype(np.float64) - mu
        return Zc @ w, Zc @ v2

    order = [("none", "fail"), ("none", "success"),
             ("full", "mismatched_type"), ("full", "random_noise"),
             ("full", "matched")]
    cents = {}
    for mode, cond in order:
        Z, _ = _stack(cap, mode, cond)
        if Z.size == 0:
            continue
        x, yv = proj(Z)
        st = COND_STYLE[cond]
        ax.scatter(x, yv, s=14, alpha=0.45, c=st["c"], marker=st["marker"],
                   label=st["label"], linewidths=0.6)
        cents[cond] = (x.mean(), yv.mean())
    if "fail" in cents and "matched" in cents:
        x0, y0 = cents["fail"]
        x1, y1 = cents["matched"]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#2471a3"))
    ax.set_xlabel("LDA retrieval axis  (fail  ->  success)")
    ax.set_ylabel("orthogonal PC")
    ax.set_title("B. Failed states enter the retrieved region\nonly under "
                 "the matched patch", fontsize=10)
    ax.legend(fontsize=7, loc="best")


def panel_battery(ax, run_dir):
    res = pd.read_csv(run_dir / "expE" / "results.csv")
    conds = ["matched", "mismatched_type", "mismatched_rand",
             "random_noise", "noop", "reverse"]
    modes = [m for m in ["full", "attn", "mlp"] if m in set(res["mode"])]
    width = 0.8 / max(len(modes), 1)
    x = np.arange(len(conds))
    for i, mode in enumerate(modes):
        vals, los, his = [], [], []
        for c in conds:
            r = res[(res["mode"] == mode) & (res["condition"] == c)]
            if r.empty:
                vals.append(0.0)
                los.append(0.0)
                his.append(0.0)
                continue
            vals.append(r.d_margin.iloc[0])
            los.append(r.d_margin.iloc[0] - r.d_margin_lo.iloc[0])
            his.append(r.d_margin_hi.iloc[0] - r.d_margin.iloc[0])
        ax.bar(x + i * width, vals, width, yerr=[los, his], capsize=2,
               color=MODE_COLOR[mode], label=mode, alpha=0.85)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("d(margin) vs baseline")
    ax.set_title("C. Control battery at each\ncomponent's best layer", fontsize=10)
    ax.legend(fontsize=8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--out", type=Path, default=Path(
        "results/parametric_retrieval_component_v1/prgc_v1_results.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cap = pd.read_parquet(args.run_dir / "expE" / "capture.parquet")
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.0])
    panel_depth(fig.add_subplot(gs[0, 0]), args.run_dir)
    panel_space(fig.add_subplot(gs[0, 1]), cap)
    panel_battery(fig.add_subplot(gs[0, 2]), args.run_dir)
    fig.suptitle("parametric_retrieval_component_v1: depth x component "
                 "same-fact patching, Qwen2.5-7B-Instruct", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
