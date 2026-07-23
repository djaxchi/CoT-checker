"""parametric_retrieval_sae_decomp figure (local, no GPU).

Reads expH/{decomp.parquet, feature_recurrence.csv, summary.json} and writes a
3-panel PNG: does the flip concentrate on a few interpretable SAE features?

  A  concentration: fraction of |df| L1 mass captured by the top-k SAE features
     (how few features carry the donor-vs-recipient difference).
  B  feature recurrence across facts: fraction of pairs each feature is a top
     "donor-added" feature (shared vs fact-specific).
  C  features needed to capture 90% of |df|: matched fact vs a random-other
     donor null, with the SAE reconstruction FVU annotated.

  python scripts/parametric_retrieval/prgd_viz.py \
      --run_dir runs/parametric_retrieval_access_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOPK_GRID = [1, 2, 4, 8, 16, 32, 64]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--out", type=Path, default=Path(
        "results/parametric_retrieval_sae_decomp/prgd_v1_results.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    exp = args.run_dir / "expH"

    d = pd.read_parquet(exp / "decomp.parquet")
    rec = pd.read_csv(exp / "feature_recurrence.csv")
    summary = json.loads((exp / "summary.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    ax = axes[0]
    means = [d[f"cap_top{k}"].mean() for k in TOPK_GRID]
    lo = [d[f"cap_top{k}"].quantile(0.25) for k in TOPK_GRID]
    hi = [d[f"cap_top{k}"].quantile(0.75) for k in TOPK_GRID]
    ax.plot(TOPK_GRID, means, "-o", color="#1f77b4")
    ax.fill_between(TOPK_GRID, lo, hi, color="#1f77b4", alpha=0.15)
    ax.axhline(0.9, color="grey", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("# top SAE features")
    ax.set_ylabel("fraction of |df| L1 mass captured")
    ax.set_title("A. Does the flip concentrate on\na few SAE features?",
                 fontsize=10)

    ax = axes[1]
    if len(rec):
        y = rec.frac_pairs.to_numpy()
        ax.plot(np.arange(1, len(y) + 1), y, color="#d62728")
        ax.fill_between(np.arange(1, len(y) + 1), 0, y, color="#d62728",
                        alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("feature rank (by pairs it is donor-added)")
    ax.set_ylabel("fraction of pairs")
    ax.set_title("B. Feature recurrence across facts:\nshared vs fact-specific",
                 fontsize=10)

    ax = axes[2]
    ax.hist(d.n_cap90, bins=30, alpha=0.7, color="#1f77b4",
            label="matched fact")
    if "n_cap90_null" in d:
        ax.hist(d.n_cap90_null, bins=30, alpha=0.5, color="#7f8c8d",
                label="random-other donor (null)")
    ax.axvline(summary["median_n_cap90"], color="#1f77b4", ls="--")
    ax.set_xlabel("# features to capture 90% of |df|")
    ax.set_ylabel("pairs")
    ax.set_title(f"C. Sparsity of the flip\n(SAE recon FVU "
                 f"{summary['recon_fvu']:.2f} [{summary['fvu_gate']}])",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("parametric_retrieval_sae_decomp: is the flip a few "
                 "interpretable SAE features? (Qwen2.5-7B-Instruct, L28 SAE)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
