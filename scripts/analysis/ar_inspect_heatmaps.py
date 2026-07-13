"""attention_routing_v0 stage 1: token-level heatmaps for the inspection
subset saved by ar_extract_forks.py (head-mean candidate attention rows).

For each inspect/<fork_id>_<role>.npz this renders one figure per fork/role:
  top    head-mean attention (selected layers) from candidate query tokens
         (rows) to all context tokens (columns), sqrt-scaled, with region
         boundaries (question | steps | candidate) marked
  bottom the token-to-region assignment strip for manual alignment checking

Usage:
  python scripts/analysis/ar_inspect_heatmaps.py \
      --run_dir runs/attention_routing/forks_attn \
      --layers 0 6 12 18 24 27
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.attention_routing import REGION_NAMES  # noqa: E402

# region strip colors (identity is also labeled in the colorbar, not
# color-alone); validated-neutral set: gray, blue, teal, orange, dark gray
REGION_COLORS = ["#9ca3af", "#2563eb", "#0e7490", "#ea580c", "#374151"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path,
                    default=Path("runs/attention_routing/forks_attn"))
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="default <run_dir>/inspect_plots")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 6, 12, 18, 24, 27])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    inspect_dir = args.run_dir / "inspect"
    out_dir = args.out_dir or args.run_dir / "inspect_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(inspect_dir.glob("*.npz"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no inspection files under {inspect_dir}")

    made = []
    for fp in files:
        z = np.load(fp)
        attn = z["attn_headmean"].astype(np.float32)  # (layers, cand, L)
        regions, c0, c1 = z["regions"], int(z["c0"]), int(z["c1"])
        layers = [li for li in args.layers if li < attn.shape[0]]
        n_rows = len(layers)

        fig, axes = plt.subplots(
            n_rows + 1, 1, figsize=(12, 1.6 * n_rows + 1.2),
            gridspec_kw={"height_ratios": [4] * n_rows + [1]}, sharex=True)
        boundaries = np.flatnonzero(np.diff(regions) != 0) + 0.5
        for ax, li in zip(axes[:-1], layers):
            im = ax.imshow(np.sqrt(attn[li]), cmap="Blues", aspect="auto",
                           interpolation="nearest",
                           extent=(-0.5, len(regions) - 0.5, c1 - 0.5,
                                   c0 - 0.5))
            for b in boundaries:
                ax.axvline(b, color="#ea580c", lw=0.6, alpha=0.7)
            ax.set_ylabel(f"L{li}", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[:-1].tolist(), label="sqrt(head-mean attn)",
                     fraction=0.02)

        strip = axes[-1]
        strip.imshow(regions[None], cmap=matplotlib.colors.ListedColormap(
            REGION_COLORS), vmin=0, vmax=len(REGION_COLORS) - 1,
            aspect="auto", interpolation="nearest")
        strip.set_yticks([])
        strip.set_xlabel("context token index "
                         f"({' | '.join(REGION_NAMES)} strip below)")
        fig.suptitle(f"{fp.stem}  (candidate rows {c0}..{c1 - 1})",
                     fontsize=10)
        out_p = out_dir / f"{fp.stem}.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(out_p)

    print("[ar-inspect] wrote:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
