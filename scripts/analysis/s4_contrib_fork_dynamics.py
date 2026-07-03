"""S4 contrib-cluster: fork dynamics — does the CORRECT continuation move the
hidden state differently than the INCORRECT one, from the identical prefix?

For each fork we have h_0 (question), h_pre (prefix end) and two alternative
next states h_cor / h_wr. Everything is compared WITHIN the pair, so problem,
prefix, and position confounds cancel. Per (fork, side, layer):

  delta      = h_side - h_pre                  (the step's displacement)
  step_norm  = ||delta||
  cos_acc    = cos(delta, h_pre - h_0)         progress along accumulated
                                               reasoning (prefix >= 1 step)
  cos_generic= cos(delta, g)                   g = mean golden contribution
                                               direction (step_index >= 2,
                                               from the 18k trajectory steps)
  dist_q     = ||h_side - h_0|| / ||h_pre - h_0||   moved away from question?
  manifold   = mean cosine to the 25 nearest golden contributions
                                               (is this a "normal" step move?)

Reported per metric: paired mean difference (correct - wrong), the fraction of
pairs where correct > wrong, and Wilcoxon p when scipy is available. This is
descriptive geometry on labels, not a trained verifier.

Outputs (under --run_dir):
  fork_dynamics.csv, fork_dynamics_summary.csv
  plots/fork_dynamics_L<layer>.png

Usage:
  python scripts/analysis/s4_contrib_fork_dynamics.py --run_dir runs/contrib_cluster
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.contrib_cluster import l2_normalize  # noqa: E402

GOOD, BAD, GRAY = "#1baf7a", "#e34948", "#b0afab"


def rowwise_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9
    return num / den


def knn_mean_cos(Q: np.ndarray, R: np.ndarray, k: int, chunk: int = 1024) -> np.ndarray:
    """Mean cosine of each (normalized) query row to its k nearest reference rows."""
    out = np.empty(Q.shape[0], dtype=np.float32)
    for s in range(0, Q.shape[0], chunk):
        sims = Q[s:s + chunk] @ R.T
        part = np.partition(sims, -k, axis=1)[:, -k:]
        out[s:s + chunk] = part.mean(axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--knn", type=int, default=25)
    args = ap.parse_args()

    fdir = args.run_dir / "forks_hidden"
    meta = pd.read_parquet(fdir / "metadata.parquet")
    roles = meta["role"].to_numpy().reshape(-1, 4)
    assert (roles == np.array(["p0", "prefix", "correct", "wrong"])).all()
    fk = meta.iloc[0::4].reset_index(drop=True)
    n = len(fk)
    n_pre = fk["n_prefix_steps"].to_numpy()
    has_prefix = n_pre >= 1
    print(f"[fork-dyn] {n} forks ({int((~has_prefix).sum())} step-1 forks "
          "excluded from cos_acc)")

    smeta = pd.read_parquet(args.run_dir / "reprs" / "step_metadata.parquet")
    later = (smeta["step_index"] >= 2).to_numpy()

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    rows, summary_rows = [], []
    for li in args.layers:
        H = np.load(fdir / f"h_layer_{li}.npy").astype(np.float32)
        p0, pre, cor, wr = H[0::4], H[1::4], H[2::4], H[3::4]
        d_cor, d_wr = cor - pre, wr - pre
        acc = pre - p0

        # generic golden step-displacement direction + manifold reference
        C = np.load(args.run_dir / "reprs" / f"repr_contribution_layer_{li}.npy") \
            .astype(np.float32)
        Cn = l2_normalize(C[later])
        g = Cn.mean(axis=0)
        g /= np.linalg.norm(g)

        met = {
            "step_norm": (np.linalg.norm(d_cor, axis=1),
                          np.linalg.norm(d_wr, axis=1), None),
            "cos_acc": (rowwise_cos(d_cor, acc), rowwise_cos(d_wr, acc), has_prefix),
            "cos_generic": (l2_normalize(d_cor) @ g, l2_normalize(d_wr) @ g, None),
            "dist_q_ratio": (
                np.linalg.norm(cor - p0, axis=1) / (np.linalg.norm(pre - p0, axis=1) + 1e-9),
                np.linalg.norm(wr - p0, axis=1) / (np.linalg.norm(pre - p0, axis=1) + 1e-9),
                has_prefix),
            "manifold_knn_cos": (knn_mean_cos(l2_normalize(d_cor), Cn, args.knn),
                                 knn_mean_cos(l2_normalize(d_wr), Cn, args.knn), None),
        }
        for name, (vc, vw, mask) in met.items():
            m = np.ones(n, bool) if mask is None else mask
            diff = vc[m] - vw[m]
            s = {
                "layer": li, "metric": name, "n_pairs": int(m.sum()),
                "correct_mean": float(vc[m].mean()), "wrong_mean": float(vw[m].mean()),
                "paired_diff_mean": float(diff.mean()),
                "paired_diff_sem": float(diff.std(ddof=1) / np.sqrt(m.sum())),
                "frac_correct_gt_wrong": float((diff > 0).mean()),
            }
            if wilcoxon is not None:
                s["wilcoxon_p"] = float(wilcoxon(diff).pvalue)
            summary_rows.append(s)
            print(f"[fork-dyn] L{li} {name:<17} cor={s['correct_mean']:+.3f} "
                  f"wr={s['wrong_mean']:+.3f} diff={s['paired_diff_mean']:+.4f} "
                  f"(±{s['paired_diff_sem']:.4f}) frac>{'':0}={s['frac_correct_gt_wrong']:.3f}"
                  + (f" p={s['wilcoxon_p']:.1e}" if wilcoxon else ""), flush=True)

        for i in range(n):
            rows.append({
                "fork_id": fk["fork_id"].iloc[i], "layer": li,
                "step_index": int(fk["step_index"].iloc[i]),
                "n_prefix_steps": int(n_pre[i]),
                **{f"{k}_correct": float(v[0][i]) for k, v in met.items()},
                **{f"{k}_wrong": float(v[1][i]) for k, v in met.items()},
            })

        # ---- figure: paired distributions ---------------------------------
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
        panels = [
            ("cos_generic", "cos(step, generic golden direction)"),
            ("cos_acc", "cos(step, accumulated reasoning h_pre - h_0)"),
            ("manifold_knn_cos", f"mean cos to {args.knn}-NN golden contributions"),
            ("step_norm", "step displacement norm"),
        ]
        for ax, (name, title) in zip(axes, panels):
            vc, vw, mask = met[name]
            m = np.ones(n, bool) if mask is None else mask
            ax.hist(vc[m], bins=50, density=True, alpha=0.6, color=GOOD,
                    label=f"correct ({vc[m].mean():.3f})")
            ax.hist(vw[m], bins=50, density=True, alpha=0.6, color=BAD,
                    label=f"wrong ({vw[m].mean():.3f})")
            ax.set_title(title, fontsize=10)
            ax.legend(frameon=False, fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
        fig.suptitle(f"Fork dynamics: correct vs incorrect continuation of the "
                     f"same prefix — layer {li}", y=1.02)
        fig.tight_layout()
        out = args.run_dir / "plots" / f"fork_dynamics_L{li}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"[fork-dyn] wrote {out}")

    pd.DataFrame(rows).to_csv(args.run_dir / "fork_dynamics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.run_dir / "fork_dynamics_summary.csv",
                                      index=False)
    print("[fork-dyn] wrote fork_dynamics.csv, fork_dynamics_summary.csv")


if __name__ == "__main__":
    main()
