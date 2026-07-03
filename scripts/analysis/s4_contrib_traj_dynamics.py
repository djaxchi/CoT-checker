"""S4 contrib-cluster: trajectory DYNAMICS — the path h_0 -> h_1 -> ... -> h_T
one reasoning chain traces through activation space.

Works on the raw prefix hidden states (hidden_states/, which include the
question-only h_0 row), per layer. Questions asked, per trajectory:

1. Shape: net displacement ||h_T - h_0||, path length sum_i ||h_i - h_{i-1}||,
   straightness = net / path (1 = a straight line, ~0 = a random walk).
2. Progress: cos(step displacement, h_T - h_0) per step — does every step move
   toward where the reasoning ends up? And ||h_i - h_T|| vs relative position —
   does the state approach its final point monotonically?
3. Shared direction across chains: cos(net displacement A, net displacement B)
   for pairs of trajectories — same question vs different questions. High
   values mean all reasoning flows along a common "reasoning axis"; low values
   mean each problem carves its own path.

Outputs (under --run_dir):
  traj_dynamics.csv            one row per (trajectory, layer)
  traj_dynamics_summary.csv    aggregates per layer
  plots/traj_dynamics_L<layer>.png

Usage:
  python scripts/analysis/s4_contrib_traj_dynamics.py --run_dir runs/contrib_cluster
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

BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
GRAY = "#b0afab"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--n_pos_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hdir = args.run_dir / "hidden_states"
    meta = pd.read_parquet(hdir / "metadata.parquet")
    meta = meta.sort_values(["trajectory_id", "step_index"]).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    groups = []  # (trajectory_id, question, row indices h_0..h_T)
    for tid, g in meta.groupby("trajectory_id", sort=True):
        idxs = g.index.to_numpy()
        if len(idxs) >= 3:  # h_0 + >=2 steps
            groups.append((tid, g["question"].iloc[0], idxs))
    print(f"[traj-dyn] {len(groups)} trajectories")

    q_codes = pd.Series([q for _, q, _ in groups]).astype("category").cat.codes.to_numpy()
    by_q: dict[int, list[int]] = {}
    for gi, qc in enumerate(q_codes):
        by_q.setdefault(int(qc), []).append(gi)
    multi_q = [v for v in by_q.values() if len(v) > 1]

    rows, summaries = [], []
    for li in args.layers:
        H = np.load(hdir / f"h_layer_{li}.npy").astype(np.float32)
        disp_dirs = np.empty((len(groups), H.shape[1]), dtype=np.float32)
        # binned curves over relative position
        bins = args.n_pos_bins
        dist_fin = np.zeros(bins)
        dist_fin_n = np.zeros(bins)
        prog_cos = np.zeros(bins)
        prog_cos_n = np.zeros(bins)
        for gi, (tid, _, idxs) in enumerate(groups):
            P = H[idxs]                       # (T+1, d): h_0..h_T
            T = len(P) - 1
            deltas = P[1:] - P[:-1]
            step_norms = np.linalg.norm(deltas, axis=1)
            disp = P[-1] - P[0]
            net = float(np.linalg.norm(disp))
            plen = float(step_norms.sum())
            disp_dirs[gi] = disp / (net + 1e-9)
            prog = [cos(deltas[i], disp) for i in range(T)]
            d0T = np.linalg.norm(P[0] - P[-1]) + 1e-9
            for i in range(T):   # i = T excluded: distance to final is 0 there
                b = min(bins - 1, int(i / T * bins)) if T > 0 else 0
                dist_fin[b] += np.linalg.norm(P[i] - P[-1]) / d0T
                dist_fin_n[b] += 1
                prog_cos[b] += prog[i]
                prog_cos_n[b] += 1
            rows.append({
                "trajectory_id": tid, "layer": li, "n_steps": T,
                "net_displacement": net, "path_length": plen,
                "straightness": net / (plen + 1e-9),
                "mean_progress_cos": float(np.mean(prog)),
                "frac_steps_toward_final": float(np.mean(np.array(prog) > 0)),
                "max_step_norm_ratio": float(step_norms.max() / (step_norms.mean() + 1e-9)),
            })

        # cross-trajectory displacement alignment
        i = rng.integers(0, len(groups), 20000)
        j = rng.integers(0, len(groups), 20000)
        m = i != j
        rand_align = (disp_dirs[i[m]] * disp_dirs[j[m]]).sum(axis=1)
        sameq_align = []
        for v in multi_q:
            for a in range(len(v)):
                for b in range(a + 1, len(v)):
                    sameq_align.append(float(disp_dirs[v[a]] @ disp_dirs[v[b]]))
        sameq_align = np.array(sameq_align)

        df = pd.DataFrame([r for r in rows if r["layer"] == li])
        summary = {
            "layer": li,
            "straightness_mean": float(df.straightness.mean()),
            "straightness_median": float(df.straightness.median()),
            "mean_progress_cos": float(df.mean_progress_cos.mean()),
            "frac_steps_toward_final": float(df.frac_steps_toward_final.mean()),
            "disp_align_same_question": float(sameq_align.mean()),
            "disp_align_random": float(rand_align.mean()),
            "n_same_question_pairs": int(len(sameq_align)),
        }
        summaries.append(summary)
        print(f"[traj-dyn] L{li}: straightness={summary['straightness_mean']:.3f} "
              f"progress_cos={summary['mean_progress_cos']:.3f} "
              f"toward_final={summary['frac_steps_toward_final']:.2f} "
              f"disp_align same-q={summary['disp_align_same_question']:.3f} "
              f"random={summary['disp_align_random']:.3f}", flush=True)

        # ---- figure ------------------------------------------------------
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
        ax[0].hist(df.straightness, bins=40, color=BLUE, alpha=0.85)
        ax[0].axvline(df.straightness.median(), color="k", lw=1, ls="--")
        ax[0].set_title(f"straightness net/path (median "
                        f"{df.straightness.median():.2f})")
        ax[0].set_xlabel("1 = straight line")

        x = (np.arange(bins) + 0.5) / bins
        ax[1].plot(x, dist_fin / np.maximum(dist_fin_n, 1), marker="o", color=BLUE)
        ax[1].set_ylim(bottom=0)
        ax[1].set_title("distance to final state (i = T excluded: trivially 0)")
        ax[1].set_xlabel("relative position in trajectory")
        ax[1].set_ylabel("||h_i - h_T|| / ||h_0 - h_T||")

        ax[2].plot(x[:-1], (prog_cos / np.maximum(prog_cos_n, 1))[:-1],
                   marker="o", color=AQUA)
        ax[2].axhline(0, color=GRAY, lw=0.8)
        ax[2].set_title("cos(step direction, net displacement)")
        ax[2].set_xlabel("relative position in trajectory")

        ax[3].hist(rand_align, bins=50, density=True, alpha=0.6, color=GRAY,
                   label=f"different question ({rand_align.mean():.2f})")
        ax[3].hist(sameq_align, bins=50, density=True, alpha=0.6, color=YELLOW,
                   label=f"same question ({sameq_align.mean():.2f})")
        ax[3].set_title("net-displacement alignment across chains")
        ax[3].set_xlabel("cosine")
        ax[3].legend(frameon=False, fontsize=9)
        for a in ax:
            a.grid(alpha=0.25, lw=0.5)
        fig.suptitle(f"Trajectory dynamics through activation space — layer {li}",
                     y=1.02)
        fig.tight_layout()
        out = args.run_dir / "plots" / f"traj_dynamics_L{li}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"[traj-dyn] wrote {out}")

    pd.DataFrame(rows).to_csv(args.run_dir / "traj_dynamics.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.run_dir / "traj_dynamics_summary.csv",
                                   index=False)
    print("[traj-dyn] wrote traj_dynamics.csv, traj_dynamics_summary.csv")


if __name__ == "__main__":
    main()
