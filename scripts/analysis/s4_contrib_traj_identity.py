"""S4 contrib-cluster: trajectory identity in step-representation space.

Each trajectory is a sequence of step vectors. This asks, per (repr, layer):

1. Retrieval: for every step, find its nearest neighbor (cosine, excluding
   itself). Does it come from the SAME trajectory, the same problem but a
   different trajectory, or a different problem? Distinguishing trajectory
   from problem matters because ~880 of the 3000 trajectories share their
   question with another sampled trajectory.
2. Pairwise similarity by relation: mean cosine within trajectory, within
   problem across trajectories, and across problems (sampled).
3. Drift: mean within-trajectory cosine as a function of step distance |i-j|
   (does the trajectory stay in one place or move?).

Outputs (under --run_dir):
  traj_identity.csv         one row per (repr, layer) with all metrics
  traj_identity_lag.csv     within-trajectory cosine by lag
  plots/traj_identity_lag.png

Usage:
  python scripts/analysis/s4_contrib_traj_identity.py --run_dir runs/contrib_cluster
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

from src.analysis.contrib_cluster import REPR_NAMES, l2_normalize  # noqa: E402

# dataviz reference categorical palette (light mode)
CAT = ["#2a78d6", "#1baf7a", "#eda100"]


def nn_relations(X: np.ndarray, traj: np.ndarray, prob: np.ndarray,
                 chunk: int = 2048) -> dict[str, float]:
    """Nearest-neighbor (cosine) relation fractions, self excluded."""
    n = X.shape[0]
    nn = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = X[s:e] @ X.T                      # (chunk, n), X is L2-normalized
        sims[np.arange(e - s), np.arange(s, e)] = -np.inf
        nn[s:e] = sims.argmax(axis=1)
    same_traj = traj[nn] == traj
    same_prob = (prob[nn] == prob) & ~same_traj
    return {
        "nn_same_trajectory": float(same_traj.mean()),
        "nn_same_problem_other_traj": float(same_prob.mean()),
        "nn_other_problem": float((~same_traj & ~same_prob).mean()),
    }


def pairwise_by_relation(X, traj, prob, groups, prob_groups, rng) -> dict[str, float]:
    """Mean cosine within trajectory / within problem across traj / across problems."""
    within = []
    for idxs in groups:
        if len(idxs) < 2:
            continue
        V = X[idxs]
        S = V @ V.T
        iu = np.triu_indices(len(idxs), k=1)
        within.append(S[iu])
    within = np.concatenate(within)

    same_prob = []
    for idx_lists in prob_groups:                # trajectories of one shared problem
        for a in range(len(idx_lists)):
            for b in range(a + 1, len(idx_lists)):
                S = X[idx_lists[a]] @ X[idx_lists[b]].T
                same_prob.append(S.ravel())
    same_prob = np.concatenate(same_prob) if same_prob else np.array([np.nan])

    i = rng.integers(0, len(X), 200000)
    j = rng.integers(0, len(X), 200000)
    m = prob[i] != prob[j]
    across = (X[i[m]] * X[j[m]]).sum(axis=1)
    return {
        "cos_within_trajectory": float(within.mean()),
        "cos_same_problem_other_traj": float(np.nanmean(same_prob)),
        "cos_other_problem": float(across.mean()),
    }


def lag_curve(X, groups, step_idx, max_lag: int = 9) -> dict[int, float]:
    sums = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)
    for idxs in groups:
        if len(idxs) < 2:
            continue
        V = X[idxs]
        S = V @ V.T
        si = step_idx[idxs]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                lag = abs(int(si[a]) - int(si[b]))
                if 1 <= lag <= max_lag:
                    sums[lag] += S[a, b]
                    counts[lag] += 1
    return {lag: float(sums[lag] / counts[lag])
            for lag in range(1, max_lag + 1) if counts[lag] > 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--reprs", type=str, nargs="+", default=list(REPR_NAMES))
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reprs_dir = args.run_dir / "reprs"
    meta = pd.read_parquet(reprs_dir / "step_metadata.parquet")
    traj = meta["trajectory_id"].astype("category").cat.codes.to_numpy()
    # group problems by question TEXT: trajectory_id's problem part embeds the raw
    # sample index, so two trajectories of the same question never share it
    prob = meta["question"].astype("category").cat.codes.to_numpy()
    step_idx = meta["step_index"].to_numpy()
    rng = np.random.default_rng(args.seed)

    groups = [g.to_numpy() for _, g in meta.groupby("trajectory_id").groups.items()]
    by_prob: dict[int, list[np.ndarray]] = {}
    for idxs in groups:
        by_prob.setdefault(int(prob[idxs[0]]), []).append(idxs)
    prob_groups = [v for v in by_prob.values() if len(v) > 1]
    n_multi = len(prob_groups)
    print(f"[traj-id] {len(groups)} trajectories, {n_multi} problems with >1 trajectory")

    rows, lag_rows = [], []
    fig, axes = plt.subplots(1, len(args.layers), figsize=(6 * len(args.layers), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, li in zip(axes, args.layers):
        for ci, name in enumerate(args.reprs):
            X = l2_normalize(np.load(reprs_dir / f"repr_{name}_layer_{li}.npy")
                             .astype(np.float32))
            print(f"[traj-id] {name} L{li}: NN retrieval ...", flush=True)
            r = {"repr": name, "layer": li}
            r.update(nn_relations(X, traj, prob))
            r.update(pairwise_by_relation(X, traj, prob, groups, prob_groups, rng))
            rows.append(r)
            lc = lag_curve(X, groups, step_idx)
            for lag, v in lc.items():
                lag_rows.append({"repr": name, "layer": li, "lag": lag, "cos": v})
            ax.plot(list(lc), list(lc.values()), marker="o", ms=4,
                    color=CAT[ci % 3], label=name)
            print(f"[traj-id]   nn_same_traj={r['nn_same_trajectory']:.3f} "
                  f"nn_same_prob={r['nn_same_problem_other_traj']:.3f} "
                  f"cos_within={r['cos_within_trajectory']:.3f} "
                  f"cos_same_prob={r['cos_same_problem_other_traj']:.3f} "
                  f"cos_other={r['cos_other_problem']:.3f}", flush=True)
        ax.set_title(f"layer {li}")
        ax.set_xlabel("step distance |i - j| within trajectory")
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("mean cosine similarity")
    axes[0].legend(frameon=False)
    fig.suptitle("Within-trajectory similarity vs step distance", y=1.0)
    fig.tight_layout()
    (args.run_dir / "plots").mkdir(exist_ok=True)
    out_png = args.run_dir / "plots" / "traj_identity_lag.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")

    pd.DataFrame(rows).to_csv(args.run_dir / "traj_identity.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(args.run_dir / "traj_identity_lag.csv", index=False)
    print(f"[traj-id] wrote traj_identity.csv, traj_identity_lag.csv, {out_png}")


if __name__ == "__main__":
    main()
