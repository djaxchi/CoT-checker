"""S4 contrib-cluster stage 3: build step representations from hidden states.

Reads metadata.parquet + h_layer_<L>.npy (from s4_contrib_extract.py) and, per
trajectory and layer, computes state / qres / contribution step vectors
(see src.analysis.contrib_cluster.compute_reprs; contribution = h_i - h_{i-1}
is the closed form of the recursion c_i = h_i - (h_0 + sum_{k<i} c_k)).

Outputs (in --out_dir, default runs/contrib_cluster/reprs/):
  step_metadata.parquet             one row per step (p_0 rows dropped), the
                                    SAME order for every repr array
  repr_<name>_layer_<L>.npy         (n_steps, hidden) float32
  repr_<name>_norm_layer_<L>.npy    row-wise L2-normalized variant
  reprs_manifest.json

Usage:
  python scripts/analysis/s4_contrib_reprs.py \
    --hidden_dir runs/contrib_cluster/hidden_states \
    --out_dir runs/contrib_cluster/reprs --layers 20 28
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prm800k_prestudy import git_commit  # noqa: E402

from src.analysis.contrib_cluster import REPR_NAMES, compute_reprs, l2_normalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden_dir", type=Path,
                    default=Path("runs/contrib_cluster/hidden_states"))
    ap.add_argument("--out_dir", type=Path, default=Path("runs/contrib_cluster/reprs"))
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--store_normalized", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = args.out_dir / "step_metadata.parquet"
    if meta_out.exists() and not args.force:
        sys.exit(f"refusing to overwrite {meta_out}; pass --force")

    meta = pd.read_parquet(args.hidden_dir / "metadata.parquet")
    meta = meta.sort_values(["trajectory_id", "step_index"]).reset_index(drop=True)

    # Row ranges per trajectory; verify each is contiguous 0..T with a p0 row.
    n_dropped_traj = 0
    groups: list[tuple[str, np.ndarray]] = []
    for tid, g in meta.groupby("trajectory_id", sort=True):
        si = g["step_index"].to_numpy()
        if si[0] != 0 or not np.array_equal(si, np.arange(len(si))) or len(si) < 3:
            n_dropped_traj += 1  # need p0 + >=2 steps, contiguous
            continue
        groups.append((tid, g.index.to_numpy()))
    print(f"[reprs] {len(groups)} trajectories usable, {n_dropped_traj} dropped "
          f"(non-contiguous or <2 steps)", flush=True)

    step_meta_rows: list[int] = []  # indices into meta of step rows, in output order
    for _, idxs in groups:
        step_meta_rows.extend(idxs[1:].tolist())
    step_meta = meta.loc[step_meta_rows].reset_index(drop=True)
    step_meta["relative_step_index"] = (
        step_meta["step_index"] / step_meta["num_steps_in_trajectory"]
    )
    step_meta.to_parquet(meta_out, index=False)
    n_steps = len(step_meta)

    for li in args.layers:
        H_all = np.load(args.hidden_dir / f"h_layer_{li}.npy").astype(np.float32)
        assert H_all.shape[0] == len(meta), (
            f"layer {li}: {H_all.shape[0]} rows vs {len(meta)} metadata rows")
        out = {name: np.empty((n_steps, H_all.shape[1]), dtype=np.float32)
               for name in REPR_NAMES}
        pos = 0
        for _, idxs in groups:
            reprs = compute_reprs(H_all[idxs])
            t = idxs.shape[0] - 1
            for name in REPR_NAMES:
                out[name][pos:pos + t] = reprs[name]
            pos += t
        assert pos == n_steps
        for name in REPR_NAMES:
            np.save(args.out_dir / f"repr_{name}_layer_{li}.npy", out[name])
            if args.store_normalized:
                np.save(args.out_dir / f"repr_{name}_norm_layer_{li}.npy",
                        l2_normalize(out[name]))
        print(f"[reprs] layer {li}: wrote {len(REPR_NAMES)} reprs x "
              f"{'2' if args.store_normalized else '1'} variants "
              f"({n_steps} steps)", flush=True)

    (args.out_dir / "reprs_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "layers": args.layers,
        "repr_names": list(REPR_NAMES),
        "n_trajectories": len(groups),
        "n_dropped_trajectories": n_dropped_traj,
        "n_steps": n_steps,
        "contribution_note": "contribution = h_i - h_{i-1}, the closed form of "
                             "c_i = h_i - (h_0 + sum_{k<i} c_k) (exact telescoping).",
    }, indent=2))
    print(f"[reprs] wrote {meta_out} ({n_steps} steps)", flush=True)


if __name__ == "__main__":
    main()
