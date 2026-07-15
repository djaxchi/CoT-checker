"""transition_operator_v0 Stage 2 analysis: z vs raw baselines on the SAME test split.

Answers the plan's decisive question apples-to-apples: does the learned operator z
organize transitions by operation better than the raw baselines, on the identical
problem-disjoint TEST forks? Loads the Stage 2 training arrays (S_prev/S_post/H_steps),
builds the five baselines and pools, loads each arm/seed z_all.npy, and runs the
Stage-1 decodability + cross-problem retrieval on the test split for all of them,
at matched dimension (PCA to d_z).

Memory: H_steps is unzipped and mmap'd, only test rows are materialized.

  python scripts/transition_operator/to_stage2_compare.py --run_dir runs/transition_operator
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "transition_operator"))

from to_stage1 import cross_problem_retrieval, decodability, matched_pca  # noqa: E402


def mmap_member(npz_path: Path, member: str, tmp: Path) -> np.ndarray:
    out = tmp / f"{member}.npy"
    if not out.exists():
        subprocess.run(["unzip", "-o", "-q", str(npz_path), f"{member}.npy",
                        "-d", str(tmp)], check=True)
    return np.load(out, mmap_mode="r")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--d_z", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = args.run_dir / "stage2" / "arrays"
    trans_rows = json.loads((a / "trans_rows.json").read_text())
    fork_rows = json.loads((a / "fork_rows.json").read_text())
    fork_pos = {r["fork_id"]: i for i, r in enumerate(fork_rows)}
    splits = json.loads((args.run_dir / "splits.json").read_text())
    test_forks = set(splits["test"]) & set(fork_pos)

    labels = pd.read_parquet(args.run_dir / "stage1" / "step_labels.parquet")
    lab = {(r.fork_id, r.branch): (r.op_symbolic, r.tag_top)
           for r in labels.itertuples()}

    test_idx, groups, op_y, tag_y = [], [], [], []
    for i, r in enumerate(trans_rows):
        if r["fork_id"] in test_forks:
            test_idx.append(i)
            groups.append(r["question"])
            o, t = lab.get((r["fork_id"], r["branch"]), (None, None))
            op_y.append(o)
            tag_y.append(t if t != "NONE" else None)
    test_idx = np.array(test_idx)
    groups = pd.factorize(np.asarray(groups, dtype=object))[0]
    op_y = pd.Series(op_y)
    tag_y = pd.Series(tag_y)
    f_of_trans = np.array([fork_pos[trans_rows[i]["fork_id"]] for i in test_idx])
    print(f"test transitions {len(test_idx)}, op-labeled {int(op_y.notna().sum())}, "
          f"tag-labeled {int(tag_y.notna().sum())}, "
          f"problems {len(np.unique(groups))}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        S_prev_all = mmap_member(a / "fork_arrays.npz", "S_prev", tmp)
        S_post_all = mmap_member(a / "trans_arrays.npz", "S_post", tmp)
        H_all = mmap_member(a / "trans_arrays.npz", "H_steps", tmp)
        n_all = mmap_member(a / "trans_arrays.npz", "n_steps", tmp)
        S_prev = np.asarray(S_prev_all[f_of_trans], np.float32)
        S_post = np.asarray(S_post_all[test_idx], np.float32)
        # pool H_steps in small chunks so the full test H (~3 GB) is never resident
        hidden = S_post.shape[1]
        meanpool = np.zeros((len(test_idx), hidden), np.float32)
        maxpool = np.zeros((len(test_idx), hidden), np.float32)
        for lo in range(0, len(test_idx), 64):
            rows = test_idx[lo:lo + 64]
            Hc = np.asarray(H_all[rows], np.float32)
            nc = np.asarray(n_all[rows])
            mask = (np.arange(Hc.shape[1])[None] < nc[:, None])[..., None]
            meanpool[lo:lo + len(rows)] = (np.where(mask, Hc, 0.0).sum(1)
                                           / np.maximum(nc[:, None], 1))
            maxpool[lo:lo + len(rows)] = np.where(mask, Hc, -np.inf).max(1)
        reps = {"S_t": S_post, "delta": S_post - S_prev,
                "concat": np.concatenate([S_prev, S_post], 1),
                "meanpool": meanpool, "maxpool": maxpool}

    for arm in ("A", "B", "AB"):
        for s in (0, 1, 2):
            zp = args.run_dir / "stage2" / f"{arm}_seed{s}" / "z_all.npy"
            if zp.exists():
                reps[f"z_{arm}{s}"] = np.load(zp)[test_idx]

    def evaluate(X, y):
        m = y.notna().to_numpy()
        if m.sum() < 30:
            return None
        Xp = matched_pca(X[m], args.d_z, args.seed)
        yy = y[m].to_numpy().astype(str)
        return (decodability(Xp, yy, groups[m], args.seed),
                cross_problem_retrieval(Xp, yy, groups[m]))

    results = {}
    print(f"\n{'repr':10} | op_F1  op_ret(ch) | tag_F1 tag_ret(ch)")
    for name, X in reps.items():
        r_op = evaluate(X, op_y)
        r_tag = evaluate(X, tag_y)
        results[name] = {"op": r_op, "tag": r_tag}
        def fmt(r):
            if r is None:
                return "  nan    nan   "
            d, rt = r
            return f"{d['macro_f1']:.3f} {rt['precision_at_k']:.3f}({rt['chance']:.2f})"
        print(f"{name:10} | {fmt(r_op)} | {fmt(r_tag)}")

    out = args.run_dir / "stage2" / "compare_baselines_vs_z.json"
    out.write_text(json.dumps(
        {k: {kk: (None if vv is None else {"decodability": vv[0],
                                           "retrieval": vv[1]})
             for kk, vv in v.items()} for k, v in results.items()}, indent=2))
    print(f"\n[compare] wrote {out}")


if __name__ == "__main__":
    main()
