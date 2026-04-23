#!/usr/bin/env python3
"""Reorganize existing encoded shards into clean train / eval splits.

Carves 25K correct + 25K incorrect (50K balanced) from shard_0 as eval.
Merges the shard_0 leftover with shards 1, 2, 3 into the training set.
No re-encoding needed.

Usage:
    python scripts/slurm/reorganize_shards.py \
        --shard-dir /scratch/d/dchikhi/cot-checker/probe_data \
        --out-dir   /scratch/d/dchikhi/cot-checker/probe_data
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--out-dir",   required=True)
    p.add_argument("--eval-per-class", type=int, default=25000,
                   help="Eval samples per class (default: 25000 → 50K total)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    shard_dir = Path(args.shard_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Split shard_0 into eval (balanced) + leftover (goes to training)
    # ------------------------------------------------------------------
    print("Loading shard_0...")
    d0 = np.load(shard_dir / "train_shard_0.npz")
    h0, y0 = d0["latents"], d0["correctness"]

    cor_idx = np.where(y0 == 1)[0]
    inc_idx = np.where(y0 == 0)[0]
    n = args.eval_per_class
    print(f"  shard_0: {len(y0):,} steps  ({len(cor_idx):,} correct / {len(inc_idx):,} incorrect)")

    if len(cor_idx) < n or len(inc_idx) < n:
        raise ValueError(
            f"shard_0 has only {len(cor_idx)} correct and {len(inc_idx)} incorrect steps "
            f"-- cannot carve {n} per class for eval."
        )

    eval_cor = rng.choice(cor_idx, size=n, replace=False)
    eval_inc = rng.choice(inc_idx, size=n, replace=False)
    eval_idx = np.concatenate([eval_cor, eval_inc])
    rng.shuffle(eval_idx)

    train_idx_0 = np.setdiff1d(np.arange(len(y0)), eval_idx)

    h_eval, y_eval = h0[eval_idx], y0[eval_idx]
    h_left, y_left = h0[train_idx_0], y0[train_idx_0]

    pos_eval = int((y_eval == 1).sum())
    print(f"  Eval   : {len(y_eval):,} steps  ({pos_eval:,} correct / {len(y_eval)-pos_eval:,} incorrect)")
    pos_left = int((y_left == 1).sum())
    print(f"  Leftover→train: {len(y_left):,} steps  ({pos_left:,} correct / {len(y_left)-pos_left:,} incorrect)")

    # Save eval
    eval_path = out_dir / "eval_held_out.npz"
    np.savez_compressed(eval_path, latents=h_eval, correctness=y_eval)
    print(f"\nSaved eval → {eval_path}")

    # ------------------------------------------------------------------
    # Merge leftover shard_0 + shards 1, 2, 3 into training set
    # ------------------------------------------------------------------
    shards_to_merge = [("shard_0 leftover", h_left, y_left)]
    for i in [1, 2, 3]:
        path = shard_dir / f"train_shard_{i}.npz"
        d = np.load(path)
        h, y = d["latents"], d["correctness"]
        pos = int((y == 1).sum())
        print(f"Loading shard_{i}: {len(y):,} steps  ({pos:,} correct / {len(y)-pos:,} incorrect)")
        shards_to_merge.append((f"shard_{i}", h, y))

    h_train = np.concatenate([s[1] for s in shards_to_merge], axis=0)
    y_train = np.concatenate([s[2] for s in shards_to_merge], axis=0)

    shuffle_idx = rng.permutation(len(y_train))
    h_train, y_train = h_train[shuffle_idx], y_train[shuffle_idx]

    total = len(y_train)
    pos   = int((y_train == 1).sum())
    print(f"\nTraining set: {total:,} steps  ({pos:,} correct {pos/total:.1%} / {total-pos:,} incorrect {(total-pos)/total:.1%})")
    print(f"Majority baseline: {max(pos, total-pos)/total:.1%}")

    train_path = out_dir / "train_final.npz"
    np.savez_compressed(train_path, latents=h_train, correctness=y_train)
    print(f"Saved training → {train_path}  ({train_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
