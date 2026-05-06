#!/usr/bin/env python3
"""Merge transition .npz shards produced by cache_transition_hidden_states.py.

The existing merge_shards.py only handles latents+correctness fields.
This script handles the full transition .npz schema:
    h_k, h_next, delta_h, problem_id, step_idx, num_steps

Usage:
    python scripts/merge_transition_shards.py \\
        --inputs shard_0.npz shard_1.npz shard_2.npz shard_3.npz \\
        --output transition_train_positive.npz
"""

import argparse
from pathlib import Path

import numpy as np

FIELDS = ["h_k", "h_next", "delta_h", "problem_id", "step_idx", "num_steps"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Shard .npz files in order")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    arrays: dict[str, list[np.ndarray]] = {f: [] for f in FIELDS}
    total = 0

    for path in args.inputs:
        d = np.load(path)
        n = len(d["h_k"])
        print(f"  {path}: {n:,} transition pairs")
        for f in FIELDS:
            arrays[f].append(d[f])
        total += n

    merged = {f: np.concatenate(arrays[f], axis=0) for f in FIELDS}
    print(f"\nMerged: {total:,} transition pairs  h_k shape: {merged['h_k'].shape}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    size_gb = out.stat().st_size / 1e9
    print(f"Saved → {out}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
