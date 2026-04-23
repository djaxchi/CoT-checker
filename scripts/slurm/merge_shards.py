#!/usr/bin/env python3
"""Merge multiple .npz probe-data shards into one file.

Usage:
    python scripts/slurm/merge_shards.py \\
        --inputs shard_0.npz shard_1.npz shard_2.npz shard_3.npz \\
        --output train_full.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Shard .npz files in order")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    latents_list, labels_list = [], []
    for path in args.inputs:
        d = np.load(path)
        latents_list.append(d["latents"])
        labels_list.append(d["correctness"])
        n = len(d["correctness"])
        pos = d["correctness"].sum()
        print(f"  {path}: {n:,} steps  ({pos/n:.1%} correct)")

    latents = np.concatenate(latents_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    total = len(labels)
    pos = labels.sum()
    print(f"\nMerged: {total:,} steps  ({pos/total:.1%} correct, {(total-pos)/total:.1%} incorrect)")
    print(f"Majority baseline: {max(pos, total-pos)/total:.1%}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, latents=latents, correctness=labels)
    print(f"Saved → {out}  ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
