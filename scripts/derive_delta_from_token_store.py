#!/usr/bin/env python3
"""Derive the delta (transition) representation from a token store, offline.

delta_step = S_t - S_{t-1} = (last-layer state at the last step token)
                          - (last-layer state at the pre-step boundary token)

Both rows live inside each stored token_seq item; the offsets are in the item's
meta (n_tokens -> last row; pre_step_boundary_idx -> the pre-step boundary). This
is pure numpy over the memory-mapped store: no model, no GPU. Vectorized per
shard via fancy indexing, then reordered by global_index.

Writes the dense-cache contract the harness already consumes:
  PRM split  -> <out>/<stem>_h.npy + <stem>_y.npy
  ProcessBench -> <out>/pb_step_h.npy + pb_step_meta.jsonl (meta passed through)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.repstore.store import RepSplit  # noqa: E402


def derive_delta_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (delta (N,d) float16, y (N,) int8, meta) in global_index order."""
    shard_dirs = sorted(glob.glob(str(split_dir / "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* under {split_dir}")
    parts_delta, parts_y, parts_gi, parts_meta = [], [], [], []
    for sd in shard_dirs:
        rs = RepSplit(sd)
        meta = rs.meta()
        pre = np.array([m["pre_step_boundary_idx"] for m in meta], dtype=np.int64)
        gi = np.array([m["global_index"] for m in meta], dtype=np.int64)
        last_abs = rs.offsets[1:] - 1                 # last row of each item
        pre_abs = rs.offsets[:-1] + pre               # pre-step boundary row
        if np.any(pre < 0):
            raise ValueError(f"{sd}: negative pre_step_boundary_idx (empty prefix?)")
        h = rs.h  # mmap (total, d)
        delta = np.asarray(h[last_abs], dtype=np.float32) - np.asarray(h[pre_abs], dtype=np.float32)
        parts_delta.append(delta.astype(np.float16))
        parts_y.append(np.asarray(rs.y, dtype=np.int8))
        parts_gi.append(gi)
        parts_meta.extend(meta)
    delta = np.concatenate(parts_delta, 0)
    y = np.concatenate(parts_y, 0)
    gi = np.concatenate(parts_gi, 0)
    order = np.argsort(gi, kind="mergesort")
    meta_sorted = [parts_meta[i] for i in order]
    return delta[order], y[order], meta_sorted


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_root", required=True, type=Path,
                   help="Token-store rep dir, e.g. <repstore>/tokens_last_layer")
    p.add_argument("--splits", nargs="+", required=True,
                   help="Split stems present under --store_root (e.g. probe_train_full val_5k test_2k)")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--mode", choices=["prm", "pb"], default="prm")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem in args.splits:
        delta, y, meta = derive_delta_split(args.store_root / stem)
        if args.mode == "prm":
            np.save(args.out_dir / f"{stem}_h.npy", delta)
            np.save(args.out_dir / f"{stem}_y.npy", y)
            print(f"[delta:{stem}] {delta.shape} -> {args.out_dir}/{stem}_h.npy", flush=True)
        else:  # pb: one subset per store split; emit pb_step contract
            sub = args.out_dir / stem
            sub.mkdir(parents=True, exist_ok=True)
            np.save(sub / "pb_step_h.npy", delta)
            with (sub / "pb_step_meta.jsonl").open("w") as f:
                for m in meta:
                    f.write(json.dumps(m) + "\n")
            print(f"[delta-pb:{stem}] {delta.shape} -> {sub}/pb_step_h.npy", flush=True)


if __name__ == "__main__":
    main()
