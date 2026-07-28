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


def _reduce_item(h, off_a, off_b, start_abs, last_abs, readout):
    """Compute one item's vector for the given readout. h is the shard mmap.
    [off_a, off_b) is the item's row range; start_abs/last_abs are the step-token
    span start and the last step token (absolute row indices)."""
    if readout == "last":
        return np.asarray(h[last_abs], dtype=np.float32)
    span = np.asarray(h[start_abs:last_abs + 1], dtype=np.float32)  # step tokens only
    if readout == "mean":
        return span.mean(0)
    if readout == "max":
        return span.max(0)
    raise ValueError(readout)


def derive_split(split_dir: Path, readout: str = "delta") -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (vectors (N,d) float16, y (N,) int8, meta) in global_index order.

    readout: 'delta' = last step-token state - pre-step boundary state;
             'last'  = last step-token state (reproduces dense_last);
             'mean'/'max' = pooled over the step-token span.
    """
    shard_dirs = sorted(glob.glob(str(split_dir / "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* under {split_dir}")
    parts_v, parts_y, parts_gi, parts_meta = [], [], [], []
    for sd in shard_dirs:
        rs = RepSplit(sd)
        meta = rs.meta()
        pre = np.array([m["pre_step_boundary_idx"] for m in meta], dtype=np.int64)
        gi = np.array([m["global_index"] for m in meta], dtype=np.int64)
        if np.any(pre < 0):
            raise ValueError(f"{sd}: negative pre_step_boundary_idx (empty prefix?)")
        last_abs = rs.offsets[1:] - 1
        pre_abs = rs.offsets[:-1] + pre
        h = rs.h
        if readout == "delta":
            vecs = np.asarray(h[last_abs], np.float32) - np.asarray(h[pre_abs], np.float32)
        elif readout == "last":
            vecs = np.asarray(h[last_abs], np.float32)
        else:  # mean/max: per-item span reduction (ragged)
            start_abs = rs.offsets[:-1] + (pre + 1)   # step span starts after boundary
            vecs = np.stack([
                _reduce_item(h, int(rs.offsets[k]), int(rs.offsets[k + 1]),
                             int(start_abs[k]), int(last_abs[k]), readout)
                for k in range(len(meta))
            ])
        parts_v.append(vecs.astype(np.float16))
        parts_y.append(np.asarray(rs.y, dtype=np.int8))
        parts_gi.append(gi)
        parts_meta.extend(meta)
    v = np.concatenate(parts_v, 0)
    y = np.concatenate(parts_y, 0)
    gi = np.concatenate(parts_gi, 0)
    order = np.argsort(gi, kind="mergesort")
    return v[order], y[order], [parts_meta[i] for i in order]


# Back-compat alias
def derive_delta_split(split_dir: Path):
    return derive_split(split_dir, "delta")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_root", required=True, type=Path,
                   help="Token-store rep dir, e.g. <repstore>/tokens_last_layer")
    p.add_argument("--splits", nargs="+", required=True,
                   help="Split stems present under --store_root (e.g. probe_train_full val_5k test_2k)")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--mode", choices=["prm", "pb"], default="prm")
    p.add_argument("--readout", choices=["delta", "last", "mean", "max"], default="delta")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem in args.splits:
        v, y, meta = derive_split(args.store_root / stem, args.readout)
        if args.mode == "prm":
            np.save(args.out_dir / f"{stem}_h.npy", v)
            np.save(args.out_dir / f"{stem}_y.npy", y)
            print(f"[{args.readout}:{stem}] {v.shape} -> {args.out_dir}/{stem}_h.npy", flush=True)
        else:  # pb: one subset per store split; emit pb_step contract
            sub = args.out_dir / stem
            sub.mkdir(parents=True, exist_ok=True)
            np.save(sub / "pb_step_h.npy", v)
            with (sub / "pb_step_meta.jsonl").open("w") as f:
                for m in meta:
                    f.write(json.dumps(m) + "\n")
            print(f"[{args.readout}-pb:{stem}] {v.shape} -> {sub}/pb_step_h.npy", flush=True)


if __name__ == "__main__":
    main()
