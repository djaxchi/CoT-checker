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
    if readout == "multistat":
        # A fixed, permutation-invariant summary of the whole token set for a
        # linear probe: concat[mean, max, min, std, last] -> 5*d.
        return np.concatenate([span.mean(0), span.max(0), span.min(0),
                               span.std(0), span[-1]])
    raise ValueError(readout)


def _out_dim(readout: str, d: int) -> int:
    return 5 * d if readout == "multistat" else d


def _shard_vecs(rs: "RepSplit", meta: list[dict], readout: str) -> np.ndarray:
    """Compute a shard's (n, out_dim) float16 vectors, memory-leanly."""
    n = len(meta)
    od = _out_dim(readout, rs.spec.dim)
    pre = np.array([m["pre_step_boundary_idx"] for m in meta], dtype=np.int64)
    if np.any(pre < 0):
        raise ValueError("negative pre_step_boundary_idx (empty prefix?)")
    last_abs = rs.offsets[1:] - 1
    h = rs.h
    if readout == "delta":
        pre_abs = rs.offsets[:-1] + pre
        return (np.asarray(h[last_abs], np.float32) - np.asarray(h[pre_abs], np.float32)).astype(np.float16)
    if readout == "last":
        return np.asarray(h[last_abs], np.float32).astype(np.float16)
    # pooled: fill a preallocated float16 array item by item (no big float32 stack)
    start_abs = rs.offsets[:-1] + (pre + 1)
    out = np.empty((n, od), dtype=np.float16)
    for k in range(n):
        out[k] = _reduce_item(h, int(rs.offsets[k]), int(rs.offsets[k + 1]),
                              int(start_abs[k]), int(last_abs[k]), readout).astype(np.float16)
    return out


def derive_split(split_dir: Path, readout: str = "delta", sort: bool = True):
    """Return (vectors (N, out_dim) float16, y (N,) int8, meta).

    readout: 'delta'|'last'|'mean'|'max'|'multistat'. With sort=True the rows are
    in global_index order (needed for ProcessBench first-error scan); with
    sort=False they are in shard order (fine for an order-invariant probe on
    PRM800K, and avoids a full-array copy on the ~1TB-derived train split).
    """
    shard_dirs = sorted(glob.glob(str(split_dir / "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* under {split_dir}")
    metas = [RepSplit(sd).meta() for sd in shard_dirs]
    ns = [len(m) for m in metas]
    N = sum(ns)
    d = RepSplit(shard_dirs[0]).spec.dim
    od = _out_dim(readout, d)
    out = np.empty((N, od), dtype=np.float16)
    y = np.empty(N, dtype=np.int8)
    gi = np.empty(N, dtype=np.int64)
    meta_all: list[dict] = []
    cur = 0
    for sd, meta in zip(shard_dirs, metas):
        rs = RepSplit(sd)
        n = len(meta)
        out[cur:cur + n] = _shard_vecs(rs, meta, readout)
        y[cur:cur + n] = np.asarray(rs.y, dtype=np.int8)
        gi[cur:cur + n] = [m["global_index"] for m in meta]
        if sort:
            meta_all.extend(meta)
        cur += n
        del rs
    if not sort:
        return out, y, []
    order = np.argsort(gi, kind="mergesort")
    return out[order], y[order], [meta_all[i] for i in order]


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
    p.add_argument("--readout", choices=["delta", "last", "mean", "max", "multistat"],
                   default="delta")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem in args.splits:
        # PRM: order-invariant probe -> skip the sort/copy. PB: keep global order
        # for the first-error scan (meta alignment).
        v, y, meta = derive_split(args.store_root / stem, args.readout,
                                  sort=(args.mode == "pb"))
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
