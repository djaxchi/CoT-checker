#!/usr/bin/env python3
"""Screen many ways of pooling a step's tokens, in one pass over the store.

The grid found that *which rows you pool* dominates everything else, and that
compression of any kind costs. So the remaining room is in the pooling rule
itself, which so far has only ever been mean / max / min / std / last.

Each pooling below is a hypothesis with a reason, not a variation for its own
sake:

  mean          the incumbent baseline
  mean_l2       mean of L2-NORMALISED token states. The project's probe-anatomy
                work found the correctness signal is carried by DIRECTION, not
                magnitude; plain mean lets high-norm tokens dominate the average,
                so normalising first should let every token vote equally on the
                thing that actually matters.
  dir           step_mean divided by its own norm. Tests whether the direction of
                the pooled vector is what carries the signal, by discarding the
                pooled magnitude entirely.
  dev           mean absolute deviation of tokens from the step's own mean. Drops
                what the step is ABOUT and keeps how much it varies internally --
                a step that wanders may be a step going wrong.
  centered      concat[mean, mean of (token - step mean) magnitudes per position].
  quantiles     per-position 10/50/90th percentiles instead of min/max. Same idea
                as step_stats but robust to a single outlier token, which max and
                min are not.
  diffs         mean and std of consecutive token differences. Within-step
                dynamics rather than within-step position: the transition
                operator work asked this at step granularity and got a negative,
                but never at token granularity inside a step.
  first_last    concat[first token, last token]. The cheapest possible test of
                whether a step's endpoints bracket the signal.

Samples the store rather than reading all of it, because a screen that costs a
full pass is not a screen.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.repstore.store import RepSplit  # noqa: E402

EPS = 1e-8


def poolings(span: np.ndarray, boundary: np.ndarray) -> dict[str, np.ndarray]:
    """span: (T, d) the step's own tokens. boundary: (d,) the pre-step state."""
    t = span.shape[0]
    mean = span.mean(0)
    unit = span / (np.linalg.norm(span, axis=1, keepdims=True) + EPS)
    dev = np.abs(span - mean).mean(0)
    out = {
        "mean": mean,
        "mean_l2": unit.mean(0),
        "dir": mean / (np.linalg.norm(mean) + EPS),
        "dev": dev,
        "centered": np.concatenate([mean, dev]),
        "quantiles": np.concatenate([np.percentile(span, q, axis=0) for q in (10, 50, 90)]),
        "first_last": np.concatenate([span[0], span[-1]]),
    }
    if t >= 2:
        d = np.diff(span, axis=0)
        out["diffs"] = np.concatenate([d.mean(0), d.std(0)])
    else:
        out["diffs"] = np.zeros(2 * span.shape[1], np.float32)
    return out


def collect(split_dir: Path, names: list[str], limit: int | None, seed: int,
            pb: bool):
    shard_dirs = sorted(glob.glob(str(split_dir / "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* under {split_dir}")
    rng = np.random.default_rng(seed)
    acc: dict[str, list] = {n: [] for n in names}
    ys, metas = [], []
    per_shard = None if limit is None else max(1, limit // len(shard_dirs))
    for sd in shard_dirs:
        rs = RepSplit(sd)
        meta = rs.meta()
        idx = np.arange(len(meta))
        if per_shard is not None and len(idx) > per_shard:
            idx = np.sort(rng.choice(len(idx), per_shard, replace=False))
        for k in idx:
            m = meta[int(k)]
            a = int(rs.offsets[k]) + int(m["step_start_idx"])
            b = int(rs.offsets[k + 1])
            if b <= a:
                a = b - 1
            span = np.asarray(rs.h[a:b], dtype=np.float32)
            bnd = np.asarray(rs.h[int(rs.offsets[k]) + int(m["pre_step_boundary_idx"])],
                             dtype=np.float32)
            p = poolings(span, bnd)
            for n in names:
                acc[n].append(p[n].astype(np.float32))
            ys.append(1 if (pb and m["label"] == m["step_idx"]) else
                      (int(rs.y[k]) if not pb else 0))
            if pb:
                metas.append(m)
        del rs
    return ({n: np.stack(v) for n, v in acc.items()},
            np.array(ys, dtype=np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prm_store", required=True, type=Path)
    p.add_argument("--pb_store", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--pb_subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--n_train", type=int, default=60000)
    p.add_argument("--n_pb", type=int, default=4000)
    p.add_argument("--names", nargs="+", default=list(poolings(
        np.zeros((3, 4), np.float32), np.zeros(4, np.float32)).keys()))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.perf_counter()
    print(f"[pool] poolings: {', '.join(args.names)}", flush=True)
    tr, ytr = collect(args.prm_store / args.train_stem, args.names, args.n_train,
                      args.seed, pb=False)
    print(f"[pool] train {len(ytr):,} ({time.perf_counter()-t0:.0f}s)", flush=True)
    va, yva = collect(args.prm_store / args.val_stem, args.names, None, args.seed,
                      pb=False)
    pbs = {}
    for sub in args.pb_subsets:
        d = args.pb_store / sub
        if not d.exists():
            continue
        pbs[sub] = collect(d, args.names, args.n_pb, args.seed, pb=True)
    print(f"[pool] pb subsets {list(pbs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in args.names:
        blob = {"x_train": tr[n], "y_train": ytr, "x_val": va[n], "y_val": yva}
        for sub, (xx, yy) in pbs.items():
            blob[f"pb_x_{sub}"] = xx[n]
            blob[f"pb_y_{sub}"] = yy
        np.savez(args.out_dir / f"{n}.npz", **blob)
        print(f"[pool] {n:<12} dim {tr[n].shape[1]:>6}  -> {args.out_dir/f'{n}.npz'}",
              flush=True)
    print(f"[pool] done in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
