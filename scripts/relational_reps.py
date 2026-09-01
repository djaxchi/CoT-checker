#!/usr/bin/env python3
"""Representations built from relations, not from the step's own content.

Every pooling screened so far answers "what does this step's activation look
like". Each is some average of the step's own token states in model space, and
they land within 0.03 of each other within length strata, which is what you would
expect if they are all reading the same thing slightly differently.

Two things the store already holds have never been used.

The pre-step boundary state. `poolings()` takes it as an argument and no pooling
touches it. But a step is not wrong in isolation, it is wrong given what came
before, and the quantity that expresses that is what the step ADDED to the
model's state rather than the state it left behind. S4 established `h_i - h_{i-1}`
as the contribution representation at step granularity; it has never been screened
as a pooling on this benchmark.

A second layer. Layer 26 was encoded for a stacking test, and concatenating it
onto layer 35 is a wider vector rather than a better idea. The interesting
quantity between two layers is not their concatenation but their DISAGREEMENT:
how much the model revised its representation of this step between the middle and
the end. A step that is going wrong may be one the late layers have to work
harder on.

Both ideas share a property worth having on purpose. They are low-dimensional, so
they sidestep the width-dependent convergence problem that made the stacking
result unreadable.

The sharpest of them is `geom`: a content-free summary of the step's geometry
with no direction in model space at all, roughly twenty numbers describing how
tightly the step's tokens cone around their own mean, how far that mean sits from
the prefix state, and how the norms are distributed. The conicity work found that
correct steps form a tight cone and incorrect ones do not, but read it through a
centroid rule that scored 0.63 against a whitened 0.82, concluding the gap was
the metric rather than the direction. Geometric features do not have a metric
problem, because an angle is already scale free. If twenty numbers of pure
geometry come close to 4,096 dimensions of activation, that is a much more
interesting sentence than any leaderboard row.

`geom` includes log token count, and step length alone scores 0.7039 on
ProcessBench, so `geom_nolen` is emitted alongside it. Without that pair the
geometry result would be unreadable for exactly the reason iteration 5 found.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.repstore.store import RepSplit  # noqa: E402

from src.harness.geom import geom_feats, _unit, _stats, EPS  # noqa: E402

def layer_feats(span_a: np.ndarray, bnd_a: np.ndarray,
                span_b: np.ndarray, bnd_b: np.ndarray) -> np.ndarray:
    """How much the model revised this step between the two layers.

    Not a concatenation. Every number here is a disagreement between the layers,
    so the vector is small and says something a single layer cannot: whether a
    step is one the later blocks left alone or one they rewrote.
    """
    ua, ub = _unit(span_a), _unit(span_b)
    cos_tok = (ua * ub).sum(1)
    na = np.linalg.norm(span_a, axis=1) + EPS
    nb = np.linalg.norm(span_b, axis=1) + EPS
    ma, mb = span_a.mean(0), span_b.mean(0)
    f = []
    f += _stats(cos_tok)
    f += _stats(np.log(na / nb), qs=(50,))
    f += [float(_unit(ma) @ _unit(mb)),
          float(_unit(bnd_a) @ _unit(bnd_b)),
          float(np.log((np.linalg.norm(ma) + EPS) / (np.linalg.norm(mb) + EPS)))]
    # the revision direction's alignment with the step's own direction: did the
    # late layers push the step further along what it was already saying, or
    # somewhere else
    f += [float(_unit(ma - mb) @ _unit(mb))]
    return np.asarray(f, dtype=np.float32)


def build(span_a, bnd_a, span_b, bnd_b) -> dict[str, np.ndarray]:
    mean = span_a.mean(0)
    contrib = mean - bnd_a
    g = geom_feats(span_a, bnd_a, with_len=True)
    out = {
        "contribution": contrib,
        "contribution_dir": _unit(contrib),
        "boundary": bnd_a,
        "geom": g,
        "geom_nolen": geom_feats(span_a, bnd_a, with_len=False),
        "dir_geom": np.concatenate([_unit(mean), g]),
        "contribution_geom": np.concatenate([contrib, g]),
    }
    if span_b is not None:
        lf = layer_feats(span_a, bnd_a, span_b, bnd_b)
        out["layer_angle"] = lf
        out["geom_layer"] = np.concatenate([g, lf])
        out["dir_geom_layer"] = np.concatenate([_unit(mean), g, lf])
    return out


def _spans(rs: RepSplit, k: int, meta_k: dict):
    a = int(rs.offsets[k]) + int(meta_k["step_start_idx"])
    b = int(rs.offsets[k + 1])
    if b <= a:
        a = b - 1
    span = np.asarray(rs.h[a:b], dtype=np.float32)
    bnd = np.asarray(rs.h[int(rs.offsets[k]) + int(meta_k["pre_step_boundary_idx"])],
                     dtype=np.float32)
    return span, bnd


def collect(dir_a: Path, dir_b: Path | None, names, limit, seed, pb: bool):
    shards_a = sorted(glob.glob(str(dir_a / "shard_*")))
    if not shards_a:
        raise FileNotFoundError(f"no shard_* under {dir_a}")
    shards_b = sorted(glob.glob(str(dir_b / "shard_*"))) if dir_b else [None] * len(shards_a)
    if len(shards_b) != len(shards_a):
        raise SystemExit(f"{dir_a} has {len(shards_a)} shards, {dir_b} has "
                         f"{len(shards_b)}; the two layers are not the same split")
    rng = np.random.default_rng(seed)
    acc = {n: [] for n in names}
    ys, lens = [], []
    per_shard = None if limit is None else max(1, limit // len(shards_a))
    for sa, sb in zip(shards_a, shards_b):
        ra = RepSplit(sa)
        meta = ra.meta()
        rb = RepSplit(sb) if sb else None
        if rb is not None:
            mb = rb.meta()
            # Compare the per-row token counts rather than an id field: the
            # ProcessBench meta has no uid, and the length vector is the stronger
            # check anyway. Two different shards would not reproduce thousands of
            # identical step lengths in the same order.
            if len(mb) != len(meta) or not np.array_equal(ra.lengths, rb.lengths):
                raise SystemExit(
                    f"shard {Path(sa).name} does not describe the same steps in "
                    f"both layers, so every relation would be computed between "
                    f"two different steps")
        idx = np.arange(len(meta))
        if per_shard is not None and len(idx) > per_shard:
            idx = np.sort(rng.choice(len(idx), per_shard, replace=False))
        for k in idx:
            m = meta[int(k)]
            span_a, bnd_a = _spans(ra, int(k), m)
            span_b = bnd_b = None
            if rb is not None:
                span_b, bnd_b = _spans(rb, int(k), mb[int(k)])
                if span_b.shape[0] != span_a.shape[0]:
                    raise SystemExit(f"token count differs between layers at {m['uid']}")
            p = build(span_a, bnd_a, span_b, bnd_b)
            for n in names:
                acc[n].append(p[n])
            lens.append(int(span_a.shape[0]))
            ys.append(1 if (pb and m["label"] == m["step_idx"]) else
                      (int(ra.y[k]) if not pb else 0))
        del ra, rb
    return ({n: np.stack(v) for n, v in acc.items()},
            np.array(ys, dtype=np.float32), np.array(lens, dtype=np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prm_store", required=True, type=Path)
    p.add_argument("--pb_store", required=True, type=Path)
    p.add_argument("--prm_store_b", type=Path, help="Second layer, optional.")
    p.add_argument("--pb_store_b", type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--pb_subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--n_train", type=int, default=60000)
    p.add_argument("--n_pb", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = 8
    probe = build(np.random.default_rng(0).normal(size=(3, d)).astype(np.float32),
                  np.zeros(d, np.float32),
                  np.random.default_rng(1).normal(size=(3, d)).astype(np.float32)
                  if args.prm_store_b else None,
                  np.zeros(d, np.float32) if args.prm_store_b else None)
    names = list(probe)
    print(f"[rel] {len(names)} representations: "
          + ", ".join(f"{n}({probe[n].shape[0] if probe[n].shape[0] < 50 else 'd'})"
                      for n in names), flush=True)

    t0 = time.perf_counter()
    sub_b = (lambda stem: args.prm_store_b / stem) if args.prm_store_b else (lambda s: None)
    tr, ytr, ltr = collect(args.prm_store / args.train_stem, sub_b(args.train_stem),
                           names, args.n_train, args.seed, pb=False)
    print(f"[rel] train {len(ytr):,} ({time.perf_counter()-t0:.0f}s)", flush=True)
    va, yva, lva = collect(args.prm_store / args.val_stem, sub_b(args.val_stem),
                           names, None, args.seed, pb=False)
    pbs = {}
    for s in args.pb_subsets:
        da = args.pb_store / s
        if not da.exists():
            continue
        pbs[s] = collect(da, (args.pb_store_b / s) if args.pb_store_b else None,
                         names, args.n_pb, args.seed, pb=True)
    print(f"[rel] pb {list(pbs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        out = {"x_train": tr[n], "y_train": ytr, "len_train": ltr,
               "x_val": va[n], "y_val": yva, "len_val": lva}
        for s, (px, py, pl) in pbs.items():
            out[f"pb_x_{s}"], out[f"pb_y_{s}"], out[f"pb_len_{s}"] = px[n], py, pl
        np.savez(args.out_dir / f"{n}.npz", **out)
        print(f"[rel] {n:<20} dim {tr[n].shape[1]:>6}  -> {args.out_dir / (n + '.npz')}")


if __name__ == "__main__":
    main()
