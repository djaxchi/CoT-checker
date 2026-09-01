#!/usr/bin/env python3
"""Represent a step by where it sits in its OWN trace, not by where it sits in
model space.

Two hypotheses, one pass, because they need the same machinery.

R1, trace-relative coordinates. This project's sharpest unexplained result is
that the pre-step boundary state alone, containing none of the step's own tokens,
scores 0.7412 in domain and 0.5035 on ProcessBench. Most of what an in-domain
probe reads is which problem this is and where in the solution we are, and none
of it survives the domain change. Meanwhile the task is relative: ProcessBench
hands you a whole solution and asks which step FIRST goes wrong, a comparison
among siblings, and every representation we have is absolute. GeoReason
(arXiv:2605.13772) arrives at the same normalisation from the other direction,
putting each trace in "a local coordinate system centered at their correct
prefix" because raw states "contain many nuisance directions such as prompt
topic, syntax, and answer length".

R2, between-step dynamics. `diffs` measured motion between TOKENS inside a step
and scored 0.7600. Nothing here has ever measured motion between STEPS.
`contribution` (h_i - h_{i-1}) is the raw velocity vector and lost to plain
content, 0.7406 against 0.7470, but only the vector was ever used: never its
length relative to the rest of the trace, never its angle to the previous step's
motion, never the second difference. Those are the seven features GeoReason
builds its detector out of.

Both are scale free and neither contains a token count, which matters because
step length alone scores 0.7039 here.

Position is kept out of both and emitted separately as `pos`. First errors sit
later in a trace on average, so relative position is a shortcut of exactly the
kind step length turned out to be, and the only way to read a trace-relative
result is against a control that is nothing but position.

The store forces the shape of this script. ProcessBench `global_index` strides by
four across shards, so one trace's steps are SPLIT ACROSS SHARDS, and the
per-shard sampling the pooling screen uses would hand back fragments of traces
with their siblings missing. So: one cheap pass over the metadata to group steps
into traces, sample whole traces, then one gather pass for the rows that survived.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.harness.geom import geom_feats  # noqa: E402
from src.repstore.store import RepSplit  # noqa: E402

EPS = 1e-8


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def _cos(a, b) -> float:
    return float(_unit(a) @ _unit(b))


def trace_key(m: dict) -> str:
    """PRM800K groups by solution, ProcessBench by problem id."""
    return m.get("solution_id") or m["id"]


def dynamics(states: np.ndarray, boundary0: np.ndarray) -> np.ndarray:
    """Ten scale-free numbers per step describing motion through the trace.

    Every quantity is either an angle or a ratio against the trace's own median,
    so nothing here can be read as a token count or an absolute activation scale.
    The first step has no predecessor inside the trace, so its incoming motion is
    measured from the pre-step boundary state, which is what the prefix actually
    was.
    """
    n = states.shape[0]
    prev = np.vstack([boundary0[None, :], states[:-1]])
    delta = states - prev
    nd = np.linalg.norm(delta, axis=1) + EPS
    med = float(np.median(nd)) + EPS
    centroid = states.mean(0)
    dev = np.linalg.norm(states - centroid, axis=1)
    med_dev = float(np.median(dev)) + EPS

    out = np.zeros((n, 10), dtype=np.float32)
    for i in range(n):
        nxt = delta[i + 1] if i + 1 < n else delta[i]
        prv = delta[i - 1] if i > 0 else delta[i]
        out[i] = [
            nd[i] / med,                              # speed, relative to the trace
            np.log(nd[i]) - np.log(med),
            _cos(delta[i], prv),                      # does the motion persist
            _cos(delta[i], nxt),                      # or does the next step turn
            np.linalg.norm(delta[i] - prv) / med,     # acceleration, relative
            _cos(delta[i], states[i]),                # motion against the state
            _cos(delta[i], states[0]),                # motion against the opening
            _cos(states[i], states[0]),
            _cos(states[i], centroid),                # how typical inside the trace
            dev[i] / med_dev,                         # how far out, relative
        ]
    return out


def position(step_idx: np.ndarray, n: int) -> np.ndarray:
    """The control. Relative position and trace length, nothing else."""
    return np.stack([step_idx / max(n - 1, 1),
                     np.full(len(step_idx), np.log(max(n, 1)))], 1).astype(np.float32)


def build_trace(states, boundary, lengths, step_idx) -> dict[str, np.ndarray]:
    n = states.shape[0]
    centroid = states.mean(0)
    sd = states.std(0) + 1e-3
    causal = np.empty_like(states)
    for i in range(n):
        causal[i] = states[i] - (states[:i].mean(0) if i else boundary[0])
    # leave-one-out mean, computed in closed form rather than by deleting a row
    # n times. Legitimate here even though it reads later steps: ProcessBench
    # hands the model a complete solution and asks which step is first wrong.
    loo = ((centroid * n)[None, :] - states) / max(n - 1, 1)
    leave_out = states - (loo if n > 1 else boundary)
    return {
        "trace_centered_causal": causal,
        "trace_centered_all": leave_out,
        "trace_z": (states - centroid) / sd,
        "dyn": dynamics(states, boundary[0]),
        "pos": position(step_idx, n),
    }


def collect(split_dir: Path, n_traces: int | None, seed: int, pb: bool):
    """Group steps into traces across shards, sample whole traces, then gather."""
    shards = sorted(glob.glob(str(split_dir / "shard_*")))
    if not shards:
        raise FileNotFoundError(f"no shard_* under {split_dir}")
    traces: dict[str, list] = defaultdict(list)
    for si, sd in enumerate(shards):
        for k, m in enumerate(RepSplit(sd).meta()):
            traces[trace_key(m)].append((si, k, int(m["step_idx"]), m))
    keys = sorted(traces)
    rng = np.random.default_rng(seed)
    if n_traces is not None and len(keys) > n_traces:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), n_traces, replace=False))]
    for k in keys:
        traces[k].sort(key=lambda r: r[2])

    want: dict[int, list] = defaultdict(list)
    for ki, k in enumerate(keys):
        for pos_in_trace, (si, row, _, _) in enumerate(traces[k]):
            want[si].append((row, ki, pos_in_trace))

    per_trace = {k: {"states": [None] * len(traces[k]), "bnd": [None] * len(traces[k]),
                     "geom": [None] * len(traces[k]), "len": [0] * len(traces[k])}
                 for k in keys}
    for si, jobs in sorted(want.items()):
        rs = RepSplit(shards[si])
        meta = rs.meta()
        for row, ki, p in sorted(jobs):
            m = meta[row]
            a = int(rs.offsets[row]) + int(m["step_start_idx"])
            b = int(rs.offsets[row + 1])
            if b <= a:
                a = b - 1
            span = np.asarray(rs.h[a:b], dtype=np.float32)
            bnd = np.asarray(rs.h[int(rs.offsets[row]) + int(m["pre_step_boundary_idx"])],
                             dtype=np.float32)
            t = per_trace[keys[ki]]
            t["states"][p] = span.mean(0)
            t["bnd"][p] = bnd
            t["geom"][p] = geom_feats(span, bnd, with_len=False)
            t["len"][p] = span.shape[0]
            if not pb and int(m["label"]) != int(rs.y[row]):
                raise SystemExit(
                    f"meta label {m['label']} disagrees with the stored y "
                    f"{int(rs.y[row])} at row {row} of {shards[si]}; the trace "
                    f"grouping reads labels from meta, so they must agree")
        del rs

    acc: dict[str, list] = defaultdict(list)
    ys, lens = [], []
    for k in keys:
        t = per_trace[k]
        states = np.stack(t["states"])
        bnd = np.stack(t["bnd"])
        recs = traces[k]
        rel = build_trace(states, bnd, np.asarray(t["len"], np.float32),
                          np.asarray([r[2] for r in recs], np.float32))
        acc["mean"].append(states)
        acc["geom_nolen"].append(np.stack(t["geom"]))
        for name, v in rel.items():
            acc[name].append(v)
        lens.extend(t["len"])
        for _, _, _, m in recs:
            ys.append(1 if (pb and m["label"] == m["step_idx"]) else
                      (int(m["label"]) if not pb else 0))
    out = {n: np.concatenate(v).astype(np.float32) for n, v in acc.items()}
    return out, np.array(ys, np.float32), np.array(lens, np.float32)


def fit_length_map(x, lengths):
    a = np.stack([np.ones_like(lengths), np.log(np.maximum(lengths, 1.0))], 1)
    coef, *_ = np.linalg.lstsq(a.astype(np.float64), x.astype(np.float64), rcond=None)
    return coef.astype(np.float32)


def apply_length_map(x, lengths, coef):
    a = np.stack([np.ones_like(lengths), np.log(np.maximum(lengths, 1.0))], 1)
    return x.astype(np.float32) - a.astype(np.float32) @ coef


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prm_store", required=True, type=Path)
    p.add_argument("--pb_store", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--pb_subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--n_traces", type=int, default=9000)
    p.add_argument("--n_pb_traces", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.perf_counter()
    tr, ytr, ltr = collect(args.prm_store / args.train_stem, args.n_traces, args.seed, False)
    print(f"[trace] train {len(ytr):,} steps ({time.perf_counter()-t0:.0f}s)", flush=True)
    va, yva, lva = collect(args.prm_store / args.val_stem, None, args.seed, False)
    pbs = {}
    for s in args.pb_subsets:
        d = args.pb_store / s
        if d.exists():
            pbs[s] = collect(d, args.n_pb_traces, args.seed, True)
    print(f"[trace] pb {list(pbs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # the current winner, recomputed on THESE rows so every comparison is within
    # one sample rather than across two different samplings of the store
    coef = fit_length_map(tr["mean"], ltr)
    def winner(d, lens):
        return np.concatenate([apply_length_map(d["mean"], lens, coef), d["geom_nolen"]], 1)
    for d, lens in [(tr, ltr), (va, lva)] + [(v[0], v[2]) for v in pbs.values()]:
        d["winner"] = winner(d, lens)
    for d in [tr, va] + [v[0] for v in pbs.values()]:
        d["winner_trace"] = np.concatenate([d["winner"], d["trace_centered_causal"]], 1)
        d["winner_dyn"] = np.concatenate([d["winner"], d["dyn"]], 1)
        d["winner_trace_dyn"] = np.concatenate([d["winner_trace"], d["dyn"]], 1)

    names = [n for n in tr if n != "geom_nolen"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        out = {"x_train": tr[n], "y_train": ytr, "len_train": ltr,
               "x_val": va[n], "y_val": yva, "len_val": lva}
        for s, (px, py, pl) in pbs.items():
            out[f"pb_x_{s}"], out[f"pb_y_{s}"], out[f"pb_len_{s}"] = px[n], py, pl
        np.savez(args.out_dir / f"{n}.npz", **out)
        print(f"[trace] {n:<26} dim {tr[n].shape[1]:>6}  -> {args.out_dir / (n + '.npz')}")


if __name__ == "__main__":
    main()
