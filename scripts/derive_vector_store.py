#!/usr/bin/env python3
"""Derive fixed-vector reps from one span-store shard into small vector stores.

judge_prompt_v1 on TamIA: the span store (~165 GB) does not fit the shared
scratch, so each encode job writes spans to node-local disk, derives the vector
reps the cells need from each shard, and keeps only those (~45 GB in all) on
shared storage. Vector cells and the geometry study then run as separate small
jobs with `--prederived`.

Output, per rep: <out_root>/<rep>/<split>/<shard>/ in the repstore format with
one row per item (lengths 1, pre_step_boundary_idx 0), so the `last` readout of
the existing reader returns the vector itself. Meta, labels and order are the
span shard's, so ProcessBench first-error scans read back in the same order.
The readout code is the trainer's own (derive_delta_from_token_store), so a
pre-derived vector is the vector the cell would have derived.

    python scripts/derive_vector_store.py --shard_dir <spans>/<split>/shard_03 \
        --out_root <vec> --reps last_token step_mean boundary_stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.derive_delta_from_token_store import _shard_vecs  # noqa: E402
from src.repstore.store import VECTOR, RepSpec, RepSplit  # noqa: E402

READOUT = {"last_token": "last", "step_mean": "mean", "step_delta": "delta",
           "step_stats": "multistat", "boundary_stats": "boundary_stats"}


def derive_shard(shard_dir: Path, out_root: Path, rep: str) -> Path:
    rs = RepSplit(shard_dir)
    meta = rs.meta()
    split, shard = shard_dir.parent.name, shard_dir.name
    out = out_root / rep / split / shard
    out.mkdir(parents=True, exist_ok=True)
    X = _shard_vecs(rs, meta, READOUT[rep])
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite {rep} vectors in {shard_dir}")
    tmp = out / "h.npy.tmp"
    with tmp.open("wb") as f:
        np.save(f, X)
    np.save(out / "lengths.npy", np.ones(len(meta), dtype=np.int32))
    np.save(out / "y.npy", np.asarray(rs.y, dtype=np.int8))
    with (out / "meta.jsonl").open("w") as f:
        for m in meta:
            row = dict(m)
            row.update({"span_n_tokens": m["n_tokens"], "n_tokens": 1,
                        "step_start_idx": 0, "pre_step_boundary_idx": 0})
            f.write(json.dumps(row) + "\n")
    spec = RepSpec(name=rep, kind=VECTOR, dim=int(X.shape[1]), layer=rs.spec.layer,
                   backbone=rs.spec.backbone, readout=rep, source_split=split,
                   prompt_style=rs.spec.prompt_style)
    (out / "spec.json").write_text(spec.to_json())
    tmp.rename(out / "h.npy")   # last: a shard counts as present only once complete
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard_dir", required=True, type=Path, nargs="+")
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument("--reps", nargs="+", required=True, choices=sorted(READOUT))
    a = p.parse_args()
    for sd in a.shard_dir:
        for rep in a.reps:
            out = derive_shard(sd, a.out_root, rep)
            print(f"[vecstore] {sd} -> {out}", flush=True)


if __name__ == "__main__":
    main()
