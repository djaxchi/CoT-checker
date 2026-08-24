#!/usr/bin/env python3
"""Compact the token store down to the rows any representation actually reads.

The master store (`tokens_last_layer`) keeps the last-layer state of *every*
token of the full sequence: question + prior steps + candidate step, 145.4M rows
for PRM800K train, 984G on disk. Every representation on the leaderboard reads
only two things out of that: the candidate step's own token span, and the single
pre-step boundary state. Measured over the 513,810 train steps, the span is 38.8
tokens on average against 283 for the full sequence, so 86% of the store is
never touched by any learner.

This script slices each item down to

    [ pre-step boundary state ] ++ [ every token of the step ]

and writes it back in the same store format. The saving is ~6.7x (147G against
984G for PRM800K train), which is what makes it affordable to train the sequence
learners on the *full* split rather than a 150k subsample, and what pulls
$SCRATCH back under quota.

The meta contract is preserved exactly: `pre_step_boundary_idx` becomes 0,
`step_start_idx` becomes 1, and `n_tokens` becomes the new item length, so every
existing reader (`read_span`, `derive_split`) works against the compact store
with no code change and returns identical vectors. The original offsets are kept
as `orig_*` fields for traceability.

What is discarded: the question and prior-step token states. No current
representation reads them; a future one that wants to attend from the step back
into the problem statement would need the master store (or a re-encode).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.repstore.store import RepSpec, RepSplit  # noqa: E402


def span_bounds(meta_row: dict, item_len: int) -> tuple[int, int]:
    """Rows [a, b) of an item to keep: the pre-step boundary through the step end.

    Returns local (start, end) offsets inside the item.
    """
    pre = int(meta_row["pre_step_boundary_idx"])
    if pre < 0:
        raise ValueError(f"negative pre_step_boundary_idx in {meta_row.get('uid')!r}")
    start = int(meta_row["step_start_idx"])
    if start != pre + 1:
        raise ValueError(
            f"step_start_idx {start} != pre_step_boundary_idx+1 {pre + 1} "
            f"in {meta_row.get('uid')!r}"
        )
    if pre >= item_len:
        raise ValueError(f"boundary {pre} outside item of length {item_len}")
    return pre, item_len


def compact_meta(meta_row: dict, new_len: int) -> dict:
    """Rewrite one meta row for the compacted item."""
    out = dict(meta_row)
    out["orig_n_tokens"] = int(meta_row["n_tokens"])
    out["orig_step_start_idx"] = int(meta_row["step_start_idx"])
    out["orig_pre_step_boundary_idx"] = int(meta_row["pre_step_boundary_idx"])
    out["n_tokens"] = int(new_len)
    out["step_start_idx"] = 1
    out["pre_step_boundary_idx"] = 0
    return out


def compact_shard(shard_dir: Path, out_dir: Path, spec_name: str) -> dict:
    """Slice one shard to the boundary+span rows, writing a standalone RepSplit."""
    rs = RepSplit(shard_dir)
    meta = rs.meta()
    n = len(meta)
    d = rs.spec.dim

    starts = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    for k, m in enumerate(meta):
        item_len = int(rs.lengths[k])
        if item_len != int(m["n_tokens"]):
            raise ValueError(
                f"lengths[{k}]={item_len} != meta n_tokens={m['n_tokens']} in {shard_dir}"
            )
        a, b = span_bounds(m, item_len)
        starts[k] = int(rs.offsets[k]) + a
        lengths[k] = b - a

    out_dir.mkdir(parents=True, exist_ok=True)
    total = int(lengths.sum())
    h_out = np.lib.format.open_memmap(
        out_dir / "h.npy", mode="w+", dtype=np.float16, shape=(total, d)
    )
    cur = 0
    for k in range(n):
        L = int(lengths[k])
        h_out[cur:cur + L] = rs.h[starts[k]:starts[k] + L]
        cur += L
    h_out.flush()
    del h_out

    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "y.npy", np.asarray(rs.y, dtype=np.int8))
    with (out_dir / "meta.jsonl").open("w") as f:
        for k, m in enumerate(meta):
            f.write(json.dumps(compact_meta(m, int(lengths[k]))) + "\n")

    spec = RepSpec(
        name=spec_name,
        kind=rs.spec.kind,
        dim=d,
        layer=rs.spec.layer,
        backbone=rs.spec.backbone,
        readout="step_span_with_boundary",
        source_split=rs.spec.source_split,
        reduce_default=rs.spec.reduce_default,
    )
    (out_dir / "spec.json").write_text(spec.to_json())
    return {"items": n, "rows_in": int(rs.offsets[-1]), "rows_out": total}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_root", required=True, type=Path,
                   help="Master token store rep dir, e.g. <repstore>/tokens_last_layer")
    p.add_argument("--out_root", required=True, type=Path,
                   help="Destination rep dir, e.g. <repstore>/step_spans")
    p.add_argument("--splits", nargs="+", required=True,
                   help="Split stems under --store_root (e.g. probe_train_full val_5k test_2k)")
    p.add_argument("--name", default="step_spans", help="Name written into spec.json")
    args = p.parse_args()

    for stem in args.splits:
        src = args.store_root / stem
        shard_dirs = sorted(glob.glob(str(src / "shard_*")))
        if not shard_dirs:
            raise SystemExit(f"no shard_* under {src}")
        tot_in = tot_out = tot_items = 0
        for sd in shard_dirs:
            sd = Path(sd)
            stats = compact_shard(sd, args.out_root / stem / sd.name, args.name)
            tot_in += stats["rows_in"]
            tot_out += stats["rows_out"]
            tot_items += stats["items"]
            print(f"[{stem}/{sd.name}] {stats['items']} items  "
                  f"{stats['rows_in']} -> {stats['rows_out']} rows", flush=True)
        ratio = tot_out / max(tot_in, 1)
        print(f"[{stem}] TOTAL {tot_items} items  {tot_in} -> {tot_out} rows "
              f"({ratio:.3f}x, mean {tot_out / max(tot_items, 1):.1f} rows/item)",
              flush=True)


if __name__ == "__main__":
    main()
