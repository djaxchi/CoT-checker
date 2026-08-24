#!/usr/bin/env python3
"""Prove the compact step-span store reproduces the master store exactly.

`build_step_span_store.py` throws away 86% of the master token store on the
argument that no representation reads those rows. That argument is only worth
acting on, and deleting a 984G store is only safe, if the vectors derived from
the compact store are identical to the ones derived from the master, on the real
data rather than on synthetic fixtures.

Two modes, and they answer different questions.

`rows` is the strong one: for every item it compares the kept rows byte for byte
against the master's `[pre_step_boundary_idx : n_tokens)` slice. If those bytes
are identical then *every* readout agrees, including ones not written yet, so it
settles the question for the whole harness rather than for the six readouts that
happen to exist today. It reads only the kept rows, ~147 GB rather than the full
984G, so it is affordable on the train split.

`readouts` is the end-to-end one: it derives both stores and compares the
vectors, the labels and the global ordering, which also exercises the offset
rewriting in the meta rather than just the bytes.

Both exit non-zero on any mismatch, so either can gate a deletion in a script.
Also prints each split's fingerprint, the value every grid cell records, so the
store the benchmark trains on is identifiable after the master is gone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.derive_delta_from_token_store import derive_split  # noqa: E402

from src.repstore import split_fingerprint  # noqa: E402
from src.repstore.store import RepSplit  # noqa: E402

READOUTS = ("last", "mean", "max", "delta", "multistat", "boundary_stats")


def compare(master: Path, compact: Path, stem: str, readout: str) -> bool:
    a, ya, ma = derive_split(master / stem, readout, sort=True)
    b, yb, mb = derive_split(compact / stem, readout, sort=True)
    same_v = a.shape == b.shape and np.array_equal(a, b)
    same_y = np.array_equal(ya, yb)
    same_o = [m["global_index"] for m in ma] == [m["global_index"] for m in mb]
    ok = same_v and same_y and same_o
    detail = "" if ok else (
        f"  vectors={same_v} labels={same_y} order={same_o}"
        f" shapes {a.shape} vs {b.shape}")
    print(f"{stem:20s} {readout:16s} {str(a.shape):>18s} "
          f"{'IDENTICAL' if ok else 'MISMATCH'}{detail}", flush=True)
    return ok


def compare_rows(master: Path, compact: Path, stem: str, report_every: int) -> bool:
    """Byte-for-byte equality of every kept row, shard by shard, item by item.

    Stronger than comparing derived vectors: if the retained rows are identical
    then any readout over them is identical too, whether or not it exists yet.
    """
    m_shards = sorted((master / stem).glob("shard_*"))
    c_shards = sorted((compact / stem).glob("shard_*"))
    if [d.name for d in m_shards] != [d.name for d in c_shards]:
        print(f"{stem}: shard sets differ {[d.name for d in m_shards]} vs "
              f"{[d.name for d in c_shards]}")
        return False

    total = bad = 0
    for m_dir, c_dir in zip(m_shards, c_shards):
        ms, cs = RepSplit(m_dir), RepSplit(c_dir)
        m_meta, c_meta = ms.meta(), cs.meta()
        if len(m_meta) != len(c_meta):
            print(f"{stem}/{m_dir.name}: item counts differ "
                  f"{len(m_meta)} vs {len(c_meta)}")
            return False
        for k, (mm, cm) in enumerate(zip(m_meta, c_meta)):
            if mm.get("uid") != cm.get("uid") or \
                    int(mm["global_index"]) != int(cm["global_index"]):
                print(f"{stem}/{m_dir.name}[{k}]: identity differs "
                      f"{mm.get('uid')} vs {cm.get('uid')}")
                bad += 1
                continue
            pre = int(mm["pre_step_boundary_idx"])
            a = int(ms.offsets[k]) + pre
            b = int(ms.offsets[k + 1])
            ca, cb = int(cs.offsets[k]), int(cs.offsets[k + 1])
            if (b - a) != (cb - ca) or not np.array_equal(ms.h[a:b], cs.h[ca:cb]):
                print(f"{stem}/{m_dir.name}[{k}] uid={mm.get('uid')}: rows differ "
                      f"({b - a} vs {cb - ca})")
                bad += 1
            total += 1
            if report_every and total % report_every == 0:
                print(f"  {stem}: {total} items checked, {bad} bad", flush=True)
        del ms, cs
    print(f"{stem:20s} rows{'':12s} {total:>10,} items  "
          f"{'IDENTICAL' if bad == 0 else f'{bad} MISMATCHES'}", flush=True)
    return bad == 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", required=True, type=Path)
    p.add_argument("--compact", required=True, type=Path)
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--readouts", nargs="+", default=list(READOUTS), choices=READOUTS)
    p.add_argument("--mode", choices=["rows", "readouts", "both"], default="both")
    p.add_argument("--report_every", type=int, default=50000,
                   help="Progress line every N items in row mode (0 = silent).")
    args = p.parse_args()

    failures = 0
    for stem in args.splits:
        if args.mode in ("rows", "both"):
            if not compare_rows(args.master, args.compact, stem, args.report_every):
                failures += 1
        if args.mode in ("readouts", "both"):
            for readout in args.readouts:
                if not compare(args.master, args.compact, stem, readout):
                    failures += 1
        print(f"{stem:20s} fingerprint(compact) "
              f"{split_fingerprint(args.compact / stem)}", flush=True)
    if failures:
        raise SystemExit(f"{failures} mismatch(es): the compact store is NOT "
                         f"equivalent, do not delete the master")
    print("the compact store is equivalent to the master on every check run")


if __name__ == "__main__":
    main()
