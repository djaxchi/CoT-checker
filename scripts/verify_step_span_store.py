#!/usr/bin/env python3
"""Prove the compact step-span store reproduces the master store exactly.

`build_step_span_store.py` throws away 86% of the master token store on the
argument that no representation reads those rows. That argument is only worth
acting on, and deleting a 984G store is only safe, if the vectors derived from
the compact store are identical to the ones derived from the master, on the real
data rather than on synthetic fixtures.

For each split and each readout this derives both stores and compares the
vectors, the labels and the global ordering. Exits non-zero on any mismatch, so
it can gate a deletion in a script.

Run it on the small splits first (val_5k, test_2k): they exercise every code path
and cost a few GB of reads, where the train split costs a terabyte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.derive_delta_from_token_store import derive_split  # noqa: E402

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", required=True, type=Path)
    p.add_argument("--compact", required=True, type=Path)
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--readouts", nargs="+", default=list(READOUTS), choices=READOUTS)
    args = p.parse_args()

    failures = 0
    for stem in args.splits:
        for readout in args.readouts:
            if not compare(args.master, args.compact, stem, readout):
                failures += 1
    if failures:
        raise SystemExit(f"{failures} mismatch(es): the compact store is NOT "
                         f"equivalent, do not delete the master")
    print("all readouts identical: the compact store is equivalent to the master")


if __name__ == "__main__":
    main()
