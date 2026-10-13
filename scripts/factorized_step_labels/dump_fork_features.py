#!/usr/bin/env python3
"""Pooled L35 step states of the fork views for the linear risk probes (risk_probe_v0).

For every item of --split in the store: the mean over the step's own token rows
and the step's last token row (the rows the `local` arm reads, without the
header). Writes <out>.npz with ids, mean, last (float32).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_contextual_token_probe import TrajStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--split", default="fsl_forks")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    store = TrajStore(a.store, splits={a.split})
    ids = store.ids(a.split)
    mean, last = [], []
    for t in ids:
        it = store.items[t]
        H = np.asarray(store.h[it["shard"]][it["a"]:it["b"]], dtype=np.float32)
        s, e = it["step_starts"][-1], it["step_ends"][-1]
        mean.append(H[s:e].mean(0)); last.append(H[e - 1])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, ids=np.array(ids), mean=np.stack(mean), last=np.stack(last))
    print(f"[dump] {len(ids)} items, d={mean[0].shape[0]} -> {a.out}")


if __name__ == "__main__":
    main()
