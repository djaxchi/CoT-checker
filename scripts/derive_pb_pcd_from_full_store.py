"""Derive the ProcessBench future-delta (pcd) representation offline.

Reads the full-solution PB store (encode_processbench_full_store.py) and, for each
step, emits the same vector the PRM encoder builds, but using ProcessBench's REAL
next step (available at eval, no generation needed):
    pcd = concat[ past boundary, mean(current step tokens),
                  within-step delta of the next step (S_next^end - S_next^preboundary) ]
Last step of a trace has no next -> zero future delta.

Writes the harness ProcessBench contract per subset: pb_step_h.npy + pb_step_meta
.jsonl, one row per step, traces in global order and steps in step order (so the
first-error scan lines up).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.repstore.store import ShardedRepSplit  # noqa: E402


def derive_subset(store: ShardedRepSplit):
    meta = store.meta()
    vecs, out_meta = [], []
    for k in range(len(store)):
        h = store.item(k)  # (n_tokens, d)
        m = meta[k]
        ss, se, ns = m["step_starts"], m["step_ends"], m["n_steps"]
        for j in range(ns):
            past = h[ss[j] - 1]
            cur = h[ss[j]:se[j]].mean(0)
            if j + 1 < ns:
                delta = h[se[j + 1] - 1] - h[ss[j + 1] - 1]
            else:
                delta = np.zeros_like(cur)
            vecs.append(np.concatenate([past, cur, delta]).astype(np.float16))
            out_meta.append({"id": m["id"], "step_idx": j, "label": m["label"],
                             "n_steps": ns, "pb_subset": m["pb_subset"]})
    return np.asarray(vecs, dtype=np.float16), out_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_root", required=True, type=Path,
                   help="full-solution store root; subsets are subdirs")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--subsets", nargs="+", default=["gsm8k", "math", "olympiadbench", "omnimath"])
    args = p.parse_args()
    for sub in args.subsets:
        sd = args.store_root / sub
        if not sd.exists():
            print(f"[skip] {sub} (no store)", flush=True)
            continue
        h, meta = derive_subset(ShardedRepSplit(sd))
        out = args.out_dir / sub
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "pb_step_h.npy", h)
        with (out / "pb_step_meta.jsonl").open("w") as f:
            for row in meta:
                f.write(json.dumps(row) + "\n")
        print(f"[pcd-pb:{sub}] {h.shape} -> {out}/pb_step_h.npy", flush=True)


if __name__ == "__main__":
    main()
