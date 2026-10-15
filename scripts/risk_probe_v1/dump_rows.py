#!/usr/bin/env python3
"""Per-view L35 rows for the risk_probe_v1 readouts (docs/risk_probe_v1_plan.md, section 2).

For every label-step view (lab_views.jsonl, across xf0..xf4) and every fork view
(fsl_forks): the row ahead of the step (`before`, cut s), the step's last row
(`after`, cut s + 1) and the mean over the step's rows (risk_probe_v0's `mean`),
from the node-local store. Writes <out>.npz with ids, split, before, after, mean
(float16).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.analysis.risk_probe_v1 import step_rows  # noqa: E402
from train_contextual_token_probe import TrajStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--lab_views", type=Path, required=True)
    ap.add_argument("--fork_split", default="fsl_forks")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lab = [json.loads(l)["view_id"] for l in open(a.lab_views) if l.strip()]
    splits = {f"xf{k}" for k in range(5)} | {a.fork_split}
    store = TrajStore(a.store, splits=splits)
    ids = lab + store.ids(a.fork_split)
    missing = [t for t in ids if t not in store.items]
    if missing:
        sys.exit(f"[dump] {len(missing)} views not in the store, e.g. {missing[:3]}")
    B, A, M, sp = [], [], [], []
    for t in ids:
        it = store.items[t]
        if len(it["step_starts"]) != 1:
            sys.exit(f"[dump] {t}: expected one stored step, got {len(it['step_starts'])}")
        H = store.h[it["shard"]][it["a"]:it["b"]]
        b, af, m = step_rows(H, it["step_starts"][0], it["step_ends"][0])
        B.append(b); A.append(af); M.append(m); sp.append(it["split"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, ids=np.array(ids), split=np.array(sp), before=np.stack(B).astype(np.float16),
             after=np.stack(A).astype(np.float16), mean=np.stack(M).astype(np.float16))
    print(f"[dump] {len(ids)} views ({len(lab)} label steps), d={B[0].shape[0]} -> {a.out}")


if __name__ == "__main__":
    main()
