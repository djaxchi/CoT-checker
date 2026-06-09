#!/usr/bin/env python3
"""Sample a subset of PRM800K forks (keeping all roles of each selected fork) so
the across-size fork visualization encodes a small, identical item set with every
backbone. CPU only.

Input: forks_*_items.jsonl (rows with fork_id, role, item_uid, problem, prefix,
candidate_step). Output: same schema, restricted to --n_forks deterministically
chosen forks, all of their anchor/positive/negative items preserved.

Usage:
    python scripts/s1ms_sample_forks.py --items $FORKS/forks_val_items.jsonl \\
        --n_forks 1000 --seed 42 --out runs/s1_model_size_dense/_forks_sample/forks_val_items.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--n_forks", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.items.exists():
        raise SystemExit(f"[sample-forks] items file not found: {args.items}")

    rows = [json.loads(l) for l in args.items.read_text().splitlines() if l.strip()]
    fork_ids = sorted({r["fork_id"] for r in rows})
    if len(fork_ids) <= args.n_forks:
        keep = set(fork_ids)
    else:
        sel = np.random.default_rng(args.seed).choice(len(fork_ids), size=args.n_forks, replace=False)
        keep = {fork_ids[i] for i in sel}

    out_rows = [r for r in rows if r["fork_id"] in keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sample-forks] {len(keep)} forks -> {len(out_rows)} items -> {args.out}")


if __name__ == "__main__":
    main()
