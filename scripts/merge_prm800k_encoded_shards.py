#!/usr/bin/env python3
"""Merge per-GPU PRM800K encoding shards into a single deterministic split.

Companion to the sharded mode of encode_prm800k_hidden_states.py
(--shard_idx/--num_shards). Each shard writes {stem}_h.npy, {stem}_y.npy and
{stem}_meta.jsonl into its own shard_NN/ directory; every meta row carries a
``global_index`` assigned over the FULL pre-shard file order. This script
concatenates all shards, sorts by global_index, and writes a single merged
{stem}_h.npy / {stem}_y.npy / {stem}_meta.jsonl identical (row-for-row) to what
a non-sharded run would have produced.

Usage:
    python scripts/merge_prm800k_encoded_shards.py \\
        --shard_root <cache>/shards \\
        --stem probe_train_40k \\
        --out_dir <cache> \\
        [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SHARD_DIR_RE = re.compile(r"^shard_(\d+)$")


def discover_shards(shard_root: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for child in sorted(shard_root.iterdir()):
        if child.is_dir() and (m := SHARD_DIR_RE.match(child.name)):
            out.append((int(m.group(1)), child))
    out.sort(key=lambda x: x[0])
    return out


def load_meta(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard_root", type=Path, required=True,
                   help="Directory containing shard_00/, shard_01/, ...")
    p.add_argument("--stem", type=str, required=True,
                   help="Split stem, e.g. probe_train_40k or val_1k.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    shards = discover_shards(args.shard_root)
    if not shards:
        sys.exit(f"[merge_prm] no shard_NN/ dirs under {args.shard_root}")

    out_h = args.out_dir / f"{args.stem}_h.npy"
    out_y = args.out_dir / f"{args.stem}_y.npy"
    out_meta = args.out_dir / f"{args.stem}_meta.jsonl"
    for f in (out_h, out_y, out_meta):
        if f.exists() and not args.force:
            sys.exit(f"[merge_prm] refusing to overwrite {f}; pass --force")

    h_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    metas: list[dict] = []
    seen_gi: set[int] = set()
    duplicates: list[int] = []
    per_shard_rows: list[int] = []

    for idx, sdir in shards:
        h_path = sdir / f"{args.stem}_h.npy"
        y_path = sdir / f"{args.stem}_y.npy"
        m_path = sdir / f"{args.stem}_meta.jsonl"
        if not (h_path.exists() and y_path.exists() and m_path.exists()):
            sys.exit(f"[merge_prm] shard {idx} missing one of {h_path}/{y_path}/{m_path}")
        h = np.load(h_path)
        y = np.load(y_path)
        m = load_meta(m_path)
        if not (h.shape[0] == y.shape[0] == len(m)):
            sys.exit(f"[merge_prm] shard {idx}: row mismatch h={h.shape[0]} y={y.shape[0]} meta={len(m)}")
        if not np.all(np.isfinite(h)):
            sys.exit(f"[merge_prm] shard {idx}: NaN/Inf in {h_path}")
        for row in m:
            gi = row.get("global_index")
            if gi is None:
                sys.exit(f"[merge_prm] shard {idx}: meta row missing global_index; re-encode with the sharded encoder.")
            gi = int(gi)
            if gi in seen_gi:
                duplicates.append(gi)
            seen_gi.add(gi)
        h_arrays.append(h)
        y_arrays.append(y)
        metas.extend(m)
        per_shard_rows.append(h.shape[0])
        print(f"[merge_prm] shard {idx:02d}: rows={h.shape[0]}")

    if duplicates:
        sys.exit(
            f"[merge_prm] FATAL: {len(duplicates)} duplicate global_index across "
            f"shards (first 10: {duplicates[:10]}). Each example must be encoded once."
        )

    big_h = np.concatenate(h_arrays, axis=0)
    big_y = np.concatenate(y_arrays, axis=0)
    if not (big_h.shape[0] == big_y.shape[0] == len(metas)):
        sys.exit("[merge_prm] concat mismatch after merge")

    order = sorted(range(len(metas)), key=lambda i: int(metas[i]["global_index"]))
    big_h = big_h[order]
    big_y = big_y[order]
    metas = [metas[i] for i in order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_h, big_h)
    np.save(out_y, big_y)
    with out_meta.open("w") as f:
        for row in metas:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[merge_prm] merged {big_h.shape[0]} rows ({per_shard_rows}) -> {out_h}")
    print(f"[merge_prm] labels -> {out_y}")
    print(f"[merge_prm] meta   -> {out_meta}")


if __name__ == "__main__":
    main()
