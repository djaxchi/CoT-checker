"""Merge per-GPU shards of the multi-token/multi-layer PRM800K encoding.

Companion to encode_prm800k_multitoken_multilayer.py (sharded mode). Each shard
writes {stem}_h.npy (n_shard, L, T, H), {stem}_y.npy, {stem}_meta.jsonl into its
own shard_NN/ dir; every meta row carries a global_index over the full pre-shard
file order. This concatenates all shards, sorts by global_index, and writes a
single merged {stem}_{h,y,meta} + manifest identical to a non-sharded run.

Usage:
    python scripts/merge_prm800k_multitoken_shards.py \
        --shard_root <out> --stem prm800k_heldout_test --out_dir <out> [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SHARD_DIR_RE = re.compile(r"^shard_(\d+)$")


def discover_shards(shard_root: Path) -> list[Path]:
    out = []
    for child in sorted(shard_root.iterdir()):
        if child.is_dir() and SHARD_DIR_RE.match(child.name):
            out.append(child)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_root", type=Path, required=True)
    ap.add_argument("--stem", type=str, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if (args.out_dir / f"{args.stem}_h.npy").exists() and not args.force:
        sys.exit(f"refusing overwrite {args.out_dir}/{args.stem}_h.npy; pass --force")

    shards = discover_shards(args.shard_root)
    if not shards:
        sys.exit(f"no shard_NN/ dirs under {args.shard_root}")

    H_parts, y_parts, meta = [], [], []
    manifest0 = None
    for sd in shards:
        h = np.load(sd / f"{args.stem}_h.npy")
        y = np.load(sd / f"{args.stem}_y.npy")
        m = [json.loads(l) for l in (sd / f"{args.stem}_meta.jsonl").read_text().splitlines() if l.strip()]
        if not (h.shape[0] == y.shape[0] == len(m)):
            sys.exit(f"{sd.name}: row mismatch h={h.shape[0]} y={y.shape[0]} meta={len(m)}")
        H_parts.append(h); y_parts.append(y); meta.extend(m)
        if manifest0 is None:
            mp = sd / f"{args.stem}_manifest.json"
            if mp.exists():
                manifest0 = json.loads(mp.read_text())
        print(f"[merge] {sd.name}: {h.shape}")

    H = np.concatenate(H_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    order = np.argsort([mm["global_index"] for mm in meta], kind="stable")
    H = H[order]; y = y[order]; meta = [meta[i] for i in order]

    gi = [mm["global_index"] for mm in meta]
    if len(set(gi)) != len(gi):
        sys.exit(f"[merge] duplicate global_index across shards (n={len(gi)}, unique={len(set(gi))})")

    np.save(args.out_dir / f"{args.stem}_h.npy", H)
    np.save(args.out_dir / f"{args.stem}_y.npy", y)
    with (args.out_dir / f"{args.stem}_meta.jsonl").open("w") as f:
        for mm in meta:
            f.write(json.dumps(mm, ensure_ascii=False) + "\n")
    if manifest0 is not None:
        manifest0.update({"merged": True, "n_steps": int(H.shape[0]),
                          "shape": list(H.shape), "num_shards_merged": len(shards)})
        (args.out_dir / f"{args.stem}_manifest.json").write_text(json.dumps(manifest0, indent=2))

    print(f"[merge] wrote {args.out_dir}/{args.stem}_h.npy shape={H.shape} "
          f"(y={y.shape}, meta={len(meta)})")


if __name__ == "__main__":
    main()
