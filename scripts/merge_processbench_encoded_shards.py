"""Merge per-GPU ProcessBench encoding shards into a single deterministic file.

Inputs::

    --shard_root <dir>/shards          (contains shard_00/, shard_01/, ...)
    --out_h      <dir>/pb_step_h.npy   (or pb_step_z.npy for SSAE latents)
    --out_meta   <dir>/pb_step_meta.jsonl

Optional::

    --array_name pb_step_h.npy | pb_step_z.npy
                (default: auto-detect by scanning the first shard)
    --sort_by   global_step_index      (default; only field currently used)
    --out_y      <dir>/split_y.npy      (optional companion label array)
    --label_name split_y.npy            (required with --out_y unless it can be
                                         derived from --array_name)

Behavior:
  * Loads every <shard_root>/shard_*/<array_name> and matching meta.
  * Verifies row counts: hidden/latent rows == meta rows within each shard.
  * Sorts the concatenated rows by global_step_index ASC.
  * If global_step_index is absent in an older shard, assigns the deterministic
    fallback shard_idx * 10**12 + local_idx before sorting and writing meta.
  * Asserts no duplicate global_step_index (a duplicate would mean two
    shards encoded the same step -- silently keeping one would be wrong).
  * Writes the final hidden/latent array and meta JSONL.
  * Refuses to overwrite existing final outputs unless --force.

Score / metadata convention: every meta row keeps id, step_idx, label,
n_steps, global_step_index, pb_subset (when present). The sorted order
matches what a non-sharded run would have produced -- callers can use
the result as a drop-in replacement.
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
        if not child.is_dir():
            continue
        m = SHARD_DIR_RE.match(child.name)
        if not m:
            continue
        out.append((int(m.group(1)), child))
    out.sort(key=lambda x: x[0])
    return out


def autodetect_array_name(shard_dir: Path) -> str:
    for cand in ("pb_step_h.npy", "pb_step_z.npy"):
        if (shard_dir / cand).exists():
            return cand
    sys.exit(f"[merge_shards] cannot auto-detect array in {shard_dir}")


def load_meta(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard_root", type=Path, required=True)
    p.add_argument("--out_h", type=Path, required=True,
                   help="Final array output (works for dense h or SSAE z).")
    p.add_argument("--out_meta", type=Path, required=True)
    p.add_argument("--array_name", default=None,
                   help="pb_step_h.npy | pb_step_z.npy. Auto if omitted.")
    p.add_argument("--sort_by", default="global_step_index")
    p.add_argument("--meta_name", default="pb_step_meta.jsonl")
    p.add_argument("--out_y", type=Path, default=None,
                   help="Optional final companion label array output.")
    p.add_argument("--label_name", default=None,
                   help="Per-shard label array name. Defaults to array_name "
                        "with _z.npy/_h.npy replaced by _y.npy.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def infer_label_name(array_name: str) -> str:
    if array_name.endswith("_z.npy") or array_name.endswith("_h.npy"):
        return array_name[:-6] + "_y.npy"
    sys.exit(
        "[merge_shards] --out_y requires --label_name when --array_name does "
        "not end in _z.npy or _h.npy"
    )


def main() -> None:
    args = parse_args()
    shards = discover_shards(args.shard_root)
    if not shards:
        sys.exit(f"[merge_shards] no shard_NN/ dirs under {args.shard_root}")

    if args.out_h.exists() and not args.force:
        sys.exit(f"[merge_shards] refusing to overwrite {args.out_h}; --force")
    if args.out_meta.exists() and not args.force:
        sys.exit(f"[merge_shards] refusing to overwrite {args.out_meta}; --force")
    if args.out_y is not None and args.out_y.exists() and not args.force:
        sys.exit(f"[merge_shards] refusing to overwrite {args.out_y}; --force")

    array_name = args.array_name or autodetect_array_name(shards[0][1])
    label_name = None
    if args.out_y is not None:
        label_name = args.label_name or infer_label_name(array_name)
    print(f"[merge_shards] {len(shards)} shard(s); array={array_name} "
          f"sort_by={args.sort_by}")

    arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    metas: list[dict] = []
    per_shard_rows: list[int] = []
    seen_gi: set[int] = set()
    duplicates: list[int] = []
    fallback_rows = 0
    for idx, sdir in shards:
        a_path = sdir / array_name
        m_path = sdir / args.meta_name
        if not a_path.exists() or not m_path.exists():
            sys.exit(f"[merge_shards] shard {idx} missing files: {a_path} / {m_path}")
        a = np.load(a_path)
        m = load_meta(m_path)
        if a.shape[0] != len(m):
            sys.exit(
                f"[merge_shards] shard {idx}: array rows ({a.shape[0]}) "
                f"!= meta rows ({len(m)})"
            )
        if not np.all(np.isfinite(a)):
            sys.exit(f"[merge_shards] shard {idx}: NaN/Inf in {a_path}")
        if args.out_y is not None:
            y_path = sdir / label_name
            if not y_path.exists():
                sys.exit(f"[merge_shards] shard {idx} missing label file: {y_path}")
            y = np.load(y_path)
            if y.shape[0] != len(m):
                sys.exit(
                    f"[merge_shards] shard {idx}: label rows ({y.shape[0]}) "
                    f"!= meta rows ({len(m)})"
                )
            if not np.all(np.isfinite(y)):
                sys.exit(f"[merge_shards] shard {idx}: NaN/Inf in {y_path}")
            label_arrays.append(y)
        for local_idx, row in enumerate(m):
            gi = row.get(args.sort_by)
            if gi is None:
                gi = idx * 10**12 + local_idx
                row[args.sort_by] = gi
                fallback_rows += 1
            gi_int = int(gi)
            if gi_int in seen_gi:
                duplicates.append(gi_int)
            seen_gi.add(gi_int)
        arrays.append(a)
        metas.extend(m)
        per_shard_rows.append(a.shape[0])
        print(f"[merge_shards] shard {idx:02d}: rows={a.shape[0]} "
              f"path={a_path}")

    if duplicates:
        sys.exit(
            f"[merge_shards] FATAL: {len(duplicates)} duplicate "
            f"{args.sort_by} across shards (first 10: {duplicates[:10]}). "
            "Each step must be encoded by exactly one shard."
        )

    big = np.concatenate(arrays, axis=0)
    if big.shape[0] != len(metas):
        sys.exit(f"[merge_shards] concat mismatch: array={big.shape[0]} "
                 f"meta={len(metas)}")
    y_big = None
    if args.out_y is not None:
        y_big = np.concatenate(label_arrays, axis=0)
        if y_big.shape[0] != len(metas):
            sys.exit(f"[merge_shards] label concat mismatch: y={y_big.shape[0]} "
                     f"meta={len(metas)}")

    # Sort by global_step_index to reconstruct the original (non-sharded) order.
    order = sorted(range(len(metas)), key=lambda i: int(metas[i][args.sort_by]))
    big_sorted = big[order]
    y_sorted = y_big[order] if y_big is not None else None
    metas_sorted = [metas[i] for i in order]

    args.out_h.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_h, big_sorted)
    if args.out_y is not None:
        args.out_y.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_y, y_sorted)
    with args.out_meta.open("w") as f:
        for row in metas_sorted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[merge_shards] merged {big_sorted.shape[0]} rows "
          f"({per_shard_rows}) -> {args.out_h}")
    if fallback_rows:
        print(
            f"[merge_shards] filled missing {args.sort_by} on "
            f"{fallback_rows} row(s) with shard_idx*10**12 + local_idx"
        )
    if args.out_y is not None:
        print(f"[merge_shards] labels -> {args.out_y}")
    print(f"[merge_shards] meta -> {args.out_meta}")


if __name__ == "__main__":
    main()
