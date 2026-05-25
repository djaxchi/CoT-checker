"""Encode every available ProcessBench subset into dense Qwen2.5-1.5B hidden states.

This is a thin wrapper that:

* discovers ProcessBench subset files under --pb_root (or accepts explicit
  --pb_files name:path entries),
* delegates the per-subset encoding to ``encode_processbench_multi.py`` so
  the encoder logic stays single-sourced.

Output layout under --out_root::

    <subset>/pb_step_h.npy
    <subset>/pb_step_meta.jsonl       (with pb_subset on every row)
    <subset>/encoding_manifest_pb.json
    combined/pb_step_h.npy
    combined/pb_step_meta.jsonl       (ids prefixed "<subset>::")
    combined/encoding_manifest_pb.json

By default refuses to overwrite an existing combined output unless FORCE=1
(env var) or --force is passed -- ProcessBench encoding is multi-GPU and
not cheap.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SUBSET_RE = re.compile(r"^processbench[_-](?P<name>[A-Za-z0-9._-]+)\.(?:jsonl|json)$",
                       re.IGNORECASE)


def infer_subset_name(path: Path) -> str | None:
    m = SUBSET_RE.match(path.name)
    if m:
        return m.group("name").lower()
    return None


def discover(pb_root: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for p in sorted(list(pb_root.glob("*.jsonl")) + list(pb_root.glob("*.json"))):
        name = infer_subset_name(p)
        if not name:
            continue
        existing = found.get(name)
        if existing is None or (existing.suffix == ".json" and p.suffix == ".jsonl"):
            found[name] = p
    return [(n, found[n]) for n in sorted(found.keys())]


def parse_pb_files(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for s in items:
        if ":" not in s:
            sys.exit(f"--pb_files entry must be name:path, got {s!r}")
        n, p = s.split(":", 1)
        out.append((n.strip().lower(), Path(p)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pb_root", type=Path,
                     help="Directory holding processbench_<subset>.jsonl files.")
    grp.add_argument("--pb_files", nargs="+",
                     help="Explicit 'name:path' entries (e.g. gsm8k:/p/x.jsonl).")

    p.add_argument("--out_root", type=Path, required=True,
                   help="Root cache dir; per-subset + combined layouts written under it.")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", default="processbench_full")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--force", action="store_true",
                   help="Allow overwriting existing per-subset and combined files.")
    p.add_argument("--multi_script", type=Path,
                   default=Path("scripts/encode_processbench_multi.py"))
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.pb_root is not None:
        pairs = discover(args.pb_root)
        if not pairs:
            sys.exit(f"[encode_pb_all] No processbench_*.jsonl|json in {args.pb_root}")
    else:
        pairs = parse_pb_files(args.pb_files)
        missing = [p for _, p in pairs if not p.exists()]
        if missing:
            sys.exit(f"[encode_pb_all] Missing PB files: {missing}")

    print(f"[encode_pb_all] Found {len(pairs)} subsets:")
    for n, p in pairs:
        print(f"    {n:<14} {p}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    force_env = os.environ.get("FORCE", "0") == "1"
    force = args.force or force_env

    # Refuse to silently overwrite existing per-subset outputs.
    if not force:
        for name, _ in pairs:
            sub_h = args.out_root / name / "pb_step_h.npy"
            if sub_h.exists():
                sys.exit(
                    f"[encode_pb_all] Refusing to overwrite {sub_h}. "
                    "Set FORCE=1 or pass --force to re-encode."
                )
        combined_h = args.out_root / "combined" / "pb_step_h.npy"
        if combined_h.exists():
            sys.exit(
                f"[encode_pb_all] Refusing to overwrite {combined_h}. "
                "Set FORCE=1 or pass --force to rebuild combined view."
            )

    subset_args = [f"{n}:{p}" for n, p in pairs]
    cmd = [
        args.python, str(args.multi_script),
        "--subsets", *subset_args,
        "--out_root", str(args.out_root),
        "--model_name_or_path", args.model_name_or_path,
        "--run_name", args.run_name,
        "--max_seq_len", str(args.max_seq_len),
        "--batch_size", str(args.batch_size),
        "--model_dtype", args.model_dtype,
        "--save_dtype", args.save_dtype,
    ]
    if args.local_files_only:
        cmd.append("--local_files_only")
    if force:
        cmd.append("--force")

    print(f"[encode_pb_all] >>> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
