"""
Encode multiple ProcessBench subsets with Qwen2.5-1.5B and assemble a 'combined'
view by concatenating per-subset hidden states + meta in a stable order.

Per-subset layout (under --out_root):
    <subset>/pb_step_h.npy
    <subset>/pb_step_meta.jsonl       (with 'pb_subset' field on every row)
    <subset>/encoding_manifest_pb.json
    combined/pb_step_h.npy
    combined/pb_step_meta.jsonl
    combined/encoding_manifest_pb.json

This wrapper shells out to encode_processbench_hidden_states.py for each
subset so the encoding logic stays single-sourced.

Discovery: pass --subsets explicitly (e.g. 'gsm8k:/path/to/gsm8k.jsonl
math:/path/to/math.jsonl'). The script does not download data and does not
guess paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Encode multiple ProcessBench subsets and combine.")
    p.add_argument("--subsets", nargs="+", required=True,
                   help="List of 'name:input_path' entries (e.g. gsm8k:/p/gsm8k.jsonl).")
    p.add_argument("--out_root", type=Path, required=True,
                   help="Root directory under which <subset>/ and combined/ live.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--encoder_script", type=Path,
                   default=Path("scripts/encode_processbench_hidden_states.py"))
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    parsed: list[tuple[str, Path]] = []
    for s in args.subsets:
        if ":" not in s:
            sys.exit(f"--subsets entry must be name:path, got {s!r}")
        name, path = s.split(":", 1)
        parsed.append((name, Path(path)))

    missing = [p for _, p in parsed if not p.exists()]
    if missing:
        sys.exit(f"Missing ProcessBench files: {missing}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    per_subset_info: list[dict] = []

    for name, src in parsed:
        subset_dir = args.out_root / name
        subset_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python, str(args.encoder_script),
            "--raw_file", str(src),
            "--out_dir", str(subset_dir),
            "--model_name_or_path", args.model_name_or_path,
            "--run_name", f"{args.run_name}_{name}",
            "--max_seq_len", str(args.max_seq_len),
            "--batch_size", str(args.batch_size),
            "--model_dtype", args.model_dtype,
            "--save_dtype", args.save_dtype,
            "--subset_name", name,
            "--output_layout", "generic",
        ]
        if args.local_files_only:
            cmd.append("--local_files_only")
        if args.force:
            cmd.append("--force")
        print(f"[multi_pb] >>> {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            sys.exit(f"[multi_pb] Subset {name} failed (exit {r.returncode}).")

        h_path = subset_dir / "pb_step_h.npy"
        m_path = subset_dir / "pb_step_meta.jsonl"
        per_subset_info.append({
            "name": name,
            "src": str(src),
            "h_path": str(h_path),
            "meta_path": str(m_path),
            "sha256_hidden": sha256_file(h_path),
        })

    # ---- Build combined ---------------------------------------------------
    combined_dir = args.out_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_h_path = combined_dir / "pb_step_h.npy"
    combined_m_path = combined_dir / "pb_step_meta.jsonl"
    if combined_h_path.exists() and not args.force:
        sys.exit(f"[multi_pb] Refusing to overwrite {combined_h_path}. Pass --force.")

    hs: list[np.ndarray] = []
    metas: list[dict] = []
    for info in per_subset_info:
        h = np.load(info["h_path"])
        hs.append(h)
        meta = [
            json.loads(line)
            for line in Path(info["meta_path"]).read_text().splitlines()
            if line.strip()
        ]
        for row in meta:
            row["pb_subset"] = row.get("pb_subset", info["name"])
            # Rewrite id to be globally unique across subsets in combined view.
            row["id"] = f"{info['name']}::{row['id']}"
        metas.extend(meta)
        if h.shape[0] != len(meta):
            sys.exit(
                f"[multi_pb] {info['name']}: h rows ({h.shape[0]}) != "
                f"meta rows ({len(meta)})"
            )

    combined_h = np.concatenate(hs, axis=0) if hs else np.zeros((0,))
    np.save(combined_h_path, combined_h)
    with combined_m_path.open("w") as f:
        for row in metas:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    combined_manifest = {
        "run_name": args.run_name,
        "subsets": per_subset_info,
        "combined": {
            "hidden_path": str(combined_h_path),
            "meta_path": str(combined_m_path),
            "n_rows": int(combined_h.shape[0]),
            "sha256_hidden": sha256_file(combined_h_path),
        },
        "id_namespacing": "combined ids are prefixed '<subset>::<id>' to stay unique",
        "total_wall_sec": time.perf_counter() - t_all,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (combined_dir / "encoding_manifest_pb.json").write_text(
        json.dumps(combined_manifest, indent=2)
    )
    print(
        f"[multi_pb] Done. combined h shape={combined_h.shape}; "
        f"manifest at {combined_dir / 'encoding_manifest_pb.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
