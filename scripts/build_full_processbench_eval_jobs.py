"""Build the full ProcessBench evaluation job list.

For every (method, subset) pair this writes a single JSON object describing
exactly which artifacts the worker must read and where the worker should
emit per-job outputs. The orchestrator (Slurm script) splits this list
round-robin across N workers and feeds each slice to
``evaluate_existing_probes_full_processbench_worker.py``.

Job record schema (matches what the worker consumes; one per line in the
output JSONL, plus a sharded JSON file per worker)::

    {
      "method": "dense_linear",
      "representation_type": "dense",
      "pb_subset": "gsm8k",
      "pb_latents": ".../qwen2_5_1_5b_processbench_full/gsm8k/pb_step_h.npy",
      "pb_meta":    ".../qwen2_5_1_5b_processbench_full/gsm8k/pb_step_meta.jsonl",
      "probe":      ".../runs/dense_linear/linear_probe.pt"   (or null for random),
      "threshold_json": ".../runs/dense_linear/threshold.json",
      "sae_repr":   null                                       (set for sae_*),
      "is_random":  false,
      "out_dir":    ".../full_processbench_eval/per_job/dense_linear/gsm8k"
    }

This file is intentionally tiny and side-effect-free so the Slurm script
can run it on the login node before allocating GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METHOD_SPECS: dict[str, tuple[str, bool, bool]] = {
    "dense_linear":                       ("dense",  False, False),
    "random":                             ("dense",  False, False),
    "sae_positive":                       ("sae",    True,  False),
    "sae_mixed":                          ("sae",    True,  False),
    "sae_contrastive":                    ("sae",    True,  False),
    "ssae_positive":                      ("ssae",   False, True),
    "ssae_mixed":                         ("ssae",   False, True),
    "ssae_contrastive":                   ("ssae",   False, True),
    "ssae_contrastive_auxlr1e-3_full":    ("ssae",   False, True),
}


def find_subsets(dense_pb_root: Path) -> list[str]:
    out: list[str] = []
    for c in sorted(dense_pb_root.iterdir()):
        if c.is_dir() and (c / "pb_step_h.npy").exists() \
                and (c / "pb_step_meta.jsonl").exists():
            out.append(c.name)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--dense_pb_cache_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Top-level out_dir (full_processbench_eval/). "
                        "Per-job results land under per_job/<method>/<subset>/.")
    p.add_argument("--methods", nargs="+", default=list(METHOD_SPECS.keys()))
    p.add_argument("--include_combined", action="store_true",
                   help="Include the pooled 'combined' subset.")
    p.add_argument("--skip_missing", action="store_true",
                   help="Silently drop (method,subset) pairs whose artifacts "
                        "are missing instead of recording them.")
    p.add_argument("--out_jobs_jsonl", type=Path, required=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shard_prefix", type=Path, required=True,
                   help="Prefix for per-worker JSON files; suffixes "
                        "_worker_<i>.json are appended.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subsets = find_subsets(args.dense_pb_cache_root)
    if not args.include_combined:
        subsets = [s for s in subsets if s != "combined"]
    if not subsets:
        sys.exit(f"[build_jobs] no subsets under {args.dense_pb_cache_root}")
    print(f"[build_jobs] subsets: {subsets}")
    print(f"[build_jobs] methods: {args.methods}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.shard_prefix.parent.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    for method in args.methods:
        if method not in METHOD_SPECS:
            print(f"[build_jobs] WARNING: unknown method {method}; skip")
            continue
        family, needs_sae, run_local = METHOD_SPECS[method]
        run_dir = args.runs_root / method
        if not run_dir.is_dir():
            msg = f"[build_jobs] missing run dir: {run_dir}"
            if args.skip_missing:
                print(msg + " -> skip")
                continue
            sys.exit(msg)

        probe = run_dir / "linear_probe.pt" if method != "random" else None
        thr = run_dir / "threshold.json"
        sae_repr = run_dir / "representation.pt" if family == "sae" else None
        for subset in subsets:
            if family == "ssae":
                lat = run_dir / "latents_full_pb" / subset / "pb_step_z.npy"
                meta = run_dir / "latents_full_pb" / subset / "pb_step_meta.jsonl"
            else:
                lat = args.dense_pb_cache_root / subset / "pb_step_h.npy"
                meta = args.dense_pb_cache_root / subset / "pb_step_meta.jsonl"
            missing = []
            if not lat.exists(): missing.append(str(lat))
            if not meta.exists(): missing.append(str(meta))
            if method != "random" and probe is not None and not probe.exists():
                missing.append(str(probe))
            if family == "sae" and sae_repr is not None and not sae_repr.exists():
                missing.append(str(sae_repr))
            if missing:
                msg = f"[build_jobs] {method}/{subset} missing: {missing}"
                if args.skip_missing:
                    print(msg + " -> skip")
                    continue
                sys.exit(msg)

            job_out = args.out_dir / "per_job" / method / subset
            jobs.append({
                "method": method,
                "representation_type": family,
                "pb_subset": subset,
                "pb_latents": str(lat),
                "pb_meta": str(meta),
                "probe": str(probe) if (probe and method != "random") else None,
                "threshold_json": str(thr) if thr.exists() else None,
                "sae_repr": str(sae_repr) if sae_repr else None,
                "is_random": method == "random",
                "out_dir": str(job_out),
            })

    # Write the master list
    with args.out_jobs_jsonl.open("w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"[build_jobs] wrote {len(jobs)} jobs -> {args.out_jobs_jsonl}")

    # Round-robin split into N worker files
    shards: list[list[dict]] = [[] for _ in range(args.num_workers)]
    for i, j in enumerate(jobs):
        shards[i % args.num_workers].append(j)
    for w, lst in enumerate(shards):
        path = Path(str(args.shard_prefix) + f"_worker_{w}.json")
        path.write_text(json.dumps(lst, indent=2))
        print(f"[build_jobs] worker {w}: {len(lst)} jobs -> {path}")


if __name__ == "__main__":
    main()
