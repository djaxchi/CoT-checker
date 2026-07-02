"""S4 contrib-cluster stage 1: PRM800K continuity audit + trajectory sampling.

Reconstructs golden-path reasoning trajectories (question + ordered selected
steps) from raw PRM800K, audits continuity, then samples N complete
trajectories with a fixed seed. Correctness ratings are never read.

Outputs (in --out_dir, default runs/contrib_cluster/):
  audit.json          continuity audit counters (raw / usable / dropped, step stats)
  audit.md            human-readable audit
  trajectories.jsonl  the sampled trajectories (materialized split; extraction
                      and all later stages read only this file)

Usage (TamIA):
  python scripts/s4_contrib_audit.py \
    --raw_dir $SCRATCH/cot_mech/raw/prm800k \
    --out_dir runs/contrib_cluster \
    --n_trajectories 3000 --max_steps 10 \
    --tokenizer_name_or_path Qwen/Qwen2.5-7B --local_files_only \
    --max_seq_len 4096
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prm800k_prestudy import git_commit, load_raw_prm800k, write_jsonl  # noqa: E402

from src.analysis.contrib_cluster import build_prefixes, fit_steps_to_length  # noqa: E402
from src.data.prm800k_trajectories import audit_trajectories  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--raw_dir", type=Path)
    grp.add_argument("--raw_file", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--n_trajectories", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=10,
                    help="cap steps per trajectory (truncate, keep the first K)")
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--tokenizer_name_or_path", type=str, default=None,
                    help="if given, trajectories whose longest prefix p_T exceeds "
                         "--max_seq_len tokens get trailing steps dropped until "
                         "they fit (dropped entirely only if even --min_steps "
                         "steps do not fit)")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_traj = args.out_dir / "trajectories.jsonl"
    if out_traj.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_traj}; pass --force")

    print("[audit] loading raw PRM800K ...", flush=True)
    samples = load_raw_prm800k(args.raw_dir, args.raw_file)
    print(f"[audit] {len(samples)} raw examples", flush=True)

    trajectories, audit = audit_trajectories(samples)
    print(f"[audit] {audit['n_usable_trajectories']} usable trajectories "
          f"({audit['n_usable_steps']} steps) | dropped: {audit['dropped']}", flush=True)

    # ---- deterministic sampling of complete trajectories -----------------
    rng = random.Random(args.seed)
    order = list(range(len(trajectories)))
    rng.shuffle(order)

    tokenizer = None
    if args.tokenizer_name_or_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, local_files_only=args.local_files_only)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    kept: list[dict] = []
    n_dropped_overlength = 0
    n_truncated_overlength = 0
    n_truncated_to_cap = 0
    for idx in order:
        if len(kept) >= args.n_trajectories:
            break
        traj = dict(trajectories[idx])
        if len(traj["steps"]) < args.min_steps:
            continue
        if len(traj["steps"]) > args.max_steps:
            traj["steps"] = traj["steps"][: args.max_steps]
            n_truncated_to_cap += 1
        if tokenizer is not None:
            fitted = fit_steps_to_length(count_tokens, traj["question"],
                                         traj["steps"], args.max_seq_len,
                                         args.min_steps)
            if fitted is None:
                n_dropped_overlength += 1
                continue
            if len(fitted) < len(traj["steps"]):
                n_truncated_overlength += 1
                traj["steps"] = fitted
            traj["max_prefix_tokens"] = count_tokens(
                build_prefixes(traj["question"], traj["steps"])[-1])
        traj["num_steps"] = len(traj["steps"])
        kept.append(traj)

    if len(kept) < args.n_trajectories:
        print(f"[audit] WARNING: only {len(kept)} trajectories available "
              f"(requested {args.n_trajectories})", flush=True)

    steps_kept = [t["num_steps"] for t in kept]
    sample_stats = {
        "n_requested": args.n_trajectories,
        "n_sampled": len(kept),
        "n_sampled_steps": int(sum(steps_kept)),
        "n_truncated_to_max_steps": n_truncated_to_cap,
        "n_truncated_overlength": n_truncated_overlength,
        "n_dropped_overlength": n_dropped_overlength,
        "max_steps_cap": args.max_steps,
        "min_steps": args.min_steps,
        "max_seq_len": args.max_seq_len if tokenizer is not None else None,
        "steps_per_sampled_trajectory": {
            "mean": float(sum(steps_kept) / len(steps_kept)) if steps_kept else 0.0,
            "median": float(sorted(steps_kept)[len(steps_kept) // 2]) if steps_kept else 0.0,
            "max": int(max(steps_kept)) if steps_kept else 0,
        },
    }
    audit_out = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": args.seed,
        "raw_source": str(args.raw_dir or args.raw_file),
        "continuity_audit": audit,
        "sampling": sample_stats,
    }

    write_jsonl(out_traj, kept)
    (args.out_dir / "audit.json").write_text(json.dumps(audit_out, indent=2))

    d = audit["dropped"]
    md = [
        "# S4 contrib-cluster: PRM800K continuity audit",
        "",
        f"- created: {audit_out['created']}  |  git: {audit_out['git_commit'][:10]}  |  seed: {args.seed}",
        f"- raw source: `{audit_out['raw_source']}`",
        "",
        "## Continuity audit (all raw examples)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| raw examples | {audit['n_raw_examples']} |",
        f"| usable trajectories (>=2 golden steps) | {audit['n_usable_trajectories']} |",
        f"| usable steps | {audit['n_usable_steps']} |",
        f"| steps/trajectory mean | {audit['steps_per_trajectory']['mean']:.2f} |",
        f"| steps/trajectory median | {audit['steps_per_trajectory']['median']:.0f} |",
        f"| steps/trajectory max | {audit['steps_per_trajectory']['max']} |",
        "",
        "## Dropped examples",
        "",
        "| reason | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in d.items()],
        "",
        "Note: `truncated_paths` counts sessions whose golden path ends early "
        "(no selected completion at some step); the steps before the break are "
        "kept when there are at least 2, so this overlaps with usable counts.",
        "",
        "## Sampled split (materialized in trajectories.jsonl)",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sample_stats.items() if not isinstance(v, dict)],
        f"| steps/traj mean (sampled) | {sample_stats['steps_per_sampled_trajectory']['mean']:.2f} |",
        f"| steps/traj median (sampled) | {sample_stats['steps_per_sampled_trajectory']['median']:.0f} |",
        f"| steps/traj max (sampled) | {sample_stats['steps_per_sampled_trajectory']['max']} |",
        "",
    ]
    (args.out_dir / "audit.md").write_text("\n".join(md))
    print(f"[audit] wrote {out_traj} ({len(kept)} trajectories, "
          f"{sum(steps_kept)} steps), audit.json, audit.md", flush=True)


if __name__ == "__main__":
    main()
