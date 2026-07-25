"""
Build full-scale PRM800K probe-training and validation JSONL files.

Reuses raw loading, candidate extraction, length filtering, and disjoint
sampling from build_prm800k_prestudy.py to guarantee identical preprocessing
semantics. This script does NOT touch the 40k prestudy artifacts; it writes
new files with templated stems (e.g., prm800k_probe_train_400k.jsonl).

Differences vs build_prm800k_prestudy.py:
- Sizes are configurable and not bound to the 20k+20k / 500+500 baseline.
- A 'full' mode emits all available pos/neg after filtering, capped by
  --val_pos / --val_neg.
- No contrastive forks are produced.
- Output filenames embed the train/val sizes so multiple builds coexist.

Schema matches prm800k_probe_train_40k.jsonl exactly (see prestudy builder).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse prestudy builder pieces so semantics stay identical.
from build_prm800k_prestudy import (  # type: ignore  (sibling module)
    build_candidates,
    filter_by_length,
    git_commit,
    load_raw_prm800k,
    sample_disjoint_train_val,
    sha256_file,
    write_jsonl,
)


def fmt_count(n: int) -> str:
    """Human-friendly numeric suffix for filenames: 400000 -> '400k', 1500000 -> '1500k'."""
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def carve_disjoint_test(
    pos_train: list[dict],
    neg_train: list[dict],
    n_pos_test: int,
    n_neg_test: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Carve a problem-disjoint balanced test split out of the train pools.

    Returns (pos_train_remaining, neg_train_remaining, pos_test, neg_test). The
    test problem_ids are held out of the returned train pools, so the test split
    is disjoint from train and (transitively, since the inputs already are) from
    val. Never mutates the input lists. Mirrors the greedy per-problem_id
    assignment used by sample_disjoint_train_val.
    """
    if n_pos_test <= 0 and n_neg_test <= 0:
        return list(pos_train), list(neg_train), [], []

    pos_by_pid: dict[str, list[dict]] = defaultdict(list)
    neg_by_pid: dict[str, list[dict]] = defaultdict(list)
    for ex in pos_train:
        pos_by_pid[ex["problem_id"]].append(ex)
    for ex in neg_train:
        neg_by_pid[ex["problem_id"]].append(ex)

    all_pids = list(set(pos_by_pid) | set(neg_by_pid))
    rng.shuffle(all_pids)

    test_pids: set[str] = set()
    tp = tn = 0
    for pid in all_pids:
        if tp >= n_pos_test and tn >= n_neg_test:
            break
        test_pids.add(pid)
        tp += len(pos_by_pid.get(pid, []))
        tn += len(neg_by_pid.get(pid, []))

    pos_test_pool = [e for pid in test_pids for e in pos_by_pid.get(pid, [])]
    neg_test_pool = [e for pid in test_pids for e in neg_by_pid.get(pid, [])]
    if len(pos_test_pool) < n_pos_test or len(neg_test_pool) < n_neg_test:
        sys.exit(
            f"Cannot carve disjoint test split (pos pool={len(pos_test_pool)} "
            f"need {n_pos_test}, neg pool={len(neg_test_pool)} need {n_neg_test})."
        )
    rng.shuffle(pos_test_pool)
    rng.shuffle(neg_test_pool)

    pos_train_rem = [
        e for pid, exs in pos_by_pid.items() if pid not in test_pids for e in exs
    ]
    neg_train_rem = [
        e for pid, exs in neg_by_pid.items() if pid not in test_pids for e in exs
    ]
    return (
        pos_train_rem,
        neg_train_rem,
        pos_test_pool[:n_pos_test],
        neg_test_pool[:n_neg_test],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build full-scale PRM800K probe-train/val JSONL splits.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--raw_dir", type=Path)
    grp.add_argument("--raw_file", type=Path)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--tokenizer_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument(
        "--train_sizes", type=int, nargs="+", default=[200000, 400000],
        help="Total train sizes (pos+neg). Each is split 50/50. Use a single "
             "value with --full to materialize all available candidates "
             "(the value is then ignored for sizing but used for the filename).",
    )
    p.add_argument(
        "--full", action="store_true",
        help="In addition to --train_sizes, emit an additional split named "
             "'prm800k_probe_train_full' containing all balanced "
             "(pos, neg) pairs after filtering.",
    )
    p.add_argument(
        "--full_safety_margin", type=int, default=1000,
        help="In --full mode, reserve this many fewer pairs per class than the "
             "theoretical max so the strict problem-disjoint val (and test) "
             "assignment has slack. Negligible loss at ~260k/class.",
    )
    p.add_argument("--val_pos", type=int, default=2500,
                   help="Positive count for validation split.")
    p.add_argument("--val_neg", type=int, default=2500,
                   help="Negative count for validation split.")
    p.add_argument("--val_name", type=str, default=None,
                   help="Override val filename stem (e.g., 'val_10k'). "
                        "Default is derived from --val_pos+--val_neg.")
    p.add_argument("--test_pos", type=int, default=0,
                   help="Positive count for a small held-out in-domain test split "
                        "(0 disables). Carved problem-disjoint from train and val.")
    p.add_argument("--test_neg", type=int, default=0,
                   help="Negative count for the in-domain test split.")
    p.add_argument("--test_name", type=str, default=None,
                   help="Override test filename stem (e.g., 'test_2k'). "
                        "Default is derived from --test_pos+--test_neg.")
    p.add_argument("--allow_problem_overlap", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output JSONLs. Default refuses.")
    args = p.parse_args()

    if args.raw_dir is None and args.raw_file is None:
        p.error("Provide --raw_dir or --raw_file.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"[build_full] Loading tokenizer from {args.tokenizer_name_or_path}", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            local_files_only=args.local_files_only,
        )
    except OSError:
        sys.exit("Tokenizer not found locally. Pre-cache Qwen/Qwen2.5-1.5B.")

    rng = random.Random(args.seed)

    print("[build_full] Loading raw PRM800K", flush=True)
    samples = load_raw_prm800k(args.raw_dir, args.raw_file)
    print(f"[build_full] Loaded {len(samples)} raw samples.", flush=True)

    counters: dict[str, Any] = defaultdict(int)
    counters.update({
        "malformed_samples": 0,
        "invalid_prefix_steps": 0,
        "candidate_total_seen": 0,
        "candidate_rating_1": 0,
        "candidate_rating_minus_1": 0,
        "candidate_rating_0": 0,
        "candidate_flagged": 0,
        "candidate_empty_text": 0,
        "candidate_missing_or_invalid_rating": 0,
        "candidate_overlength": 0,
        "valid_forks_found": 0,
    })

    print("[build_full] Constructing candidate examples", flush=True)
    all_examples, _fork_map = build_candidates(samples, counters)
    print(
        f"[build_full] Raw candidates: {len(all_examples)} "
        f"(pos={counters['candidate_rating_1']}, neg={counters['candidate_rating_minus_1']})",
        flush=True,
    )

    print("[build_full] Length filter", flush=True)
    all_examples = filter_by_length(all_examples, tokenizer, args.max_seq_len, counters)
    print(
        f"[build_full] After length filter: {len(all_examples)} "
        f"({counters['candidate_overlength']} discarded as overlength)",
        flush=True,
    )

    pos_examples = [e for e in all_examples if e["label"] == 0]
    neg_examples = [e for e in all_examples if e["label"] == 1]
    print(f"[build_full] Available pos={len(pos_examples)} neg={len(neg_examples)}", flush=True)

    # Reserve the validation split first using the full disjointness mechanism.
    if len(pos_examples) < args.val_pos or len(neg_examples) < args.val_neg:
        sys.exit(
            f"Not enough examples for val split: have pos={len(pos_examples)} "
            f"neg={len(neg_examples)}, need val_pos={args.val_pos} val_neg={args.val_neg}"
        )

    # Find the largest train size requested for the disjoint reservation.
    largest_train = max(args.train_sizes)
    largest_pos = largest_train // 2
    largest_neg = largest_train - largest_pos

    # If 'full' mode requested, expand to all available minus the val reservation.
    full_pos = len(pos_examples) - args.val_pos
    full_neg = len(neg_examples) - args.val_neg
    if args.full:
        balanced_full = min(full_pos, full_neg)
        # Strict problem-disjoint val assignment overshoots its exact quota by up
        # to one problem's worth of candidates, so reserving the full theoretical
        # balance leaves no slack and the reservation fails by a few examples.
        # Back off a small safety margin; the loss is negligible at ~260k/class.
        balanced_full = max(0, balanced_full - args.full_safety_margin)
        # The full split will consume all balanced pairs available beyond val.
        largest_pos = max(largest_pos, balanced_full)
        largest_neg = max(largest_neg, balanced_full)

    print(
        f"[build_full] Reserving disjoint val (pos={args.val_pos}, neg={args.val_neg}) "
        f"and train pool (pos={largest_pos}, neg={largest_neg})",
        flush=True,
    )
    pos_train_all, neg_train_all, pos_val, neg_val = sample_disjoint_train_val(
        pos_examples, neg_examples,
        largest_pos, largest_neg,
        args.val_pos, args.val_neg,
        rng, args.allow_problem_overlap,
    )

    # Carve a small balanced in-domain test split out of the train pool, kept
    # problem-disjoint from both train and val. Done before emitting train so the
    # size-based and --full train splits exclude the test problems.
    pos_test: list[dict] = []
    neg_test: list[dict] = []
    if args.test_pos > 0 or args.test_neg > 0:
        pos_train_all, neg_train_all, pos_test, neg_test = carve_disjoint_test(
            pos_train_all, neg_train_all, args.test_pos, args.test_neg, rng
        )
        print(
            f"[build_full] Carved in-domain test (pos={len(pos_test)}, "
            f"neg={len(neg_test)}); train pool now pos={len(pos_train_all)}, "
            f"neg={len(neg_train_all)}",
            flush=True,
        )

    combined_val = pos_val + neg_val
    rng.shuffle(combined_val)

    val_total = args.val_pos + args.val_neg
    val_stem = args.val_name or f"val_{fmt_count(val_total)}"
    val_jsonl_name = f"prm800k_{val_stem}.jsonl"

    # Write the requested train sizes (each is a prefix of the shuffled pos/neg pools).
    files_meta: dict[str, dict] = {}

    def emit(name: str, rows: list[dict]) -> None:
        path = args.out_dir / name
        if path.exists() and not args.force:
            sys.exit(f"Refusing to overwrite existing {path}. Pass --force.")
        write_jsonl(path, rows)
        files_meta[name] = {
            "path": str(path),
            "n_rows": len(rows),
            "n_pos": sum(1 for r in rows if r["label"] == 0),
            "n_neg": sum(1 for r in rows if r["label"] == 1),
            "sha256": sha256_file(path),
        }
        print(f"[build_full] wrote {name}: {len(rows)} rows", flush=True)

    emit(val_jsonl_name, combined_val)

    test_stem = None
    if pos_test or neg_test:
        combined_test = pos_test + neg_test
        rng.shuffle(combined_test)
        test_total = args.test_pos + args.test_neg
        test_stem = args.test_name or f"test_{fmt_count(test_total)}"
        emit(f"prm800k_{test_stem}.jsonl", combined_test)

    train_sizes_to_emit = list(args.train_sizes)
    for total in train_sizes_to_emit:
        n_pos = total // 2
        n_neg = total - n_pos
        if n_pos > len(pos_train_all) or n_neg > len(neg_train_all):
            sys.exit(
                f"Requested train_size={total} exceeds reserved pool "
                f"(pos={len(pos_train_all)}, neg={len(neg_train_all)})."
            )
        rows = pos_train_all[:n_pos] + neg_train_all[:n_neg]
        rng.shuffle(rows)
        emit(f"prm800k_probe_train_{fmt_count(total)}.jsonl", rows)

    if args.full:
        balanced_full = min(len(pos_train_all), len(neg_train_all))
        rows = pos_train_all[:balanced_full] + neg_train_all[:balanced_full]
        rng.shuffle(rows)
        emit("prm800k_probe_train_full.jsonl", rows)

    # Overlap checks
    val_uids = {r["uid"] for r in combined_val}
    train_uids = {r["uid"] for r in pos_train_all + neg_train_all}
    val_pids = {r["problem_id"] for r in combined_val}
    train_pids = {r["problem_id"] for r in pos_train_all + neg_train_all}

    overlap = {
        "train_val_uid_overlap": len(train_uids & val_uids),
        "train_val_problem_id_overlap": len(train_pids & val_pids),
    }
    if overlap["train_val_uid_overlap"] != 0:
        sys.exit(f"BUG: train/val UID overlap = {overlap['train_val_uid_overlap']}")
    if not args.allow_problem_overlap and overlap["train_val_problem_id_overlap"] != 0:
        sys.exit(f"BUG: train/val problem_id overlap = {overlap['train_val_problem_id_overlap']}")

    if pos_test or neg_test:
        combined_test = pos_test + neg_test
        test_uids = {r["uid"] for r in combined_test}
        test_pids = {r["problem_id"] for r in combined_test}
        overlap["test_train_uid_overlap"] = len(test_uids & train_uids)
        overlap["test_val_uid_overlap"] = len(test_uids & val_uids)
        overlap["test_train_problem_id_overlap"] = len(test_pids & train_pids)
        overlap["test_val_problem_id_overlap"] = len(test_pids & val_pids)
        if overlap["test_train_uid_overlap"] or overlap["test_val_uid_overlap"]:
            sys.exit(f"BUG: test UID overlaps train/val: {overlap}")
        if not args.allow_problem_overlap and (
            overlap["test_train_problem_id_overlap"]
            or overlap["test_val_problem_id_overlap"]
        ):
            sys.exit(f"BUG: test problem_id overlaps train/val: {overlap}")

    manifest = {
        "run_name": args.run_name,
        "seed": args.seed,
        "source_paths": {
            "raw_dir": str(args.raw_dir) if args.raw_dir else None,
            "raw_file": str(args.raw_file) if args.raw_file else None,
        },
        "label_mapping": {"rating_1": 0, "rating_minus_1": 1},
        "discarded_ratings": [0],
        "discarded_conditions": [
            "flagged=true", "empty_text", "missing_rating",
            "rating_not_in_{-1,1}", "tokenized_length_exceeds_max_seq_len",
        ],
        "length_policy": "no truncation",
        "max_seq_len": args.max_seq_len,
        "prefix_policy": "previous human_completion if available else chosen_completion",
        "split_policy": (
            "problem_id disjoint train/val"
            if not args.allow_problem_overlap
            else "problem_id overlap allowed (--allow_problem_overlap)"
        ),
        "available_after_filter": {
            "pos": len(pos_examples),
            "neg": len(neg_examples),
        },
        "reserved_pool": {
            "pos_train": len(pos_train_all),
            "neg_train": len(neg_train_all),
            "pos_val": len(pos_val),
            "neg_val": len(neg_val),
        },
        "filtering_counts": {k: int(v) for k, v in counters.items() if k.startswith(("candidate_", "invalid_", "malformed_"))},
        "overlap_checks": overlap,
        "files": files_meta,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }

    manifest_path = args.out_dir / "build_full_manifest.json"
    if manifest_path.exists() and not args.force:
        sys.exit(f"Refusing to overwrite existing {manifest_path}. Pass --force.")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[build_full] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
