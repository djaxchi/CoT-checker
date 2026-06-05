"""Build a full-scale PRM800K fork dataset for Sprint 2 representation shaping.

This reuses the *exact* candidate-construction and length-filtering logic from
``build_prm800k_prestudy.py`` (same prefix policy, same label mapping, same
tokenization), but instead of extracting 20 debug forks it materializes a
large, problem-disjoint train/val split of forks together with the
positive/negative pairs that the ranking and triplet objectives consume.

Outputs (under --out_dir):
  forks_full.jsonl              all valid forks (audit), each with full
                                positive/negative sibling lists.
  forks_train_items.jsonl       encodable rows for the train split: one row per
  forks_val_items.jsonl         unique (anchor | positive | negative) item.
  forks_train_pairs.jsonl       {fork_id, anchor_uid, positive_uid,
  forks_val_pairs.jsonl         negative_uid} pairs under the chosen pair_mode.
  forks_manifest.json           counts, split sizes, sibling distribution, and
                                pair counts under BOTH one/all modes.

Item roles:
  anchor   -> candidate_step="" ; encoder embeds the reasoning prefix.
  positive -> a rating +1 continuation (label 0).
  negative -> a rating -1 continuation (label 1).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_prm800k_prestudy import (  # noqa: E402
    build_candidates,
    filter_by_length,
    git_commit,
    load_raw_prm800k,
    sha256_file,
    write_jsonl,
)
from src.repr.objectives import enumerate_fork_pairs  # noqa: E402


def anchor_uid(fork_id: str) -> str:
    return f"{fork_id}::anchor"


def item_uid(fork_id: str, role: str, completion_idx: int) -> str:
    return f"{fork_id}::{role}::{completion_idx}"


def valid_forks(fork_map: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    """Keep forks with >=1 positive (label 0) and >=1 negative (label 1)."""
    out: dict[str, dict[str, list[dict]]] = {}
    for key, exs in fork_map.items():
        pos = [e for e in exs if e["label"] == 0]
        neg = [e for e in exs if e["label"] == 1]
        if pos and neg:
            out[key] = {"pos": pos, "neg": neg}
    return out


def split_forks_by_problem(
    forks: dict[str, dict[str, list[dict]]],
    n_train: int,
    n_val: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Assign whole problem_ids to val or train so no problem leaks across the
    split, then cap each side. Extra forks are left held out (not returned)."""
    by_problem: dict[str, list[str]] = defaultdict(list)
    for key in forks:
        problem_id = key.split("::", 1)[0]
        by_problem[problem_id].append(key)

    problem_ids = list(by_problem.keys())
    rng.shuffle(problem_ids)

    val_keys: list[str] = []
    train_keys: list[str] = []
    for pid in problem_ids:
        keys = by_problem[pid]
        if len(val_keys) < n_val:
            val_keys.extend(keys)
        else:
            train_keys.extend(keys)

    rng.shuffle(val_keys)
    rng.shuffle(train_keys)
    val_keys = val_keys[:n_val]
    train_keys = train_keys[:n_train]
    return train_keys, val_keys


def build_split_artifacts(
    split_name: str,
    fork_keys: list[str],
    forks: dict[str, dict[str, list[dict]]],
    pair_mode: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Return (items, pairs, stats) for one split.

    items dedup the anchor and every referenced sibling. pairs reference items
    by uid and always carry the fork anchor (used by triplet; ignored by rank).
    """
    items: list[dict] = []
    pairs: list[dict] = []
    seen_uids: set[str] = set()
    n_pairs_all = 0
    n_pairs_one = 0

    def add_item(uid: str, fork_id: str, role: str, ex: dict | None, anchor_ex: dict) -> None:
        if uid in seen_uids:
            return
        seen_uids.add(uid)
        src = anchor_ex if ex is None else ex
        items.append({
            "item_uid": uid,
            "fork_id": fork_id,
            "role": role,
            "problem_id": src["problem_id"],
            "solution_id": src["solution_id"],
            "step_idx": src["step_idx"],
            "problem": src["problem"],
            "ground_truth_answer": src.get("ground_truth_answer", ""),
            "prefix": src["prefix"],
            "candidate_step": "" if role == "anchor" else src["candidate_step"],
            "rating": 0 if role == "anchor" else src["rating"],
            "label": -1 if role == "anchor" else src["label"],
        })

    for key in fork_keys:
        pid, sid, sidx = key.split("::", 2)
        fork_id = f"prm800k::{pid}::{sid}::{sidx}"
        pos_exs = forks[key]["pos"]
        neg_exs = forks[key]["neg"]
        anchor_ex = pos_exs[0]  # any sibling carries the shared prefix/problem
        a_uid = anchor_uid(fork_id)

        # Track both-mode pair counts for the design report (independent of choice).
        n_pairs_all += len(pos_exs) * len(neg_exs)
        n_pairs_one += 1

        chosen = enumerate_fork_pairs(pos_exs, neg_exs, mode=pair_mode, rng=rng)

        add_item(a_uid, fork_id, "anchor", None, anchor_ex)
        for pos_ex, neg_ex in chosen:
            p_uid = item_uid(fork_id, "positive", pos_ex["completion_idx"])
            n_uid = item_uid(fork_id, "negative", neg_ex["completion_idx"])
            add_item(p_uid, fork_id, "positive", pos_ex, anchor_ex)
            add_item(n_uid, fork_id, "negative", neg_ex, anchor_ex)
            pairs.append({
                "fork_id": fork_id,
                "anchor_uid": a_uid,
                "positive_uid": p_uid,
                "negative_uid": n_uid,
            })

    stats = {
        f"{split_name}_forks": len(fork_keys),
        f"{split_name}_items": len(items),
        f"{split_name}_pairs_materialized": len(pairs),
        f"{split_name}_pairs_if_one": n_pairs_one,
        f"{split_name}_pairs_if_all": n_pairs_all,
    }
    return items, pairs, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build full-scale PRM800K fork dataset.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--raw_dir", type=Path)
    g.add_argument("--raw_file", type=Path)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--tokenizer_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--n_train_forks", type=int, default=40000)
    p.add_argument("--n_val_forks", type=int, default=5000)
    p.add_argument("--pair_mode", choices=["one", "all"], default="one",
                   help="'one': one balanced (pos,neg) pair per fork. "
                        "'all': every positive against every negative.")
    args = p.parse_args()

    if args.raw_dir is None and args.raw_file is None:
        p.error("Provide --raw_dir or --raw_file.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[forks] Loading tokenizer {args.tokenizer_name_or_path} ...", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, local_files_only=args.local_files_only
        )
    except OSError:
        sys.exit("Tokenizer not found locally. Pre-cache the model before running offline.")

    rng = random.Random(args.seed)

    print("[forks] Loading raw PRM800K ...", flush=True)
    samples = load_raw_prm800k(args.raw_dir, args.raw_file)
    print(f"[forks] Loaded {len(samples)} raw samples.", flush=True)

    counters: dict[str, Any] = defaultdict(int)
    all_examples, fork_map = build_candidates(samples, counters)
    print(f"[forks] Raw candidates: {len(all_examples)}", flush=True)

    all_examples = filter_by_length(all_examples, tokenizer, args.max_seq_len, counters)
    survived = {e["uid"] for e in all_examples}
    fork_map = {
        k: [e for e in exs if e["uid"] in survived]
        for k, exs in fork_map.items()
    }
    fork_map = {k: v for k, v in fork_map.items() if v}

    forks = valid_forks(fork_map)
    print(f"[forks] Valid forks (>=1 pos & >=1 neg): {len(forks)}", flush=True)
    if len(forks) < args.n_train_forks + args.n_val_forks:
        print(
            f"[forks] WARNING: only {len(forks)} valid forks for requested "
            f"{args.n_train_forks}+{args.n_val_forks}; splits will be capped.",
            flush=True,
        )

    train_keys, val_keys = split_forks_by_problem(
        forks, args.n_train_forks, args.n_val_forks, rng
    )
    print(f"[forks] Split: train={len(train_keys)} val={len(val_keys)}", flush=True)

    train_items, train_pairs, train_stats = build_split_artifacts(
        "train", train_keys, forks, args.pair_mode, rng
    )
    val_items, val_pairs, val_stats = build_split_artifacts(
        "val", val_keys, forks, args.pair_mode, rng
    )

    # Problem-disjointness check across splits.
    train_pids = {k.split("::", 1)[0] for k in train_keys}
    val_pids = {k.split("::", 1)[0] for k in val_keys}
    pid_overlap = len(train_pids & val_pids)

    # Sibling distribution over all valid forks (grounds the one-vs-all report).
    sib_pos = [len(v["pos"]) for v in forks.values()]
    sib_neg = [len(v["neg"]) for v in forks.values()]

    # Audit dump of full forks (selected only, to keep it bounded).
    full_rows: list[dict] = []
    for key in train_keys + val_keys:
        pid, sid, sidx = key.split("::", 2)
        fork_id = f"prm800k::{pid}::{sid}::{sidx}"
        v = forks[key]
        full_rows.append({
            "fork_id": fork_id,
            "problem_id": v["pos"][0]["problem_id"],
            "step_idx": v["pos"][0]["step_idx"],
            "prefix": v["pos"][0]["prefix"],
            "n_positive": len(v["pos"]),
            "n_negative": len(v["neg"]),
            "positives": [
                {"completion_idx": e["completion_idx"], "text": e["candidate_step"]}
                for e in v["pos"]
            ],
            "negatives": [
                {"completion_idx": e["completion_idx"], "text": e["candidate_step"]}
                for e in v["neg"]
            ],
        })

    files: dict[str, dict] = {}

    def dump(name: str, rows: list[dict]) -> None:
        path = args.out_dir / name
        write_jsonl(path, rows)
        files[name] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}

    dump("forks_full.jsonl", full_rows)
    dump("forks_train_items.jsonl", train_items)
    dump("forks_val_items.jsonl", val_items)
    dump("forks_train_pairs.jsonl", train_pairs)
    dump("forks_val_pairs.jsonl", val_pairs)

    manifest = {
        "run_name": args.run_name,
        "seed": args.seed,
        "source_dataset": "PRM800K",
        "pair_mode": args.pair_mode,
        "label_mapping": {"rating_1": 0, "rating_minus_1": 1, "anchor": -1},
        "max_seq_len": args.max_seq_len,
        "n_valid_forks_total": len(forks),
        "split_policy": "problem_id disjoint train/val; surplus forks held out",
        "problem_id_overlap_train_val": pid_overlap,
        "counts": {**train_stats, **val_stats},
        "sibling_distribution": {
            "positives_per_fork_mean": (sum(sib_pos) / len(sib_pos)) if sib_pos else 0.0,
            "positives_per_fork_max": max(sib_pos) if sib_pos else 0,
            "negatives_per_fork_mean": (sum(sib_neg) / len(sib_neg)) if sib_neg else 0.0,
            "negatives_per_fork_max": max(sib_neg) if sib_neg else 0,
            "total_pairs_if_all_over_valid_forks": sum(
                len(v["pos"]) * len(v["neg"]) for v in forks.values()
            ),
        },
        "raw_counts": dict(counters),
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / "forks_manifest.json").write_text(json.dumps(manifest, indent=2))

    if pid_overlap != 0:
        sys.exit(f"BUG: problem_id overlap between train and val forks = {pid_overlap}")

    print("[forks] Done. Counts:", flush=True)
    for k, val in manifest["counts"].items():
        print(f"  {k}: {val}", flush=True)
    print(f"[forks] Manifest: {args.out_dir / 'forks_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
