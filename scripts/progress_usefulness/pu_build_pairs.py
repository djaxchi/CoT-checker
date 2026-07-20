"""progress_usefulness_v0 P0: build same-prefix PROGRESS(+1) vs NEUTRAL(0) pairs.

Sibling of ``build_prm800k_forks.py``. The correctness fork line pairs rating +1
(label 0) against rating -1 (label 1); this thread instead pairs, at the SAME
reasoning prefix, a rating +1 "progress-making" candidate against a rating 0
"neutral / no-progress" candidate. The rating-0 candidate is the object the
correctness pipeline discards; here it is the negative.

The clean unit is the PRM800K phase pool: a fork key
``{problem_id}::{solution_id}::{step_idx}`` that holds multiple candidate
completions evaluated from the same reasoning state. Pairing (s+, s0) within a
fork removes problem difficulty, reasoning state, step number and context length
as confounders (they are shared by construction).

Phase-1 first: point ``--raw_dir`` / ``--raw_file`` at PRM800K phase-1 data,
where alternatives were collected before conditioning on a first mistake. This
script does not itself split by collection phase; it consumes whatever raw files
it is given.

Reuses the EXACT candidate construction, tokenization, length filter and
problem-disjoint split policy of the correctness line. The only new behavior is
``build_candidates(keep_neutral=True)`` (retain rating 0) and pairing on
``rating`` rather than ``label``.

Outputs (under --out_dir):
  pu_pairs_full.jsonl        selected forks (audit): prefix + full progress/
                             neutral sibling lists.
  pu_train_items.jsonl       unique encodable rows for train (one per
  pu_val_items.jsonl         anchor | progress | neutral item).
  pu_train_pairs.jsonl       {fork_id, anchor_uid, progress_uid, neutral_uid}.
  pu_val_pairs.jsonl
  pu_manifest.json           counts, split sizes, sibling distribution, pair
                             counts under BOTH one/all modes, seed, git hash.

Item roles / progress_label:
  anchor   -> candidate_step="" ; embeds the shared reasoning prefix. label -1.
  progress -> a rating +1 continuation.  progress_label 1.
  neutral  -> a rating  0 continuation.  progress_label 0.
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_prm800k_forks import (  # noqa: E402
    anchor_uid,
    item_uid,
    split_forks_by_problem,
)
from scripts.build_prm800k_prestudy import (  # noqa: E402
    build_candidates,
    filter_by_length,
    git_commit,
    load_raw_prm800k,
    sha256_file,
    write_jsonl,
)
from src.repr.objectives import enumerate_fork_pairs  # noqa: E402


def progress_forks(fork_map: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    """Keep forks with >=1 progress (rating +1) and >=1 neutral (rating 0)."""
    out: dict[str, dict[str, list[dict]]] = {}
    for key, exs in fork_map.items():
        pos = [e for e in exs if e["rating"] == 1]
        neu = [e for e in exs if e["rating"] == 0]
        if pos and neu:
            out[key] = {"pos": pos, "neu": neu}
    return out


def build_split_artifacts(
    split_name: str,
    fork_keys: list[str],
    forks: dict[str, dict[str, list[dict]]],
    pair_mode: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Return (items, pairs, stats) for one split.

    items dedup the anchor and every referenced sibling; pairs reference items by
    uid and carry the shared fork anchor. Mirrors build_prm800k_forks with
    progress/neutral roles in place of positive/negative.
    """
    items: list[dict] = []
    pairs: list[dict] = []
    seen_uids: set[str] = set()
    n_pairs_all = 0
    n_pairs_one = 0
    prog_of = {"anchor": -1, "progress": 1, "neutral": 0}

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
            "progress_label": prog_of[role],
        })

    for key in fork_keys:
        pid, sid, sidx = key.split("::", 2)
        fork_id = f"prm800k::{pid}::{sid}::{sidx}"
        pos_exs = forks[key]["pos"]
        neu_exs = forks[key]["neu"]
        anchor_ex = pos_exs[0]  # any sibling carries the shared prefix/problem
        a_uid = anchor_uid(fork_id)

        # Track both-mode pair counts for the design report (independent of choice).
        n_pairs_all += len(pos_exs) * len(neu_exs)
        n_pairs_one += 1

        chosen = enumerate_fork_pairs(pos_exs, neu_exs, mode=pair_mode, rng=rng)

        add_item(a_uid, fork_id, "anchor", None, anchor_ex)
        for pos_ex, neu_ex in chosen:
            p_uid = item_uid(fork_id, "progress", pos_ex["completion_idx"])
            u_uid = item_uid(fork_id, "neutral", neu_ex["completion_idx"])
            add_item(p_uid, fork_id, "progress", pos_ex, anchor_ex)
            add_item(u_uid, fork_id, "neutral", neu_ex, anchor_ex)
            pairs.append({
                "fork_id": fork_id,
                "anchor_uid": a_uid,
                "progress_uid": p_uid,
                "neutral_uid": u_uid,
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
    p = argparse.ArgumentParser(
        description="Build PRM800K same-prefix progress(+1) vs neutral(0) pairs.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--raw_dir", type=Path)
    g.add_argument("--raw_file", type=Path)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--tokenizer_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--n_train_forks", type=int, default=4000)
    p.add_argument("--n_val_forks", type=int, default=500)
    p.add_argument("--pair_mode", choices=["one", "all"], default="one",
                   help="'one': one balanced (progress,neutral) pair per fork. "
                        "'all': every progress candidate against every neutral.")
    args = p.parse_args()

    if args.raw_dir is None and args.raw_file is None:
        p.error("Provide --raw_dir or --raw_file.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pu] Loading tokenizer {args.tokenizer_name_or_path} ...", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, local_files_only=args.local_files_only
        )
    except OSError:
        sys.exit("Tokenizer not found locally. Pre-cache the model before running offline.")

    rng = random.Random(args.seed)

    print("[pu] Loading raw PRM800K ...", flush=True)
    samples = load_raw_prm800k(args.raw_dir, args.raw_file)
    print(f"[pu] Loaded {len(samples)} raw samples.", flush=True)

    counters: dict[str, Any] = defaultdict(int)
    all_examples, fork_map = build_candidates(samples, counters, keep_neutral=True)
    print(f"[pu] Raw candidates (incl. neutral): {len(all_examples)}", flush=True)

    all_examples = filter_by_length(all_examples, tokenizer, args.max_seq_len, counters)
    survived = {e["uid"] for e in all_examples}
    fork_map = {
        k: [e for e in exs if e["uid"] in survived]
        for k, exs in fork_map.items()
    }
    fork_map = {k: v for k, v in fork_map.items() if v}

    forks = progress_forks(fork_map)
    print(f"[pu] Valid forks (>=1 progress & >=1 neutral): {len(forks)}", flush=True)
    if len(forks) < args.n_train_forks + args.n_val_forks:
        print(
            f"[pu] WARNING: only {len(forks)} valid forks for requested "
            f"{args.n_train_forks}+{args.n_val_forks}; splits will be capped.",
            flush=True,
        )

    train_keys, val_keys = split_forks_by_problem(
        forks, args.n_train_forks, args.n_val_forks, rng
    )
    print(f"[pu] Split: train={len(train_keys)} val={len(val_keys)}", flush=True)

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
    sib_neu = [len(v["neu"]) for v in forks.values()]

    # Audit dump of selected forks (bounded to the chosen train+val keys).
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
            "n_progress": len(v["pos"]),
            "n_neutral": len(v["neu"]),
            "progress": [
                {"completion_idx": e["completion_idx"], "text": e["candidate_step"]}
                for e in v["pos"]
            ],
            "neutral": [
                {"completion_idx": e["completion_idx"], "text": e["candidate_step"]}
                for e in v["neu"]
            ],
        })

    files: dict[str, dict] = {}

    def dump(name: str, rows: list[dict]) -> None:
        path = args.out_dir / name
        write_jsonl(path, rows)
        files[name] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}

    dump("pu_pairs_full.jsonl", full_rows)
    dump("pu_train_items.jsonl", train_items)
    dump("pu_val_items.jsonl", val_items)
    dump("pu_train_pairs.jsonl", train_pairs)
    dump("pu_val_pairs.jsonl", val_pairs)

    manifest = {
        "run_name": args.run_name,
        "seed": args.seed,
        "source_dataset": "PRM800K",
        "thread": "progress_usefulness_v0",
        "pair_mode": args.pair_mode,
        "pair_semantics": "positive=rating+1 (progress), negative=rating 0 (neutral)",
        "progress_label_mapping": {"rating_1": 1, "rating_0": 0, "anchor": -1},
        "max_seq_len": args.max_seq_len,
        "n_valid_forks_total": len(forks),
        "split_policy": "problem_id disjoint train/val; surplus forks held out",
        "problem_id_overlap_train_val": pid_overlap,
        "counts": {**train_stats, **val_stats},
        "sibling_distribution": {
            "progress_per_fork_mean": (sum(sib_pos) / len(sib_pos)) if sib_pos else 0.0,
            "progress_per_fork_max": max(sib_pos) if sib_pos else 0,
            "neutral_per_fork_mean": (sum(sib_neu) / len(sib_neu)) if sib_neu else 0.0,
            "neutral_per_fork_max": max(sib_neu) if sib_neu else 0,
            "total_pairs_if_all_over_valid_forks": sum(
                len(v["pos"]) * len(v["neu"]) for v in forks.values()
            ),
        },
        "raw_counts": dict(counters),
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / "pu_manifest.json").write_text(json.dumps(manifest, indent=2))

    if pid_overlap != 0:
        sys.exit(f"BUG: problem_id overlap between train and val forks = {pid_overlap}")

    print("[pu] Done. Counts:", flush=True)
    for k, val in manifest["counts"].items():
        print(f"  {k}: {val}", flush=True)
    print(f"[pu] Manifest: {args.out_dir / 'pu_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
