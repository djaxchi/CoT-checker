"""
Build fixed PRM800K dataset artifacts for prestudy_v1.

All design decisions follow the MPB specification exactly.
Do not change counts, labels, truncation policy, split policy, or artifact names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_prm800k(raw_dir: Path | None, raw_file: Path | None) -> list[dict]:
    """Load PRM800K from a directory of JSONL/JSON files or a single file."""
    paths: list[Path] = []

    if raw_file is not None:
        if not raw_file.exists():
            sys.exit(
                "PRM800K raw files not found. "
                "This job does not download datasets. "
                f"Provide --raw_dir or --raw_file. (checked: {raw_file})"
            )
        paths = [raw_file]
    elif raw_dir is not None:
        if not raw_dir.exists():
            sys.exit(
                "PRM800K raw files not found. "
                "This job does not download datasets. "
                f"Provide --raw_dir or --raw_file. (checked: {raw_dir})"
            )
        for ext in ("*.jsonl", "*.json"):
            paths.extend(sorted(raw_dir.glob(ext)))
        if not paths:
            sys.exit(
                f"PRM800K raw files not found. No .jsonl or .json files in {raw_dir}. "
                "This job does not download datasets."
            )
    else:
        sys.exit("Must supply --raw_dir or --raw_file.")

    samples: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            loaded = json.loads(content)
            if isinstance(loaded, list):
                samples.extend(loaded)
            else:
                print(f"[WARN] {p}: JSON root is not a list, skipping.", file=sys.stderr)
        else:
            for lineno, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] {p}:{lineno}: JSON decode error: {e}", file=sys.stderr)
    return samples


# ---------------------------------------------------------------------------
# Text formatting (shared with encoder — must stay in sync)
# ---------------------------------------------------------------------------

def make_input_text(problem: str, prefix: str, candidate_step: str) -> str:
    prefix_section = f"Previous reasoning:\n{prefix}\n\n" if prefix else "Previous reasoning:\n\n"
    return f"Problem:\n{problem}\n\n{prefix_section}Current step:\n{candidate_step}"


def build_prompt_prefix(problem: str, prefix: str) -> str:
    """The prompt prefix text up to and including 'Current step:\n'.

    Must match the encoder's build_prompt_prefix exactly so that split
    tokenization in the length filter and in the encoder agree.
    """
    prefix_section = f"Previous reasoning:\n{prefix}\n\n" if prefix else "Previous reasoning:\n\n"
    return f"Problem:\n{problem}\n\n{prefix_section}Current step:\n"


# ---------------------------------------------------------------------------
# Candidate example construction
# ---------------------------------------------------------------------------

def build_candidates(
    samples: list[dict],
    counters: dict,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Extract all valid candidate-step examples and populate the fork map.

    Returns (all_examples, fork_map).
    fork_map maps "{problem_id}::{solution_id}::{step_idx}" -> list of example dicts.
    """
    all_examples: list[dict] = []
    fork_map: dict[str, list[dict]] = defaultdict(list)

    for sample_idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            counters["malformed_samples"] += 1
            continue

        # Raw PRM800K nests problem under "question" and steps under "label".
        # Some mirrors/preprocessed versions hoist these to the top level.
        # Support both layouts transparently.
        question = sample.get("question")
        if isinstance(question, dict):
            problem = question.get("problem")
            ground_truth = question.get("ground_truth_answer", "")
        else:
            problem = sample.get("problem")
            ground_truth = sample.get("ground_truth_answer", "")

        label_field = sample.get("label")
        if isinstance(label_field, dict):
            steps = label_field.get("steps")
        else:
            steps = sample.get("steps")

        if not problem or not isinstance(problem, str):
            counters["malformed_samples"] += 1
            continue
        if not isinstance(steps, list):
            counters["malformed_samples"] += 1
            continue

        problem_id = sample.get("problem_id") or f"p{sample_idx}_{stable_hash(problem)}"
        solution_id = sample.get("solution_id") or f"s{sample_idx}"

        prefix_parts: list[str] = []

        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                counters["invalid_prefix_steps"] += 1
                continue

            completions = step.get("completions")
            human_completion = step.get("human_completion")
            chosen_completion = step.get("chosen_completion")

            # Enumerate candidate completions at this step
            if isinstance(completions, list):
                current_prefix = "\n\n".join(prefix_parts)

                for comp_idx, comp in enumerate(completions):
                    counters["candidate_total_seen"] += 1

                    if not isinstance(comp, dict):
                        continue

                    text = comp.get("text")
                    rating = comp.get("rating")
                    flagged = comp.get("flagged", False)

                    if not text or not isinstance(text, str) or text.strip() == "":
                        counters["candidate_empty_text"] += 1
                        continue
                    if rating is None or rating not in (-1, 0, 1):
                        counters["candidate_missing_or_invalid_rating"] += 1
                        continue
                    if rating == 0:
                        counters["candidate_rating_0"] += 1
                        continue
                    if flagged:
                        counters["candidate_flagged"] += 1
                        continue

                    if rating == 1:
                        counters["candidate_rating_1"] += 1
                        label = 0
                    else:
                        counters["candidate_rating_minus_1"] += 1
                        label = 1

                    uid = (
                        f"prm800k::{problem_id}::{solution_id}"
                        f"::{step_idx}::{comp_idx}"
                    )
                    example = {
                        "uid": uid,
                        "problem_id": problem_id,
                        "solution_id": solution_id,
                        "step_idx": step_idx,
                        "completion_idx": comp_idx,
                        "problem": problem,
                        "ground_truth_answer": ground_truth,
                        "prefix": current_prefix,
                        "candidate_step": text,
                        "input_text": make_input_text(problem, current_prefix, text),
                        "rating": rating,
                        "label": label,
                        "source": "prm800k",
                    }
                    all_examples.append(example)
                    fork_map[f"{problem_id}::{solution_id}::{step_idx}"].append(example)

            # Advance prefix using selected completion
            if human_completion is not None:
                sel_text = human_completion.get("text") if isinstance(human_completion, dict) else None
                if sel_text and isinstance(sel_text, str):
                    prefix_parts.append(sel_text)
                else:
                    counters["invalid_prefix_steps"] += 1
            elif chosen_completion is not None and isinstance(completions, list):
                idx = chosen_completion
                if isinstance(idx, int) and 0 <= idx < len(completions):
                    sel = completions[idx]
                    sel_text = sel.get("text") if isinstance(sel, dict) else None
                    if sel_text and isinstance(sel_text, str):
                        prefix_parts.append(sel_text)
                    else:
                        counters["invalid_prefix_steps"] += 1
                else:
                    counters["invalid_prefix_steps"] += 1
            else:
                counters["invalid_prefix_steps"] += 1

    return all_examples, fork_map


# ---------------------------------------------------------------------------
# Length filtering
#
# Uses the SAME split tokenization as the encoder so that n_tokens counts
# are identical in both scripts and no example can pass this filter but
# then exceed max_seq_len at encoding time.
# ---------------------------------------------------------------------------

def filter_by_length(
    examples: list[dict],
    tokenizer,
    max_seq_len: int,
    counters: dict,
) -> list[dict]:
    kept: list[dict] = []
    for ex in examples:
        prefix_ids = tokenizer(
            build_prompt_prefix(ex["problem"], ex["prefix"]),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        cand_ids = tokenizer(
            ex["candidate_step"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(prefix_ids) + len(cand_ids) <= max_seq_len:
            kept.append(ex)
        else:
            counters["candidate_overlength"] += 1
    return kept


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def sample_disjoint_train_val(
    pos_examples: list[dict],
    neg_examples: list[dict],
    n_pos_train: int,
    n_neg_train: int,
    n_pos_val: int,
    n_neg_val: int,
    rng: random.Random,
    allow_problem_overlap: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Returns (pos_train, neg_train, pos_val, neg_val).

    No problem_id overlap between train and val unless allow_problem_overlap is set.
    Never mutates the input lists.
    """
    total_pos = n_pos_train + n_pos_val
    total_neg = n_neg_train + n_neg_val

    if len(pos_examples) < total_pos:
        sys.exit(
            f"Not enough positive examples after filtering: "
            f"need {total_pos}, have {len(pos_examples)}."
        )
    if len(neg_examples) < total_neg:
        sys.exit(
            f"Not enough negative examples after filtering: "
            f"need {total_neg}, have {len(neg_examples)}."
        )

    if allow_problem_overlap:
        pos = list(pos_examples)
        neg = list(neg_examples)
        rng.shuffle(pos)
        rng.shuffle(neg)
        return (
            pos[n_pos_val : n_pos_val + n_pos_train],
            neg[n_neg_val : n_neg_val + n_neg_train],
            pos[:n_pos_val],
            neg[:n_neg_val],
        )

    # Strict problem-level disjointness.
    # Assign each problem_id to either val or train once, shared across pos and neg.
    # This prevents a problem from appearing in pos_train and neg_val (or vice versa).
    pos_by_pid: dict[str, list[dict]] = defaultdict(list)
    neg_by_pid: dict[str, list[dict]] = defaultdict(list)
    for ex in pos_examples:
        pos_by_pid[ex["problem_id"]].append(ex)
    for ex in neg_examples:
        neg_by_pid[ex["problem_id"]].append(ex)

    all_pids = list(set(pos_by_pid.keys()) | set(neg_by_pid.keys()))
    rng.shuffle(all_pids)

    # Greedily fill val until both pos_val and neg_val quotas are met.
    val_pids: set[str] = set()
    val_pos_count = 0
    val_neg_count = 0
    for pid in all_pids:
        if val_pos_count >= n_pos_val and val_neg_count >= n_neg_val:
            break
        val_pids.add(pid)
        val_pos_count += len(pos_by_pid.get(pid, []))
        val_neg_count += len(neg_by_pid.get(pid, []))

    train_pids = [pid for pid in all_pids if pid not in val_pids]

    pos_val_pool = [e for pid in val_pids for e in pos_by_pid.get(pid, [])]
    neg_val_pool = [e for pid in val_pids for e in neg_by_pid.get(pid, [])]
    pos_train_pool = [e for pid in train_pids for e in pos_by_pid.get(pid, [])]
    neg_train_pool = [e for pid in train_pids for e in neg_by_pid.get(pid, [])]

    if len(pos_val_pool) < n_pos_val or len(pos_train_pool) < n_pos_train:
        sys.exit(
            f"Strict problem-level disjointness cannot be satisfied for positives "
            f"(val pool={len(pos_val_pool)}, train pool={len(pos_train_pool)}). "
            f"Pass --allow_problem_overlap to relax."
        )
    if len(neg_val_pool) < n_neg_val or len(neg_train_pool) < n_neg_train:
        sys.exit(
            f"Strict problem-level disjointness cannot be satisfied for negatives "
            f"(val pool={len(neg_val_pool)}, train pool={len(neg_train_pool)}). "
            f"Pass --allow_problem_overlap to relax."
        )

    rng.shuffle(pos_val_pool)
    rng.shuffle(pos_train_pool)
    rng.shuffle(neg_val_pool)
    rng.shuffle(neg_train_pool)

    return (
        pos_train_pool[:n_pos_train],
        neg_train_pool[:n_neg_train],
        pos_val_pool[:n_pos_val],
        neg_val_pool[:n_neg_val],
    )


# ---------------------------------------------------------------------------
# Contrastive fork extraction
# ---------------------------------------------------------------------------

def extract_forks(
    fork_map: dict[str, list[dict]],
    n_forks: int,
    train_uids: set[str],
    val_uids: set[str],
    rng: random.Random,
    counters: dict,
) -> list[dict]:
    """
    Find fork keys with at least one positive and one negative.
    Prefer forks whose examples are not in train/val.
    """
    valid_keys: list[str] = []
    for key, exs in fork_map.items():
        pos_exs = [e for e in exs if e["rating"] == 1]
        neg_exs = [e for e in exs if e["rating"] == -1]
        if pos_exs and neg_exs:
            valid_keys.append(key)

    counters["valid_forks_found"] = len(valid_keys)

    all_uids = train_uids | val_uids
    preferred = [k for k in valid_keys if not any(e["uid"] in all_uids for e in fork_map[k])]
    fallback = [k for k in valid_keys if k not in preferred]

    rng.shuffle(preferred)
    rng.shuffle(fallback)
    candidates = preferred + fallback

    if len(candidates) < n_forks:
        sys.exit(
            f"Cannot find {n_forks} contrastive forks: "
            f"only {len(candidates)} valid forks exist."
        )

    selected_keys = candidates[:n_forks]
    forks: list[dict] = []
    for key in selected_keys:
        exs = fork_map[key]
        pos_exs = [e for e in exs if e["rating"] == 1]
        neg_exs = [e for e in exs if e["rating"] == -1]
        pos = rng.choice(pos_exs)
        neg = rng.choice(neg_exs)
        pid, sid, sidx_str = key.split("::", 2)
        fork_id = f"prm800k::{pid}::{sid}::{sidx_str}"
        forks.append({
            "fork_id": fork_id,
            "problem_id": pos["problem_id"],
            "solution_id": pos["solution_id"],
            "step_idx": pos["step_idx"],
            "problem": pos["problem"],
            "ground_truth_answer": pos["ground_truth_answer"],
            "prefix": pos["prefix"],
            "positive_completion": {
                "completion_idx": pos["completion_idx"],
                "text": pos["candidate_step"],
                "rating": 1,
                "label": 0,
            },
            "negative_completion": {
                "completion_idx": neg["completion_idx"],
                "text": neg["candidate_step"],
                "rating": -1,
                "label": 1,
            },
        })
    return forks


def flatten_forks(forks: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for fork in forks:
        for role, comp_key, label in [
            ("positive", "positive_completion", 0),
            ("negative", "negative_completion", 1),
        ]:
            comp = fork[comp_key]
            uid = (
                f"prm800k::{fork['problem_id']}::{fork['solution_id']}"
                f"::{fork['step_idx']}::{comp['completion_idx']}"
            )
            flat.append({
                "uid": uid,
                "fork_id": fork["fork_id"],
                "pair_role": role,
                "problem_id": fork["problem_id"],
                "solution_id": fork["solution_id"],
                "step_idx": fork["step_idx"],
                "completion_idx": comp["completion_idx"],
                "problem": fork["problem"],
                "ground_truth_answer": fork["ground_truth_answer"],
                "prefix": fork["prefix"],
                "candidate_step": comp["text"],
                "input_text": make_input_text(
                    fork["problem"], fork["prefix"], comp["text"]
                ),
                "rating": comp["rating"],
                "label": label,
                "source": "prm800k_contrastive_fork",
            })
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build PRM800K prestudy dataset artifacts.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--raw_dir", type=Path)
    group.add_argument("--raw_file", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--n_pos_base", type=int, default=20000)
    parser.add_argument("--n_neg_base", type=int, default=20000)
    parser.add_argument("--n_pos_val", type=int, default=500)
    parser.add_argument("--n_neg_val", type=int, default=500)
    parser.add_argument("--n_forks", type=int, default=20)
    parser.add_argument("--allow_problem_overlap", action="store_true")
    args = parser.parse_args()

    if args.raw_dir is None and args.raw_file is None:
        parser.error("Provide --raw_dir or --raw_file.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] Loading tokenizer from {args.tokenizer_name_or_path} ...", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            local_files_only=args.local_files_only,
        )
    except OSError:
        sys.exit(
            "Model not found locally. This job runs offline on TamIA. "
            "Pre-cache Qwen/Qwen2.5-1.5B before submitting."
        )

    rng = random.Random(args.seed)

    print("[build] Loading raw PRM800K ...", flush=True)
    samples = load_raw_prm800k(args.raw_dir, args.raw_file)
    print(f"[build] Loaded {len(samples)} raw samples.", flush=True)

    counters: dict[str, Any] = {
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
    }

    print("[build] Constructing candidate examples ...", flush=True)
    all_examples, fork_map = build_candidates(samples, counters)
    print(
        f"[build] Raw candidates: {len(all_examples)} "
        f"(pos={counters['candidate_rating_1']}, "
        f"neg={counters['candidate_rating_minus_1']})",
        flush=True,
    )

    # Length filter — uses split tokenization (same as encoder) so counts agree.
    print("[build] Filtering by token length ...", flush=True)
    all_examples = filter_by_length(all_examples, tokenizer, args.max_seq_len, counters)
    print(
        f"[build] After length filter: {len(all_examples)} "
        f"({counters['candidate_overlength']} discarded as overlength)",
        flush=True,
    )

    # Rebuild fork_map to only include examples that survived the length filter.
    # This prevents overlength examples from appearing in the contrastive fork files,
    # which would cause encoding failures.
    survived_uids = {e["uid"] for e in all_examples}
    filtered_fork_map: dict[str, list[dict]] = defaultdict(list)
    for key, exs in fork_map.items():
        kept = [e for e in exs if e["uid"] in survived_uids]
        if kept:
            filtered_fork_map[key] = kept
    fork_map = filtered_fork_map

    pos_examples = [e for e in all_examples if e["label"] == 0]
    neg_examples = [e for e in all_examples if e["label"] == 1]
    print(f"[build] Positives available: {len(pos_examples)}", flush=True)
    print(f"[build] Negatives available: {len(neg_examples)}", flush=True)

    print("[build] Constructing train/val splits ...", flush=True)
    pos_train, neg_train, pos_val, neg_val = sample_disjoint_train_val(
        pos_examples, neg_examples,
        args.n_pos_base, args.n_neg_base,
        args.n_pos_val, args.n_neg_val,
        rng, args.allow_problem_overlap,
    )

    combined_train = pos_train + neg_train
    rng.shuffle(combined_train)
    combined_val = pos_val + neg_val
    rng.shuffle(combined_val)

    train_uids = {e["uid"] for e in pos_train + neg_train}
    val_uids = {e["uid"] for e in pos_val + neg_val}

    print("[build] Extracting contrastive forks ...", flush=True)
    forks = extract_forks(fork_map, args.n_forks, train_uids, val_uids, rng, counters)
    forks_flat = flatten_forks(forks)

    # Overlap tracking
    fork_uids = {r["uid"] for r in forks_flat}
    fork_train_overlap = len(fork_uids & train_uids)
    fork_val_overlap = len(fork_uids & val_uids)
    if fork_train_overlap or fork_val_overlap:
        print(
            f"[WARN] Fork/train UID overlap: {fork_train_overlap}, "
            f"fork/val UID overlap: {fork_val_overlap}",
            file=sys.stderr,
        )

    train_val_uid_overlap = len(train_uids & val_uids)
    train_pids = {e["problem_id"] for e in pos_train + neg_train}
    val_pids = {e["problem_id"] for e in pos_val + neg_val}
    train_val_pid_overlap = len(train_pids & val_pids)

    # Write files
    files_meta: dict[str, dict] = {}

    def write_and_hash(name: str, path: Path, rows: list[dict]) -> None:
        write_jsonl(path, rows)
        files_meta[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    print("[build] Writing JSONL files ...", flush=True)
    write_and_hash("prm800k_pos_base_20k", args.out_dir / "prm800k_pos_base_20k.jsonl", pos_train)
    write_and_hash("prm800k_neg_base_20k", args.out_dir / "prm800k_neg_base_20k.jsonl", neg_train)
    write_and_hash("prm800k_probe_train_40k", args.out_dir / "prm800k_probe_train_40k.jsonl", combined_train)
    write_and_hash("prm800k_mixed_train_40k", args.out_dir / "prm800k_mixed_train_40k.jsonl", list(combined_train))
    write_and_hash("prm800k_val_1k", args.out_dir / "prm800k_val_1k.jsonl", combined_val)
    write_and_hash("prm800k_contrastive_forks_20", args.out_dir / "prm800k_contrastive_forks_20.jsonl", forks)
    write_and_hash("prm800k_contrastive_forks_20_flat", args.out_dir / "prm800k_contrastive_forks_20_flat.jsonl", forks_flat)

    manifest = {
        "run_name": args.run_name,
        "seed": args.seed,
        "source_dataset": "PRM800K",
        "label_mapping": {
            "rating_1": 0,
            "rating_minus_1": 1,
        },
        "discarded_ratings": [0],
        "discarded_conditions": [
            "flagged=true",
            "empty_text",
            "missing_rating",
            "rating_not_in_{-1,1}",
            "tokenized_length_exceeds_max_seq_len",
        ],
        "length_policy": "no truncation",
        "max_seq_len": args.max_seq_len,
        "prefix_policy": "previous human_completion if available else chosen_completion",
        "split_policy": (
            "problem_id disjoint train/val"
            if not args.allow_problem_overlap
            else "problem_id overlap allowed (--allow_problem_overlap)"
        ),
        "counts": {
            "pos_base_20k": len(pos_train),
            "neg_base_20k": len(neg_train),
            "probe_train_40k": len(combined_train),
            "mixed_train_40k": len(combined_train),
            "val_1k": len(combined_val),
            "contrastive_forks": len(forks),
            "contrastive_forks_flat": len(forks_flat),
        },
        "overlap_checks": {
            "train_val_uid_overlap": train_val_uid_overlap,
            "train_val_problem_id_overlap": train_val_pid_overlap,
            "fork_train_uid_overlap": fork_train_overlap,
            "fork_val_uid_overlap": fork_val_overlap,
        },
        "raw_counts": {
            "candidate_total_seen": counters["candidate_total_seen"],
            "candidate_rating_1": counters["candidate_rating_1"],
            "candidate_rating_minus_1": counters["candidate_rating_minus_1"],
            "candidate_rating_0": counters["candidate_rating_0"],
            "candidate_flagged": counters["candidate_flagged"],
            "candidate_empty_text": counters["candidate_empty_text"],
            "candidate_overlength": counters["candidate_overlength"],
            "malformed_samples": counters["malformed_samples"],
            "invalid_prefix_steps": counters["invalid_prefix_steps"],
            "valid_forks_found": counters["valid_forks_found"],
        },
        "files": files_meta,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }

    manifest_path = args.out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[build] Done. Manifest: {manifest_path}", flush=True)
    print("[build] Counts:", flush=True)
    for k, v in manifest["counts"].items():
        print(f"  {k}: {v}", flush=True)

    # Hard assertions — these must all hold; assert is fine here because
    # count violations are bugs, not user errors.
    if len(pos_train) != args.n_pos_base:
        sys.exit(f"BUG: pos_train count = {len(pos_train)}, expected {args.n_pos_base}")
    if len(neg_train) != args.n_neg_base:
        sys.exit(f"BUG: neg_train count = {len(neg_train)}, expected {args.n_neg_base}")
    if len(combined_val) != args.n_pos_val + args.n_neg_val:
        sys.exit(f"BUG: val count = {len(combined_val)}, expected {args.n_pos_val + args.n_neg_val}")
    if len(forks) != args.n_forks:
        sys.exit(f"BUG: fork count = {len(forks)}, expected {args.n_forks}")
    if len(forks_flat) != args.n_forks * 2:
        sys.exit(f"BUG: forks_flat count = {len(forks_flat)}, expected {args.n_forks * 2}")
    if not args.allow_problem_overlap and train_val_pid_overlap != 0:
        sys.exit(f"BUG: problem_id overlap = {train_val_pid_overlap}")
    if train_val_uid_overlap != 0:
        sys.exit(f"BUG: UID overlap = {train_val_uid_overlap}")

    print("[build] All assertions passed.", flush=True)


if __name__ == "__main__":
    main()
