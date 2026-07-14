"""transition_operator_v0 stage 0a: build fork rows WITH answers, plus complete
golden trajectories for the suffix-selection gate.

Mirrors scripts/analysis/s4_contrib_forks.py (same fork definition, dedup key,
seeding, overlength filter) but additionally carries, per fork:
  gt_answer               question.ground_truth_answer from the raw session
  pre_generated_answer    phase-2 model answer (distractor source 1), else null
  wrong_finals            answers stated in rating -1 completions of the SAME
                          session (distractor source 2)
  phase                   1 or 2 (from the source filename)
Forks whose session has no ground-truth answer are dropped (Target B needs it).

Also writes golden.jsonl: complete golden trajectories (finish_reason == "solution",
gt answer present) for Stage 0 gate 1 (elicitation-suffix selection).

Outputs (in --run_dir, default runs/transition_operator/):
  forks.jsonl, golden.jsonl, build_manifest.json

Usage:
  python scripts/transition_operator/to_build_forks.py --n_forks 1000 --n_golden 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.transition_operator import extract_wrong_finals  # noqa: E402
from src.data.prm800k_trajectories import (  # noqa: E402
    extract_fork_pairs,
    reconstruct_trajectory,
)


def h16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--dataset", type=str, default="tasksource/PRM800K")
    ap.add_argument("--files", type=str, nargs="+",
                    default=["phase1_test.jsonl", "phase1_train.jsonl",
                             "phase2_test.jsonl", "phase2_train.jsonl"])
    ap.add_argument("--n_forks", type=int, default=1000)
    ap.add_argument("--n_golden", type=int, default=500)
    ap.add_argument("--max_prefix_steps", type=int, default=9)
    ap.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    forks_path = args.run_dir / "forks.jsonl"
    golden_path = args.run_dir / "golden.jsonl"
    for p in (forks_path, golden_path):
        if p.exists() and not args.force:
            sys.exit(f"refusing to overwrite {p}; pass --force")

    from huggingface_hub import hf_hub_download
    rng = random.Random(args.seed)
    counters = {"skipped_sessions": 0, "forks_found": 0, "no_gt_answer": 0}
    traj_counters: dict[str, int] = {
        "malformed_samples": 0, "missing_problem": 0, "missing_steps": 0,
        "truncated_paths": 0, "too_few_steps": 0}
    candidates: dict[str, dict] = {}
    golden: dict[str, dict] = {}
    n_rows = 0
    for fname in args.files:
        phase = 2 if "phase2" in fname else 1
        path = hf_hub_download(args.dataset, fname, repo_type="dataset")
        print(f"[to_forks] scanning {fname} ...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_rows += 1
                q = row.get("question")
                gt = q.get("ground_truth_answer") if isinstance(q, dict) else None
                pre_gen = q.get("pre_generated_answer") if isinstance(q, dict) else None
                label = row.get("label")
                finish = label.get("finish_reason") if isinstance(label, dict) else None

                # ---- golden trajectories for gate 1
                if gt and finish == "solution" and len(golden) < 10 * args.n_golden:
                    traj = reconstruct_trajectory(row, n_rows, traj_counters)
                    if traj is not None and traj["trajectory_id"] not in golden:
                        golden[traj["trajectory_id"]] = {
                            "trajectory_id": traj["trajectory_id"],
                            "question": traj["question"],
                            "steps": traj["steps"],
                            "gt_answer": str(gt),
                            "phase": phase,
                        }

                # ---- forks with answers
                forks = extract_fork_pairs(row, counters)
                if forks and not gt:
                    counters["no_gt_answer"] += len(forks)
                    continue
                wrong_finals = extract_wrong_finals(row) if forks else []
                for fk in forks:
                    if len(fk["prefix_steps"]) > args.max_prefix_steps:
                        continue
                    wrong = rng.choice(fk["wrongs"])
                    key = h16("\x1e".join([fk["question"], *fk["prefix_steps"],
                                           fk["correct"], wrong]))
                    if key not in candidates:
                        candidates[key] = {
                            "fork_id": key,
                            "question": fk["question"],
                            "prefix_steps": fk["prefix_steps"],
                            "step_index": fk["step_index"],
                            "correct": fk["correct"],
                            "wrong": wrong,
                            "gt_answer": str(gt),
                            "pre_generated_answer":
                                str(pre_gen) if pre_gen is not None else None,
                            "wrong_finals": wrong_finals,
                            "phase": phase,
                        }
    print(f"[to_forks] {n_rows} sessions -> {counters['forks_found']} fork steps, "
          f"{len(candidates)} unique pairs with gt answer "
          f"({counters['no_gt_answer']} forks dropped: no gt answer); "
          f"{len(golden)} complete golden trajectories", flush=True)

    pool = [candidates[k] for k in sorted(candidates)]
    rng.shuffle(pool)
    golden_pool = [golden[k] for k in sorted(golden)]
    rng.shuffle(golden_pool)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                        local_files_only=True)

    def n_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    kept: list[dict] = []
    n_overlength = 0
    for fk in pool:
        if len(kept) >= args.n_forks:
            break
        base = "\n".join([fk["question"], *fk["prefix_steps"]])
        longest = base + "\n" + max(fk["correct"], fk["wrong"], key=len)
        if n_tokens(longest) > args.max_seq_len:
            n_overlength += 1
            continue
        kept.append(fk)

    kept_golden: list[dict] = []
    for tr in golden_pool:
        if len(kept_golden) >= args.n_golden:
            break
        if n_tokens("\n".join([tr["question"], *tr["steps"]])) <= args.max_seq_len:
            kept_golden.append(tr)

    with open(forks_path, "w", encoding="utf-8") as f:
        for fk in kept:
            f.write(json.dumps(fk, ensure_ascii=False) + "\n")
    with open(golden_path, "w", encoding="utf-8") as f:
        for tr in kept_golden:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    n_wf = sum(1 for fk in kept if fk["wrong_finals"])
    n_pg = sum(1 for fk in kept if fk["pre_generated_answer"])
    (args.run_dir / "build_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "n_sessions_scanned": n_rows,
        "n_fork_steps_found": counters["forks_found"],
        "n_forks_dropped_no_gt": counters["no_gt_answer"],
        "n_unique_pairs": len(candidates),
        "n_forks_sampled": len(kept),
        "n_dropped_overlength": n_overlength,
        "n_golden": len(kept_golden),
        "frac_forks_with_wrong_finals": n_wf / len(kept) if kept else 0,
        "frac_forks_with_pre_generated": n_pg / len(kept) if kept else 0,
        "max_prefix_steps": args.max_prefix_steps,
        "max_seq_len": args.max_seq_len,
    }, indent=2))
    print(f"[to_forks] wrote {forks_path} ({len(kept)} forks; "
          f"{n_wf} with wrong finals, {n_pg} with pre-generated answers) and "
          f"{golden_path} ({len(kept_golden)} trajectories)")


if __name__ == "__main__":
    main()
