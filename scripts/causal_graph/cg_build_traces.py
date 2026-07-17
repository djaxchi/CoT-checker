#!/usr/bin/env python3
"""cot_causal_graph_v0 Stage 0: build intervention-ready traces (CPU, local).

Scans raw PRM800K sessions and keeps those that contain BOTH a complete golden
trajectory (finish_reason == "solution", ground-truth answer present) AND at
least one fork pair on that golden path (spec: docs/cot_causal_graph_v0_plan.md).
Session-level pairing guarantees the fork's golden prefix lies on the kept
trajectory, so every intervention site has a real teacher-forced continuation.
(A file-level join of the separately sampled transition_operator forks.jsonl x
golden.jsonl was tried first and yields ~1 trace: 28/911 question overlap.)

Per trace, frozen at build time:
  wrong_step      the fork's rating -1 sibling (the ERROR intervention)
  alt_pos_step    alternative +1 sibling when distinct from the golden step
                  (paraphrase-level control), else null
  xprob_step      off-topic control: golden step from another problem,
                  word-count matched to wrong_step
  candidates      the frozen 8-answer margin-readout set (gold first)

Also emits the Arm-G problem list and problem-disjoint splits (val for the
detection threshold, test for the cross-tab).

Outputs (in --run_dir): traces_forks.jsonl, onpolicy_problems.jsonl, splits.json,
build_manifest.json.

Usage:
  python scripts/causal_graph/cg_build_traces.py --n_traces 800
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

from scripts.encode_prm800k_hidden_states import git_commit, write_jsonl  # noqa: E402
from src.analysis.causal_graph import (  # noqa: E402
    join_forks_to_golden,
    length_matched_step,
)
from src.analysis.transition_operator import (  # noqa: E402
    build_candidates,
    extract_wrong_finals,
    stable_seed,
)
from src.data.prm800k_trajectories import (  # noqa: E402
    extract_fork_pairs,
    reconstruct_trajectory,
)


def h16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def session_traces(row: dict, sample_idx: int, phase: int, rng: random.Random,
                   counters: dict, min_downstream: int) -> list[dict]:
    """All joined (fork x golden) traces of one raw PRM800K session."""
    for k in ("malformed_samples", "missing_problem", "missing_steps",
              "truncated_paths", "too_few_steps", "skipped_sessions",
              "forks_found"):
        counters.setdefault(k, 0)
    label = row.get("label")
    if not isinstance(label, dict) or label.get("finish_reason") != "solution":
        counters["not_solution"] = counters.get("not_solution", 0) + 1
        return []
    q = row.get("question")
    gt = q.get("ground_truth_answer") if isinstance(q, dict) else None
    if not gt or not str(gt).strip():
        counters["no_gt_answer"] = counters.get("no_gt_answer", 0) + 1
        return []
    traj = reconstruct_trajectory(row, sample_idx, counters)
    if traj is None:
        return []
    traj["gt_answer"] = str(gt).strip()
    forks = extract_fork_pairs(row, counters)
    if not forks:
        return []
    pre_gen = q.get("pre_generated_answer") if isinstance(q, dict) else None
    wrong_finals = extract_wrong_finals(row)
    fork_rows = []
    for fk in forks:
        wrong = rng.choice(fk["wrongs"]) if fk.get("wrongs") else fk.get("wrong")
        if not wrong:
            continue
        fork_rows.append({
            "fork_id": h16(f'{fk["question"]}|{fk["step_index"]}|{fk["correct"]}|{wrong}'),
            "question": fk["question"],
            "step_index": fk["step_index"],
            "prefix_steps": fk["prefix_steps"],
            "correct": fk["correct"],
            "wrong": wrong,
            "gt_answer": traj["gt_answer"],
            "pre_generated_answer": pre_gen,
            "wrong_finals": wrong_finals,
            "phase": phase,
        })
    joined = join_forks_to_golden(fork_rows, [traj], min_downstream=min_downstream,
                                  counters=counters)
    # paraphrase-level control: a +1-rated sibling other than the golden step
    # (extract_fork_pairs' "correct" is the golden step itself whenever the chosen
    # completion is rated +1, so alt_pos must come from the raw completions)
    pos_by_index: dict[int, list[str]] = {}
    for i, step in enumerate(label.get("steps") or []):
        if not isinstance(step, dict):
            break
        comps = step.get("completions")
        if isinstance(comps, list):
            pos_by_index[i] = [
                c["text"].strip() for c in comps
                if isinstance(c, dict) and not c.get("flagged")
                and c.get("rating") == 1 and isinstance(c.get("text"), str)
                and c["text"].strip()]
    for tr in joined:
        if tr["alt_pos_step"] is None:
            alts = [p for p in pos_by_index.get(tr["fork_t"], [])
                    if p != tr["steps"][tr["fork_t"]]]
            if alts:
                tr["alt_pos_step"] = rng.choice(alts)
    return joined


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--dataset", type=str, default="tasksource/PRM800K")
    ap.add_argument("--files", type=str, nargs="+",
                    default=["phase2_test.jsonl", "phase2_train.jsonl",
                             "phase1_test.jsonl", "phase1_train.jsonl"])
    ap.add_argument("--n_traces", type=int, default=800)
    ap.add_argument("--min_downstream", type=int, default=2)
    ap.add_argument("--max_per_question", type=int, default=2)
    ap.add_argument("--max_total_words", type=int, default=1200,
                    help="proxy overlength filter; exact token filter runs in stage 1")
    ap.add_argument("--k_candidates", type=int, default=8)
    ap.add_argument("--n_onpolicy_problems", type=int, default=300)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out_traces = args.run_dir / "traces_forks.jsonl"
    if out_traces.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_traces}; pass --force")

    from huggingface_hub import hf_hub_download
    rng = random.Random(args.seed)
    counters: dict[str, int] = {}
    traces: list[dict] = []
    per_q: dict[str, int] = {}
    # oversample x3 before the shuffle/cap so early files don't dominate ordering
    target_pool = args.n_traces * 3
    for fname in args.files:
        phase = 2 if "phase2" in fname else 1
        path = hf_hub_download(args.dataset, fname, repo_type="dataset")
        print(f"[cg_build] scanning {fname} ...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                traces.extend(session_traces(row, idx, phase, rng, counters,
                                             args.min_downstream))
                if len(traces) >= target_pool:
                    break
        if len(traces) >= target_pool:
            break

    rng.shuffle(traces)
    kept: list[dict] = []
    seen_ids: set[str] = set()
    for tr in traces:
        if tr["trace_id"] in seen_ids:
            continue
        if per_q.get(tr["question"], 0) >= args.max_per_question:
            counters["capped_per_question"] = counters.get("capped_per_question", 0) + 1
            continue
        n_words = len(tr["question"].split()) + sum(len(s.split()) for s in tr["steps"])
        if n_words > args.max_total_words:
            counters["overlength_proxy"] = counters.get("overlength_proxy", 0) + 1
            continue
        seen_ids.add(tr["trace_id"])
        per_q[tr["question"]] = per_q.get(tr["question"], 0) + 1
        kept.append(tr)
        if len(kept) >= args.n_traces:
            break
    traces = kept

    # off-topic control pool + frozen candidate sets
    pool = [(tr["question"], s) for tr in traces for s in tr["steps"]]
    corpus_answers = tuple(tr["gt_answer"] for tr in traces if tr.get("gt_answer"))
    for tr in traces:
        r = random.Random(stable_seed(tr["trace_id"], args.seed))
        tr["xprob_step"] = length_matched_step(
            r, pool, target_words=len(tr["wrong_step"].split()),
            exclude_key=tr["question"])
        tr["candidates"] = build_candidates(
            tr["gt_answer"], tr.get("pre_generated_answer"),
            tr.get("wrong_finals") or [], corpus_answers,
            k=args.k_candidates, seed=stable_seed(tr["trace_id"], args.seed))

    # problem-disjoint splits by question hash order
    questions = sorted({tr["question"] for tr in traces},
                       key=lambda q: hashlib.sha1(q.encode()).hexdigest())
    n_val = int(round(args.val_frac * len(questions)))
    val_q = set(questions[:n_val])
    for tr in traces:
        tr["split"] = "val" if tr["question"] in val_q else "test"

    # Arm G problem list, sharing the split
    probs = []
    seen_q: set[str] = set()
    for tr in traces:
        if tr["question"] in seen_q:
            continue
        seen_q.add(tr["question"])
        probs.append({"problem": tr["question"],
                      "ground_truth_answer": tr["gt_answer"],
                      "split": tr["split"],
                      "fork_id": tr["trace_id"].split("::")[0]})
        if len(probs) >= args.n_onpolicy_problems:
            break

    write_jsonl(out_traces, traces)
    write_jsonl(args.run_dir / "onpolicy_problems.jsonl", probs)
    (args.run_dir / "splits.json").write_text(json.dumps({
        "val_questions_sha1": sorted(hashlib.sha1(q.encode()).hexdigest()
                                     for q in val_q),
        "n_val_questions": len(val_q),
        "n_test_questions": len(questions) - len(val_q)}, indent=2))
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "counters": counters,
        "n_traces": len(traces),
        "n_traces_val": sum(t["split"] == "val" for t in traces),
        "n_with_alt_pos": sum(t["alt_pos_step"] is not None for t in traces),
        "n_onpolicy_problems": len(probs),
        "downstream_steps_hist": {
            str(k): sum(len(tr["steps"]) - tr["fork_t"] - 1 == k for tr in traces)
            for k in sorted({len(tr["steps"]) - tr["fork_t"] - 1 for tr in traces})},
    }
    (args.run_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
