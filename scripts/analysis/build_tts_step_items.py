#!/usr/bin/env python3
"""Per-step items (verifier-template rows) for the TTS traces of questions with a mixed pair.

external_reader_v0 (docs/external_reader_v0_plan.md) needs every step of every
candidate on the questions where at least one correct and one incorrect final
answer coexist (the only questions that enter the mixed-pair analysis of
§21.34). Steps follow scripts/generate_onpolicy_steps.split_into_steps, the
segmentation the §21.13 score files use; the step count is checked against them.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from src.eval.math_grade import answer_key  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("cot-checker-results/tts_regraded_v2/instruct_t10"))
    p.add_argument("--dataset", default="math500")
    p.add_argument("--check_cell", default="prm_qwen25_math_7b")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    trajs = [r for f in sorted(glob.glob(str(a.root / f"tts_{a.dataset}.shard*_trajectories.jsonl")))
             for r in map(json.loads, open(f)) if r.get("gradeable")]
    nsteps = {r["traj_uid"]: r["n_steps"] for f in glob.glob(str(a.root / f"scores/tts_{a.dataset}__{a.check_cell}.shard*.jsonl"))
              for r in map(json.loads, open(f))}
    by = defaultdict(list)
    for r in trajs:
        by[r["fork_id"]].append(r)
    keep = {q for q, rs in by.items()
            if any(r["correct"] for r in rs) and
            len({answer_key(r["pred"]) for r in rs if not r["correct"]} - {answer_key(r["pred"]) for r in rs if r["correct"]})}
    n_items = n_bad = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for q in sorted(keep):
            for r in by[q]:
                steps = split_into_steps(r["solution"])
                if nsteps.get(r["traj_uid"]) != len(steps):
                    n_bad += 1
                    continue
                for k, s in enumerate(steps):
                    f.write(json.dumps({"uid": f"{r['traj_uid']}::{k}", "traj_uid": r["traj_uid"], "step_idx": k,
                                        "problem": r["problem"], "prefix": "\n\n".join(steps[:k]),
                                        "candidate_step": s, "label": 0}) + "\n")
                    n_items += 1
    print(f"{len(keep)} questions, {n_items} step items, {n_bad} traces skipped for step-count mismatch")


if __name__ == "__main__":
    main()
