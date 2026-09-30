#!/usr/bin/env python3
"""Pack online_reject_v2 rollouts into the explorer's data file.

One record per (arm, problem): the kept chain of steps, and at each step every
draft that was tried, with its text, all panel scores and whether it was kept.
Problem text and gold answer come from the problem files. Arm metadata (active
checker, threshold) is read from each arm's timing manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True,
                   help="directory with one subdirectory (or <arm>.jsonl file) per arm")
    p.add_argument("--problems", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    probs = {}
    for f in a.problems:
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                probs[r["problem_id"]] = {"dataset": r.get("dataset", ""), "problem": r["problem"],
                                          "gold": r.get("ground_truth_answer", r.get("gold"))}
    files = defaultdict(list)
    for f in sorted(a.runs.rglob("*.jsonl")):
        arm = f.parent.name if f.parent != a.runs else f.stem
        files[arm].append(f)

    arms, used = {}, set()
    for arm, fs in sorted(files.items()):
        meta, rolls = {}, {}
        for f in fs:
            t = f.with_suffix(".timing.json")
            if t.exists():
                m = json.loads(t.read_text())
                meta = {"active": m.get("active"), "tau": m.get("reject_tau"),
                        "blind_rate": m.get("blind_retry_rate")}
            for l in f.read_text().splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                texts = r.get("draft_texts") or [[s] for s in r["steps"]]
                scores = r.get("draft_scores") or [[{}] for _ in r["steps"]]
                kept = r.get("kept_attempt") or [0] * len(r["steps"])
                steps = []
                for k, step in enumerate(r["steps"]):
                    tx = texts[k] if k < len(texts) else [step]
                    sc = scores[k] if k < len(scores) else [{}]
                    steps.append([{"t": tx[j], "s": {n: round(v, 4) for n, v in
                                                    (sc[j] if j < len(sc) else {}).items()},
                                   "k": j == (kept[k] if k < len(kept) else 0)}
                                  for j in range(len(tx))])
                rolls[r["problem_id"]] = {"c": bool(r["correct"]), "p": r.get("pred"),
                                          "tok": r.get("gen_tokens"), "steps": steps}
                used.add(r["problem_id"])
        arms[arm] = {"meta": meta, "rollouts": rolls}
    out = {"problems": {k: v for k, v in probs.items() if k in used}, "arms": arms}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"[explorer] {len(out['problems'])} problems, arms {list(arms)} -> {a.out} "
          f"({a.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
