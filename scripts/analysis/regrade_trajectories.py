#!/usr/bin/env python3
"""Regrade a saved TTS pool with the current grader, into a separate directory.

The trajectories keep `pred`, `correct` and `gradeable` from generation time, so a
grader fix changes nothing downstream until the rows are regraded. This rewrites
only those three fields, from each row's own solution text and gold, into
`--out_root`, and links the confidence and score atoms beside them unchanged so
the frontier scripts read the regraded pool exactly as they read the original.
The source pool is never modified. A summary of what flipped is printed, since
a grader change that moves many answers in both directions is a new bug, not a
fix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.math_grade import grade  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--stems", nargs="+", default=["tts_gsm8k", "tts_math500"])
    a = p.parse_args()
    a.out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for stem in a.stems:
        up = down = n = 0
        for f in sorted(a.run_root.glob(f"{stem}.shard*_trajectories.jsonl")):
            out = []
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                g = grade(r["solution"], r["gold"])
                up += (not r["correct"]) and g["correct"]
                down += r["correct"] and not g["correct"]
                r["pred"], r["correct"], r["gradeable"] = g["pred"], g["correct"], g["gradeable"]
                out.append(json.dumps(r))
                n += 1
            (a.out_root / f.name).write_text("\n".join(out) + "\n")
        for f in sorted(a.run_root.glob(f"{stem}.shard*_conf.jsonl")):
            dst = a.out_root / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
        acc = sum(json.loads(l)["correct"] for f in a.out_root.glob(f"{stem}.shard*_trajectories.jsonl")
                  for l in f.read_text().splitlines() if l.strip()) / max(n, 1)
        summary[stem] = {"n": n, "wrong_to_right": int(up), "right_to_wrong": int(down),
                         "pass1": acc}
        print(f"[regrade] {stem}: n={n} wrong->right={up} right->wrong={down} pass@1={acc:.4f}")
    scores = a.out_root / "scores"
    if not scores.exists() and (a.run_root / "scores").exists():
        os.symlink((a.run_root / "scores").resolve(), scores)
    (a.out_root / "regrade_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
