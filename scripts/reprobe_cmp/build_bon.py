#!/usr/bin/env python3
"""Best-of-N candidates of ReProbe's released pools as scoring inputs (reprobe_comparison_v2, B1).

For each set, the first N candidates per question of `JingweiNi/test_*_Qwen3-8B_texts_uhead_prm`
(N = 10, 5 for planning, as in the paper's Table 3), written in the formats the step
scorers already read:

  <out>/<set>.traces.jsonl  ReProbe-head traces (scripts/reprobe_cmp/reprobe_probe.py):
                            prompt_text, reply_ids, the pool's own steps (saved_stats.claims),
                            labels all -1 (no step labels)
  <out>/<set>.items.jsonl   read-token items (scripts/read_token_inline.py --layout reprobe),
                            label 0 as a placeholder
  <out>/<set>.meta.jsonl    trace_id, question index qi, candidate index k, and the pool's
                            stored trace scores (ReProbe Attn+Logit DeepSeek, PRMs, UQ)

Candidate order and question indices are those of bon_d0.py, whose graded.jsonl the
analysis joins on (qi, k).

    python scripts/reprobe_cmp/build_bon.py --pools $SCRATCH/cot_mech/reprobe_bon/pools --out $R/bon/inputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.bon_d0 import load_pool  # noqa: E402
from src.data.reprobe_steps import _claim_steps, step_item  # noqa: E402
from src.eval.reprobe_bon_grade import GRADED  # noqa: E402
from src.onpolicy.prompts import reprobe_prompt  # noqa: E402

SCORE_COLS = ("JingweiNi/", "Qwen/", "peiyi", "RLHFlow/", "universalprm/", "HuggingFaceH4/",
              "MaximumSequenceProbability", "MeanTokenEntropy", "Perplexity", "PTrue")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sets", default=",".join(GRADED))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.sets.split(","):
        n = 5 if name in ("calendar", "trips", "meetings") else 10
        df = load_pool(a.pools, name)
        df = df[df.k < n].reset_index(drop=True)
        cols = [c for c in df.columns if c.startswith(SCORE_COLS)]
        bad = 0
        with open(a.out / f"{name}.traces.jsonl", "w") as ft, open(a.out / f"{name}.items.jsonl", "w") as fi, \
                open(a.out / f"{name}.meta.jsonl", "w") as fm:
            for r in df.to_dict("records"):
                ss = r["saved_stats"]
                if reprobe_prompt(r["q"]) != r["question"]:
                    raise SystemExit(f"{name} q{r['qi']}: prompt is not reprobe_prompt(question)")
                steps = _claim_steps(r["reply"], list(ss["claims"]))
                if not steps:
                    bad += 1
                    continue
                tid = f"{name}_q{r['qi']:04d}_k{r['k']:02d}"
                ft.write(json.dumps({"trace_id": tid, "set": name, "prompt_text": r["question"],
                                     "reply_ids": [int(t) for t in ss["greedy_tokens"]],
                                     "steps": [{"aligned_token_ids": s["aligned_token_ids"]} for s in steps],
                                     "labels": [-1] * len(steps)}) + "\n")
                for j, st in enumerate(steps):
                    prefix, body = step_item(r["reply"], st)
                    fi.write(json.dumps({"uid": f"{tid}::{j}", "id": tid, "step_idx": j, "problem": r["q"],
                                         "prefix": prefix, "candidate_step": body, "label": 0},
                                        ensure_ascii=False) + "\n")
                fm.write(json.dumps({"trace_id": tid, "qi": int(r["qi"]), "k": int(r["k"]), "n_steps": len(steps),
                                     "stored": {c: float(r[c]) for c in cols}}) + "\n")
        print(f"[bon] {name}: {df.qi.nunique()} questions x {n}, {len(df) - bad} candidates "
              f"({bad} without steps)", flush=True)


if __name__ == "__main__":
    main()
