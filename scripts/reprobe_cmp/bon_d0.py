#!/usr/bin/env python3
"""Gate D0 of reprobe_comparison_v2 (docs/reprobe_comparison_v2_plan.md): does our grading
of ReProbe's released best-of-N pools reproduce its Table 3?

For each programmatically graded set, takes the first N candidates per question (N = 10,
5 for planning, as in the paper), grades them (src/eval/reprobe_bon_grade.py) and prints
pass@1 (mean candidate accuracy), pass@N, majority vote and the best-of-N accuracy of
every stored trace score (lowest score kept; ReProbe and the PRMs are stored as
uncertainties), next to the paper's numbers. Writes <out>/<set>.graded.jsonl (question
index, candidate index, correct, answer key) for the B1 analysis.

    python scripts/reprobe_cmp/bon_d0.py --pools $SCRATCH/cot_mech/reprobe_bon/pools --out $R/bon
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.reprobe_bon_grade import GRADED, grade  # noqa: E402

UHEAD = "JingweiNi/uhead_claim_Qwen3-8B_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3"
# Table 3 (Qwen3-8B): pass@1, pass@N, majority, ReProbe Attn+Logit DeepSeek, Qwen2.5-Math-PRM-7B
TABLE3 = {"math": (92.4, 99.3, None, 92.7, 93.7), "gsm8k": (95.6, 99.2, 97.6, 97.8, 97.8),
          "strategyqa": (86.8, 98.3, 86.6, 88.6, 88.1), "scienceqa": (92.7, 99.3, 92.5, 96.9, 96.9),
          "calendar": (23.5, 60.5, None, 26.0, 26.5)}
QUESTION = re.compile(r"<Question>:\s*(.*?)<\|im_end\|>", re.S)


def load_pool(root: Path, name: str):
    import pandas as pd
    files = sorted(glob.glob(str(root / name / "data" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
    df["q"] = df.question.str.extract(QUESTION)[0]
    df["qi"] = df.groupby("q", sort=False).ngroup()
    df["k"] = df.groupby("q", sort=False).cumcount()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sets", default=",".join(GRADED))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in a.sets.split(","):
        n = 5 if name == "calendar" else 10
        df = load_pool(a.pools, name)
        df = df[df.k < n].reset_index(drop=True)
        g = [grade(name, r, str(gold), q) for r, gold, q in zip(df.reply, df.answer, df.q)]
        df["ok"], df["key"] = [x[0] for x in g], [x[1] for x in g]
        with open(a.out / f"{name}.graded.jsonl", "w") as f:
            for r in df[["qi", "k", "ok", "key"]].itertuples(index=False):
                f.write(json.dumps({"qi": int(r.qi), "k": int(r.k), "ok": bool(r.ok), "key": r.key}) + "\n")
        by = df.groupby("qi")
        maj = by.apply(lambda x: bool(x.ok.iloc[[k == Counter(x.key.dropna()).most_common(1)[0][0]
                                                 for k in x.key].index(True)]) if x.key.notna().any() else False)
        row = {"questions": int(df.qi.nunique()), "N": n, "no_answer": float(df.key.isna().mean()),
               "pass@1": 100 * float(df.ok.mean()), "pass@N": 100 * float(by.ok.max().mean()),
               "majority": 100 * float(maj.mean())}
        for c in [UHEAD] + [c for c in df.columns if c.startswith(("Qwen/", "peiyi", "RLHFlow/", "universalprm/",
                                                                     "HuggingFaceH4/"))]:
            row[c] = 100 * float(by.apply(lambda x: bool(x.ok.loc[x[c].idxmin()])).mean())
        report[name] = row
        t = TABLE3[name]
        print(f"[D0] {name} ({row['questions']} questions, N={n}, no answer {row['no_answer']:.3f}): "
              f"pass@1 {row['pass@1']:.1f} (paper {t[0]}), pass@N {row['pass@N']:.1f} ({t[1]}), "
              f"majority {row['majority']:.1f} ({t[2]}), ReProbe A+L DS {row[UHEAD]:.1f} ({t[3]}), "
              f"Qwen2.5-Math-PRM-7B {row.get('Qwen/Qwen2.5-Math-PRM-7B', float('nan')):.1f} ({t[4]})", flush=True)
    (a.out / "d0.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
