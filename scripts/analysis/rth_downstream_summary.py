#!/usr/bin/env python3
"""RTH downstream summary (docs/read_token_v1_plan.md, "Stage E: RTH downstream").

Reads the saved Instruct pools of §21.13 with RTH's score files added, and the
outputs of `tts_rule_contrast.py` (baseline majority) for each pool:

  selection   accuracy lift over majority (points) with the paired question-cluster
              interval, for tie-break, best-of-N (rerank) and weighted vote at N = 2,
              4, 10, worst-step aggregation
  ranking     within-problem AUROC of the trajectory score for a correct final answer
              (problem-macro, ties half credit, `src/analysis/downstream_audit.within_auc`),
              worst step and mean step, all traces of each problem with both outcomes

    python scripts/analysis/rth_downstream_summary.py --pool cot-checker-results/tts_regraded_v2 \
        --contrast t10=results/read_token_v1/downstream/rule_contrast_t10.json \
        --out results/read_token_v1/downstream/summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.downstream_audit import within_auc  # noqa: E402

SCORERS = {"rth_rth_correct_seed1": "RTH",
           "prm_qwen25_math_7b": "PRM (Qwen2.5-Math-PRM-7B)",
           "boundary_stats__mlp_h1024x2__seed42__gen": "best probe (boundary_stats x mlp h1024x2)",
           "step_tokens__transformer_d512_l2_f2048_h8__seed44__gen": "step_tokens x d512"}
POOLS = {"t10": "instruct_t10", "t07": "instruct_t07"}
RULES = ("tiebreak", "rerank", "wvote")


def ranking(root: Path, ds: str) -> dict:
    trajs = [r for f in glob.glob(str(root / f"tts_{ds}.shard*_trajectories.jsonl"))
             for r in map(json.loads, open(f)) if r.get("gradeable")]
    scores = {c: {r["traj_uid"]: r["scores"] for f in glob.glob(str(root / f"scores/tts_{ds}__{c}.shard*.jsonl"))
                  for r in map(json.loads, open(f)) if r.get("scores")} for c in SCORERS}
    by = defaultdict(list)
    for r in trajs:
        if all(r["traj_uid"] in s for s in scores.values()):
            by[r["fork_id"]].append(r)
    out = {}
    for c, s in scores.items():
        for agg, f in (("worst", max), ("mean", np.mean)):
            vals = []
            for rs in by.values():
                y = np.array([bool(r["correct"]) for r in rs])
                if y.all() or not y.any():
                    continue
                vals.append(within_auc(y, np.array([f(s[r["traj_uid"]]) for r in rs], float)))
            out[(c, agg)] = (float(np.mean(vals)), len(vals))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, required=True, help="dir holding instruct_t10/ and instruct_t07/")
    p.add_argument("--contrast", nargs="+", required=True, help="t10=path t07=path (baseline majority)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    lines = ["# RTH downstream (Stage E)", "",
             "Lift over majority in accuracy points [paired question-cluster 95% interval]; "
             "worst-step aggregation. Ranking: within-problem AUROC for a correct final answer.", ""]
    for spec in a.contrast:
        tag, path = spec.split("=", 1)
        rows = json.loads(Path(path).read_text())
        res = {(r["dataset"], r["n"], r["rule"]): r for r in rows}
        lines += [f"## Pool {tag} ({POOLS[tag]})", ""]
        for ds in ("math500", "gsm8k"):
            base = {n: res[(ds, n, f"tiebreak::probe::{next(iter(SCORERS))}::worst")]["baseline_acc"]
                    for n in (2, 4, 10)}
            lines += [f"### {ds}: selection (majority {', '.join(f'N={n} {100 * v:.2f}' for n, v in base.items())})", "",
                      "| scorer | " + " | ".join(f"{rule} N={n}" for rule in RULES for n in (2, 4, 10)) + " |",
                      "|---|" + "---|" * (3 * len(RULES))]
            for c, name in SCORERS.items():
                cells = []
                for rule in RULES:
                    for n in (2, 4, 10):
                        r = res[(ds, n, f"{rule}::probe::{c}::worst")]
                        lo, hi = r["ci95"]
                        cells.append(f"{-100 * r['delta']:+.2f} [{-100 * hi:+.2f}, {-100 * lo:+.2f}]")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            rk = ranking(a.pool / POOLS[tag], ds)
            lines += ["", f"### {ds}: within-problem AUROC ({rk[(next(iter(SCORERS)), 'worst')][1]} problems)", "",
                      "| scorer | worst step | mean step |", "|---|---|---|"]
            for c, name in SCORERS.items():
                lines.append(f"| {name} | {rk[(c, 'worst')][0]:.3f} | {rk[(c, 'mean')][0]:.3f} |")
            lines.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
