#!/usr/bin/env python3
"""Zero-shot verdict of the backbone under the judge prompt (judge_prompt_v1).

No probe: the score of a step is P(No) renormalized over {Yes, No} at the last
token of the verdict question, read from the logits the judge encode stored in
the judge-token store's meta. Written as a leaderboard cell (results.json plus
pb_step_scores_<subset>.jsonl) so in-domain AUROC / F1 and ProcessBench
val / oracle / calib-20 come from the same code as every trained cell.

The val threshold is chosen on val_5k like a cell's, so nothing is fitted on
test_2k or ProcessBench.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train_easy_probe_method import (  # noqa: E402
    auroc_numpy, evaluate_processbench, resolve_threshold_grid,
    select_threshold, step_binary_metrics,
)
from src.repstore.store import ShardedRepSplit  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def p_no(meta: list[dict]) -> np.ndarray:
    """P(No | Yes or No) = sigmoid(logit_no - logit_yes)."""
    d = np.array([m["judge_logit_no"] - m["judge_logit_yes"] for m in meta], dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-d))


def load(split_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """(y, meta) in global_index order."""
    view = ShardedRepSplit(split_dir)
    meta, y = view.meta(), np.asarray(view.y, dtype=np.int8)
    order = np.argsort([m["global_index"] for m in meta], kind="mergesort")
    return y[order], [meta[i] for i in order]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judge_store", required=True, type=Path,
                   help="PRM800K judge-token store root (holds val_5k, test_2k)")
    p.add_argument("--pb_judge_store", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--test_stem", default="test_2k")
    p.add_argument("--threshold_grid", default="0.01")
    a = p.parse_args()
    grid = resolve_threshold_grid(a.threshold_grid)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    y_val, m_val = load(a.judge_store / a.val_stem)
    y_te, m_te = load(a.judge_store / a.test_stem)
    s_val, s_te = p_no(m_val), p_no(m_te)
    t_val, val_bacc, _ = select_threshold(s_val, y_val, grid)
    t_or, _, _ = select_threshold(s_te, y_te, grid)
    in_domain = {
        "auroc": float(auroc_numpy(y_te, s_te)), "stem": a.test_stem,
        "is_validation": False, "val_threshold": float(t_val), "val_bacc": float(val_bacc),
        "fixed_0.5": step_binary_metrics(y_te, s_te, 0.5),
        "val_selected": step_binary_metrics(y_te, s_te, t_val),
        "oracle": step_binary_metrics(y_te, s_te, t_or),
        "val_auroc": float(auroc_numpy(y_val, s_val)),
    }
    print(f"[in_domain] AUROC={in_domain['auroc']:.4f} (val {in_domain['val_auroc']:.4f}) "
          f"t_val={t_val:.2f}", flush=True)

    pb = {}
    for sub in PB_SUBSETS:
        if not (a.pb_judge_store / sub).exists():
            print(f"[pb] skip {sub}: missing", flush=True)
            continue
        _, meta = load(a.pb_judge_store / sub)
        s = p_no(meta)
        rows, mv = evaluate_processbench(s, meta, t_val)
        best_f1, best_t = -1.0, grid[0]
        for t in grid:
            _, mt = evaluate_processbench(s, meta, t)
            if mt["F1_PB"] > best_f1:
                best_f1, best_t = mt["F1_PB"], t
        pb[sub] = {"val_selected": mv, "oracle_F1_PB": float(best_f1),
                   "oracle_threshold": float(best_t)}
        with (a.out_dir / f"pb_step_scores_{sub}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[pb:{sub}] F1_PB@val={mv['F1_PB']:.4f} oracle={best_f1:.4f}", flush=True)

    res = {"rep": "judge_zeroshot", "learner": "none", "seed": 0, "n_params": 0,
           "judge_store": str(a.judge_store), "pb_judge_store": str(a.pb_judge_store),
           "score": "P(No) over {Yes, No} at the verdict token",
           "in_domain": in_domain, "processbench": pb}
    (a.out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"[zeroshot] wrote {a.out_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
