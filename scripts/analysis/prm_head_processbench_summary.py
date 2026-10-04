"""ProcessBench F1 of Qwen2.5-Math-PRM-7B's own head, on the probes' metrics.

Reads the per-trace suspicion files written by scripts/score_processbench_with_prm.py
and reports, per subset and averaged over the four:
  fixed_0.5  Qwen's published protocol (first step with P(correct) < 0.5)
  oracle     best single threshold on the subset itself (a ceiling)
  calib20    threshold chosen on 20 held-out traces, 20 splits (the leaderboard metric)

    python scripts/analysis/prm_head_processbench_summary.py \
        --scores_dir cot-checker-results/prm_backbone_v1/prm_head_scores \
        --out results/prm_backbone_v1/prm_head_processbench.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.pb_threshold_calibration import load_traces  # noqa: E402
from scripts.merge_rep_grid_leaderboard import (CALIB_GRID, calib20_subset,  # noqa: E402
                                                f1_pb_from_preds, pred_matrix, quantile_grid)

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def subset_metrics(traces: list) -> dict:
    preds, labels = pred_matrix(traces, np.array([0.5]))
    is_cor = labels == -1
    acc_err = float((preds[~is_cor, 0] == labels[~is_cor]).mean())
    acc_cor = float((preds[is_cor, 0] == -1).mean())
    grid = np.unique(np.concatenate([quantile_grid(traces), CALIB_GRID]))
    p_all, _ = pred_matrix(traces, grid)
    return {"n_traces": len(traces),
            "fixed_0.5": {"Acc_error": acc_err, "Acc_correct": acc_cor,
                          "F1_PB": float(f1_pb_from_preds(preds, labels)[0])},
            "oracle_F1_PB": float(f1_pb_from_preds(p_all, labels).max()),
            "calib20_F1_PB": float(calib20_subset(traces))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    res = {s: subset_metrics(load_traces(a.scores_dir / f"pb_step_scores_{s}.jsonl"))
           for s in SUBSETS}
    res["avg"] = {"fixed_0.5_F1_PB": float(np.mean([res[s]["fixed_0.5"]["F1_PB"] for s in SUBSETS])),
                  "oracle_F1_PB": float(np.mean([res[s]["oracle_F1_PB"] for s in SUBSETS])),
                  "calib20_F1_PB": float(np.mean([res[s]["calib20_F1_PB"] for s in SUBSETS]))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{'subset':14s} {'F1@0.5':>8s} {'oracle':>8s} {'calib20':>8s}  Acc_err Acc_cor")
    for s in SUBSETS:
        r = res[s]
        print(f"{s:14s} {r['fixed_0.5']['F1_PB']:8.4f} {r['oracle_F1_PB']:8.4f} "
              f"{r['calib20_F1_PB']:8.4f}  {r['fixed_0.5']['Acc_error']:.3f}   "
              f"{r['fixed_0.5']['Acc_correct']:.3f}")
    v = res["avg"]
    print(f"{'avg':14s} {v['fixed_0.5_F1_PB']:8.4f} {v['oracle_F1_PB']:8.4f} {v['calib20_F1_PB']:8.4f}")


if __name__ == "__main__":
    main()
