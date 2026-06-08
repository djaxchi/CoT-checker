#!/usr/bin/env python3
"""Stage D of the model-size DenseLinear ablation: aggregate one model's row.

Reads the 4 per-subset metrics files written by
evaluate_saved_probe_on_processbench.py (each has the val_selected metrics at
top level plus an ``oracle`` block) and writes, into the model directory:

    processbench_val_threshold.json    per-subset + macro F1_PB at the PRM800K
                                       val-selected threshold (deployable)
    processbench_oracle_threshold.json per-subset + macro F1_PB at the per-subset
                                       oracle threshold (0.005 grid; not deployable)
    per_subset_metrics.json            combined val + oracle, per subset + macro

Macro = unweighted mean of F1_PB across the 4 subsets (Sprint 1 convention).

Smoke gate: when --expect_val_macro / --expect_oracle_macro are given (the 1.5B
reproduction), the script asserts the macro values land within --tol; otherwise
it exits NON-ZERO so the afterok-chained launcher stops the sweep for debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[aggregate] missing per-subset metrics: {path}")
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", type=Path, required=True,
                   help="processbench_eval_shards/ dir containing <subset>/metrics.json.")
    p.add_argument("--out_dir", type=Path, required=True, help="The model directory.")
    p.add_argument("--metrics_name", type=str, default="metrics.json")
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    p.add_argument("--expect_val_macro", type=float, default=None)
    p.add_argument("--expect_oracle_macro", type=float, default=None)
    p.add_argument("--tol", type=float, default=0.01,
                   help="Absolute tolerance for the smoke-gate reproduction check.")
    args = p.parse_args()

    val_rows: dict[str, dict] = {}
    oracle_rows: dict[str, dict] = {}
    combined: dict[str, dict] = {}
    eval_time_total = 0.0

    for subset in args.subsets:
        m = load_json(args.eval_dir / subset / args.metrics_name)
        if "oracle" not in m:
            sys.exit(
                f"[aggregate] {subset} metrics lack an 'oracle' block. Re-run "
                "evaluate_saved_probe_on_processbench.py with --also_oracle."
            )
        oracle = m["oracle"]
        om = oracle["metrics"]
        eval_time_total += float(m.get("eval_time_sec", 0.0) or 0.0)

        val_rows[subset] = {
            "threshold": m["threshold"],
            "F1_PB": m["F1_PB"],
            "Acc_error": m["Acc_error"],
            "Acc_correct": m["Acc_correct"],
            "n_traces": m["n_traces"],
        }
        oracle_rows[subset] = {
            "threshold": oracle["threshold"],
            "threshold_step": oracle.get("threshold_step"),
            "n_grid_points": oracle.get("n_grid_points"),
            "F1_PB": om["F1_PB"],
            "Acc_error": om["Acc_error"],
            "Acc_correct": om["Acc_correct"],
        }
        combined[subset] = {
            "val_selected": val_rows[subset],
            "oracle": oracle_rows[subset],
        }

    def macro(rows: dict[str, dict]) -> float:
        return sum(rows[s]["F1_PB"] for s in args.subsets) / len(args.subsets)

    macro_val = macro(val_rows)
    macro_oracle = macro(oracle_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "processbench_val_threshold.json").write_text(json.dumps({
        "threshold_type": "val_selected",
        "macro_f1_pb": macro_val,
        "per_subset": val_rows,
        "subsets": args.subsets,
    }, indent=2))
    (args.out_dir / "processbench_oracle_threshold.json").write_text(json.dumps({
        "threshold_type": "oracle",
        "macro_f1_pb": macro_oracle,
        "per_subset": oracle_rows,
        "subsets": args.subsets,
    }, indent=2))
    (args.out_dir / "per_subset_metrics.json").write_text(json.dumps({
        "macro_f1_val_threshold": macro_val,
        "macro_f1_oracle": macro_oracle,
        "eval_walltime_sec": eval_time_total,
        "per_subset": combined,
        "subsets": args.subsets,
    }, indent=2))

    print(f"[aggregate] macro F1_PB  val_selected={macro_val:.4f}  oracle={macro_oracle:.4f}")
    for s in args.subsets:
        print(f"           {s:>14}: val={val_rows[s]['F1_PB']:.4f}  "
              f"oracle={oracle_rows[s]['F1_PB']:.4f} (t={oracle_rows[s]['threshold']})")

    # ---- Smoke gate (1.5B reproduction) ----------------------------------
    failed = False
    if args.expect_val_macro is not None:
        d = abs(macro_val - args.expect_val_macro)
        ok = d <= args.tol
        print(f"[gate] val macro {macro_val:.4f} vs expected {args.expect_val_macro:.4f} "
              f"(|Δ|={d:.4f}, tol={args.tol}) -> {'OK' if ok else 'FAIL'}")
        failed = failed or not ok
    if args.expect_oracle_macro is not None:
        d = abs(macro_oracle - args.expect_oracle_macro)
        ok = d <= args.tol
        print(f"[gate] oracle macro {macro_oracle:.4f} vs expected {args.expect_oracle_macro:.4f} "
              f"(|Δ|={d:.4f}, tol={args.tol}) -> {'OK' if ok else 'FAIL'}")
        failed = failed or not ok
    if failed:
        sys.exit(
            "[gate] FATAL: 1.5B did not reproduce the known Sprint 1 DenseLinear "
            "result within tolerance. Stopping the sweep (downstream afterok jobs "
            "will not run). Debug before scaling up."
        )


if __name__ == "__main__":
    main()
