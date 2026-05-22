"""Recompute ProcessBench-GSM8K metrics from a saved pb_step_scores.jsonl.

Reads predictions already made by the training script (threshold + per-step
scores) and emits a refreshed eval_metrics.json. Useful for re-evaluating with
a different threshold without retraining.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pb_step_scores", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path. Defaults to eval_metrics.json next to the input.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the threshold stored in pb_step_scores.jsonl.")
    p.add_argument("--method", type=str, default=None,
                   help="Method tag to include in the output JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.pb_step_scores.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {args.pb_step_scores}")

    threshold = args.threshold if args.threshold is not None else float(rows[0]["threshold"])
    n_error = n_correct = 0
    acc_error_hits = acc_correct_hits = 0
    exact_all = 0
    n_steps_total = 0

    t0 = time.time()
    for row in rows:
        scores = row["scores"]
        n_steps_total += len(scores)
        pred = -1
        for t_idx, s in enumerate(scores):
            if s > threshold:
                pred = t_idx
                break
        label = int(row["label"])
        if label == -1:
            n_correct += 1
            if pred == -1:
                acc_correct_hits += 1
        else:
            n_error += 1
            if pred == label:
                acc_error_hits += 1
        if pred == label:
            exact_all += 1
    eval_time = time.time() - t0

    acc_error = acc_error_hits / max(n_error, 1) if n_error else 0.0
    acc_correct = acc_correct_hits / max(n_correct, 1) if n_correct else 0.0
    denom = acc_error + acc_correct
    f1_pb = (2 * acc_error * acc_correct / denom) if denom > 0 else 0.0

    metrics = {
        "method": args.method,
        "threshold": threshold,
        "n_traces": len(rows),
        "n_error_traces": n_error,
        "n_correct_traces": n_correct,
        "Acc_error": acc_error,
        "Acc_correct": acc_correct,
        "F1_PB": f1_pb,
        "Exact_match_all": exact_all / max(len(rows), 1),
        "eval_time_sec": eval_time,
        "mean_step_latency_ms": eval_time * 1000.0 / max(n_steps_total, 1),
        "mean_trace_latency_ms": eval_time * 1000.0 / max(len(rows), 1),
    }
    out_path = args.out if args.out is not None else args.pb_step_scores.parent / "eval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
