#!/usr/bin/env python3
"""Inspect ProcessBench GSM8K step/error distribution from the raw jsonl.

Answers: for how many incorrect solutions does h_{k+1} exist
(i.e., the error step is NOT the last step of the full solution)?

Usage:
    python scripts/inspect_pb_step_distribution.py \
        --data-file $SCRATCH/cot-checker/processbench/processbench_gsm8k.jsonl
"""

import argparse
import json
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True)
    args = p.parse_args()

    correct, incorrect = [], []
    error_pos_dist = Counter()   # error_step_position -> count
    steps_after_err = Counter()  # n_steps_after_error -> count

    with open(args.data_file) as f:
        for line in f:
            row = json.loads(line)
            label = row["label"]
            n_steps = len(row["steps"])
            if label == -1:
                correct.append(n_steps)
            else:
                incorrect.append(n_steps)
                error_pos_dist[label] += 1
                after = n_steps - label - 1   # steps after the error step
                steps_after_err[after] += 1

    n_c = len(correct)
    n_i = len(incorrect)
    n_total = n_c + n_i

    print(f"ProcessBench GSM8K summary")
    print(f"  Total solutions  : {n_total}")
    print(f"  Correct  (-1)    : {n_c}  ({n_c/n_total:.1%})")
    print(f"  Incorrect        : {n_i}  ({n_i/n_total:.1%})")
    print()

    if correct:
        avg_c = sum(correct) / len(correct)
        print(f"  Correct solutions  avg_steps={avg_c:.2f}  "
              f"min={min(correct)}  max={max(correct)}")
    if incorrect:
        avg_i = sum(incorrect) / len(incorrect)
        print(f"  Incorrect solutions avg_steps={avg_i:.2f}  "
              f"min={min(incorrect)}  max={max(incorrect)}")
    print()

    print("  Error step position distribution (0-indexed):")
    for pos in sorted(error_pos_dist):
        print(f"    error at step {pos}: {error_pos_dist[pos]:>4} solutions")
    print()

    print("  Steps AFTER the error step:")
    usable = sum(cnt for after, cnt in steps_after_err.items() if after > 0)
    for after in sorted(steps_after_err):
        marker = "  <- h_{k+1} exists" if after > 0 else "  <- error is last step (PTB cannot score)"
        print(f"    {after} steps after error: {steps_after_err[after]:>4} solutions{marker}")
    print()
    print(f"  --> Usable for PTB recon-error (>=1 step after error): "
          f"{usable}/{n_i} = {usable/max(n_i,1):.1%}")
    print()

    # Also show how many encoded steps the new npz would add vs current
    current_encoded = sum(label + 1 for label, _ in
                          [(row["label"], row["steps"])
                           for row in (json.loads(l) for l in open(args.data_file))
                           if row["label"] != -1])
    # recount
    extra = 0
    with open(args.data_file) as f:
        for line in f:
            row = json.loads(line)
            if row["label"] != -1:
                extra += len(row["steps"]) - (row["label"] + 1)
    print(f"  Current encoding (truncated): ~{sum(incorrect)} steps from incorrect solutions")
    print(f"  Extra steps added by full encoding: {extra}")


if __name__ == "__main__":
    main()
