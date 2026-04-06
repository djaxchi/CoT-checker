#!/usr/bin/env python3
"""Label GSM8K-Aug steps with symbolic arithmetic verification.

Downloads Miaow-Lab/SSAE-Dataset (gsm8k config, train split) and labels
each reasoning step using symbolic_step_judge() -- the same approach the
paper uses: check whether every arithmetic equation in the step is correct.

No model is needed. Labeling is pure Python and runs on CPU in seconds.

Output JSONL (one record per problem):
    {"question": "...", "steps": ["...", ...], "labels": [1, 0, ...]}

Labels: 1 = step is arithmetically correct (or has no verifiable equations)
        0 = step contains at least one arithmetic error

The script targets the paper's majority baseline (70.49% correct steps).
It processes at least --min-problems problems, then keeps going until the
correct-step ratio is within --tolerance of --target-baseline, or the
dataset is exhausted.

Usage:
    python scripts/label_gsm8k_aug.py
    python scripts/label_gsm8k_aug.py --min-problems 10000 --output results/labeled_data/gsm8k_aug_10k.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k_dataset import split_answer_into_steps, symbolic_step_judge

# Match the GSM8K final-answer line: "#### 42" or "#### 42."
_FINAL_ANSWER_RE = re.compile(r"^#{2,}\s*[-+]?\d")


def clean_steps(raw_steps: list[str]) -> list[str]:
    """Remove the final '#### <answer>' line and blank steps."""
    return [s.strip() for s in raw_steps if s.strip() and not _FINAL_ANSWER_RE.match(s.strip())]


def label_problem(question: str, answer: str) -> dict | None:
    """Parse and label all steps in one GSM8K-Aug problem.

    Returns a dict with question, steps, labels -- or None if no steps found.
    """
    raw = split_answer_into_steps(answer)
    steps = clean_steps(raw)
    if not steps:
        return None

    labels = [symbolic_step_judge(step) for step in steps]
    return {"question": question, "steps": steps, "labels": labels}


def parse_args():
    p = argparse.ArgumentParser(description="Label GSM8K-Aug steps with symbolic verifier")
    p.add_argument(
        "--output",
        default="results/labeled_data/gsm8k_aug_half.jsonl",
        help="Output JSONL path",
    )
    p.add_argument(
        "--min-problems",
        type=int,
        default=192_310,
        help="Minimum problems to process before checking baseline (default: half of 384,620)",
    )
    p.add_argument(
        "--target-baseline",
        type=float,
        default=0.7049,
        help="Paper's majority baseline: fraction of correct steps (default: 0.7049)",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.002,
        help="Acceptable deviation from target baseline (default: 0.2 pp)",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="Dataset split to use (default: train)",
    )
    p.add_argument(
        "--dataset",
        default="Miaow-Lab/SSAE-Dataset",
        help="HuggingFace dataset ID",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} (gsm8k / {args.split}) …")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, "gsm8k", split=args.split, streaming=True)

    n_problems = 0
    n_skipped = 0
    total_steps = 0
    total_correct = 0
    total_incorrect = 0

    print(f"Target baseline  : {args.target_baseline:.2%}  (paper: 70.49%)")
    print(f"Minimum problems : {args.min_problems:,}")
    print(f"Tolerance        : ±{args.tolerance:.2%}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Labeling"):
            record = label_problem(row["question"], row["answer"])
            if record is None:
                n_skipped += 1
                continue

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_problems += 1

            n_steps = len(record["steps"])
            n_pos = sum(record["labels"])
            total_steps += n_steps
            total_correct += n_pos
            total_incorrect += n_steps - n_pos

            # After the minimum, stop as soon as the baseline is on target
            if n_problems >= args.min_problems and total_steps > 0:
                current_correct_ratio = total_correct / total_steps
                if abs(current_correct_ratio - args.target_baseline) <= args.tolerance:
                    break

    correct_ratio = total_correct / total_steps if total_steps else 0
    majority_baseline = max(total_correct, total_incorrect) / total_steps if total_steps else 0
    on_target = abs(correct_ratio - args.target_baseline) <= args.tolerance

    print(f"\nProblems written : {n_problems:,}")
    print(f"Problems skipped : {n_skipped:,}  (no parseable steps)")
    print(f"Total steps      : {total_steps:,}")
    print(f"  Correct (+)    : {total_correct:,}  ({correct_ratio:.2%})")
    print(f"  Incorrect (-)  : {total_incorrect:,}  ({1 - correct_ratio:.2%})")
    print(f"Majority baseline: {majority_baseline:.2%}  (target: {args.target_baseline:.2%})  {'OK' if on_target else 'OFF TARGET'}")
    print(f"\nSaved to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
