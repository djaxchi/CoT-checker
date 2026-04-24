#!/usr/bin/env python3
"""Evaluate trained probes on ProcessBench GSM8K encoded latents.

Run encode_processbench.py first to produce the .npz input.

Two evaluation modes:

1. Per-step binary classification (our primary metric, comparable to Math-Shepherd):
   - Treats each encoded step as an independent binary classification instance.
   - Reports accuracy, F1 correct, F1 incorrect, Macro F1 at multiple thresholds.
   - NOTE: step-level distribution is heavily skewed (mostly correct steps), so
     macro F1 is more informative than accuracy here.

2. ProcessBench F1 (for direct comparison with published PRM results):
   - For each solution, predict the first step classified as incorrect (P(correct)
     below threshold). If all steps pass, predict "all correct" (-1).
   - Accuracy on correct solutions  = fraction where probe predicts "all correct"
   - Accuracy on incorrect solutions = fraction where probe's first-error index
     matches the ground-truth label exactly
   - ProcessBench F1 = harmonic mean of the two accuracies
   - This is the metric reported by Math-Shepherd-PRM-7B (47.9) and
     Qwen2.5-Math-7B-PRM800K (68.2) in the ProcessBench paper (arXiv:2412.06559).

Usage:
    # Single checkpoint
    python scripts/eval_processbench.py \\
        --latents results/probe_data/processbench_gsm8k.npz \\
        --checkpoints results/probes/probe_seed42.pt \\
        --device cuda

    # Multiple checkpoints (aggregated)
    python scripts/eval_processbench.py \\
        --latents results/probe_data/processbench_gsm8k.npz \\
        --checkpoints results/probes/probe_seed4{2,3,4,5}.pt \\
        --linear-checkpoints results/probes/linear_seed4{2,3,4,5}.pt \\
        --device cuda
"""

import argparse
import sys
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.probes.classifier import StepCorrectnessClassifier


# ---------------------------------------------------------------------------
# Linear probe loader (mirrors experiment_linear_probe.py's save format)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(input_dim=ckpt["input_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


def load_probe(path: str, device: str):
    """Auto-detect MLP vs linear probe from checkpoint keys."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in ckpt and "config" in ckpt:
        return StepCorrectnessClassifier.load(path, device=device), "MLP"
    elif "state_dict" in ckpt and "input_dim" in ckpt:
        return LinearProbe.load(path, device=device), "Linear"
    else:
        raise ValueError(f"Unrecognised checkpoint format in {path}: keys={list(ckpt.keys())}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_p_correct(model, latents: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    """Run probe on all encoded steps, return P(correct) array."""
    model.eval()
    h = torch.from_numpy(latents.astype(np.float32))
    results = []
    for i in range(0, len(h), batch_size):
        x = h[i:i + batch_size].to(device)
        logits = model(x).squeeze(-1)
        results.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(results)


# ---------------------------------------------------------------------------
# Metric 1: per-step binary classification
# ---------------------------------------------------------------------------

def step_metrics(p_correct: np.ndarray, step_labels: np.ndarray, threshold: float):
    preds = (p_correct >= threshold).astype(np.int64)
    labels = step_labels.astype(np.int64)
    acc = (preds == labels).mean()
    out = {}
    for cls, name in [(1, "correct"), (0, "incorrect")]:
        tp = ((preds == cls) & (labels == cls)).sum()
        fp = ((preds == cls) & (labels != cls)).sum()
        fn = ((preds != cls) & (labels == cls)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = {"precision": prec, "recall": rec, "f1": f1}
    macro_f1 = (out["correct"]["f1"] + out["incorrect"]["f1"]) / 2
    return acc, out, macro_f1


# ---------------------------------------------------------------------------
# Metric 2: ProcessBench F1
# ---------------------------------------------------------------------------

def processbench_f1(
    p_correct: np.ndarray,
    solution_ids: np.ndarray,
    step_positions: np.ndarray,
    solution_labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Compute ProcessBench F1 at a given threshold.

    For each solution, scan steps left-to-right (sorted by step_positions).
    Prediction = index of first step where P(correct) < threshold, or -1.
    Compare to ground-truth solution_labels.

    Returns (acc_correct, acc_incorrect, f1).
    """
    n_solutions = len(solution_labels)
    correct_hits = []   # for solutions where label == -1
    incorrect_hits = [] # for solutions where label >= 0

    for sol_idx in range(n_solutions):
        mask = (solution_ids == sol_idx)
        if not mask.any():
            continue

        positions = step_positions[mask]
        scores = p_correct[mask]
        # Sort by position (should already be ordered, but be safe)
        order = np.argsort(positions)
        scores_ordered = scores[order]

        # First step below threshold = predicted error position
        below = np.where(scores_ordered < threshold)[0]
        pred_label = int(below[0]) if len(below) > 0 else -1

        gt_label = int(solution_labels[sol_idx])

        if gt_label == -1:
            correct_hits.append(pred_label == -1)
        else:
            incorrect_hits.append(pred_label == gt_label)

    acc_correct = float(np.mean(correct_hits)) if correct_hits else 0.0
    acc_incorrect = float(np.mean(incorrect_hits)) if incorrect_hits else 0.0
    if acc_correct + acc_incorrect > 0:
        f1 = 2 * acc_correct * acc_incorrect / (acc_correct + acc_incorrect)
    else:
        f1 = 0.0
    return acc_correct, acc_incorrect, f1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_step_sweep(p_correct, step_labels, majority):
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"\n  Per-step binary classification (N={len(step_labels):,} steps):")
    print(f"  Majority baseline: {majority:.1%}")
    print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'F1 correct':>12}  {'F1 incorrect':>13}  {'Macro F1':>10}")
    best_macro, best_t = 0.0, 0.5
    for t in thresholds:
        acc, res, macro = step_metrics(p_correct, step_labels, t)
        marker = " <--" if macro > best_macro else ""
        if macro > best_macro:
            best_macro = macro
            best_t = t
        print(
            f"  {t:>10.1f}  {acc*100:>9.2f}%"
            f"  {res['correct']['f1']:>12.3f}"
            f"  {res['incorrect']['f1']:>13.3f}"
            f"  {macro:>10.3f}{marker}"
        )
    return best_t, best_macro


def print_pb_sweep(p_correct, solution_ids, step_positions, solution_labels):
    n_correct_sols = int((solution_labels == -1).sum())
    n_incorrect_sols = int((solution_labels != -1).sum())
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"\n  ProcessBench F1 (solution-level):")
    print(f"  Correct solutions: {n_correct_sols}  |  Incorrect solutions: {n_incorrect_sols}")
    print(f"  {'Threshold':>10}  {'Acc(correct sols)':>18}  {'Acc(error sols)':>16}  {'PB-F1':>8}")
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        ac, ai, f1 = processbench_f1(p_correct, solution_ids, step_positions, solution_labels, t)
        marker = " <--" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        print(
            f"  {t:>10.1f}  {ac*100:>17.1f}%  {ai*100:>15.1f}%  {f1*100:>7.1f}{marker}"
        )
    return best_t, best_f1


def run_checkpoint(ckpt_path: str, latents, step_labels, solution_ids, step_positions, solution_labels, majority, device):
    model, model_type = load_probe(ckpt_path, device)
    name = Path(ckpt_path).name

    print(f"\n{'='*64}")
    print(f"  Checkpoint : {name}  [{model_type}]")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    p_correct = get_p_correct(model, latents, device)

    best_step_t, best_step_macro = print_step_sweep(p_correct, step_labels, majority)
    best_pb_t, best_pb_f1 = print_pb_sweep(p_correct, solution_ids, step_positions, solution_labels)

    return {
        "name": name,
        "model_type": model_type,
        "best_step_macro": best_step_macro,
        "best_step_t": best_step_t,
        "best_pb_f1": best_pb_f1,
        "best_pb_t": best_pb_t,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True,
                   help="Path to .npz from encode_processbench.py")
    p.add_argument("--checkpoints", nargs="+", default=[],
                   help="MLP probe checkpoint(s) to evaluate")
    p.add_argument("--linear-checkpoints", nargs="+", default=[],
                   help="Linear probe checkpoint(s) to evaluate")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading latents from {args.latents}...")
    d = np.load(args.latents)
    latents = d["latents"].astype(np.float32)
    step_labels = d["step_labels"]
    solution_ids = d["solution_ids"]
    step_positions = d["step_positions"]
    solution_labels = d["solution_labels"]

    n_total = len(step_labels)
    n_step_correct = int((step_labels == 1).sum())
    majority = max(n_step_correct, n_total - n_step_correct) / n_total

    print(f"  Steps: {n_total}  |  Correct: {n_step_correct} ({n_step_correct/n_total:.1%})  |  Incorrect: {n_total-n_step_correct} ({(n_total-n_step_correct)/n_total:.1%})")
    print(f"  Solutions: {len(solution_labels)}  |  Majority baseline (steps): {majority:.1%}")

    all_ckpts = list(args.checkpoints) + list(args.linear_checkpoints)
    if not all_ckpts:
        print("No checkpoints provided. Use --checkpoints or --linear-checkpoints.")
        return

    rows = []
    for ckpt in all_ckpts:
        row = run_checkpoint(
            ckpt, latents, step_labels, solution_ids, step_positions, solution_labels, majority, args.device
        )
        rows.append(row)

    if len(rows) > 1:
        print(f"\n{'='*64}")
        print("  AGGREGATE SUMMARY (best threshold per metric per checkpoint)")
        print(f"  {'Checkpoint':<30}  {'Type':>8}  {'Step Macro F1':>14}  {'PB-F1 (%)':>10}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*14}  {'-'*10}")
        for r in rows:
            print(
                f"  {r['name']:<30}  {r['model_type']:>8}"
                f"  {r['best_step_macro']:>14.3f}"
                f"  {r['best_pb_f1']*100:>9.1f}%"
            )

        # Aggregate by model type
        for mtype in ["MLP", "Linear"]:
            group = [r for r in rows if r["model_type"] == mtype]
            if len(group) < 2:
                continue
            step_vals = [r["best_step_macro"] for r in group]
            pb_vals = [r["best_pb_f1"] * 100 for r in group]
            print(f"\n  {mtype} aggregate ({len(group)} seeds):")
            print(f"    Step Macro F1 : {statistics.mean(step_vals):.3f} +/- {statistics.stdev(step_vals):.3f}")
            print(f"    PB-F1         : {statistics.mean(pb_vals):.1f}% +/- {statistics.stdev(pb_vals):.1f}%")

        print(f"\n  Reference results from ProcessBench paper (arXiv:2412.06559):")
        print(f"  {'Method':<30}  {'Type':>8}  {'PB-F1 GSM8K (%)':>16}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*16}")
        print(f"  {'Math-Shepherd-PRM-7B':<30}  {'PRM 7B':>8}  {'47.9':>15}%")
        print(f"  {'Qwen2.5-Math-7B-PRM800K':<30}  {'PRM 7B':>8}  {'68.2':>15}%")
        print(f"  {'ActPRM (SOTA, Apr 2025)':<30}  {'PRM':>8}  {'~75.0':>15}%")


if __name__ == "__main__":
    main()
