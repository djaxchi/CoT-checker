#!/usr/bin/env python3
"""Model-size ablation: train linear probe and evaluate on MS + ProcessBench.

Loads pre-extracted .npz files (from model_size_ablation_extract.py) and:
  1. Applies optional L2 normalisation via --repr {raw,l2}.
  2. Trains a sklearn LogisticRegression on Math-Shepherd train hidden states.
  3. Selects threshold on Math-Shepherd eval (maximises Macro-F1).
  4. Evaluates the same probe + threshold on ProcessBench (OOD transfer).
  5. Reports Macro-F1, AUROC, AUPRC, PPR, confusion matrix, collapse flag.
  6. Saves metrics JSON and per-example score arrays for downstream plots.

Run this script twice per model -- once with --repr raw, once with --repr l2 --
to test whether L2 normalisation helps or hides a magnitude signal.

Outputs (repr-tagged to avoid collisions):
  {output_dir}/metrics_{tag}_{repr}.json
  {output_dir}/scores_{tag}_{repr}.npz
      ms_eval_scores   float32  (N,)
      ms_eval_labels   int8     (N,)
      pb_scores        float32  (N,)
      pb_labels        int8     (N,)
      probe_coef       float32  (1, d)
      probe_intercept  float32  (1,)

Usage:
    python scripts/model_size_ablation_probe.py \\
        --model-tag 7b --repr l2 \\
        --data-dir  $SCRATCH/cot-checker/ms_ablation \\
        --output-dir $SCRATCH/cot-checker/ms_ablation/probes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ms_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    h = d["hidden_states"].astype(np.float32)
    y = d["labels"].astype(np.int32)
    return h, y


def load_pb_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    h = d["hidden_states"].astype(np.float32)
    y = d["step_labels"].astype(np.int32)
    return h, y


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    dataset_name: str,
) -> dict:
    """Return metrics dict at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)

    pos_rate   = float(y_pred.mean())
    n_pos      = int((y_true == 1).sum())
    n_neg      = int((y_true == 0).sum())
    total      = len(y_true)
    pred_pos   = int((y_pred == 1).sum())
    pred_neg   = total - pred_pos

    macro_f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_cor     = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_inc     = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true, y_score))
    except ValueError:
        auprc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # Collapse: probe always predicts the same class
    collapse = pred_pos == 0 or pred_neg == 0
    collapse_class = None
    if pred_pos == 0:
        collapse_class = "all_incorrect"
    elif pred_neg == 0:
        collapse_class = "all_correct"

    return {
        "dataset":       dataset_name,
        "n_correct":     n_pos,
        "n_incorrect":   n_neg,
        "threshold":     threshold,
        "macro_f1":      round(macro_f1, 4),
        "f1_correct":    round(f1_cor,   4),
        "f1_incorrect":  round(f1_inc,   4),
        "auroc":         round(auroc,    4),
        "auprc":         round(auprc,    4),
        "pos_pred_rate": round(pos_rate, 4),
        "confusion_matrix": cm,
        "collapse":      collapse,
        "collapse_class": collapse_class,
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (best_threshold, best_macro_f1) by sweeping thresholds on y_true."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if macro > best_f1:
            best_f1 = macro
            best_t  = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train linear probe for model-size ablation")
    p.add_argument("--model-tag",  required=True, help="Model tag, e.g. 7b")
    p.add_argument("--repr",       default="l2", choices=["raw", "l2"],
                   help="'l2' L2-normalises hidden states before probing (tests directional "
                        "geometry only). 'raw' keeps magnitude intact (may carry scale signal).")
    p.add_argument("--data-dir",   required=True, help="Directory with pre-extracted .npz files")
    p.add_argument("--output-dir", required=True, help="Directory to write metrics + scores")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--lr-C",       type=float, default=1.0, help="Inverse regularisation for LR")
    p.add_argument("--max-iter",   type=int,   default=1000)
    return p.parse_args()


def apply_repr(h: np.ndarray, repr_mode: str) -> np.ndarray:
    """Apply representation transform: 'raw' is identity, 'l2' unit-normalises rows."""
    if repr_mode == "l2":
        return sk_normalize(h, norm="l2")
    return h


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag  = args.model_tag
    repr_mode = args.repr
    suffix = f"{tag}_{repr_mode}"

    print(f"\n[{tag}  repr={repr_mode}]")

    # --- Load data ---
    print(f"Loading Math-Shepherd train …")
    h_train, y_train = load_ms_npz(str(data_dir / f"ms_train_{tag}.npz"))
    h_train = apply_repr(h_train, repr_mode)
    print(f"  Shape: {h_train.shape}  correct: {(y_train==1).sum():,}  incorrect: {(y_train==0).sum():,}")

    print(f"Loading Math-Shepherd eval …")
    h_eval, y_eval = load_ms_npz(str(data_dir / f"ms_eval_{tag}.npz"))
    h_eval = apply_repr(h_eval, repr_mode)
    print(f"  Shape: {h_eval.shape}  correct: {(y_eval==1).sum():,}  incorrect: {(y_eval==0).sum():,}")

    print(f"Loading ProcessBench …")
    h_pb, y_pb = load_pb_npz(str(data_dir / f"pb_{tag}.npz"))
    h_pb = apply_repr(h_pb, repr_mode)
    print(f"  Shape: {h_pb.shape}  correct: {(y_pb==1).sum():,}  incorrect: {(y_pb==0).sum():,}")

    # --- Train linear probe ---
    print(f"\nTraining LogisticRegression (C={args.lr_C}) on {len(y_train):,} steps …")
    clf = LogisticRegression(
        C=args.lr_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=args.max_iter,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(h_train, y_train)
    print(f"  Converged: {clf.n_iter_[0] < args.max_iter}  iterations: {clf.n_iter_[0]}")

    # P(correct) = P(class=1)
    ms_eval_scores = clf.predict_proba(h_eval)[:, 1].astype(np.float32)
    pb_scores      = clf.predict_proba(h_pb)[:,   1].astype(np.float32)

    # --- Threshold selection on MS eval ---
    print(f"\n[{tag}] Threshold sweep on MS eval …")
    best_t, best_ms_f1 = threshold_sweep(y_eval, ms_eval_scores)
    print(f"  Best threshold: {best_t:.3f}  Macro-F1: {best_ms_f1:.4f}")

    # --- Compute all metrics ---
    ms_metrics = compute_metrics(y_eval, ms_eval_scores, best_t, "math_shepherd_eval")
    pb_metrics = compute_metrics(y_pb,   pb_scores,      best_t, "processbench_gsm8k")

    print(f"\n[{suffix}] Math-Shepherd eval:")
    print(f"  Macro-F1 : {ms_metrics['macro_f1']:.4f}")
    print(f"  AUROC    : {ms_metrics['auroc']:.4f}")
    print(f"  AUPRC    : {ms_metrics['auprc']:.4f}")
    print(f"  PPR      : {ms_metrics['pos_pred_rate']:.4f}")

    print(f"\n[{suffix}] ProcessBench (OOD transfer, threshold={best_t:.3f}):")
    print(f"  Macro-F1 : {pb_metrics['macro_f1']:.4f}")
    print(f"  AUROC    : {pb_metrics['auroc']:.4f}")
    print(f"  AUPRC    : {pb_metrics['auprc']:.4f}")
    print(f"  PPR      : {pb_metrics['pos_pred_rate']:.4f}")
    if pb_metrics["collapse"]:
        print(f"  ** COLLAPSE: {pb_metrics['collapse_class']} **")

    # --- Save metrics JSON ---
    metrics = {
        "model_tag":           tag,
        "repr":                repr_mode,
        "hidden_dim":          int(h_train.shape[1]),
        "n_train":             int(len(y_train)),
        "n_eval":              int(len(y_eval)),
        "n_pb":                int(len(y_pb)),
        "best_threshold":      best_t,
        "math_shepherd_eval":  ms_metrics,
        "processbench":        pb_metrics,
    }
    metrics_path = out_dir / f"metrics_{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved metrics → {metrics_path}")

    # --- Save scores for downstream plots ---
    scores_path = out_dir / f"scores_{suffix}.npz"
    np.savez_compressed(
        scores_path,
        ms_eval_scores   = ms_eval_scores,
        ms_eval_labels   = y_eval.astype(np.int8),
        pb_scores        = pb_scores,
        pb_labels        = y_pb.astype(np.int8),
        probe_coef       = clf.coef_.astype(np.float32),
        probe_intercept  = clf.intercept_.astype(np.float32),
    )
    print(f"  Saved scores   → {scores_path}")

    print(f"\n[{suffix}] Done.")


if __name__ == "__main__":
    main()
