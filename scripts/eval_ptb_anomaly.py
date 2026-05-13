#!/usr/bin/env python3
"""PTB reconstruction-error anomaly detection on ProcessBench.

Tests the hypothesis: steps where PTB poorly reconstructs the transition
(high ||delta_h_hat - delta_h_actual||) are the error-introducing steps.

For each step k in a solution:
    delta_h_actual  = h[k+1] - h[k]
    delta_h_hat     = PTB.decode(PTB.encode(h[k]))
    recon_err[k]    = ||delta_h_hat - delta_h_actual||  (L2 norm)

The step with the highest recon_err is the PTB's predicted first-error step.
The last step of each solution has no successor and is excluded.

Compares against a dense-h linear probe (--probe-ckpt) if provided.

Usage:
    python scripts/eval_ptb_anomaly.py \
        --pb-dense   $SCRATCH/cot-checker/processbench/processbench_dense_gsm8k.npz \
        --ptb-ckpt   $STORE/results/checkpoints/ptb_robust/no_l1/best.pt \
        --probe-ckpt $STORE/results/ptb_robust_probes/dense_h_linear_probe_seed42.pt \
        --n-solutions 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.saes.ptb import PredictiveTransitionBottleneck


# ---------------------------------------------------------------------------
# Minimal probe loader (mirrors the class in eval_predictive_transition_probes)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pb_solutions(pb_path: str, n_solutions: int) -> dict[int, dict]:
    """Load ProcessBench dense npz, return dict keyed by solution_id."""
    d = np.load(pb_path)
    h          = d["latents"].astype(np.float32)
    sol_ids    = d["solution_ids"].astype(int)
    step_pos   = d["step_positions"].astype(int)
    step_lbls  = d["step_labels"].astype(int)    # 1=correct, 0=wrong
    sol_lbls   = d["solution_labels"].astype(int) # 1=clean, 0=has error

    # Group rows by solution
    raw: dict[int, list] = {}
    for i in range(len(h)):
        sid = sol_ids[i]
        if sid not in raw:
            raw[sid] = []
        raw[sid].append({
            "h": h[i], "step_pos": step_pos[i],
            "step_lbl": step_lbls[i], "sol_lbl": sol_lbls[i],
        })

    solutions: dict[int, dict] = {}
    for sid in sorted(raw.keys())[:n_solutions]:
        rows = sorted(raw[sid], key=lambda r: r["step_pos"])
        solutions[sid] = {
            "h":          np.stack([r["h"]        for r in rows]),
            "step_pos":   np.array([r["step_pos"] for r in rows], dtype=int),
            "step_lbls":  np.array([r["step_lbl"] for r in rows], dtype=int),
            "sol_lbl":    rows[0]["sol_lbl"],
        }
    return solutions


# ---------------------------------------------------------------------------
# Per-step computations
# ---------------------------------------------------------------------------

@torch.no_grad()
def recon_errors(ptb: PredictiveTransitionBottleneck, h: np.ndarray) -> np.ndarray:
    """Return per-step L2 reconstruction error. Last step = NaN (no successor)."""
    n = len(h)
    errs = np.full(n, np.nan)
    h_t = torch.from_numpy(h)
    for k in range(n - 1):
        h_k    = h_t[k].unsqueeze(0)
        h_next = h_t[k + 1].unsqueeze(0)
        delta_actual = h_next - h_k
        delta_hat    = ptb.decode(ptb.encode(h_k))
        errs[k]      = (delta_hat - delta_actual).norm().item()
    return errs


@torch.no_grad()
def probe_p_correct(probe: LinearProbe, h: np.ndarray) -> np.ndarray:
    """Return per-step P(correct) from dense linear probe."""
    h_t = torch.from_numpy(h)
    return torch.sigmoid(probe(h_t).squeeze(-1)).numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def first_error_step(step_lbls: np.ndarray) -> int | None:
    """Index of first wrong step (label=0), or None if solution is clean."""
    wrong = np.where(step_lbls == 0)[0]
    return int(wrong[0]) if len(wrong) > 0 else None


def evaluate_anomaly_method(
    solutions: dict[int, dict],
    scores: dict[int, np.ndarray],   # per-solution, per-step "incorrectness" scores
    thresholds: list[float],
) -> list[dict]:
    """Compute PB metrics across thresholds.

    For clean solutions: correct if max(score) < threshold.
    For error solutions: predicted first error = first step where score > threshold.
    Correct if predicted first error == true first error.
    Also reports "any error detected" separately from "first error located correctly".
    """
    results = []
    for thr in thresholds:
        n_clean, n_clean_ok = 0, 0
        n_err, n_err_detected, n_err_located = 0, 0, 0

        for sid, s in solutions.items():
            errs     = scores[sid]
            lbls     = s["step_lbls"]
            has_err  = (s["sol_lbl"] == 0)
            valid    = ~np.isnan(errs)

            if not has_err:
                n_clean += 1
                if not np.any(errs[valid] > thr):
                    n_clean_ok += 1
            else:
                n_err += 1
                true_fe = first_error_step(lbls)
                # First step (by position) where score > threshold
                pred_fe = None
                for k in range(len(errs)):
                    if not np.isnan(errs[k]) and errs[k] > thr:
                        pred_fe = k
                        break
                if pred_fe is not None:
                    n_err_detected += 1
                if pred_fe == true_fe:
                    n_err_located += 1

        acc_c   = n_clean_ok   / max(n_clean, 1)
        acc_e   = n_err_located / max(n_err, 1)
        det_e   = n_err_detected / max(n_err, 1)
        pb_f1   = (2 * acc_c * acc_e / (acc_c + acc_e)) if (acc_c + acc_e) > 0 else 0.0
        results.append({
            "threshold": thr,
            "acc_clean": acc_c, "acc_error_located": acc_e,
            "error_detected": det_e, "pb_f1": pb_f1,
            "n_clean": n_clean, "n_err": n_err,
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pb-dense",    required=True, help="processbench_dense_gsm8k.npz")
    p.add_argument("--ptb-ckpt",    required=True, help="PTB best.pt checkpoint")
    p.add_argument("--probe-ckpt",  default=None,  help="Dense probe .pt (optional)")
    p.add_argument("--n-solutions", type=int, default=50)
    p.add_argument("--device",      default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading ProcessBench data ({args.n_solutions} solutions) ...")
    solutions = load_pb_solutions(args.pb_dense, args.n_solutions)
    n_err   = sum(1 for s in solutions.values() if s["sol_lbl"] == 0)
    n_clean = len(solutions) - n_err
    print(f"  {len(solutions)} solutions  |  {n_err} with errors  |  {n_clean} clean")

    print(f"\nLoading PTB from {args.ptb_ckpt} ...")
    ptb = PredictiveTransitionBottleneck.from_checkpoint(args.ptb_ckpt, device=args.device)
    ptb.eval()

    probe = None
    if args.probe_ckpt and Path(args.probe_ckpt).exists():
        print(f"Loading dense probe from {args.probe_ckpt} ...")
        ckpt  = torch.load(args.probe_ckpt, map_location=args.device, weights_only=False)
        probe = LinearProbe(ckpt["input_dim"])
        probe.load_state_dict(ckpt["state_dict"])
        probe.eval()

    # -----------------------------------------------------------------------
    # Compute per-solution scores
    # -----------------------------------------------------------------------
    ptb_err_scores:   dict[int, np.ndarray] = {}
    probe_inc_scores: dict[int, np.ndarray] = {}

    for sid, s in solutions.items():
        ptb_err_scores[sid] = recon_errors(ptb, s["h"])
        if probe is not None:
            probe_inc_scores[sid] = 1.0 - probe_p_correct(probe, s["h"])

    # -----------------------------------------------------------------------
    # Per-solution step tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  PER-SOLUTION STEP DETAIL")
    print("=" * 72)

    for sol_idx, (sid, s) in enumerate(solutions.items()):
        has_err  = (s["sol_lbl"] == 0)
        true_fe  = first_error_step(s["step_lbls"])
        errs     = ptb_err_scores[sid]
        valid    = errs[~np.isnan(errs)]
        pred_ptb = int(np.nanargmax(errs)) if len(valid) > 0 else None

        # PTB verdict
        if has_err:
            ptb_verdict = "HIT" if pred_ptb == true_fe else f"MISS (predicted s{pred_ptb}, true s{true_fe})"
        else:
            ptb_verdict = "clean (no true error)"

        print(f"\nSolution {sol_idx+1:>3}  id={sid}  "
              f"has_error={has_err}  steps={len(s['h'])}  "
              f"PTB: {ptb_verdict}")

        hdr  = f"  {'s':>2}  {'label':>8}  {'recon_err':>10}"
        if probe is not None:
            hdr += f"  {'probe_inc':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for k in range(len(s["h"])):
            lbl_str = "WRONG" if s["step_lbls"][k] == 0 else "correct"
            err_str = f"{errs[k]:.4f}" if not np.isnan(errs[k]) else "    --"

            flags = []
            if k == pred_ptb:
                flags.append("← PTB max")
            if k == true_fe:
                flags.append("← TRUE first error")

            row = f"  {k:>2}  {lbl_str:>8}  {err_str:>10}"
            if probe is not None:
                row += f"  {probe_inc_scores[sid][k]:>9.4f}"
            row += ("  " + "  ".join(flags)) if flags else ""
            print(row)

    # -----------------------------------------------------------------------
    # Aggregate: reconstruction error distribution by step label
    # -----------------------------------------------------------------------
    correct_errs, wrong_errs = [], []
    for sid, s in solutions.items():
        for k, (err, lbl) in enumerate(zip(ptb_err_scores[sid], s["step_lbls"])):
            if not np.isnan(err):
                (correct_errs if lbl == 1 else wrong_errs).append(err)

    correct_errs = np.array(correct_errs)
    wrong_errs   = np.array(wrong_errs)

    print("\n" + "=" * 72)
    print("  RECONSTRUCTION ERROR BY STEP LABEL")
    print("=" * 72)
    print(f"\n  Correct steps  n={len(correct_errs):>5} :  "
          f"mean={correct_errs.mean():.4f}  std={correct_errs.std():.4f}  "
          f"median={np.median(correct_errs):.4f}  p95={np.percentile(correct_errs,95):.4f}")
    if len(wrong_errs) > 0:
        print(f"  Wrong   steps  n={len(wrong_errs):>5} :  "
              f"mean={wrong_errs.mean():.4f}  std={wrong_errs.std():.4f}  "
              f"median={np.median(wrong_errs):.4f}  p95={np.percentile(wrong_errs,95):.4f}")
        ratio = wrong_errs.mean() / max(correct_errs.mean(), 1e-8)
        print(f"\n  Mean ratio wrong/correct: {ratio:.3f}")
        if ratio > 1.15:
            print("  --> Wrong steps have higher recon error  [hypothesis SUPPORTED]")
        elif ratio < 0.85:
            print("  --> Wrong steps have LOWER recon error   [hypothesis REFUTED]")
        else:
            print("  --> No meaningful difference             [hypothesis NOT supported]")
    else:
        print("  (no wrong steps with predecessors in this subset)")

    # -----------------------------------------------------------------------
    # PB-F1 across thresholds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  PB METRICS ACROSS THRESHOLDS")
    print("=" * 72)

    ptb_thresholds = np.percentile(correct_errs, [50, 75, 90, 95, 99]).tolist() if len(correct_errs) > 0 else [0.5, 1.0, 1.5, 2.0, 3.0]

    print(f"\n  PTB reconstruction error  (thresholds = percentiles of correct-step distribution)")
    print(f"  {'threshold':>10}  {'acc_clean':>9}  {'err_detect':>10}  {'err_locate':>10}  {'PB-F1':>7}")
    print("  " + "-" * 55)
    ptb_results = evaluate_anomaly_method(solutions, ptb_err_scores, ptb_thresholds)
    for r in ptb_results:
        print(f"  {r['threshold']:>10.4f}  {r['acc_clean']:>9.3f}  "
              f"{r['error_detected']:>10.3f}  {r['acc_error_located']:>10.3f}  {r['pb_f1']:>7.3f}")

    if probe is not None:
        probe_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print(f"\n  Dense probe (1 - p_correct)  [trained on MS, transferred to PB]")
        print(f"  {'threshold':>10}  {'acc_clean':>9}  {'err_detect':>10}  {'err_locate':>10}  {'PB-F1':>7}")
        print("  " + "-" * 55)
        probe_results = evaluate_anomaly_method(solutions, probe_inc_scores, probe_thresholds)
        for r in probe_results:
            print(f"  {r['threshold']:>10.2f}  {r['acc_clean']:>9.3f}  "
                  f"{r['error_detected']:>10.3f}  {r['acc_error_located']:>10.3f}  {r['pb_f1']:>7.3f}")

    # Simple hit-rate summary
    print("\n" + "=" * 72)
    print("  FIRST-ERROR LOCALISATION: argmax(recon_err) HIT RATE")
    print("=" * 72)
    hits, total = 0, 0
    for sid, s in solutions.items():
        if s["sol_lbl"] == 0:  # error solution
            true_fe  = first_error_step(s["step_lbls"])
            errs     = ptb_err_scores[sid]
            pred_fe  = int(np.nanargmax(errs)) if not np.all(np.isnan(errs)) else None
            total   += 1
            if pred_fe == true_fe:
                hits += 1
    if total > 0:
        print(f"\n  PTB argmax hit rate: {hits}/{total} = {hits/total:.1%}")
        print(f"  Random baseline:     1/avg_steps ≈ {1/np.mean([len(s['h']) for s in solutions.values()]):.1%}")
    print()


if __name__ == "__main__":
    main()
