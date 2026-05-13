#!/usr/bin/env python3
"""End-to-end PTB reconstruction-error vs dense-probe comparison on ProcessBench.

Encodes ProcessBench solutions directly from the raw JSONL (all steps, no
truncation), then evaluates two anomaly signals side-by-side:

  PTB recon error  : ||PTB.decode(PTB.encode(h_k)) - (h_{k+1} - h_k)||
                     High error = PTB thinks this transition is out-of-distribution.
  Dense probe      : 1 - sigmoid(probe(h_k))  [if --probe-ckpt is given]
                     High score = probe thinks this step is incorrect.

Usage:
    python scripts/eval_ptb_vs_probe.py \
        --data-file  $SCRATCH/cot-checker/processbench/processbench_gsm8k.jsonl \
        --ssae-ckpt  /project/aip-azouaq/dchikhi/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt \
        --ptb-ckpt   $STORE/results/checkpoints/ptb_robust/no_l1/best.pt \
        --probe-ckpt $STORE/results/ptb_robust_probes/dense_h_linear_probe_seed42.pt \
        --n-solutions 50 \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.saes.ssae import SSAE
from src.saes.ptb import PredictiveTransitionBottleneck


# ---------------------------------------------------------------------------
# Inline encoder (dense h_k from backbone, no SSAE projector)
# ---------------------------------------------------------------------------

def encode_steps(model: SSAE, tokenizer, steps: list[str], problem: str,
                 device: str, max_seq_len: int = 2048) -> np.ndarray:
    """Return (n_steps, d) float32 dense hidden states for all steps."""
    sep_id = tokenizer.sep_token_id
    latents = []
    prior: list[str] = []
    for step in steps:
        ctx = (problem + " " + " ".join(prior)).strip() if prior else problem
        ctx_ids  = tokenizer.encode(ctx,  add_special_tokens=False)
        step_ids = tokenizer.encode(step, add_special_tokens=False)
        seq = ctx_ids + [sep_id] + step_ids + [tokenizer.eos_token_id]
        if len(seq) > max_seq_len:
            keep = max_seq_len - len(step_ids) - 2
            seq = ctx_ids[-max(keep, 0):] + [sep_id] + step_ids + [tokenizer.eos_token_id]
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        msk = torch.ones_like(ids)
        with torch.no_grad():
            h = model.encode_dense(ids, msk)
        latents.append(h.squeeze().cpu().float().numpy())
        prior.append(step)
    return np.stack(latents)


# ---------------------------------------------------------------------------
# PTB reconstruction error
# ---------------------------------------------------------------------------

@torch.no_grad()
def recon_errors(ptb: PredictiveTransitionBottleneck, h: np.ndarray) -> np.ndarray:
    n = len(h)
    errs = np.full(n, np.nan)
    h_t = torch.from_numpy(h)
    for k in range(n - 1):
        delta_actual = h_t[k + 1] - h_t[k]
        delta_hat    = ptb.decode(ptb.encode(h_t[k].unsqueeze(0)))
        errs[k]      = (delta_hat - delta_actual).norm().item()
    return errs


# ---------------------------------------------------------------------------
# Dense probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def probe_incorrectness(probe: LinearProbe, h: np.ndarray) -> np.ndarray:
    h_t = torch.from_numpy(h)
    return (1.0 - torch.sigmoid(probe(h_t).squeeze(-1))).numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def first_error_idx(step_labels: list[int]) -> int | None:
    for i, lbl in enumerate(step_labels):
        if lbl == 0:
            return i
    return None


def pb_metrics(solutions: list[dict], scores: dict[int, np.ndarray],
               thresholds: list[float]) -> list[dict]:
    results = []
    for thr in thresholds:
        n_clean, n_clean_ok = 0, 0
        n_err, n_err_det, n_err_loc = 0, 0, 0
        for sol in solutions:
            sid   = sol["sid"]
            errs  = scores[sid]
            valid = ~np.isnan(errs)
            has_err = sol["has_error"]
            if not has_err:
                n_clean += 1
                if not np.any(errs[valid] > thr):
                    n_clean_ok += 1
            else:
                n_err += 1
                true_fe = sol["first_error"]
                pred_fe = next((k for k in range(len(errs))
                                if not np.isnan(errs[k]) and errs[k] > thr), None)
                if pred_fe is not None:
                    n_err_det += 1
                if pred_fe == true_fe:
                    n_err_loc += 1
        acc_c  = n_clean_ok / max(n_clean, 1)
        acc_e  = n_err_loc  / max(n_err, 1)
        det_e  = n_err_det  / max(n_err, 1)
        pb_f1  = 2 * acc_c * acc_e / (acc_c + acc_e) if (acc_c + acc_e) > 0 else 0.0
        results.append(dict(threshold=thr, acc_clean=acc_c, err_detect=det_e,
                            err_locate=acc_e, pb_f1=pb_f1,
                            n_clean=n_clean, n_err=n_err))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file",   required=True, help="ProcessBench JSONL")
    p.add_argument("--ssae-ckpt",   required=True, help="SSAE checkpoint (backbone source)")
    p.add_argument("--ptb-ckpt",    required=True, help="PTB best.pt")
    p.add_argument("--probe-ckpt",  default=None,  help="Dense linear probe .pt (optional)")
    p.add_argument("--n-solutions", type=int, default=50)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--max-seq-len", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    with open(args.data_file) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = rows[:args.n_solutions]
    print(f"Loaded {len(rows)} solutions from {args.data_file}")

    # Load models
    print(f"Loading backbone from {args.ssae_ckpt} ...")
    backbone = SSAE.from_checkpoint(args.ssae_ckpt, device=args.device)
    backbone.eval()
    tokenizer = backbone.tokenizer

    print(f"Loading PTB from {args.ptb_ckpt} ...")
    ptb = PredictiveTransitionBottleneck.from_checkpoint(args.ptb_ckpt, device=args.device)
    ptb.eval()

    probe = None
    if args.probe_ckpt and Path(args.probe_ckpt).exists():
        print(f"Loading probe from {args.probe_ckpt} ...")
        ckpt  = torch.load(args.probe_ckpt, map_location=args.device, weights_only=False)
        probe = LinearProbe(ckpt["input_dim"])
        probe.load_state_dict(ckpt["state_dict"])
        probe.eval()

    # Encode + score all solutions
    solutions: list[dict] = []
    ptb_scores:   dict[int, np.ndarray] = {}
    probe_scores: dict[int, np.ndarray] = {}

    print(f"\nEncoding {len(rows)} solutions ...")
    for sid, row in enumerate(rows):
        label  = row["label"]          # -1 = all correct, k = first error index
        steps  = row["steps"]
        n      = len(steps)
        lbls   = [1] * n               # all correct by default
        if label != -1:
            for i in range(label, n):
                lbls[i] = 0            # error step and everything after

        has_error  = (label != -1)
        first_err  = label if has_error else None

        h = encode_steps(backbone, tokenizer, steps, row["problem"],
                         args.device, args.max_seq_len)

        ptb_scores[sid]  = recon_errors(ptb, h)
        if probe is not None:
            probe_scores[sid] = probe_incorrectness(probe, h)

        solutions.append(dict(sid=sid, has_error=has_error, first_error=first_err,
                              step_labels=lbls, n_steps=n))

    # -----------------------------------------------------------------------
    # Per-solution detail table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  PER-SOLUTION STEP DETAIL")
    print("=" * 72)

    for sol in solutions:
        sid  = sol["sid"]
        errs = ptb_scores[sid]
        fe   = sol["first_error"]
        pred = int(np.nanargmax(errs)) if not np.all(np.isnan(errs)) else None

        if sol["has_error"]:
            verdict = "HIT" if pred == fe else f"MISS (pred={pred}, true={fe})"
        else:
            verdict = "clean"

        print(f"\nSol {sid+1:>3}  label={row['label'] if sid < len(rows) else '?'}  "
              f"steps={sol['n_steps']}  PTB: {verdict}")

        hdr = f"  {'k':>2}  {'label':>8}  {'recon_err':>10}"
        if probe is not None:
            hdr += f"  {'probe_inc':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for k in range(sol["n_steps"]):
            lbl_str = "WRONG" if sol["step_labels"][k] == 0 else "correct"
            err_str = f"{errs[k]:.4f}" if not np.isnan(errs[k]) else "    --"
            flags = []
            if k == pred:       flags.append("← PTB max")
            if k == fe:         flags.append("← TRUE first error")
            row_str = f"  {k:>2}  {lbl_str:>8}  {err_str:>10}"
            if probe is not None:
                row_str += f"  {probe_scores[sid][k]:>9.4f}"
            if flags:
                row_str += "  " + "  ".join(flags)
            print(row_str)

    # -----------------------------------------------------------------------
    # Reconstruction error distribution by step label
    # -----------------------------------------------------------------------
    correct_errs, wrong_errs = [], []
    for sol in solutions:
        sid = sol["sid"]
        for k, (err, lbl) in enumerate(zip(ptb_scores[sid], sol["step_labels"])):
            if not np.isnan(err):
                (correct_errs if lbl == 1 else wrong_errs).append(err)

    correct_errs = np.array(correct_errs) if correct_errs else np.array([])
    wrong_errs   = np.array(wrong_errs)   if wrong_errs   else np.array([])

    print("\n" + "=" * 72)
    print("  RECONSTRUCTION ERROR BY STEP LABEL")
    print("=" * 72)
    if len(correct_errs):
        print(f"\n  Correct steps  n={len(correct_errs):>4} :  "
              f"mean={correct_errs.mean():.4f}  std={correct_errs.std():.4f}  "
              f"median={np.median(correct_errs):.4f}  p95={np.percentile(correct_errs,95):.4f}")
    if len(wrong_errs):
        print(f"  Wrong   steps  n={len(wrong_errs):>4} :  "
              f"mean={wrong_errs.mean():.4f}  std={wrong_errs.std():.4f}  "
              f"median={np.median(wrong_errs):.4f}  p95={np.percentile(wrong_errs,95):.4f}")
        ratio = wrong_errs.mean() / max(correct_errs.mean(), 1e-8)
        print(f"\n  Mean ratio wrong/correct: {ratio:.3f}")
        if ratio > 1.15:
            print("  --> Wrong steps have HIGHER recon error  [hypothesis SUPPORTED]")
        elif ratio < 0.85:
            print("  --> Wrong steps have LOWER recon error   [hypothesis REFUTED]")
        else:
            print("  --> No meaningful difference             [hypothesis NOT supported]")
    else:
        print("  (no wrong steps with a successor h_{k+1} in this subset)")

    # -----------------------------------------------------------------------
    # PB-F1 across thresholds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  PB METRICS ACROSS THRESHOLDS")
    print("=" * 72)

    ptb_thresholds = (np.percentile(correct_errs, [50, 75, 90, 95, 99]).tolist()
                      if len(correct_errs) else [0.5, 1.0, 1.5, 2.0, 3.0])

    print(f"\n  PTB recon error  (thresholds = percentiles of correct-step distribution)")
    print(f"  {'threshold':>10}  {'acc_clean':>9}  {'err_detect':>10}  {'err_locate':>10}  {'PB-F1':>7}")
    print("  " + "-" * 55)
    for r in pb_metrics(solutions, ptb_scores, ptb_thresholds):
        print(f"  {r['threshold']:>10.4f}  {r['acc_clean']:>9.3f}  "
              f"{r['err_detect']:>10.3f}  {r['err_locate']:>10.3f}  {r['pb_f1']:>7.3f}")

    if probe is not None:
        print(f"\n  Dense probe  (1 - p_correct)")
        print(f"  {'threshold':>10}  {'acc_clean':>9}  {'err_detect':>10}  {'err_locate':>10}  {'PB-F1':>7}")
        print("  " + "-" * 55)
        for r in pb_metrics(solutions, probe_scores, [0.3, 0.4, 0.5, 0.6, 0.7]):
            print(f"  {r['threshold']:>10.2f}  {r['acc_clean']:>9.3f}  "
                  f"{r['err_detect']:>10.3f}  {r['err_locate']:>10.3f}  {r['pb_f1']:>7.3f}")

    # -----------------------------------------------------------------------
    # Argmax hit rate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  ARGMAX HIT RATE")
    print("=" * 72)

    hits, total, skipped = 0, 0, 0
    for sol in solutions:
        if not sol["has_error"]:
            continue
        sid  = sol["sid"]
        errs = ptb_scores[sid]
        if np.all(np.isnan(errs)):
            skipped += 1
            continue
        pred = int(np.nanargmax(errs))
        total += 1
        if pred == sol["first_error"]:
            hits += 1

    avg_steps = np.mean([s["n_steps"] for s in solutions])
    print(f"\n  PTB argmax hit rate : {hits}/{total} = {hits/max(total,1):.1%}"
          f"  ({skipped} skipped -- error is last step)")
    print(f"  Random baseline     : 1/avg_steps ≈ {1/avg_steps:.1%}")
    print()


if __name__ == "__main__":
    main()
