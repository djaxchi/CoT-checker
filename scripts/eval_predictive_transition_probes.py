#!/usr/bin/env python3
"""Benchmark linear probes on PTB-experiment representations.

Focused objective: answer two questions about PTB.
  1. Does transition prediction produce a useful correctness representation?
  2. Does sparsity help or hurt the PTB bottleneck?

Representations benchmarked (SSAE excluded by default):
  dense_h        raw backbone hidden state
  dense_delta    delta_h = h_{k+1} - h_k
  dense_concat   [h_k ; delta_h]
  random_bln     random Linear+ReLU (same arch as PTB, untrained)
  ptb_no_l1      PTB trained with no sparsity penalty
  ptb_*          any other PTB variant found in --rep-dir

Primary comparisons:
  PTB_no_l1 vs random_bln   -- does the transition objective learn anything?
  PTB_no_l1 vs dense_h      -- does transition prediction improve over raw states?
  PTB_DWA vs PTB_no_l1      -- does sparsity help or remove useful information?
  PTB_topk vs PTB_DWA       -- is the issue DWA or sparsity itself?

Metric suite per representation (macro-F1 is the primary metric):
  Math-Shepherd balanced: macro-F1, AUROC, AUPRC, best threshold,
    pos/neg rate, confusion matrix, collapse flag
  ProcessBench: solution-level F1, step-level macro-F1, pos rate, collapse flag
  Transfer degradation: MS macro-F1 - PB macro-F1

Randomization controls (--controls):
  random_labels      same h, randomly permuted y
  shuffled_pairs     permuted (h, y) -- breaks alignment
  random_projection  random Linear+ReLU of same dim

Class-balance sweep (--balance-sweep):
  70_30, 50_50, 30_70, natural, class_weighted, focal

Usage:
    python scripts/eval_predictive_transition_probes.py \\
        --rep-dir      ptb_representations/ \\
        --dense-train  dense_train_full.npz \\
        --dense-eval   dense_eval_held_out.npz \\
        --pb-dense     processbench_dense_gsm8k.npz \\
        --output-dir   results/ptb_probes/ \\
        --device       cuda \\
        --controls \\
        --balance-sweep

    # With SSAE for comparison (optional):
        --ssae-train train_final.npz --ssae-eval eval_held_out.npz \\
        --pb-ssae    processbench_gsm8k.npz
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available; AUROC/AUPRC will be NaN.")

from scripts.eval_processbench import (  # type: ignore
    processbench_f1,
    step_metrics,
    get_p_correct,
)


# ---------------------------------------------------------------------------
# Minimal linear probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "input_dim": self.fc.in_features}, path)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    h = d["latents"].astype(np.float32)
    y = d["correctness"] if "correctness" in d.files else d["step_labels"]
    return h, y.astype(np.int64)


def _subsample_class_ratio(
    h: np.ndarray, y: np.ndarray,
    frac_correct: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to a target correct fraction.

    frac_correct=0.7 -> 70% correct / 30% incorrect (etc.)
    Caps at the minority class; does not oversample.
    """
    rng = np.random.default_rng(seed)
    cor = np.where(y == 1)[0]
    inc = np.where(y == 0)[0]
    # Determine limiting class
    if frac_correct >= 0.5:
        n_inc = len(inc)
        n_cor = min(int(n_inc * frac_correct / (1.0 - frac_correct)), len(cor))
    else:
        n_cor = len(cor)
        n_inc = min(int(n_cor * (1.0 - frac_correct) / frac_correct), len(inc))
    sel = np.concatenate([rng.choice(cor, n_cor, replace=False),
                          rng.choice(inc, n_inc, replace=False)])
    rng.shuffle(sel)
    return h[sel], y[sel]


def _balanced_eval_subset(
    h: np.ndarray, y: np.ndarray, seed: int, n_per_class: int = 25000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cor = np.where(y == 1)[0]
    inc = np.where(y == 0)[0]
    n   = min(n_per_class, len(cor), len(inc))
    sel = np.concatenate([rng.choice(cor, n, replace=False), rng.choice(inc, n, replace=False)])
    rng.shuffle(sel)
    return h[sel], y[sel]


def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits).detach()
    pt = torch.where(targets == 1, p, 1 - p)
    return (((1 - pt) ** gamma) * bce).mean()


# ---------------------------------------------------------------------------
# Full metric suite at a given threshold
# ---------------------------------------------------------------------------

def _compute_metrics_at_threshold(
    probs: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict:
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n  = len(y)

    prec_c  = tp / max(tp + fp, 1)
    rec_c   = tp / max(tp + fn, 1)
    f1_c    = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-9)

    prec_i  = tn / max(tn + fn, 1)
    rec_i   = tn / max(tn + fp, 1)
    f1_i    = 2 * prec_i * rec_i / max(prec_i + rec_i, 1e-9)

    acc     = (tp + tn) / max(n, 1)
    macro   = (f1_c + f1_i) / 2.0
    pos_rate = (tp + fp) / max(n, 1)
    neg_rate = (fn + tn) / max(n, 1)

    return {
        "threshold": threshold,
        "accuracy":  acc,
        "macro_f1":  macro,
        "f1_correct": f1_c,
        "f1_incorrect": f1_i,
        "prec_correct": prec_c,
        "rec_correct":  rec_c,
        "prec_incorrect": prec_i,
        "rec_incorrect":  rec_i,
        "pos_rate":  pos_rate,
        "neg_rate":  neg_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _sweep_thresholds(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    return [_compute_metrics_at_threshold(probs, y, t) for t in thresholds]


def _best_at_threshold(sweep: list[dict]) -> dict:
    """Return metrics at the threshold that maximises macro F1."""
    return max(sweep, key=lambda r: r["macro_f1"])


def _collapse_flag(sweep: list[dict]) -> str | None:
    """Return a collapse description if the probe predicts only one class."""
    best = _best_at_threshold(sweep)
    if best["pos_rate"] < 0.02:
        return f"COLLAPSE: probe predicts all-negative (pos_rate={best['pos_rate']:.3f})"
    if best["pos_rate"] > 0.98:
        return f"COLLAPSE: probe predicts all-positive (pos_rate={best['pos_rate']:.3f})"
    return None


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------

def _get_class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    w_pos = len(y) / (2.0 * max(n_pos, 1))
    w_neg = len(y) / (2.0 * max(n_neg, 1))
    return torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)


def train_and_eval_probe(
    train_h: np.ndarray,
    train_y: np.ndarray,
    eval_ms_h: np.ndarray,
    eval_ms_y: np.ndarray,
    pb_npz_path: str | None,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    out_path: Path,
    class_balance: str = "70_30",
    loss_type: str = "bce",
    reuse: bool = False,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if reuse and out_path.exists():
        ckpt = torch.load(out_path, map_location=device, weights_only=False)
        model = LinearProbe(ckpt["input_dim"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        h_sub = train_h  # not used for training, only for reporting counts
        y_sub = train_y
    else:
        # ---- Build training subset ----
        if class_balance == "natural":
            h_sub, y_sub = train_h, train_y
        elif class_balance == "50_50":
            h_sub, y_sub = _subsample_class_ratio(train_h, train_y, 0.5, seed)
        elif class_balance == "70_30":
            h_sub, y_sub = _subsample_class_ratio(train_h, train_y, 0.7, seed)
        elif class_balance == "30_70":
            h_sub, y_sub = _subsample_class_ratio(train_h, train_y, 0.3, seed)
        elif class_balance in ("class_weighted", "focal"):
            h_sub, y_sub = train_h, train_y
        else:
            h_sub, y_sub = _subsample_class_ratio(train_h, train_y, 0.7, seed)

        h_t = torch.from_numpy(h_sub).to(device)
        y_t = torch.from_numpy(y_sub.astype(np.float32)).to(device)

        model = LinearProbe(h_sub.shape[1]).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = DataLoader(TensorDataset(h_t, y_t), batch_size=batch_size, shuffle=True)

        class_weights = _get_class_weights(y_sub, device) if class_balance == "class_weighted" else None

        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                logits = model(xb).squeeze(-1)
                if loss_type == "focal":
                    loss = _focal_loss(logits, yb)
                elif class_weights is not None:
                    w_per_sample = torch.where(yb == 1, class_weights[1], class_weights[0])
                    loss = F.binary_cross_entropy_with_logits(logits, yb, weight=w_per_sample)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        model.eval()
        model.save(out_path)

    # ---- Math-Shepherd eval (balanced) ----
    h_ev, y_ev = _balanced_eval_subset(eval_ms_h, eval_ms_y, seed)
    with torch.no_grad():
        h_ev_t = torch.from_numpy(h_ev).to(device)
        logits_ev = model(h_ev_t).squeeze(-1)
        probs_ms  = torch.sigmoid(logits_ev).cpu().numpy()

    sweep_ms = _sweep_thresholds(probs_ms, y_ev)
    best_ms  = _best_at_threshold(sweep_ms)
    collapse_ms = _collapse_flag(sweep_ms)

    auroc_ms = auprc_ms = float("nan")
    if SKLEARN_AVAILABLE:
        try:
            auroc_ms = float(roc_auc_score(y_ev, probs_ms))
            auprc_ms = float(average_precision_score(y_ev, probs_ms))
        except Exception:
            pass

    result = {
        "seed":         seed,
        "class_balance": class_balance,
        "loss_type":    loss_type,
        "train_n":      len(y_sub),
        "train_pos":    int((y_sub == 1).sum()),
        "train_neg":    int((y_sub == 0).sum()),
        # Best-threshold MS metrics
        "ms_best_threshold":  best_ms["threshold"],
        "ms_acc":             best_ms["accuracy"],
        "ms_macro_f1":        best_ms["macro_f1"],
        "ms_f1_correct":      best_ms["f1_correct"],
        "ms_f1_incorrect":    best_ms["f1_incorrect"],
        "ms_pos_rate":        best_ms["pos_rate"],
        "ms_auroc":           auroc_ms,
        "ms_auprc":           auprc_ms,
        "ms_collapse":        collapse_ms,
        # Fixed-threshold t=0.5 (for comparability)
        "ms_acc_05":          _compute_metrics_at_threshold(probs_ms, y_ev, 0.5)["accuracy"],
        "ms_macro_f1_05":     _compute_metrics_at_threshold(probs_ms, y_ev, 0.5)["macro_f1"],
        # Threshold sweep for plotting
        "ms_sweep":           sweep_ms,
        "pb_f1":              float("nan"),
        "pb_macro_f1":        float("nan"),
        "pb_pos_rate":        float("nan"),
        "pb_collapse":        None,
    }

    # ---- ProcessBench eval ----
    if pb_npz_path and Path(pb_npz_path).exists():
        pb = np.load(pb_npz_path)
        pb_h           = pb["latents"].astype(np.float32)
        pb_step_labels = pb["step_labels"]
        pb_sol_ids     = pb["solution_ids"]
        pb_step_pos    = pb["step_positions"]
        pb_sol_labels  = pb["solution_labels"]

        with torch.no_grad():
            pb_h_t    = torch.from_numpy(pb_h).to(device)
            logits_pb = model(pb_h_t).squeeze(-1)
            probs_pb  = torch.sigmoid(logits_pb).cpu().numpy()

        sweep_pb    = _sweep_thresholds(probs_pb, pb_step_labels)
        best_pb     = _best_at_threshold(sweep_pb)
        collapse_pb = _collapse_flag(sweep_pb)

        # ProcessBench solution-level F1 (sweep thresholds)
        best_pb_f1 = 0.0
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            _, _, f1 = processbench_f1(probs_pb, pb_sol_ids, pb_step_pos, pb_sol_labels, threshold=t)
            if f1 > best_pb_f1:
                best_pb_f1 = f1

        result["pb_f1"]       = float(best_pb_f1)
        result["pb_macro_f1"] = float(best_pb["macro_f1"])
        result["pb_pos_rate"] = float(best_pb["pos_rate"])
        result["pb_collapse"] = collapse_pb
        result["pb_sweep"]    = sweep_pb

    return result


# ---------------------------------------------------------------------------
# Randomization controls
# ---------------------------------------------------------------------------

def run_randomization_controls(
    label: str,
    train_h: np.ndarray,
    train_y: np.ndarray,
    eval_ms_h: np.ndarray,
    eval_ms_y: np.ndarray,
    pb_npz_path: str | None,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    out_dir: Path,
) -> list[dict]:
    """Run randomization controls for a single representation."""
    results = []

    # Random projection (same dim as the representation)
    d = train_h.shape[1]
    rng = np.random.default_rng(0)
    W = rng.standard_normal((d, d)).astype(np.float32) * (1.0 / d**0.5)
    b = np.zeros(d, dtype=np.float32)
    train_h_rand = np.maximum(train_h @ W.T + b, 0)
    eval_h_rand  = np.maximum(eval_ms_h @ W.T + b, 0)

    for seed in seeds:
        # Control 1: random labels (same h_k)
        rng_ctrl = np.random.default_rng(seed + 1000)
        train_y_shuffled = rng_ctrl.permutation(train_y.copy())
        r = train_and_eval_probe(
            train_h, train_y_shuffled, eval_ms_h, eval_ms_y,
            pb_npz_path, seed, epochs // 2, batch_size, lr, device,
            out_dir / f"{label}_ctrl_random_labels_seed{seed}.pt",
            class_balance="natural",
        )
        r["control"] = "random_labels"
        r["representation"] = label
        results.append(r)

        # Control 2: shuffled (h, y) pairs -- breaks alignment
        rng_ctrl2 = np.random.default_rng(seed + 2000)
        perm = rng_ctrl2.permutation(len(train_h))
        r2 = train_and_eval_probe(
            train_h[perm], train_y, eval_ms_h, eval_ms_y,
            pb_npz_path, seed, epochs // 2, batch_size, lr, device,
            out_dir / f"{label}_ctrl_shuffled_pairs_seed{seed}.pt",
            class_balance="natural",
        )
        r2["control"] = "shuffled_pairs"
        r2["representation"] = label
        results.append(r2)

        # Control 3: random projection (same dim)
        r3 = train_and_eval_probe(
            train_h_rand, train_y, eval_h_rand, eval_ms_y,
            None, seed, epochs // 2, batch_size, lr, device,
            out_dir / f"{label}_ctrl_random_proj_seed{seed}.pt",
            class_balance="natural",
        )
        r3["control"] = "random_projection"
        r3["representation"] = label
        results.append(r3)

    return results


# ---------------------------------------------------------------------------
# Representation registry
# ---------------------------------------------------------------------------

def build_rep_registry(args: argparse.Namespace) -> list[dict]:
    rep_dir = Path(args.rep_dir)

    # Core representations (SSAE included only if explicitly provided)
    reps = [
        {"label": "dense_h",   "train": args.dense_train, "eval_ms": args.dense_eval, "pb": args.pb_dense},
        {"label": "dense_delta",  "train": str(rep_dir / "dense_delta_ms_train.npz"),
                                  "eval_ms": str(rep_dir / "dense_delta_ms_eval.npz"),
                                  "pb": str(rep_dir / "dense_delta_pb.npz")},
        {"label": "dense_concat", "train": str(rep_dir / "dense_concat_ms_train.npz"),
                                  "eval_ms": str(rep_dir / "dense_concat_ms_eval.npz"),
                                  "pb": str(rep_dir / "dense_concat_pb.npz")},
        {"label": "random_bln",   "train": str(rep_dir / "random_bln_ms_train.npz"),
                                  "eval_ms": str(rep_dir / "random_bln_ms_eval.npz"),
                                  "pb": str(rep_dir / "random_bln_pb.npz")},
    ]

    # SSAE only when explicitly supplied
    if args.ssae_train and args.ssae_eval:
        reps.insert(1, {
            "label":   "ssae_z",
            "train":   args.ssae_train,
            "eval_ms": args.ssae_eval,
            "pb":      args.pb_ssae,
        })

    # Auto-discover PTB variants: any file ptb_*_ms_train.npz in rep_dir
    for f in sorted(rep_dir.glob("ptb_*_ms_train.npz")):
        # Derive variant name from filename: ptb_{name}_ms_train.npz
        stem = f.stem  # e.g. ptb_no_l1_ms_train
        # Strip _ms_train suffix to get base: ptb_no_l1
        base = stem.replace("_ms_train", "")
        label = base  # e.g. ptb_no_l1
        reps.append({
            "label":   label,
            "train":   str(f),
            "eval_ms": str(rep_dir / f"{base}_ms_eval.npz"),
            "pb":      str(rep_dir / f"{base}_pb.npz"),
        })

    available = []
    for r in reps:
        if Path(r["train"]).exists() and Path(r["eval_ms"]).exists():
            available.append(r)
        else:
            print(f"  SKIP {r['label']} (missing {r['train']} or {r['eval_ms']})")
    return available


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _nan_str(v: float, fmt: str = ".3f") -> str:
    if v != v:
        return "  n/a  "
    return format(v, fmt)


def _collapse_str(c: str | None) -> str:
    return "COLLAPSE" if c else "ok"


def print_table(rows: list[dict]) -> None:
    # Primary metric is macro-F1, not raw accuracy.
    # Accuracy shown in parentheses for reference only.
    hdr = (
        f"  {'Representation':<26}  {'MS MacroF1':>10}  {'AUROC':>6}  {'AUPRC':>6}"
        f"  {'MS thr':>6}  {'MS pos%':>7}  {'PB-F1':>6}  {'PB MacroF1':>10}"
        f"  {'MS':>8}  {'PB':>8}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in rows:
        pb_f1_s  = _nan_str(r["pb_f1"] * 100 if r["pb_f1"] == r["pb_f1"] else float("nan"), ".1f")
        ms_pos   = _nan_str(r.get("ms_pos_rate", float("nan")), ".2f")
        print(
            f"  {r['label']:<26}"
            f"  {_nan_str(r['ms_macro_f1'], '.3f'):>10}"
            f"  {_nan_str(r['ms_auroc']):>6}"
            f"  {_nan_str(r['ms_auprc']):>6}"
            f"  {_nan_str(r['ms_best_threshold'], '.2f'):>6}"
            f"  {ms_pos:>7}"
            f"  {pb_f1_s:>6}"
            f"  {_nan_str(r['pb_macro_f1'], '.3f'):>10}"
            f"  {_collapse_str(r['ms_collapse']):>8}"
            f"  {_collapse_str(r['pb_collapse']):>8}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rep-dir",      required=True)
    p.add_argument("--dense-train",  required=True)
    p.add_argument("--dense-eval",   required=True)
    # SSAE is optional in the PTB-focused experiment
    p.add_argument("--ssae-train",   default=None, help="SSAE train .npz (optional)")
    p.add_argument("--ssae-eval",    default=None, help="SSAE eval .npz (optional)")
    p.add_argument("--pb-dense",     default=None)
    p.add_argument("--pb-ssae",      default=None)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seeds",        type=int, nargs="+", default=[42, 43, 44, 45])
    p.add_argument("--device",       default="cuda", choices=["cpu", "cuda", "mps"])
    # New flags
    p.add_argument("--controls",         action="store_true",
                   help="Run randomization controls for each representation")
    p.add_argument("--balance-sweep",    action="store_true",
                   help="Run multiple class-balance experiments")
    p.add_argument("--balance-configs",  nargs="+",
                   default=["70_30", "50_50", "30_70", "natural", "class_weighted", "focal"],
                   help="Which class-balance variants to run")
    p.add_argument("--reuse-probes",     action="store_true",
                   help="If a probe .pt already exists at the output path, load it and skip training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = build_rep_registry(args)
    print(f"\nEvaluating {len(reps)} representations x {len(args.seeds)} seeds")

    summary_rows     = []
    all_seed_results = {}
    control_rows     = []

    for rep in reps:
        label = rep["label"]
        print(f"\n{'='*60}\n  {label}\n{'='*60}")

        train_h, train_y = _load_npz(rep["train"])
        eval_h,  eval_y  = _load_npz(rep["eval_ms"])
        print(f"  train={len(train_y):,}  eval={len(eval_y):,}  dim={train_h.shape[1]}")
        print(f"  train_pos={int((train_y==1).sum())}  train_neg={int((train_y==0).sum())}")

        # Primary experiment: 70/30 class balance
        seed_results = []
        for seed in args.seeds:
            probe_path = out_dir / f"{label}_linear_probe_seed{seed}.pt"
            res = train_and_eval_probe(
                train_h, train_y, eval_h, eval_y,
                pb_npz_path  = rep.get("pb"),
                seed         = seed,
                epochs       = args.epochs,
                batch_size   = args.batch_size,
                lr           = args.lr,
                device       = args.device,
                out_path     = probe_path,
                class_balance = "70_30",
                reuse        = args.reuse_probes,
            )
            res["representation"] = label
            seed_results.append(res)
            print(
                f"  seed={seed}"
                f"  macro_f1={res['ms_macro_f1']:.3f}"
                f"  auroc={_nan_str(res['ms_auroc'])}"
                f"  pos_rate={_nan_str(res['ms_pos_rate'], '.2f')}"
                f"  pb_f1={_nan_str(res['pb_f1']*100 if res['pb_f1']==res['pb_f1'] else float('nan'), '.1f')}%"
                f"  collapse={_collapse_str(res['ms_collapse'])}"
            )

        all_seed_results[label] = seed_results

        # Aggregate
        ms_accs    = [r["ms_acc"]      for r in seed_results]
        ms_macros  = [r["ms_macro_f1"] for r in seed_results]
        ms_aurocs  = [r["ms_auroc"]    for r in seed_results if r["ms_auroc"] == r["ms_auroc"]]
        ms_auprcs  = [r["ms_auprc"]    for r in seed_results if r["ms_auprc"] == r["ms_auprc"]]
        pb_f1s     = [r["pb_f1"]       for r in seed_results if r["pb_f1"]    == r["pb_f1"]]
        pb_mf1s    = [r["pb_macro_f1"] for r in seed_results if r["pb_macro_f1"] == r["pb_macro_f1"]]
        ms_thrs    = [r["ms_best_threshold"] for r in seed_results]
        ms_pos_rates = [r["ms_pos_rate"] for r in seed_results if r["ms_pos_rate"] == r["ms_pos_rate"]]
        ms_collapses = [r["ms_collapse"] for r in seed_results if r["ms_collapse"] is not None]
        pb_collapses = [r["pb_collapse"] for r in seed_results if r["pb_collapse"] is not None]

        row = {
            "label":           label,
            "ms_acc":          statistics.mean(ms_accs),
            "ms_acc_std":      statistics.stdev(ms_accs) if len(ms_accs) > 1 else 0.0,
            "ms_macro_f1":     statistics.mean(ms_macros),
            "ms_mf1_std":      statistics.stdev(ms_macros) if len(ms_macros) > 1 else 0.0,
            "ms_auroc":        statistics.mean(ms_aurocs) if ms_aurocs else float("nan"),
            "ms_auprc":        statistics.mean(ms_auprcs) if ms_auprcs else float("nan"),
            "ms_best_threshold": statistics.mean(ms_thrs),
            "ms_pos_rate":     statistics.mean(ms_pos_rates) if ms_pos_rates else float("nan"),
            "pb_f1":           statistics.mean(pb_f1s) if pb_f1s else float("nan"),
            "pb_f1_std":       statistics.stdev(pb_f1s) if len(pb_f1s) > 1 else 0.0,
            "pb_macro_f1":     statistics.mean(pb_mf1s) if pb_mf1s else float("nan"),
            "ms_collapse":     ms_collapses[0] if ms_collapses else None,
            "pb_collapse":     pb_collapses[0] if pb_collapses else None,
            "transfer_gap":    (statistics.mean(ms_macros) - statistics.mean(pb_mf1s))
                               if pb_mf1s else float("nan"),
        }
        summary_rows.append(row)

        print(
            f"  AGGREGATE:"
            f"  macro_f1={row['ms_macro_f1']:.3f}+-{row['ms_mf1_std']:.3f}"
            f"  auroc={_nan_str(row['ms_auroc'])}"
            f"  pos_rate={_nan_str(row['ms_pos_rate'], '.2f')}"
            f"  pb_f1={_nan_str(row['pb_f1']*100 if row['pb_f1']==row['pb_f1'] else float('nan'), '.1f')}%"
            f"  transfer_gap={_nan_str(row['transfer_gap'])}"
        )

        # ---- Balance sweep ----
        if args.balance_sweep:
            bal_results = []
            for balance in args.balance_configs:
                for seed in args.seeds[:2]:  # 2 seeds for balance sweep
                    loss_t = "focal" if balance == "focal" else "bce"
                    bal_bal = "natural" if balance == "focal" else balance
                    r = train_and_eval_probe(
                        train_h, train_y, eval_h, eval_y,
                        rep.get("pb"), seed, args.epochs, args.batch_size, args.lr,
                        args.device,
                        out_dir / f"{label}_bal_{balance}_seed{seed}.pt",
                        class_balance=bal_bal, loss_type=loss_t,
                    )
                    r["representation"] = label
                    bal_results.append(r)
                    print(f"  balance={balance} seed={seed}: "
                          f"ms_macro_f1={r['ms_macro_f1']:.3f}  "
                          f"collapse={_collapse_str(r['ms_collapse'])}")

            bal_path = out_dir / f"{label}_balance_sweep.json"
            bal_path.write_text(json.dumps(bal_results, indent=2))
            print(f"  Balance sweep -> {bal_path}")

        # ---- Randomization controls ----
        if args.controls:
            ctrl = run_randomization_controls(
                label, train_h, train_y, eval_h, eval_y,
                rep.get("pb"), args.seeds[:2], args.epochs // 2,
                args.batch_size, args.lr, args.device, out_dir,
            )
            control_rows.extend(ctrl)
            for r in ctrl:
                print(f"  control={r['control']}  seed={r['seed']}: "
                      f"ms_macro_f1={r['ms_macro_f1']:.3f}  "
                      f"ms_auroc={_nan_str(r['ms_auroc'])}  "
                      f"collapse={_collapse_str(r['ms_collapse'])}")

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  FINAL BENCHMARK TABLE")
    print_table(summary_rows)

    # Transfer degradation column
    print("\n  Transfer degradation (MS macro_F1 - PB macro_F1):")
    for r in summary_rows:
        gap = r["transfer_gap"]
        print(f"    {r['label']:<24}  {_nan_str(gap)}")

    # ------------------------------------------------------------------
    # Focused 4-question conclusion
    # ------------------------------------------------------------------
    baseline_random  = next((r for r in summary_rows if r["label"] == "random_bln"),    None)
    baseline_dense   = next((r for r in summary_rows if r["label"] == "dense_h"),       None)
    ptb_no_l1        = next((r for r in summary_rows if r["label"] == "ptb_no_l1"),     None)
    ptb_sparse_best  = max(
        (r for r in summary_rows if r["label"].startswith("ptb_") and r["label"] != "ptb_no_l1"),
        key=lambda r: r["ms_macro_f1"] if r["ms_macro_f1"] == r["ms_macro_f1"] else -1,
        default=None,
    )

    print("\n\n" + "=" * 70)
    print("  CONCLUSION: 4 RESEARCH QUESTIONS")
    print("=" * 70)

    def _mf1(r):
        return r["ms_macro_f1"] if r else float("nan")

    def _yn(cond: bool) -> str:
        return "YES" if cond else "NO"

    def _margin(a, b) -> str:
        if a != a or b != b:
            return "n/a"
        return f"{(a - b)*100:+.1f}pp"

    # Q1: Does PTB beat random?
    q1_rep  = ptb_no_l1
    q1_beat = q1_rep is not None and baseline_random is not None and _mf1(q1_rep) > _mf1(baseline_random)
    print(f"\n  Q1. Does PTB beat random_bln?  -->  {_yn(q1_beat)}")
    if q1_rep and baseline_random:
        print(f"      ptb_no_l1={_mf1(q1_rep):.3f}  random_bln={_mf1(baseline_random):.3f}"
              f"  margin={_margin(_mf1(q1_rep), _mf1(baseline_random))}")
    if q1_rep is None:
        print("      ptb_no_l1 not found -- cannot answer")
    if not q1_beat and q1_rep:
        print("      INTERPRETATION: transition prediction does not improve over random "
              "projection; PTB latent is no better than an untrained encoder.")

    # Q2: Does PTB beat dense_h?
    q2_beat = q1_rep is not None and baseline_dense is not None and _mf1(q1_rep) > _mf1(baseline_dense)
    print(f"\n  Q2. Does PTB beat dense_h?  -->  {_yn(q2_beat)}")
    if q1_rep and baseline_dense:
        print(f"      ptb_no_l1={_mf1(q1_rep):.3f}  dense_h={_mf1(baseline_dense):.3f}"
              f"  margin={_margin(_mf1(q1_rep), _mf1(baseline_dense))}")
    if not q2_beat and q1_rep:
        print("      INTERPRETATION: the transition bottleneck compresses information "
              "relative to the raw hidden state without adding discriminative signal.")

    # Q3: Does sparsity improve or hurt PTB?
    q3_helps = (ptb_sparse_best is not None and q1_rep is not None
                and _mf1(ptb_sparse_best) > _mf1(q1_rep))
    print(f"\n  Q3. Does sparsity improve PTB?  -->  {_yn(q3_helps)}")
    if ptb_sparse_best and q1_rep:
        print(f"      best_sparse={ptb_sparse_best['label']}  macro_f1={_mf1(ptb_sparse_best):.3f}"
              f"  ptb_no_l1={_mf1(q1_rep):.3f}"
              f"  margin={_margin(_mf1(ptb_sparse_best), _mf1(q1_rep))}")
    elif ptb_sparse_best is None:
        print("      No sparse PTB variant found -- run dwa_calibrated / topk variants.")

    # Q4: Does any representation transfer to ProcessBench?
    pb_candidates = [(r["label"], r["pb_f1"]) for r in summary_rows if r["pb_f1"] == r["pb_f1"]]
    pb_candidates.sort(key=lambda x: -x[1])
    best_pb_label, best_pb_f1 = pb_candidates[0] if pb_candidates else (None, float("nan"))
    q4_transfers  = best_pb_f1 > 0.25  # >25% PB-F1 = non-trivial localization
    print(f"\n  Q4. Does any representation transfer to ProcessBench?  -->  {_yn(q4_transfers)}")
    if pb_candidates:
        print(f"      Best: {best_pb_label}  PB-F1={best_pb_f1*100:.1f}%")
        print("      Top 3 by PB-F1:")
        for lbl, f1 in pb_candidates[:3]:
            coll = next((r["pb_collapse"] for r in summary_rows if r["label"] == lbl), None)
            print(f"        {lbl:<26}  PB-F1={f1*100:.1f}%  collapse={_collapse_str(coll)}")
    else:
        print("      No ProcessBench results available.")

    # Per-PTB validity flags
    print("\n  Per-PTB variant validity:")
    for r in summary_rows:
        if not r["label"].startswith("ptb_"):
            continue
        flags = []
        if r["ms_collapse"]:
            flags.append(r["ms_collapse"])
        if r["pb_collapse"]:
            flags.append(f"PB: {r['pb_collapse']}")
        if baseline_random and _mf1(r) <= _mf1(baseline_random):
            flags.append("does not beat random_bln")
        status = "VALID" if not flags else "INVALID"
        print(f"    {r['label']:<26}  {status}" + ("  -- " + "; ".join(flags) if flags else ""))
    print()

    # ------------------------------------------------------------------
    # Save JSON + markdown summary
    # ------------------------------------------------------------------
    json_path = out_dir / "summary_results.json"
    json_path.write_text(json.dumps({
        "summary_rows":    summary_rows,
        "seed_results":    all_seed_results,
        "control_rows":    control_rows,
    }, indent=2))
    print(f"\n  Full results -> {json_path}")

    md_lines = [
        "| Representation | MS MacroF1 | AUROC | AUPRC | MS thr | MS pos% | PB-F1 | PB MacroF1 | Transfer gap | MS collapse | PB collapse |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        pb_f1_str = f"{r['pb_f1']*100:.1f}%" if r["pb_f1"] == r["pb_f1"] else "n/a"
        md_lines.append(
            f"| {r['label']} "
            f"| {r['ms_macro_f1']:.3f} +- {r['ms_mf1_std']:.3f} "
            f"| {_nan_str(r['ms_auroc'])} "
            f"| {_nan_str(r['ms_auprc'])} "
            f"| {_nan_str(r['ms_best_threshold'], '.2f')} "
            f"| {_nan_str(r.get('ms_pos_rate', float('nan')), '.2f')} "
            f"| {pb_f1_str} "
            f"| {_nan_str(r['pb_macro_f1'])} "
            f"| {_nan_str(r['transfer_gap'])} "
            f"| {_collapse_str(r['ms_collapse'])} "
            f"| {_collapse_str(r['pb_collapse'])} |"
        )
    md_path = out_dir / "summary_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  Summary table -> {md_path}")

    if control_rows:
        ctrl_path = out_dir / "control_results.json"
        ctrl_path.write_text(json.dumps(control_rows, indent=2))
        print(f"  Control results -> {ctrl_path}")


if __name__ == "__main__":
    main()
