#!/usr/bin/env python3
"""Train linear probes on all PTB-experiment representations and print comparison table.

Trains 4 linear probes (seeds 42-45) for each representation, evaluates on
Math-Shepherd eval + ProcessBench, and prints a markdown comparison table.

Representations evaluated (each needs a train .npz and eval .npz):
    dense_h       — raw backbone h_k (existing files)
    ssae_z        — SSAE reconstruction latent (existing files)
    ptb_z         — PTB transition latent (from extract_predictive_transition_latents.py)
    dense_delta   — Δh_k = h_{k+1} - h_k (from extraction script)
    dense_concat  — [h_k ; Δh_k] input_dim=1792 (from extraction script)
    random_bln    — random bottleneck z_k (from extraction script)

Usage:
    python scripts/eval_predictive_transition_probes.py \\
        --rep-dir      $SCRATCH/cot-checker/probe_data/ptb_representations \\
        --dense-train  $SCRATCH/cot-checker/probe_data/dense_train_full.npz \\
        --dense-eval   $SCRATCH/cot-checker/probe_data/dense_eval_held_out.npz \\
        --ssae-train   $SCRATCH/cot-checker/probe_data/train_final.npz \\
        --ssae-eval    $SCRATCH/cot-checker/probe_data/eval_held_out.npz \\
        --pb-dense     $SCRATCH/cot-checker/processbench/processbench_dense_gsm8k.npz \\
        --pb-ssae      $SCRATCH/cot-checker/processbench/processbench_gsm8k.npz \\
        --output-dir   $STORE/results/ptb_probes \\
        --device cuda
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_processbench import (  # type: ignore
    processbench_f1,
    step_metrics,
    get_p_correct,
)


# ---------------------------------------------------------------------------
# Minimal linear probe (avoids import from experiment_linear_probe.py which
# has extra dependencies on experiment_full_clean.py)
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


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    h = d["latents"].astype(np.float32)
    # Support both field names
    y = d["correctness"] if "correctness" in d.files else d["step_labels"]
    return h, y.astype(np.int64)


def _build_70_30_subset(h: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to 70% correct / 30% incorrect (same as experiment_linear_probe.py)."""
    rng = np.random.default_rng(seed)
    cor = np.where(y == 1)[0]
    inc = np.where(y == 0)[0]
    n_inc = len(inc)
    n_cor = min(int(n_inc * 0.7 / 0.3), len(cor))
    sel = np.concatenate([rng.choice(cor, n_cor, replace=False), inc])
    rng.shuffle(sel)
    return h[sel], y[sel]


def _balanced_eval_subset(h: np.ndarray, y: np.ndarray, seed: int, n_per_class: int = 25000
                           ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cor = np.where(y == 1)[0]
    inc = np.where(y == 0)[0]
    n = min(n_per_class, len(cor), len(inc))
    sel = np.concatenate([rng.choice(cor, n, replace=False), rng.choice(inc, n, replace=False)])
    rng.shuffle(sel)
    return h[sel], y[sel]


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
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build training subset
    h_sub, y_sub = _build_70_30_subset(train_h, train_y, seed)
    h_t = torch.from_numpy(h_sub).to(device)
    y_t = torch.from_numpy(y_sub.astype(np.float32)).to(device)

    model = LinearProbe(h_sub.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(h_t, y_t), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            logits = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    model.save(out_path)

    # Math-Shepherd eval (balanced subset)
    h_ev, y_ev = _balanced_eval_subset(eval_ms_h, eval_ms_y, seed)
    p_correct_ms = get_p_correct(model, h_ev, device)
    acc_ms, res_ms, macro_f1_ms = step_metrics(p_correct_ms, y_ev, threshold=0.5)

    result = {
        "seed":         seed,
        "ms_acc":       float(acc_ms),
        "ms_f1_cor":    float(res_ms["correct"]["f1"]),
        "ms_f1_inc":    float(res_ms["incorrect"]["f1"]),
        "ms_macro_f1":  float(macro_f1_ms),
        "pb_f1":        float("nan"),
    }

    # ProcessBench eval — sweep thresholds and report the best PB-F1
    if pb_npz_path and Path(pb_npz_path).exists():
        pb = np.load(pb_npz_path)
        pb_h = pb["latents"].astype(np.float32)
        pb_step_labels  = pb["step_labels"]
        pb_sol_ids      = pb["solution_ids"]
        pb_step_pos     = pb["step_positions"]
        pb_sol_labels   = pb["solution_labels"]
        p_correct_pb = get_p_correct(model, pb_h, device)
        best_pb_f1 = 0.0
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            _, _, f1 = processbench_f1(
                p_correct_pb, pb_sol_ids, pb_step_pos, pb_sol_labels, threshold=t
            )
            if f1 > best_pb_f1:
                best_pb_f1 = f1
        result["pb_f1"] = float(best_pb_f1)

    return result


# ---------------------------------------------------------------------------
# Representation registry
# ---------------------------------------------------------------------------

def build_rep_registry(args: argparse.Namespace) -> list[dict]:
    """Return a list of representation configs.

    Each entry: {label, train_npz, eval_ms_npz, pb_npz}
    Only entries where both train_npz and eval_ms_npz exist are included.
    """
    rep_dir = Path(args.rep_dir)
    reps = [
        {
            "label":    "dense_h",
            "train":    args.dense_train,
            "eval_ms":  args.dense_eval,
            "pb":       args.pb_dense,
        },
        {
            "label":    "ssae_z",
            "train":    args.ssae_train,
            "eval_ms":  args.ssae_eval,
            "pb":       args.pb_ssae,
        },
        {
            "label":    "ptb_z",
            "train":    str(rep_dir / "ptb_z_ms_train.npz"),
            "eval_ms":  str(rep_dir / "ptb_z_ms_eval.npz"),
            "pb":       str(rep_dir / "ptb_z_pb.npz"),
        },
        {
            "label":    "dense_delta",
            "train":    str(rep_dir / "dense_delta_ms_train.npz"),
            "eval_ms":  str(rep_dir / "dense_delta_ms_eval.npz"),
            "pb":       str(rep_dir / "dense_delta_pb.npz"),
        },
        {
            "label":    "dense_concat",
            "train":    str(rep_dir / "dense_concat_ms_train.npz"),
            "eval_ms":  str(rep_dir / "dense_concat_ms_eval.npz"),
            "pb":       str(rep_dir / "dense_concat_pb.npz"),
        },
        {
            "label":    "random_bln",
            "train":    str(rep_dir / "random_bln_ms_train.npz"),
            "eval_ms":  str(rep_dir / "random_bln_ms_eval.npz"),
            "pb":       str(rep_dir / "random_bln_pb.npz"),
        },
    ]
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

def print_table(rows: list[dict]) -> None:
    header = f"{'Representation':<16}  {'MS Acc':>8}  {'MS Macro F1':>12}  {'PB-F1':>8}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        pb_str = f"{r['pb_f1']*100:7.1f}%" if not (r['pb_f1'] != r['pb_f1']) else "     n/a"
        print(
            f"  {r['label']:<14}  {r['ms_acc']*100:7.2f}%"
            f"  {r['ms_macro_f1']:>12.3f}  {pb_str}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train linear probes on all PTB representations")
    p.add_argument("--rep-dir",      required=True, help="Dir with ptb_z_*.npz etc.")
    p.add_argument("--dense-train",  required=True, help="dense_train_full.npz")
    p.add_argument("--dense-eval",   required=True, help="dense_eval_held_out.npz")
    p.add_argument("--ssae-train",   required=True, help="Existing SSAE train .npz")
    p.add_argument("--ssae-eval",    required=True, help="Existing SSAE eval .npz")
    p.add_argument("--pb-dense",     default=None,  help="processbench_dense_gsm8k.npz")
    p.add_argument("--pb-ssae",      default=None,  help="processbench_gsm8k.npz")
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seeds",        type=int, nargs="+", default=[42, 43, 44, 45])
    p.add_argument("--device",       default="cuda", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = build_rep_registry(args)
    print(f"\nEvaluating {len(reps)} representations × {len(args.seeds)} seeds "
          f"= {len(reps) * len(args.seeds)} probes\n")

    summary_rows = []

    for rep in reps:
        label = rep["label"]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        train_h, train_y = _load_npz(rep["train"])
        eval_h,  eval_y  = _load_npz(rep["eval_ms"])
        print(f"  train={len(train_y):,}  eval={len(eval_y):,}  dim={train_h.shape[1]}")

        seed_results = []
        for seed in args.seeds:
            probe_path = out_dir / f"{label}_linear_probe_seed{seed}.pt"
            res = train_and_eval_probe(
                train_h      = train_h,
                train_y      = train_y,
                eval_ms_h    = eval_h,
                eval_ms_y    = eval_y,
                pb_npz_path  = rep.get("pb"),
                seed         = seed,
                epochs       = args.epochs,
                batch_size   = args.batch_size,
                lr           = args.lr,
                device       = args.device,
                out_path     = probe_path,
            )
            seed_results.append(res)
            print(f"  seed={seed}  ms_acc={res['ms_acc']*100:.2f}%  "
                  f"macro_f1={res['ms_macro_f1']:.3f}  pb_f1={res['pb_f1']*100:.1f}%")

        # Aggregate across seeds
        ms_accs     = [r["ms_acc"]      for r in seed_results]
        ms_macros   = [r["ms_macro_f1"] for r in seed_results]
        pb_f1s      = [r["pb_f1"]       for r in seed_results if r["pb_f1"] == r["pb_f1"]]

        row = {
            "label":       label,
            "ms_acc":      statistics.mean(ms_accs),
            "ms_acc_std":  statistics.stdev(ms_accs) if len(ms_accs) > 1 else 0.0,
            "ms_macro_f1": statistics.mean(ms_macros),
            "ms_mf1_std":  statistics.stdev(ms_macros) if len(ms_macros) > 1 else 0.0,
            "pb_f1":       statistics.mean(pb_f1s) if pb_f1s else float("nan"),
            "pb_f1_std":   statistics.stdev(pb_f1s) if len(pb_f1s) > 1 else 0.0,
        }
        summary_rows.append(row)

        print(f"  AGGREGATE: ms_acc={row['ms_acc']*100:.2f}±{row['ms_acc_std']*100:.2f}%  "
              f"macro_f1={row['ms_macro_f1']:.3f}±{row['ms_mf1_std']:.3f}  "
              f"pb_f1={row['pb_f1']*100:.1f}±{row['pb_f1_std']*100:.1f}%")

    # Final comparison table
    print("\n\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print_table(summary_rows)

    # Save summary table as markdown
    md_lines = [
        "| Representation | MS Acc | MS Macro F1 | PB-F1 |",
        "| --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        pb_str = f"{r['pb_f1']*100:.1f}%" if r['pb_f1'] == r['pb_f1'] else "n/a"
        md_lines.append(
            f"| {r['label']} | {r['ms_acc']*100:.2f}% ± {r['ms_acc_std']*100:.2f}% "
            f"| {r['ms_macro_f1']:.3f} ± {r['ms_mf1_std']:.3f} | {pb_str} |"
        )
    md_path = out_dir / "summary_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nSummary table → {md_path}")


if __name__ == "__main__":
    main()
