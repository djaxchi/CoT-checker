#!/usr/bin/env python3
"""Stage B of the model-size DenseLinear ablation: train the probe only.

This is the SAME DenseLinear probe as Sprint 1. To guarantee identical math we
IMPORT the training/threshold functions from train_easy_probe_method.py rather
than reimplementing them; we only decouple probe training from ProcessBench
evaluation so the pipeline fits the staged SLURM DAG (PB encoding happens later,
in Stage C, on the GPU workers).

Inputs (the merged PRM800K cache produced by merge_prm800k_encoded_shards.py):
    <cache_dir>/probe_train_40k_h.npy + _y.npy
    <cache_dir>/val_1k_h.npy          + _y.npy

Outputs (into <out_dir>):
    linear_probe.pt            state_dict {fc.weight, fc.bias}  (input_dim = hidden_size)
    threshold.json             {selected_threshold, ...}  (consumed by
                               evaluate_saved_probe_on_processbench.py --threshold_json)
    val_threshold_metrics.json {selected_threshold, val_balanced_accuracy, val_f1, ...}
    train_metrics.json         {method, hidden_dim, n_train, n_val, timings, ...}
    val_scores.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the EXACT Sprint 1 DenseLinear probe + threshold logic.
from train_easy_probe_method import (  # type: ignore  # noqa: E402
    LinearProbe,
    load_npy_pair,
    probe_scores,
    seed_everything,
    select_threshold,
    train_linear_probe,
    resolve_threshold_grid,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", type=Path, required=True,
                   help="Merged PRM800K cache dir with {stem}_h.npy/_y.npy.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--probe_train_stem", type=str, default="probe_train_40k")
    p.add_argument("--val_stem", type=str, default="val_1k")
    p.add_argument("--seed", type=int, default=42)
    # Defaults match train_easy_probe_method.py's dense_linear path exactly.
    p.add_argument("--epochs_probe", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_probe", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="AdamW L2. 0.0 = original pipeline. Helps the wide multilayer "
                        "concat probe (100k-dim) where the offline bake-off needed strong L2.")
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--threshold_grid", type=str, default=None,
                   help="Float step in (0,1) or comma list. Default 0.1..1.0 "
                        "(the Sprint 1 val-selection grid).")
    p.add_argument("--model_name", type=str, default=None,
                   help="Recorded in train_metrics.json for provenance.")
    p.add_argument("--skip_size_asserts", action="store_true",
                   help="Skip the 40k/1k row-count guard (smoke runs only).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""

    probe_train_h, probe_train_y = load_npy_pair(args.cache_dir, args.probe_train_stem)
    val_h, val_y = load_npy_pair(args.cache_dir, args.val_stem)
    hidden_dim = probe_train_h.shape[1]

    if not args.skip_size_asserts:
        assert probe_train_h.shape[0] == 40000, (
            f"{args.probe_train_stem} must have 40000 rows, got {probe_train_h.shape[0]}. "
            "Pass --skip_size_asserts only for smoke runs."
        )
        assert val_h.shape[0] == 1000, (
            f"{args.val_stem} must have 1000 rows, got {val_h.shape[0]}."
        )

    threshold_grid = resolve_threshold_grid(args.threshold_grid)

    # ---- Train the linear probe directly on the raw hidden states (DenseLinear).
    t0 = time.time()
    probe = train_linear_probe(
        z_train=probe_train_h,
        y_train=probe_train_y,
        z_val=val_h,
        y_val=val_y,
        epochs=args.epochs_probe,
        batch_size=args.batch_size,
        lr=args.lr_probe,
        patience=args.early_stopping_patience,
        device=device,
        seed=args.seed,
        weight_decay=args.weight_decay,
    )
    probe_train_time = time.time() - t0
    torch.save(probe.state_dict(), args.out_dir / "linear_probe.pt")

    # ---- Threshold selection on PRM800K val (balanced-accuracy, Sprint 1).
    val_scores = probe_scores(probe, val_h, args.batch_size, device)
    np.save(args.out_dir / "val_scores.npy", val_scores)
    sel_t, best_bacc, val_f1 = select_threshold(val_scores, val_y, threshold_grid)

    threshold_payload = {
        "selected_threshold": sel_t,
        "selection_metric": "balanced_accuracy",
        "best_val_balanced_accuracy": best_bacc,
        "val_f1_binary": val_f1,
        "threshold_grid": threshold_grid,
        "fixed_threshold_0p5": 0.5,
    }
    (args.out_dir / "threshold.json").write_text(json.dumps(threshold_payload, indent=2))
    (args.out_dir / "val_threshold_metrics.json").write_text(json.dumps({
        "method": "dense_linear",
        "selected_threshold": sel_t,
        "selection_metric": "balanced_accuracy",
        "val_balanced_accuracy": best_bacc,
        "val_balanced_accuracy_pct": round(100.0 * best_bacc, 4),
        "val_f1_binary": val_f1,
        "val_n": int(val_h.shape[0]),
        "threshold_grid": threshold_grid,
    }, indent=2))

    (args.out_dir / "train_metrics.json").write_text(json.dumps({
        "method": "dense_linear",
        "model_name": args.model_name,
        "seed": args.seed,
        "hidden_dim": int(hidden_dim),
        "probe_train_n": int(probe_train_h.shape[0]),
        "val_n": int(val_h.shape[0]),
        "probe_train_stem": args.probe_train_stem,
        "val_stem": args.val_stem,
        "epochs_probe": args.epochs_probe,
        "batch_size": args.batch_size,
        "lr_probe": args.lr_probe,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "probe_train_time_sec": probe_train_time,
        "selected_threshold": sel_t,
        "val_balanced_accuracy": best_bacc,
        "gpu_name": gpu_name,
        "device": device.type,
    }, indent=2))

    print(
        f"[train] dense_linear hidden_dim={hidden_dim} "
        f"val_bacc={best_bacc:.4f} val_f1={val_f1:.4f} selected_t={sel_t} "
        f"(train {probe_train_time:.1f}s)"
    )


if __name__ == "__main__":
    main()
