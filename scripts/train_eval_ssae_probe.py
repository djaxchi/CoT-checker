"""Train a fresh linear probe on SSAE latents, select threshold on val_1k,
evaluate on ProcessBench-GSM8K.

Reuses the easy-probe helpers (LinearProbe, train_linear_probe, probe_scores,
select_threshold, evaluate_processbench, THRESHOLD_GRID) verbatim via import,
to guarantee identical metric/threshold logic across leaderboards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the easy-probe module file by path (script has no package context).
_EASY_PATH = ROOT / "scripts" / "train_easy_probe_method.py"
_spec = importlib.util.spec_from_file_location("easy_probe_module", _EASY_PATH)
_easy = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_easy)

LinearProbe = _easy.LinearProbe
train_linear_probe = _easy.train_linear_probe
probe_scores = _easy.probe_scores
select_threshold = _easy.select_threshold
evaluate_processbench = _easy.evaluate_processbench
THRESHOLD_GRID = _easy.THRESHOLD_GRID
seed_everything = _easy.seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--latents_dir", required=True, type=Path,
                   help="Directory containing probe_train_40k_z.npy etc.")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs_probe", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_probe", type=float, default=1e-3)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--smoke", action="store_true",
                   help="Skip 40k-row assertions for smoke runs.")
    return p.parse_args()


def load_pair(latents_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(latents_dir / f"{stem}_z.npy").astype(np.float32)
    y = np.load(latents_dir / f"{stem}_y.npy").astype(np.int64)
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"{stem}: z/y row mismatch {z.shape[0]} vs {y.shape[0]}")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError(f"{stem}: y must be in {{0,1}}, got {np.unique(y).tolist()}")
    return z, y


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""

    z_train, y_train = load_pair(args.latents_dir, "probe_train_40k")
    z_val, y_val = load_pair(args.latents_dir, "val_1k")

    if not args.smoke:
        if z_train.shape[0] != 40000:
            raise ValueError(f"probe_train_40k must have 40000 rows, got {z_train.shape[0]}")
        if z_val.shape[0] != 1000:
            raise ValueError(f"val_1k must have 1000 rows, got {z_val.shape[0]}")

    # ProcessBench latents + meta
    pb_z = np.load(args.latents_dir / "pb_gsm8k_step_z.npy").astype(np.float32)
    with (args.latents_dir / "pb_gsm8k_step_meta.jsonl").open() as f:
        pb_meta = [json.loads(line) for line in f if line.strip()]
    if len(pb_meta) != pb_z.shape[0]:
        raise ValueError(
            f"PB meta rows ({len(pb_meta)}) != z rows ({pb_z.shape[0]})"
        )
    # Validate trace-level label consistency, mirroring easy-probe loader.
    seen: dict[str, int] = {}
    for row in pb_meta:
        tid = row["id"]
        lbl = int(row["label"])
        if tid in seen and seen[tid] != lbl:
            raise ValueError(
                f"ProcessBench trace {tid} has inconsistent labels: "
                f"{seen[tid]} vs {lbl}"
            )
        seen[tid] = lbl

    # ---- Probe training ---------------------------------------------------
    t0 = time.time()
    probe = train_linear_probe(
        z_train=z_train, y_train=y_train,
        z_val=z_val, y_val=y_val,
        epochs=args.epochs_probe, batch_size=args.batch_size,
        lr=args.lr_probe, patience=args.early_stopping_patience,
        device=device, seed=args.seed,
    )
    probe_train_time = time.time() - t0
    torch.save(probe.state_dict(), args.out_dir / "linear_probe.pt")

    val_scores = probe_scores(probe, z_val, args.batch_size, device)
    np.save(args.out_dir / "val_scores.npy", val_scores)
    threshold, best_bacc, val_f1 = select_threshold(val_scores, y_val)
    (args.out_dir / "threshold.json").write_text(json.dumps(
        {
            "selected_threshold": threshold,
            "selection_metric": "balanced_accuracy",
            "best_val_balanced_accuracy": best_bacc,
            "val_f1_binary": val_f1,
            "threshold_grid": THRESHOLD_GRID,
        }, indent=2,
    ))

    # ---- ProcessBench eval -----------------------------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_t0 = time.time()
    pb_scores_arr = probe_scores(probe, pb_z, args.batch_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    pb_rows, pb_metrics = evaluate_processbench(pb_scores_arr, pb_meta, threshold)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_time = time.time() - eval_t0

    with (args.out_dir / "pb_step_scores.jsonl").open("w") as f:
        for row in pb_rows:
            f.write(json.dumps(row) + "\n")
    with (args.out_dir / "pb_predictions.jsonl").open("w") as f:
        for row in pb_rows:
            f.write(json.dumps({
                "id": row["id"], "label": row["label"],
                "prediction": row["prediction"], "threshold": row["threshold"],
            }) + "\n")

    n_steps = pb_z.shape[0]
    n_traces = pb_metrics["n_traces"]
    eval_metrics = {
        "method": args.method,
        "threshold": threshold,
        **pb_metrics,
        "eval_time_sec": eval_time,
        "mean_step_latency_ms": eval_time * 1000.0 / max(n_steps, 1),
        "mean_trace_latency_ms": eval_time * 1000.0 / max(n_traces, 1),
        "latency_scope": (
            "includes: linear probe scoring on cached SSAE latents and "
            "trace aggregation. excludes: SSAE forward (covered by "
            "extract_ssae_latents) and disk I/O."
        ),
    }
    (args.out_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    probe_metrics = {
        "method": args.method,
        "seed": args.seed,
        "n_latents": int(z_train.shape[1]),
        "probe_train_n": int(z_train.shape[0]),
        "val_n": int(z_val.shape[0]),
        "probe_train_time_sec": probe_train_time,
        "best_val_balanced_accuracy": best_bacc,
        "val_f1_binary": val_f1,
        "gpu_name": gpu_name,
        "device": device.type,
    }
    (args.out_dir / "probe_train_metrics.json").write_text(json.dumps(probe_metrics, indent=2))

    print(
        f"[{args.method}] threshold={threshold} "
        f"F1_PB={eval_metrics['F1_PB']:.4f} "
        f"Acc_err={eval_metrics['Acc_error']:.4f} "
        f"Acc_corr={eval_metrics['Acc_correct']:.4f} "
        f"val_bacc={best_bacc:.4f}"
    )


if __name__ == "__main__":
    main()
