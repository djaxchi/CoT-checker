"""Train reuse-only probe variants over an existing SSAE latent cache.

This audit keeps the original paper Qwen2.5-0.5B SSAE checkpoint fixed and
changes only the lightweight scoring protocol over cached latents.

Variants:
  * positive: one-class/OOD baseline. Fit a viable centroid using only y=0
    rows from probe_train_40k. Score by cosine similarity to that centroid.
  * contrastive: supervised linear probe on all probe_train_40k rows. The
    probe is trained to emit viable/correct scores with BCE plus a batchwise
    label-margin term.

IMPORTANT: for SSAE PRM800K JSONL, label convention is y=0 viable and
y=1 non-viable, as defined in src/ssae/dataset.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THRESHOLD_GRID = [round(0.1 * i, 1) for i in range(1, 11)]
EPS = 1e-12


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iter_minibatches(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, EPS)


def cosine_viable_scores(z: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    z_norm = l2_normalize(z.astype(np.float32), axis=1)
    c = centroid.astype(np.float32)
    c = c / max(float(np.linalg.norm(c)), EPS)
    cos = z_norm @ c
    return ((cos + 1.0) * 0.5).astype(np.float32)


def probe_viable_scores(
    probe: LinearProbe, z: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    probe.eval()
    out = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            chunk = torch.from_numpy(z[i:i + batch_size]).to(device)
            out.append(torch.sigmoid(probe(chunk)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def select_error_threshold(error_scores: np.ndarray, y_error: np.ndarray) -> tuple[float, float, float]:
    """Select threshold for evaluator convention: higher score means error."""
    best_t = THRESHOLD_GRID[0]
    best_bacc = -1.0
    best_f1 = 0.0
    for t in THRESHOLD_GRID:
        pred = (error_scores > t).astype(np.int64)
        tp = int(((pred == 1) & (y_error == 1)).sum())
        tn = int(((pred == 0) & (y_error == 0)).sum())
        fp = int(((pred == 1) & (y_error == 0)).sum())
        fn = int(((pred == 0) & (y_error == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bacc = 0.5 * (tpr + tnr)
        prec = tp / max(tp + fp, 1)
        rec = tpr
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if bacc > best_bacc + 1e-12:
            best_bacc = bacc
            best_t = t
            best_f1 = f1
    return best_t, best_bacc, best_f1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True)
    p.add_argument("--variant", required=True, choices=["positive", "contrastive"])
    p.add_argument("--latents_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs_probe", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_probe", type=float, default=1e-3)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--expected_dim", type=int, default=896)
    return p.parse_args()


def load_meta_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_pair(latents_dir: Path, stem: str, expected_dim: int) -> tuple[np.ndarray, np.ndarray, int]:
    z_path = latents_dir / f"{stem}_z.npy"
    y_path = latents_dir / f"{stem}_y.npy"
    meta_path = latents_dir / f"{stem}_meta.jsonl"
    z = np.load(z_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    n_meta = load_meta_count(meta_path)
    if z.ndim != 2 or z.shape[1] != expected_dim:
        raise ValueError(f"{stem}: expected z dim {expected_dim}, got shape {z.shape}")
    if z.shape[0] != y.shape[0] or z.shape[0] != n_meta:
        raise ValueError(
            f"{stem}: row mismatch z={z.shape[0]} y={y.shape[0]} meta={n_meta}"
        )
    if not np.all(np.isfinite(z)):
        raise ValueError(f"{stem}: NaN/Inf in {z_path}")
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{stem}: y must be in {{0,1}}, got {sorted(labels)}")
    return z, y, n_meta


def contrastive_margin_loss(logits: torch.Tensor, y_error: torch.Tensor) -> torch.Tensor:
    viable = logits[y_error == 0]
    nonviable = logits[y_error == 1]
    if viable.numel() == 0 or nonviable.numel() == 0:
        return logits.new_tensor(0.0)
    return F.relu(1.0 - (viable.mean() - nonviable.mean()))


def train_contrastive_probe(
    z_train: np.ndarray,
    y_error_train: np.ndarray,
    z_val: np.ndarray,
    y_error_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
    seed: int,
) -> LinearProbe:
    rng = np.random.default_rng(seed)
    probe = LinearProbe(z_train.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)

    z_train_t = torch.from_numpy(z_train).to(device)
    y_error_train_t = torch.from_numpy(y_error_train.astype(np.float32)).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)
    y_viable_val_t = torch.from_numpy((1 - y_error_val).astype(np.float32)).to(device)

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    bad_epochs = 0

    for _epoch in range(epochs):
        probe.train()
        for batch_idx in iter_minibatches(z_train_t.shape[0], batch_size, rng):
            ib = torch.from_numpy(batch_idx).to(device)
            logits = probe(z_train_t.index_select(0, ib))
            y_error_b = y_error_train_t.index_select(0, ib)
            y_viable_b = 1.0 - y_error_b
            loss = F.binary_cross_entropy_with_logits(logits, y_viable_b)
            loss = loss + 0.1 * contrastive_margin_loss(logits, y_error_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(z_val_t)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_viable_val_t).item()
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    probe.load_state_dict(best_state)
    return probe


def print_variant_counts(method: str, variant: str, y_used: np.ndarray, score_direction: str) -> None:
    print(f"method_name={method}", flush=True)
    print(f"variant_type={variant}", flush=True)
    print(f"n_total={y_used.shape[0]}", flush=True)
    print(f"n_viable_y0={int((y_used == 0).sum())}", flush=True)
    print(f"n_nonviable_y1={int((y_used == 1).sum())}", flush=True)
    print(f"y_mean={float(y_used.mean()) if y_used.size else 0.0:.6f}", flush=True)
    print(f"score_direction={score_direction}", flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""

    z_train_all, y_train_all, train_meta_n = load_pair(
        args.latents_dir, "probe_train_40k", args.expected_dim
    )
    z_val, y_val, val_meta_n = load_pair(args.latents_dir, "val_1k", args.expected_dim)
    if z_train_all.shape[0] != 40000:
        raise ValueError(f"probe_train_40k must have 40000 rows, got {z_train_all.shape[0]}")
    if z_val.shape[0] != 1000:
        raise ValueError(f"val_1k must have 1000 rows, got {z_val.shape[0]}")

    t0 = time.time()
    score_direction = "viable_higher"
    if args.variant == "positive":
        keep = np.flatnonzero(y_train_all == 0)
        y_used = y_train_all[keep]
        if keep.shape[0] != 20000:
            raise ValueError(f"positive expected 20000 viable y=0 rows, got {keep.shape[0]}")
        print_variant_counts(args.method, args.variant, y_used, score_direction)
        z_viable_norm = l2_normalize(z_train_all[keep], axis=1)
        centroid = z_viable_norm.mean(axis=0).astype(np.float32)
        centroid = centroid / max(float(np.linalg.norm(centroid)), EPS)
        np.savez(args.out_dir / "centroid_scorer.npz", centroid=centroid)
        val_scores_viable = cosine_viable_scores(z_val, centroid)
        train_time = time.time() - t0
        probe_protocol = "one-class viable centroid fit on y=0 rows only"
    else:
        y_used = y_train_all
        n0 = int((y_used == 0).sum())
        n1 = int((y_used == 1).sum())
        if n0 != 20000 or n1 != 20000:
            raise ValueError(
                f"contrastive expected 20000 y=0 and 20000 y=1 rows, got {n0}/{n1}"
            )
        print_variant_counts(args.method, args.variant, y_used, score_direction)
        probe = train_contrastive_probe(
            z_train=z_train_all,
            y_error_train=y_train_all,
            z_val=z_val,
            y_error_val=y_val,
            epochs=args.epochs_probe,
            batch_size=args.batch_size,
            lr=args.lr_probe,
            patience=args.early_stopping_patience,
            device=device,
            seed=args.seed,
        )
        train_time = time.time() - t0
        torch.save(probe.state_dict(), args.out_dir / "linear_probe.pt")
        val_scores_viable = probe_viable_scores(probe, z_val, args.batch_size, device)
        probe_protocol = "all rows with viable-target BCE plus 0.1 * batchwise label-margin loss"

    if not np.all(np.isfinite(val_scores_viable)):
        raise ValueError("NaN/Inf in viable validation scores")
    val_scores_error = 1.0 - val_scores_viable
    np.save(args.out_dir / "val_scores_viable.npy", val_scores_viable)
    np.save(args.out_dir / "val_scores.npy", val_scores_error)
    threshold_error, best_bacc, val_f1 = select_error_threshold(val_scores_error, y_val)
    selected_viable_threshold = 1.0 - threshold_error
    (args.out_dir / "threshold.json").write_text(json.dumps({
        "selected_threshold": threshold_error,
        "selected_threshold_score_direction": "error_higher_for_processbench_evaluator",
        "selected_viable_threshold": selected_viable_threshold,
        "raw_score_direction": score_direction,
        "selection_metric": "balanced_accuracy",
        "best_val_balanced_accuracy": best_bacc,
        "val_f1_binary": val_f1,
        "threshold_grid": THRESHOLD_GRID,
        "latent_source": str(args.latents_dir),
        "fixed_ssae_checkpoint_note": (
            "Probe/scorer-only variant over the same original paper Qwen2.5-0.5B "
            "SSAE latents; this is not a distinct SSAE representation checkpoint."
        ),
    }, indent=2))

    y_counts = {
        "train_all_y0": int((y_train_all == 0).sum()),
        "train_all_y1": int((y_train_all == 1).sum()),
        "train_used_y0": int((y_used == 0).sum()),
        "train_used_y1": int((y_used == 1).sum()),
        "val_y0": int((y_val == 0).sum()),
        "val_y1": int((y_val == 1).sum()),
    }
    (args.out_dir / "probe_train_metrics.json").write_text(json.dumps({
        "method": args.method,
        "variant": args.variant,
        "seed": args.seed,
        "n_latents": int(z_train_all.shape[1]),
        "probe_train_n_all": int(z_train_all.shape[0]),
        "probe_train_n_used": int(y_used.shape[0]),
        "probe_train_meta_rows": train_meta_n,
        "val_n": int(z_val.shape[0]),
        "val_meta_rows": val_meta_n,
        "probe_train_time_sec": train_time,
        "best_val_balanced_accuracy": best_bacc,
        "val_f1_binary": val_f1,
        "selected_threshold": threshold_error,
        "selected_viable_threshold": selected_viable_threshold,
        "raw_score_direction": score_direction,
        "label_counts": y_counts,
        "gpu_name": gpu_name,
        "device": device.type,
        "probe_protocol": probe_protocol,
    }, indent=2))

    print(
        f"[{args.method}] variant={args.variant} "
        f"threshold_error={threshold_error} "
        f"threshold_viable={selected_viable_threshold:.6f} "
        f"val_bacc={best_bacc:.4f} val_f1={val_f1:.4f} "
        f"train_used={y_used.shape[0]} dim={z_train_all.shape[1]} "
        f"score_direction={score_direction}",
        flush=True,
    )


if __name__ == "__main__":
    main()
