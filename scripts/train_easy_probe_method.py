"""Train and evaluate one easy-probe method end-to-end.

Methods: random, dense_linear, sae_positive, sae_mixed, sae_contrastive.

Trains the representation (if any) on PRM800K cached hidden states, trains a
fresh linear probe on probe_train_40k, selects a threshold on val_1k using the
10-point grid 0.1..1.0, then scores ProcessBench-GSM8K steps and writes the
official ProcessBench first-error metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

METHODS = ["random", "dense_linear", "sae_positive", "sae_mixed", "sae_contrastive"]
THRESHOLD_GRID = [round(0.1 * i, 1) for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_npy_pair(cache_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    h = np.load(cache_dir / f"{stem}_h.npy")
    y = np.load(cache_dir / f"{stem}_y.npy")
    if h.shape[0] != y.shape[0]:
        raise ValueError(f"{stem}: h/y row mismatch {h.shape[0]} vs {y.shape[0]}")
    y = y.astype(np.int64)
    unique = np.unique(y)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"{stem}: y must contain only {{0, 1}}, got {unique.tolist()}")
    return h.astype(np.float32), y


def require_pb_cache(pb_cache_dir: Path) -> tuple[np.ndarray, list[dict]]:
    h_path = pb_cache_dir / "pb_gsm8k_step_h.npy"
    meta_path = pb_cache_dir / "pb_gsm8k_step_meta.jsonl"
    if not h_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"ProcessBench cached hidden states missing: expected\n"
            f"  {h_path}\n  {meta_path}\n"
            "Cannot evaluate. Re-run the ProcessBench encoder."
        )
    h = np.load(h_path).astype(np.float32)
    meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if len(meta) != h.shape[0]:
        raise ValueError(
            f"ProcessBench meta rows ({len(meta)}) != h rows ({h.shape[0]})"
        )
    # Each ProcessBench trace must have a unique label across its rows.
    seen: dict[str, int] = {}
    for row in meta:
        tid = row["id"]
        lbl = int(row["label"])
        if tid in seen and seen[tid] != lbl:
            raise ValueError(
                f"ProcessBench trace {tid} has inconsistent labels: "
                f"{seen[tid]} vs {lbl}"
            )
        seen[tid] = lbl
    return h, meta


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SAE(nn.Module):
    """Single-layer SAE: z = ReLU(W_enc h + b_enc); h_hat = W_dec z + b_dec."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=True)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(h))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        h_hat = self.decoder(z)
        return z, h_hat


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def iter_minibatches(
    n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True
) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def train_sae(
    h_train: np.ndarray,
    y_train: np.ndarray | None,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_weight: float,
    bce_weight: float,
    contrastive: bool,
    device: torch.device,
    seed: int,
) -> tuple[SAE, dict]:
    rng = np.random.default_rng(seed)
    sae = SAE(hidden_dim, latent_dim).to(device)
    aux: nn.Linear | None = None
    params = list(sae.parameters())
    if contrastive:
        aux = nn.Linear(latent_dim, 1).to(device)
        params += list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    h_tensor = torch.from_numpy(h_train).to(device)
    y_tensor = (
        torch.from_numpy(y_train.astype(np.float32)).to(device)
        if y_train is not None
        else None
    )

    n = h_tensor.shape[0]
    final_mse = float("nan")
    final_l1 = float("nan")
    final_bce = float("nan")
    for epoch in range(epochs):
        sae.train()
        if aux is not None:
            aux.train()
        ep_mse, ep_l1, ep_bce, n_batches = 0.0, 0.0, 0.0, 0
        for batch_idx in iter_minibatches(n, batch_size, rng):
            ib = torch.from_numpy(batch_idx).to(device)
            h_b = h_tensor.index_select(0, ib)
            z, h_hat = sae(h_b)
            mse = F.mse_loss(h_hat, h_b)
            l1 = z.abs().mean()
            loss = mse + l1_weight * l1
            bce_val = torch.tensor(0.0, device=device)
            if contrastive and aux is not None and y_tensor is not None:
                y_b = y_tensor.index_select(0, ib)
                logit = aux(z).squeeze(-1)
                bce_val = F.binary_cross_entropy_with_logits(logit, y_b)
                loss = loss + bce_weight * bce_val
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_mse += mse.item()
            ep_l1 += l1.item()
            ep_bce += bce_val.item()
            n_batches += 1
        final_mse = ep_mse / max(n_batches, 1)
        final_l1 = ep_l1 / max(n_batches, 1)
        final_bce = ep_bce / max(n_batches, 1)
    stats = {
        "final_reconstruction_mse": final_mse,
        "final_l1_mean": final_l1,
        "final_aux_bce": final_bce if contrastive else None,
    }
    return sae, stats


def train_linear_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
    seed: int,
) -> LinearProbe:
    rng = np.random.default_rng(seed)
    in_dim = z_train.shape[1]
    probe = LinearProbe(in_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)

    z_train_t = torch.from_numpy(z_train).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    n = z_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    bad_epochs = 0

    for epoch in range(epochs):
        probe.train()
        for batch_idx in iter_minibatches(n, batch_size, rng):
            ib = torch.from_numpy(batch_idx).to(device)
            zb = z_train_t.index_select(0, ib)
            yb = y_train_t.index_select(0, ib)
            logit = probe(zb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val_logit = probe(z_val_t)
            val_loss = F.binary_cross_entropy_with_logits(val_logit, y_val_t).item()
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


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def encode_with_sae(sae: SAE, h: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    sae.eval()
    out = []
    with torch.no_grad():
        for i in range(0, h.shape[0], batch_size):
            chunk = torch.from_numpy(h[i : i + batch_size]).to(device)
            z = sae.encode(chunk).cpu().numpy()
            out.append(z)
    return np.concatenate(out, axis=0)


def probe_scores(probe: LinearProbe, z: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    probe.eval()
    out = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            chunk = torch.from_numpy(z[i : i + batch_size]).to(device)
            logit = probe(chunk)
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    best_t = THRESHOLD_GRID[0]
    best_bacc = -1.0
    best_f1 = 0.0
    for t in THRESHOLD_GRID:
        pred = (scores > t).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
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


# ---------------------------------------------------------------------------
# ProcessBench evaluation
# ---------------------------------------------------------------------------

def evaluate_processbench(
    scores: np.ndarray, meta: list[dict], threshold: float
) -> tuple[list[dict], dict]:
    groups: dict[str, list[tuple[int, float]]] = {}
    labels: dict[str, int] = {}
    n_steps_map: dict[str, int] = {}
    for i, row in enumerate(meta):
        tid = row["id"]
        groups.setdefault(tid, []).append((int(row["step_idx"]), float(scores[i])))
        labels[tid] = int(row["label"])
        n_steps_map[tid] = int(row["n_steps"])

    rows: list[dict] = []
    n_error = n_correct = 0
    acc_error_hits = 0
    acc_correct_hits = 0
    exact_all = 0
    for tid, items in groups.items():
        items.sort(key=lambda x: x[0])
        step_scores = [s for _, s in items]
        pred = -1
        for t_idx, s in items:
            if s > threshold:
                pred = t_idx
                break
        label = labels[tid]
        rows.append(
            {
                "id": tid,
                "label": label,
                "n_steps": n_steps_map[tid],
                "scores": step_scores,
                "threshold": threshold,
                "prediction": pred,
            }
        )
        if label == -1:
            n_correct += 1
            if pred == -1:
                acc_correct_hits += 1
        else:
            n_error += 1
            if pred == label:
                acc_error_hits += 1
        if pred == label:
            exact_all += 1

    acc_error = acc_error_hits / max(n_error, 1) if n_error else 0.0
    acc_correct = acc_correct_hits / max(n_correct, 1) if n_correct else 0.0
    denom = acc_error + acc_correct
    f1_pb = (2 * acc_error * acc_correct / denom) if denom > 0 else 0.0
    exact_match_all = exact_all / max(len(groups), 1)

    metrics = {
        "n_traces": len(groups),
        "n_error_traces": n_error,
        "n_correct_traces": n_correct,
        "Acc_error": acc_error,
        "Acc_correct": acc_correct,
        "F1_PB": f1_pb,
        "Exact_match_all": exact_match_all,
    }
    return rows, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--cache_dir", required=True, type=Path)
    p.add_argument("--pb_cache_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs_sae", type=int, default=20)
    p.add_argument("--epochs_probe", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_sae", type=float, default=1e-3)
    p.add_argument("--lr_probe", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--l1_weight", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=None,
                   help="Default: hidden_dim (no overcomplete unless explicitly set).")
    p.add_argument("--early_stopping_patience", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""

    # Persist config
    cfg = vars(args).copy()
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    cfg["device"] = device.type
    cfg["gpu_name"] = gpu_name
    (args.out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    # Load PRM800K data
    probe_train_h, probe_train_y = load_npy_pair(args.cache_dir, "probe_train_40k")
    val_h, val_y = load_npy_pair(args.cache_dir, "val_1k")
    hidden_dim = probe_train_h.shape[1]
    latent_dim = args.latent_dim if args.latent_dim is not None else hidden_dim

    assert probe_train_h.shape[0] == 40000, (
        f"probe_train_40k must have 40000 rows, got {probe_train_h.shape[0]}"
    )
    assert val_h.shape[0] == 1000, (
        f"val_1k must have 1000 rows, got {val_h.shape[0]}"
    )

    # Require ProcessBench cache up-front
    pb_h, pb_meta = require_pb_cache(args.pb_cache_dir)
    assert pb_h.shape[1] == hidden_dim, (
        f"ProcessBench hidden_dim ({pb_h.shape[1]}) != PRM800K hidden_dim ({hidden_dim})"
    )

    method = args.method

    representation_train_time = 0.0
    probe_train_time = 0.0
    rep_train_n = 0
    sae_stats = {
        "final_reconstruction_mse": None,
        "final_l1_mean": None,
        "final_aux_bce": None,
    }

    total_t0 = time.time()

    # ---- Representation step ----------------------------------------------
    sae: SAE | None = None
    if method in ("sae_positive", "sae_mixed", "sae_contrastive"):
        if method == "sae_positive":
            rep_h, rep_y = load_npy_pair(args.cache_dir, "pos_base_20k")
            contrastive = False
            rep_y_for_train = None
        elif method == "sae_mixed":
            rep_h, rep_y = load_npy_pair(args.cache_dir, "mixed_train_40k")
            contrastive = False
            rep_y_for_train = None
        else:  # sae_contrastive
            rep_h, rep_y = load_npy_pair(args.cache_dir, "mixed_train_40k")
            contrastive = True
            rep_y_for_train = rep_y
        rep_train_n = rep_h.shape[0]
        t0 = time.time()
        sae, sae_stats = train_sae(
            h_train=rep_h,
            y_train=rep_y_for_train,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=args.epochs_sae,
            batch_size=args.batch_size,
            lr=args.lr_sae,
            l1_weight=args.l1_weight,
            bce_weight=args.bce_weight,
            contrastive=contrastive,
            device=device,
            seed=args.seed,
        )
        representation_train_time = time.time() - t0
        torch.save(sae.state_dict(), args.out_dir / "representation.pt")
        for p in sae.parameters():
            p.requires_grad_(False)
        sae.eval()

    # ---- Build probe inputs for training/val (not ProcessBench yet) -------
    if sae is not None:
        z_probe_train = encode_with_sae(sae, probe_train_h, args.batch_size, device)
        z_val = encode_with_sae(sae, val_h, args.batch_size, device)
    else:
        z_probe_train = probe_train_h
        z_val = val_h

    # ---- Final probe -------------------------------------------------------
    if method == "random":
        rng = np.random.default_rng(args.seed)
        val_scores = rng.uniform(0.0, 1.0, size=val_h.shape[0]).astype(np.float32)
        probe = None
    else:
        t0 = time.time()
        probe = train_linear_probe(
            z_train=z_probe_train,
            y_train=probe_train_y,
            z_val=z_val,
            y_val=val_y,
            epochs=args.epochs_probe,
            batch_size=args.batch_size,
            lr=args.lr_probe,
            patience=args.early_stopping_patience,
            device=device,
            seed=args.seed,
        )
        probe_train_time = time.time() - t0
        torch.save(probe.state_dict(), args.out_dir / "linear_probe.pt")
        val_scores = probe_scores(probe, z_val, args.batch_size, device)

    # Random method: zero out timing per spec
    if method == "random":
        representation_train_time = 0.0
        probe_train_time = 0.0

    # ---- Threshold selection ----------------------------------------------
    np.save(args.out_dir / "val_scores.npy", val_scores)
    threshold, best_bacc, val_f1 = select_threshold(val_scores, val_y)
    (args.out_dir / "threshold.json").write_text(
        json.dumps(
            {
                "selected_threshold": threshold,
                "selection_metric": "balanced_accuracy",
                "best_val_balanced_accuracy": best_bacc,
                "val_f1_binary": val_f1,
                "threshold_grid": THRESHOLD_GRID,
            },
            indent=2,
        )
    )

    # ---- ProcessBench eval ------------------------------------------------
    # eval_time_sec includes: SAE encoding of pb_h (when applicable),
    # probe inference on pb_z, and per-trace aggregation/prediction.
    # Excludes: PRM800K data load, training, disk I/O for outputs.
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_t0 = time.time()

    if sae is not None:
        z_pb = encode_with_sae(sae, pb_h, args.batch_size, device)
    else:
        z_pb = pb_h

    if method == "random":
        # Deterministic random scores; counted in eval timing as the
        # "inference" step for consistency with other methods.
        pb_rng = np.random.default_rng(args.seed + 1)
        pb_scores = pb_rng.uniform(0.0, 1.0, size=pb_h.shape[0]).astype(np.float32)
    else:
        pb_scores = probe_scores(probe, z_pb, args.batch_size, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    pb_rows, pb_metrics = evaluate_processbench(pb_scores, pb_meta, threshold)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_time = time.time() - eval_t0

    with (args.out_dir / "pb_step_scores.jsonl").open("w") as f:
        for row in pb_rows:
            f.write(json.dumps(row) + "\n")
    with (args.out_dir / "pb_predictions.jsonl").open("w") as f:
        for row in pb_rows:
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": row["prediction"],
                        "threshold": row["threshold"],
                    }
                )
                + "\n"
            )

    n_steps_total = pb_h.shape[0]
    n_traces = pb_metrics["n_traces"]
    mean_step_latency_ms = (eval_time * 1000.0 / max(n_steps_total, 1))
    mean_trace_latency_ms = (eval_time * 1000.0 / max(n_traces, 1))

    eval_metrics = {
        "method": method,
        "threshold": threshold,
        **pb_metrics,
        "eval_time_sec": eval_time,
        "mean_step_latency_ms": mean_step_latency_ms,
        "mean_trace_latency_ms": mean_trace_latency_ms,
        "latency_scope": (
            "includes: representation encoding of ProcessBench hidden states, "
            "probe scoring, and trace aggregation. "
            "excludes: disk loading and training."
        ),
    }
    (args.out_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    # ---- Train metrics ----------------------------------------------------
    total_time = time.time() - total_t0
    if method == "random":
        total_time = 0.0

    train_metrics = {
        "method": method,
        "seed": args.seed,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "representation_train_n": rep_train_n,
        "probe_train_n": int(probe_train_h.shape[0]),
        "val_n": int(val_h.shape[0]),
        "representation_train_time_sec": representation_train_time,
        "probe_train_time_sec": probe_train_time,
        "total_train_time_sec": total_time,
        "avg_representation_train_latency_ms_per_example": (
            (representation_train_time * 1000.0 / rep_train_n) if rep_train_n else 0.0
        ),
        "avg_probe_train_latency_ms_per_example": (
            (probe_train_time * 1000.0 / probe_train_h.shape[0])
            if method != "random"
            else 0.0
        ),
        "final_reconstruction_mse": (
            sae_stats["final_reconstruction_mse"] if method != "dense_linear" and method != "random" else None
        ),
        "final_l1_mean": (
            sae_stats["final_l1_mean"] if method != "dense_linear" and method != "random" else None
        ),
        "final_aux_bce": sae_stats["final_aux_bce"],
        "gpu_name": gpu_name,
        "device": device.type,
    }
    (args.out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2))

    print(
        f"[{method}] threshold={threshold} F1_PB={eval_metrics['F1_PB']:.4f} "
        f"Acc_err={eval_metrics['Acc_error']:.4f} Acc_corr={eval_metrics['Acc_correct']:.4f} "
        f"val_bacc={best_bacc:.4f}"
    )


if __name__ == "__main__":
    main()
