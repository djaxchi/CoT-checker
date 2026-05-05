#!/usr/bin/env python3
"""Train a standard activation-level SAE on pre-saved dense h_k vectors (E4).

Architecture: Linear encoder (D→D) + TopK(k) activation + Linear decoder (D→D).
Loss: MSE(decode(topk(encode(h))), h)  -- reconstructs activations, not text.
Output is L2-normalised sparse codes to match the normalisation in SSAE.encode().

Contrast with SSAE (E2/E3): SSAE reconstructs step *text* given context.
This SAE reconstructs the *hidden state* h_k -- no language modelling involved.

Pipeline this script runs:
  1. Train SAE on dense_train_full.npz["latents"]
  2. Encode dense_train_full.npz  → actsae_train_full.npz  (latents + correctness)
  3. Encode dense_eval_held_out.npz → actsae_eval_held_out.npz
  4. Train 4 linear probes (seeds 42-45) on actsae_train_full.npz
  5. Optionally encode extra npz files (--encode-extra src:dst ...)

Checkpoint format (actsae.pt):
  {"model": state_dict, "config": {"n_inputs": D, "n_latents": D, "k": k}}

Linear probe checkpoint format (matches experiment_linear_probe.py):
  {"state_dict": ..., "input_dim": D}

Usage:
  python scripts/train_activation_sae.py \\
      --train-data  $SCRATCH/cot-checker/probe_data/dense_train_full.npz \\
      --eval-data   $SCRATCH/cot-checker/probe_data/dense_eval_held_out.npz \\
      --output-dir  $SCRATCH/cot-checker/results_actsae \\
      --device      cpu

  # Later: apply saved SAE to ProcessBench dense npz
  python scripts/train_activation_sae.py \\
      --encode-only \\
      --actsae-checkpoint $SCRATCH/.../actsae.pt \\
      --encode-extra processbench_dense.npz:processbench_actsae.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ActivationSAE(nn.Module):
    """TopK sparse autoencoder on dense activation vectors.

    Encoder: Linear(D, D) + ReLU + TopK(k) -> sparse codes
    Decoder: Linear(D, D) -> reconstruction
    Output codes are L2-normalised for consistency with SSAE.encode().
    """

    def __init__(self, n_inputs: int, n_latents: int, k: int) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.k = k
        self.encoder = nn.Linear(n_inputs, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, n_inputs, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, D) -> sparse (B, D), L2-normalised."""
        pre = self.encoder(h).relu()
        k = min(self.k, pre.shape[-1])
        topk_vals, topk_idx = torch.topk(pre, k, dim=-1)
        sparse = torch.zeros_like(pre).scatter_(-1, topk_idx, topk_vals)
        return F.normalize(sparse, p=2, dim=-1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (sparse_codes, mse_loss)."""
        sparse = self.encode(h)
        recon = self.decoder(sparse)
        loss = F.mse_loss(recon, h)
        return sparse, loss


# ---------------------------------------------------------------------------
# Linear probe (matches experiment_linear_probe.py save format)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dense_npz(path: str, max_samples: int | None = None, seed: int = 42
                   ) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    h = d["latents"].astype(np.float32)
    y = d["correctness"].astype(np.int64)
    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        h, y = h[idx], y[idx]
    print(f"  Loaded {len(y):,} steps from {path}  dim={h.shape[1]}")
    print(f"    correct={int((y==1).sum()):,}  incorrect={int((y==0).sum()):,}")
    return h, y


def balance_and_subsample(h: np.ndarray, y: np.ndarray, n: int, seed: int = 42
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Return at most n balanced samples (n//2 per class)."""
    rng = np.random.default_rng(seed)
    per_class = n // 2
    cor = np.where(y == 1)[0]
    inc = np.where(y == 0)[0]
    sel = np.concatenate([
        rng.choice(cor, min(per_class, len(cor)), replace=False),
        rng.choice(inc, min(per_class, len(inc)), replace=False),
    ])
    rng.shuffle(sel)
    return h[sel], y[sel]


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------

def train_sae(
    train_h: np.ndarray,
    n_latents: int,
    k: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> ActivationSAE:
    D = train_h.shape[1]
    model = ActivationSAE(D, n_latents, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    h_t = torch.from_numpy(train_h).to(device)
    loader = DataLoader(TensorDataset(h_t), batch_size=batch_size, shuffle=True)

    print(f"\n  Training ActivationSAE: D={D}, n_latents={n_latents}, k={k}")
    print(f"  {len(train_h):,} samples  |  {epochs} epochs  |  batch={batch_size}  |  lr={lr}")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            _, loss = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        avg = total_loss / len(train_h)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  mse={avg:.6f}  ({time.time()-t0:.0f}s)")
    print(f"  SAE training done in {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_with_sae(model: ActivationSAE, h: np.ndarray, batch_size: int = 4096,
                    device: str = "cpu") -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(h), batch_size):
        x = torch.from_numpy(h[i:i + batch_size]).to(device)
        out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def encode_npz(model: ActivationSAE, src_path: str, dst_path: str,
               device: str = "cpu") -> None:
    """Encode a dense .npz (must have 'latents') and write a new .npz."""
    d = np.load(src_path)
    h = d["latents"].astype(np.float32)
    print(f"  Encoding {len(h):,} steps from {src_path}")
    sparse = encode_with_sae(model, h, device=device)
    save_kw = {"latents": sparse.astype(np.float32)}
    # Forward all other arrays unchanged (correctness, step_labels, solution_ids, ...)
    for key in d.files:
        if key != "latents":
            save_kw[key] = d[key]
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **save_kw)
    print(f"  Saved actsae latents {sparse.shape} -> {dst_path}")


# ---------------------------------------------------------------------------
# Linear probe training
# ---------------------------------------------------------------------------

def train_linear_probe(
    train_h: np.ndarray,
    train_y: np.ndarray,
    eval_h: np.ndarray,
    eval_y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    out_path: str,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = train_h.shape[1]

    # Build balanced 70/30 split (70% correct, 30% incorrect -- mirrors existing probes)
    rng = np.random.default_rng(seed)
    cor = np.where(train_y == 1)[0]
    inc = np.where(train_y == 0)[0]
    n_inc = len(inc)
    n_cor = min(int(n_inc * 0.7 / 0.3), len(cor))
    sel = np.concatenate([rng.choice(cor, n_cor, replace=False), inc])
    rng.shuffle(sel)
    h_tr = torch.from_numpy(train_h[sel]).to(device)
    y_tr = torch.from_numpy(train_y[sel].astype(np.float32)).to(device)

    model = LinearProbe(D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(h_tr, y_tr), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            logits = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    # Eval
    model.eval()
    with torch.no_grad():
        h_ev = torch.from_numpy(eval_h).to(device)
        logits = model(h_ev).squeeze(-1)
        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.int64)
    acc = (preds == eval_y).mean()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": D}, out_path)
    print(f"  seed={seed}  acc={acc*100:.2f}%  -> {out_path}")
    return float(acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=False)

    # --- train mode (default) ---
    p.add_argument("--train-data",  default=None)
    p.add_argument("--eval-data",   default=None)
    p.add_argument("--output-dir",  default="results/actsae")
    p.add_argument("--n-latents",   type=int, default=896)
    p.add_argument("--k",           type=int, default=40,
                   help="TopK sparsity (same as SSAE c=4 default)")
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch-size",  type=int, default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--probe-epochs",type=int, default=50)
    p.add_argument("--probe-lr",    type=float, default=1e-3)
    p.add_argument("--probe-batch", type=int, default=512)
    p.add_argument("--seeds",       type=int, nargs="+", default=[42, 43, 44, 45])
    p.add_argument("--device",      default="cpu",
                   help="'cuda' for GPU, 'cpu' for CPU-only (SAE training is fast on CPU)")
    # --- encode-only mode ---
    p.add_argument("--encode-only", action="store_true",
                   help="Skip training; just apply a saved SAE checkpoint to --encode-extra files")
    p.add_argument("--actsae-checkpoint", default=None,
                   help="Path to actsae.pt (required with --encode-only)")
    p.add_argument("--encode-extra", nargs="+", default=[],
                   metavar="SRC:DST",
                   help="Extra npz files to encode: pass as src.npz:dst.npz pairs")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # encode-only mode: apply existing SAE to extra npz files             #
    # ------------------------------------------------------------------ #
    if args.encode_only:
        if not args.actsae_checkpoint:
            raise ValueError("--actsae-checkpoint is required with --encode-only")
        ckpt = torch.load(args.actsae_checkpoint, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        model = ActivationSAE(cfg["n_inputs"], cfg["n_latents"], cfg["k"])
        model.load_state_dict(ckpt["model"])
        model.eval()
        for pair in args.encode_extra:
            src, dst = pair.split(":", 1)
            encode_npz(model, src, dst, device=args.device)
        return

    # ------------------------------------------------------------------ #
    # Full training mode                                                   #
    # ------------------------------------------------------------------ #
    if not args.train_data or not args.eval_data:
        raise ValueError("--train-data and --eval-data are required")

    print("=" * 64)
    print("  Activation SAE (E4) — training")
    print("=" * 64)

    print("\n[1/4] Loading data...")
    h_train, y_train = load_dense_npz(args.train_data)
    h_eval,  y_eval  = load_dense_npz(args.eval_data)

    # Use a balanced subsample of the training set to speed up SAE training
    # (the full 360k is useful for probes but the SAE converges on a subset)
    h_train_sae, _ = balance_and_subsample(h_train, y_train, n=200_000)
    print(f"  SAE will train on {len(h_train_sae):,} balanced samples")

    print("\n[2/4] Training SAE...")
    model = train_sae(
        h_train_sae,
        n_latents=args.n_latents,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    model.eval()

    cfg = {"n_inputs": model.n_inputs, "n_latents": model.n_latents, "k": model.k}
    ckpt_path = outdir / "actsae.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"  SAE saved -> {ckpt_path}")

    print("\n[3/4] Encoding train + eval splits...")
    # Encode full train set (all labels, for probe training)
    train_sparse = encode_with_sae(model, h_train, device=args.device)
    np.savez_compressed(
        outdir / "actsae_train_full.npz",
        latents=train_sparse.astype(np.float32),
        correctness=y_train.astype(np.int64),
    )
    print(f"  Train encoded: {train_sparse.shape} -> {outdir}/actsae_train_full.npz")

    eval_sparse = encode_with_sae(model, h_eval, device=args.device)
    np.savez_compressed(
        outdir / "actsae_eval_held_out.npz",
        latents=eval_sparse.astype(np.float32),
        correctness=y_eval.astype(np.int64),
    )
    print(f"  Eval  encoded: {eval_sparse.shape} -> {outdir}/actsae_eval_held_out.npz")

    # Encode any extra npz files requested upfront
    for pair in args.encode_extra:
        src, dst = pair.split(":", 1)
        encode_npz(model, src, dst, device=args.device)

    print("\n[4/4] Training linear probes...")
    accs = []
    for seed in args.seeds:
        acc = train_linear_probe(
            train_sparse, y_train,
            eval_sparse, y_eval,
            seed=seed,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch,
            lr=args.probe_lr,
            device=args.device,
            out_path=str(outdir / f"actsae_linear_probe_seed{seed}.pt"),
        )
        accs.append(acc)

    import statistics
    print(f"\n  Linear probe mean acc: {statistics.mean(accs)*100:.2f}% "
          f"+/- {(statistics.stdev(accs)*100 if len(accs)>1 else 0.0):.2f}%")
    print("\n  Done. Outputs:")
    print(f"    SAE checkpoint  : {ckpt_path}")
    print(f"    Encoded train   : {outdir}/actsae_train_full.npz")
    print(f"    Encoded eval    : {outdir}/actsae_eval_held_out.npz")
    for seed in args.seeds:
        print(f"    Linear probe s{seed}: {outdir}/actsae_linear_probe_seed{seed}.pt")


if __name__ == "__main__":
    main()
