#!/usr/bin/env python3
"""Train a bottleneck on a base representation and emit it for the screen.

The sparse-dictionary result said a bottleneck trained to reconstruct spends its
budget on high-variance directions and discards step correctness, which is a
~0.01%-variance margin. This trains bottlenecks that are asked to preserve
something else, and writes their codes in the format the screen consumes, so a
new objective costs a minute to judge rather than a grid run.

The baseline that matters is the base representation itself, uncompressed. A
bottleneck that does not beat the thing it compresses is not interesting however
good its absolute number looks, so `--objective none` emits exactly that and it
should always be screened alongside.

Reads the dense vector caches the grid already derives, so no re-encoding.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.harness import rescale as rs  # noqa: E402
from src.harness.bottleneck import Bottleneck, signal_share  # noqa: E402


def find_cache(cache_dir: Path, rep: str, stem: str, what: str = "h"):
    """The grid names caches {rep}__{stem}__{fingerprint}_{h,y}.npy."""
    hits = sorted(glob.glob(str(cache_dir / f"{rep}__{stem}__*_{what}.npy")))
    if not hits:
        hits = sorted(glob.glob(str(cache_dir / f"{rep}__{stem}_{what}.npy")))
    if not hits:
        raise FileNotFoundError(
            f"no cache for {rep}/{stem} under {cache_dir}; run a grid cell on "
            f"{rep} first so the vectors are derived")
    return Path(hits[-1])


def load_pb(cache_dir: Path, rep: str, subs) -> list[tuple[np.ndarray, np.ndarray]]:
    """(vectors, is-first-error) per ProcessBench subset."""
    out = []
    for sub in subs:
        h = find_cache(cache_dir, rep, sub, "h")
        meta_hits = sorted(glob.glob(str(cache_dir / f"{rep}__{sub}__*_meta.jsonl")))
        if not meta_hits:
            print(f"[bneck] no meta for {sub}, skipping", flush=True)
            continue
        meta = [json.loads(l) for l in Path(meta_hits[-1]).read_text().splitlines() if l.strip()]
        y = np.array([1 if m["label"] == m["step_idx"] else 0 for m in meta], dtype=np.float32)
        out.append((np.load(h, mmap_mode="r"), y))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vec_cache", required=True, type=Path)
    p.add_argument("--pb_cache", type=Path, default=None,
                   help="Defaults to --vec_cache.")
    p.add_argument("--base_rep", default="step_mean")
    p.add_argument("--objective", required=True,
                   choices=["none", "recon", "recon_white", "mixed", "ib"])
    p.add_argument("--d_code", type=int, default=256)
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=1e-3)
    p.add_argument("--rescale", choices=["none", "zscore"], default="zscore")
    p.add_argument("--n_train", type=int, default=150000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--pb_subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pb_cache = args.pb_cache or args.vec_cache
    t0 = time.perf_counter()

    Xtr = np.load(find_cache(args.vec_cache, args.base_rep, args.train_stem), mmap_mode="r")
    ytr = np.load(find_cache(args.vec_cache, args.base_rep, args.train_stem, "y"))
    Xva = np.load(find_cache(args.vec_cache, args.base_rep, args.val_stem), mmap_mode="r")
    yva = np.load(find_cache(args.vec_cache, args.base_rep, args.val_stem, "y"))
    pb = load_pb(pb_cache, args.base_rep, args.pb_subsets)
    n = min(args.n_train, Xtr.shape[0])
    d = Xtr.shape[1]
    print(f"[bneck] base {args.base_rep} d={d}  train {n:,}/{Xtr.shape[0]:,}  "
          f"val {len(yva):,}  pb subsets {len(pb)}  device {device.type}", flush=True)

    x = torch.as_tensor(np.asarray(Xtr[:n], dtype=np.float32))
    y = torch.as_tensor(ytr[:n].astype(np.float32))
    tfm = None
    if args.rescale == "zscore":
        st = rs.fit(x.numpy())
        tfm = rs.to_torch(st, device)

    def prep(a):
        t = torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
        return rs.apply_torch(t, tfm) if tfm is not None else t

    if args.objective == "none":
        enc = lambda t: t                                             # noqa: E731
        parts, share_after = {}, None
    else:
        whiten = None
        if args.objective == "recon_white":
            w = rs.fit_whiten(x.numpy())
            whiten = torch.from_numpy(w["W"]).to(device)
            print(f"[bneck] whitened metric, covariance condition {w['cond']:.3g}",
                  flush=True)
        torch.manual_seed(args.seed)
        model = Bottleneck(d, args.d_code, args.objective, hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        rng = np.random.default_rng(args.seed)
        xg, yg = prep(x), y.to(device)
        model.train()
        for ep in range(args.epochs):
            order = rng.permutation(n)
            for i in range(0, n, args.batch):
                idx = torch.from_numpy(order[i:i + args.batch]).to(device)
                loss, parts = model.loss(xg.index_select(0, idx), yg.index_select(0, idx),
                                         beta=args.beta, whiten=whiten,
                                         kl_weight=args.kl_weight)
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"[bneck] epoch {ep+1}/{args.epochs} " +
                  " ".join(f"{k}={v:.4f}" for k, v in parts.items()), flush=True)
        model.eval()
        enc = lambda t: model.encode(t)                               # noqa: E731

    @torch.no_grad()
    def code(a, batch=8192):
        outs = []
        for i in range(0, len(a), batch):
            outs.append(enc(prep(a[i:i + batch])).cpu().numpy().astype(np.float32))
        return np.concatenate(outs)

    z_tr = code(Xtr[:n]); z_va = code(Xva)
    share_before = signal_share(torch.as_tensor(np.asarray(Xtr[:20000], dtype=np.float32)),
                                torch.as_tensor(ytr[:20000]))
    share_after = signal_share(torch.as_tensor(z_tr[:20000]), torch.as_tensor(ytr[:20000]))
    out = {"x_train": z_tr, "y_train": ytr[:n].astype(np.float32),
           "x_val": z_va, "y_val": yva.astype(np.float32)}
    for (xx, yy), sub in zip(pb, args.pb_subsets):
        out[f"pb_x_{sub}"] = code(xx)
        out[f"pb_y_{sub}"] = yy
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    print(f"[bneck] {args.objective} d_code={z_tr.shape[1]}  "
          f"signal share {share_before:.5f} -> {share_after:.5f} "
          f"(x{share_after/max(share_before,1e-12):.2f})  "
          f"{time.perf_counter()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
