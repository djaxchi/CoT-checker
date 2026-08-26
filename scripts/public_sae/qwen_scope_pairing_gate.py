#!/usr/bin/env python3
"""Which Qwen-Scope SAE layer, if any, reconstructs the states we actually stored?

The SAE arm only works if the tensor in the store is the tensor the SAE was
trained on. That pairing is easy to get subtly wrong, because HF's
`output_hidden_states` is offset by one and its last entry is special:

    hidden_states[0]      embeddings          = resid_pre of block 0
    hidden_states[i]      input to block i    = resid_post of block i-1
    hidden_states[-1]     resid_post of the LAST block, AFTER the final RMSNorm

So `resid_post_layer_N` pairs with `hidden_states[N+1]` for every block except
the last, whose raw resid_post `output_hidden_states` never exposes. Encoding
with `--layer -1` therefore stores a post-norm tensor, which no SAE was trained
on unless Qwen-Scope happens to have normalized too.

This scores candidate SAE layers against a sample of real stored rows by fraction
of variance unexplained, alongside two references that make the number readable:
the FVU of predicting the mean (1.0 by construction) and of a rank-k PCA, so
"good" is not judged against nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.harness.qwen_scope import TopKSAE, find_snapshot  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402


def sample_rows(split_dir: Path, n_items: int, seed: int) -> np.ndarray:
    view = ShardedRepSplit(split_dir)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(view), min(n_items, len(view)), replace=False)
    return np.concatenate([np.asarray(view.item(int(k))) for k in pick])


def pca_fvu(h: torch.Tensor, k: int) -> float:
    """FVU of the best rank-k linear reconstruction. A dictionary that cannot beat
    this on its own training distribution is not paired correctly."""
    x = h - h.mean(0)
    q = torch.linalg.svd(x, full_matrices=False)[2][:k]
    return float(((x - (x @ q.T) @ q) ** 2).sum() / (x ** 2).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split_dir", required=True, type=Path)
    p.add_argument("--hf_cache", required=True)
    p.add_argument("--repo_id", default="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--n_items", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = sample_rows(args.split_dir, args.n_items, args.seed)
    h = torch.from_numpy(rows).float().to(device)
    print(f"[gate] {h.shape[0]:,} stored rows, dim {h.shape[1]}, "
          f"|h|max {h.abs().max():.1f}, std {h.std():.3f}", flush=True)

    snap = find_snapshot(args.hf_cache, args.repo_id)
    results = {"split": str(args.split_dir), "repo": args.repo_id,
               "n_rows": int(h.shape[0]), "layers": {}}

    ref = pca_fvu(h.cpu(), 50)
    results["pca50_fvu"] = ref
    print(f"[ref ] FVU of predicting the mean      1.0000")
    print(f"[ref ] FVU of a rank-50 PCA            {ref:.4f}", flush=True)

    best = (None, 2.0)
    for layer in args.layers:
        try:
            sae = TopKSAE.from_snapshot(snap, layer=layer, device=device)
        except FileNotFoundError as e:
            print(f"[skip] layer {layer}: {e}")
            continue
        f = sae.fvu(h)
        results["layers"][str(layer)] = f
        print(f"[sae ] layer{layer:<3d} (k={sae.k})  FVU {f:.4f}"
              f"{'   <- reconstructs' if f < 0.5 else ''}", flush=True)
        if f < best[1]:
            best = (layer, f)
        del sae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["best_layer"], results["best_fvu"] = best
    print()
    if best[0] is None or best[1] >= 0.5:
        print("[VERDICT] no candidate layer reconstructs these states. The stored "
              "tensor is not what any tested SAE was trained on; re-encode at the "
              "hidden_states index that matches a real resid_post.")
    else:
        print(f"[VERDICT] layer{best[0]} pairs with this store (FVU {best[1]:.4f}).")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"[gate] wrote {args.out}")


if __name__ == "__main__":
    main()
