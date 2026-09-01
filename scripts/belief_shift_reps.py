#!/usr/bin/env python3
"""Read the stored states through the unembedding instead of around it.

Every representation in this study treats the residual stream as a vector to be
pooled. None has been pushed through the unembedding, which is the one place the
state becomes a statement about what the model thinks comes next.

Two shifts, both from states already on disk:

  along the step   the belief at the pre-step boundary against the belief at the
                   step's last token. A step that goes wrong may be one that
                   moves the model somewhere it was not heading.
  across layers    the step's last token read at layer 26 and at layer 35. Layer
                   26 already beats layer 35 as a probe input here, so the two
                   demonstrably disagree; this asks whether the disagreement is
                   about the prediction itself.

This is a logit-lens reading, stated plainly: the store holds resid_post of block
34 for a 36-block model, so applying the final norm and the unembedding to it is
not the model's true output distribution. That is fine for a SHIFT between two
positions read the same way, and it would not be fine for a claim about the
model's actual next token.

Only divergences, rank changes and entropies are kept, never raw logits, so
nothing here carries an activation scale or a token count. The risk is on the
record in advance: this project measured step incorrectness against per-token
entropy at -0.20, and latent_memory_v0 found the answer-belief readout behaves as
an answer shortcut. The hypothesis is the shift, not the uncertainty.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.harness.beliefshift import N_BELIEF, belief_feats  # noqa: E402
from src.repstore.store import RepSplit  # noqa: E402


def load_head(model_id: str, cache: str, device, dtype=torch.float32):
    """The final norm and the unembedding, without the 36 blocks in front."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache, torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True)
    norm_w = m.model.norm.weight.detach().to(device, dtype).clone()
    eps = float(getattr(m.model.norm, "variance_epsilon", 1e-6))
    w = m.lm_head.weight.detach().to(device, dtype).clone()
    del m
    return norm_w, eps, w


def logits_of(h: torch.Tensor, norm_w, eps, w) -> torch.Tensor:
    """RMSNorm then unembed, matching how the model reads its own final state."""
    h = h.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
    return h @ w.T


def collect(dir_a: Path, dir_b: Path | None, limit, seed, pb, head, device,
            batch: int = 256):
    norm_w, eps, w = head
    shards_a = sorted(glob.glob(str(dir_a / "shard_*")))
    if not shards_a:
        raise FileNotFoundError(f"no shard_* under {dir_a}")
    shards_b = sorted(glob.glob(str(dir_b / "shard_*"))) if dir_b else [None] * len(shards_a)
    rng = np.random.default_rng(seed)
    per_shard = None if limit is None else max(1, limit // len(shards_a))
    feats, ys, lens = [], [], []

    for sa, sb in zip(shards_a, shards_b):
        ra = RepSplit(sa)
        meta = ra.meta()
        rb = RepSplit(sb) if sb else None
        if rb is not None and not np.array_equal(ra.lengths, rb.lengths):
            raise SystemExit(f"{sa} and {sb} are not the same steps in the same order")
        idx = np.arange(len(meta))
        if per_shard is not None and len(idx) > per_shard:
            idx = np.sort(rng.choice(len(idx), per_shard, replace=False))
        rows_b, rows_e, rows_m, keep = [], [], [], []
        for k in idx:
            m = meta[int(k)]
            lo = int(ra.offsets[k])
            hi = int(ra.offsets[k + 1])
            st = lo + int(m["step_start_idx"])
            if hi <= st:
                st = hi - 1
            rows_b.append(np.asarray(ra.h[lo + int(m["pre_step_boundary_idx"])], np.float32))
            rows_e.append(np.asarray(ra.h[hi - 1], np.float32))
            rows_m.append(np.asarray(rb.h[hi - 1], np.float32) if rb is not None else None)
            lens.append(hi - st)
            ys.append(1 if (pb and m["label"] == m["step_idx"]) else
                      (int(ra.y[k]) if not pb else 0))
            keep.append(k)

        for s in range(0, len(keep), batch):
            b = torch.from_numpy(np.stack(rows_b[s:s + batch])).to(device)
            e = torch.from_numpy(np.stack(rows_e[s:s + batch])).to(device)
            lb = logits_of(b, norm_w, eps, w).cpu().numpy()
            le = logits_of(e, norm_w, eps, w).cpu().numpy()
            lm = None
            if rows_m[0] is not None:
                mm = torch.from_numpy(np.stack(rows_m[s:s + batch])).to(device)
                lm = logits_of(mm, norm_w, eps, w).cpu().numpy()
            for j in range(lb.shape[0]):
                feats.append(belief_feats(lb[j], le[j], None if lm is None else lm[j]))
        del ra, rb
    return (np.stack(feats).astype(np.float32), np.array(ys, np.float32),
            np.array(lens, np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prm_store", required=True, type=Path)
    p.add_argument("--pb_store", required=True, type=Path)
    p.add_argument("--prm_store_b", type=Path)
    p.add_argument("--pb_store_b", type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--model_id", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--hf_cache", required=True)
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--pb_subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--n_train", type=int, default=60000)
    p.add_argument("--n_pb", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    head = load_head(args.model_id, args.hf_cache, device)
    print(f"[belief] head loaded, vocab {head[2].shape[0]:,} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    sb = (lambda stem: args.prm_store_b / stem) if args.prm_store_b else (lambda s: None)
    tr, ytr, ltr = collect(args.prm_store / args.train_stem, sb(args.train_stem),
                           args.n_train, args.seed, False, head, device)
    print(f"[belief] train {len(ytr):,} ({time.perf_counter()-t0:.0f}s)", flush=True)
    va, yva, lva = collect(args.prm_store / args.val_stem, sb(args.val_stem),
                           None, args.seed, False, head, device)
    pbs = {}
    for s in args.pb_subsets:
        d = args.pb_store / s
        if d.exists():
            pbs[s] = collect(d, (args.pb_store_b / s) if args.pb_store_b else None,
                             args.n_pb, args.seed, True, head, device)
    print(f"[belief] pb {list(pbs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"x_train": tr, "y_train": ytr, "len_train": ltr,
           "x_val": va, "y_val": yva, "len_val": lva}
    for s, (px, py, pl) in pbs.items():
        out[f"pb_x_{s}"], out[f"pb_y_{s}"], out[f"pb_len_{s}"] = px, py, pl
    np.savez(args.out_dir / "belief.npz", **out)
    print(f"[belief] belief dim {N_BELIEF} -> {args.out_dir / 'belief.npz'}")


if __name__ == "__main__":
    main()
