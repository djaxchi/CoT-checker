#!/usr/bin/env python3
"""Derive sparse SAE representations of a step, offline from the span store.

The dense grid asks which rows of the forward pass carry step correctness. This
adds the sparse-dictionary answer: run each token's residual state through the
Qwen-Scope TopK SAE and pool the resulting codes, so the learner reads 65,536
interpretable feature activations instead of 4,096 raw dimensions.

Two readouts, deliberately mirroring two dense ones so the contrast is clean:

    sae_last            code of the step's final token          <-> last_token
    sae_mean            mean of the codes of every token        <-> step_mean
    sae_delta           code(last) - code(pre-step boundary)     <-> step_delta
    sae_stats           concat[mean, max, min, std, last]        <-> step_stats
    sae_boundary_stats  boundary code prepended to sae_stats     <-> boundary_stats

Every dense vector representation in the grid has a sparse twin, so the sparsity
question can be asked of the whole leaderboard rather than of two rows.

Same pooling rule, same layer, same activations, same protocol: the only thing
that changes is whether the learner sees the raw state or its sparse code. That
is what makes "does sparsity help?" answerable rather than a vibe.

No re-encoding: the SAE is a matmul over states already in the store, exactly
like the other offline readouts. Output is CSR, because a pooled code stays
sparse and storing it dense would cost 67 GB for the train split against ~1.5 GB.

Note what the SAE costs going in: reconstruction leaves FVU 0.336 at this layer,
so a third of the variance is discarded before the probe sees anything. If step
correctness lives in the low-variance directions an SAE trained for
reconstruction has no reason to keep, this should lose to the dense readouts.
That is a real prediction and the point of running it.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.harness.qwen_scope import TopKSAE, find_snapshot  # noqa: E402
from src.harness.sparse_vec import write_csr  # noqa: E402
from src.repstore.store import RepSplit  # noqa: E402

READOUTS = ("sae_last", "sae_mean", "sae_delta", "sae_stats",
            "sae_boundary_stats")
# multiplier on d_sae for each readout's output width
WIDTH = {"sae_last": 1, "sae_mean": 1, "sae_delta": 1,
         "sae_stats": 5, "sae_boundary_stats": 6}


def _cat_sparse(parts: list[torch.Tensor], d: int):
    """Concatenate blocks into one sparse (indices, values), offsetting each block.

    Keeps the wide readouts sparse: sae_stats is 327,680 wide but only about
    2,050 entries are non-zero, because `min` over a step's tokens is empty
    wherever any token is silent and `std` shares `mean`'s support.
    """
    idx, val = [], []
    for b, z in enumerate(parts):
        nz = torch.nonzero(z, as_tuple=True)[0]
        idx.append(nz + b * d)
        val.append(z[nz])
    return torch.cat(idx).cpu().numpy(), torch.cat(val).cpu().numpy()


def pooled_codes(sae: TopKSAE, span: torch.Tensor, boundary: torch.Tensor,
                 readout: str):
    """(indices, values) of one step's pooled sparse code.

    `span` is the step's own token states; `boundary` is the single pre-step
    state, the same rows the dense readouts use.
    """
    d = sae.d_sae
    if readout == "sae_last":
        return _cat_sparse([sae.encode(span[-1:])[0]], d)
    codes = sae.encode(span)                          # (T, d_sae)
    if readout == "sae_mean":
        return _cat_sparse([codes.mean(0)], d)
    if readout == "sae_delta":
        return _cat_sparse([codes[-1] - sae.encode(boundary)[0]], d)
    stats = [codes.mean(0), codes.max(0).values, codes.min(0).values,
             codes.std(0) if codes.shape[0] > 1 else torch.zeros_like(codes[0]),
             codes[-1]]
    if readout == "sae_stats":
        return _cat_sparse(stats, d)
    if readout == "sae_boundary_stats":
        return _cat_sparse([sae.encode(boundary)[0]] + stats, d)
    raise ValueError(readout)


def derive_split(split_dir: Path, sae: TopKSAE, readout: str, device, batch: int,
                 sort: bool):
    shard_dirs = sorted(glob.glob(str(split_dir / "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* under {split_dir}")
    rows, labels, metas, gi = [], [], [], []
    t0 = time.perf_counter()
    for sd in shard_dirs:
        rs = RepSplit(sd)
        meta = rs.meta()
        for k, m in enumerate(meta):
            a = int(rs.offsets[k]) + int(m["step_start_idx"])
            b = int(rs.offsets[k + 1])
            if b <= a:
                a = b - 1
            span = torch.from_numpy(np.asarray(rs.h[a:b], dtype=np.float32)).to(device)
            bi = int(rs.offsets[k]) + int(m["pre_step_boundary_idx"])
            boundary = torch.from_numpy(
                np.asarray(rs.h[bi:bi + 1], dtype=np.float32)).to(device)
            idx, val = pooled_codes(sae, span, boundary, readout)
            rows.append((idx, val))
            labels.append(int(rs.y[k]))
            gi.append(int(m["global_index"]))
            if sort:
                metas.append(m)
            if len(rows) % 20000 == 0:
                print(f"  {len(rows):,} items ({time.perf_counter()-t0:.0f}s)", flush=True)
        del rs
    if sort:
        order = np.argsort(np.array(gi), kind="mergesort")
        rows = [rows[i] for i in order]
        labels = [labels[i] for i in order]
        metas = [metas[i] for i in order]
    return rows, np.array(labels, dtype=np.int8), metas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_root", required=True, type=Path)
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--readout", choices=READOUTS, required=True)
    p.add_argument("--mode", choices=["prm", "pb"], default="prm")
    p.add_argument("--hf_cache", required=True)
    p.add_argument("--repo_id", default="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50")
    p.add_argument("--sae_layer", type=int, required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--fvu_check", type=int, default=2000,
                   help="Rows to score reconstruction on before deriving; a wrong "
                        "layer or convention shows up here, not in the results.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snap = find_snapshot(args.hf_cache, args.repo_id)
    sae = TopKSAE.from_snapshot(snap, layer=args.sae_layer, device=device)
    print(f"[sae] {args.repo_id} layer{args.sae_layer}  d_model {sae.d_model} "
          f"d_sae {sae.d_sae} k {sae.k}  device {device.type}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"repo": args.repo_id, "sae_layer": args.sae_layer,
             "readout": args.readout, "splits": {}}

    first = args.store_root / args.splits[0]
    rs = RepSplit(sorted(glob.glob(str(first / "shard_*")))[0])
    sample = torch.from_numpy(np.asarray(rs.h[:args.fvu_check], dtype=np.float32)).to(device)
    fvu = sae.fvu(sample)
    stats["fvu"] = fvu
    print(f"[gate] reconstruction FVU on {args.fvu_check} stored rows: {fvu:.4f}", flush=True)
    if fvu >= 1.0:
        raise SystemExit(f"[FATAL] FVU {fvu:.3f} >= 1: this SAE does not reconstruct "
                         f"these states, check the layer pairing before deriving")
    del sample, rs

    for stem in args.splits:
        print(f"[derive] {args.readout} :: {stem}", flush=True)
        rows, y, metas = derive_split(args.store_root / stem, sae, args.readout,
                                      device, args.batch, sort=(args.mode == "pb"))
        if args.mode == "prm":
            out = args.out_dir / f"{args.readout}__{stem}.npz"
            s = write_csr(out, rows, y, sae.d_sae * WIDTH[args.readout])
        else:
            sub = args.out_dir / stem
            sub.mkdir(parents=True, exist_ok=True)
            s = write_csr(sub / f"{args.readout}.npz", rows, y,
                          sae.d_sae * WIDTH[args.readout])
            with (sub / f"{args.readout}_meta.jsonl").open("w") as f:
                for m in metas:
                    f.write(json.dumps(m) + "\n")
        stats["splits"][stem] = s
        print(f"  {s['items']:,} items  mean nnz {s['mean_nnz']:.1f} "
              f"({100*s['density']:.3f}% of {sae.d_sae*WIDTH[args.readout]:,})  "
              f"{s['bytes']/1e9:.2f} GB",
              flush=True)

    (args.out_dir / f"{args.readout}_manifest.json").write_text(json.dumps(stats, indent=2))
    print(f"[derive] wrote {args.out_dir}/{args.readout}_manifest.json")


if __name__ == "__main__":
    main()
