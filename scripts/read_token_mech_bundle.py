#!/usr/bin/env python3
"""Every layer at the end of each step, for a mechanistic study of the read token.

For the items of a reference split (all of test_2k, val_5k and each ProcessBench
subset; a seeded subsample of train, for fitting per-layer readouts), one forward
of each model writes, per item:

  last_step.npy (N, 37, 4096)  Qwen3-8B, last step token: block 0..35 outputs, then
                               the post-norm final state. Row l + 1 is HF
                               hidden_states[l + 1]; row 34 is the Instruct
                               leaderboard's layer 35.
  read.npy      (N, 37, 4096)  the read token right after the step, same rows
                               (offsets included). The step's own states are the
                               ones above: the read token comes after them.
  prm.npy       (N, 29, 3584)  Qwen2.5-Math-PRM-7B, last step token, block 0..27
                               outputs and its final state (row 26 is the PRM
                               leaderboard's layer 27).
  head_logit.npy (N,)          the read token's own head on its final state
  drv_logit.npy (N,), drv_z.npy (N, 512)
                               DRV (the d512 cell over blocks 3, 7, ..., 31, 34 +
                               final): its logit and the pooled vector its head reads
  y.npy, meta.jsonl, index.npy  label, reference meta, position in the reference split

float16. Each worker writes <out>/<split>/part_<i>/; load_bundle() puts the parts
back in reference order. Item order, labels and texts come from the reference
store exactly as in scripts/read_token_store.py.

    python scripts/read_token_mech_bundle.py --ckpt tokens/dq_correct/read_token.pt \
        --drv_cell <stage_f>/cells_fab/step_tokens__transformer_d512_l2_f2048_h8__seed42 \
        --ref <repstore>/step_spans/test_2k --source prm800k:<data>/prm800k_test_2k.jsonl \
        --out <mech_bundle> --shards 0/4
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
from scripts.analysis.drv_embed import embed, load_drv  # noqa: E402
from scripts.read_token_probe import (  # noqa: E402
    batches, capture_hooks, read_all_states, reduce_states, tokenize)
from scripts.read_token_store import load_model, load_read_token, match, source_index  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402

DRV_BLOCKS = (3, 7, 11, 15, 19, 23, 27, 31, 34)
ARRAYS = ("last_step", "read", "prm", "head_logit", "drv_logit", "drv_z", "y", "index")


def last_token_states(model, ids_list, pad, dev) -> torch.Tensor:
    """(B, n_blocks + 1, d): every block's output at the last token, then the final state."""
    B, L = len(ids_list), max(len(t) for t in ids_list)
    ids = torch.full((B, L), pad, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for r, t in enumerate(ids_list):
        ids[r, L - len(t):] = torch.tensor(t)
        att[r, L - len(t):] = 1
    ids, att = ids.to(dev), att.to(dev)
    blocks = list(range(len(model.model.layers)))
    cap: dict = {}
    with torch.no_grad(), capture_hooks(model, blocks, cap):
        out = model.model(input_ids=ids, attention_mask=att,
                          position_ids=(att.cumsum(1) - 1).clamp(min=0))
    return torch.stack([cap[l] for l in blocks] + [out.last_hidden_state[:, -1].float()], 1)


def pick(n: int, sample: int, seed: int) -> np.ndarray:
    if not sample or sample >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))


def load_bundle(split_dir: Path, arrays=ARRAYS, mmap: bool = True) -> dict:
    """The parts of one split, concatenated in reference order."""
    parts = sorted(split_dir.glob("part_*"))
    idx = np.concatenate([np.load(p / "index.npy") for p in parts])
    order = np.argsort(idx, kind="stable")
    out = {a: np.concatenate([np.load(p / f"{a}.npy", mmap_mode="r" if mmap else None)
                              for p in parts])[order] for a in arrays}
    metas = [json.loads(l) for p in parts for l in (p / "meta.jsonl").read_text().splitlines()]
    out["meta"] = [metas[i] for i in order]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prm_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--drv_cell", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=0, help="seeded subsample of the split (0: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shards", default="0/1", help="i/n: this worker takes items %% n == i")
    p.add_argument("--drv_blocks", default=",".join(map(str, DRV_BLOCKS)),
                   help="the blocks DRV reads, before the final state (its rt_layers rows)")
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--soft_attn", choices=["no_cudnn"], default=None)
    a = p.parse_args()

    kind, index = source_index(a.source)
    ref = ShardedRepSplit(a.ref)
    meta_all, y_all = ref.meta(), np.asarray(ref.y)
    sel = pick(len(meta_all), a.sample, a.seed)
    i, n = map(int, a.shards.split("/"))
    mine = sel[i::n]
    meta = [meta_all[k] for k in mine]
    items = match(kind, index, meta, y_all[mine])
    out = a.out / a.ref.name / f"part_{i}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rt, ck = load_read_token(a.ckpt, model.config.hidden_size, dev)
    soft_attn = a.soft_attn or ck.get("soft_attn")
    nb = len(model.model.layers)
    toks = [tokenize(tok, it) for it in items]
    N, d = len(items), model.config.hidden_size
    LS = np.lib.format.open_memmap(out / "last_step.npy", "w+", np.float16, (N, nb + 1, d))
    RD = np.lib.format.open_memmap(out / "read.npy", "w+", np.float16, (N, nb + 1, d))
    head = np.zeros(N, np.float32)
    for b in batches([len(t) for t in toks], a.max_tokens, None):
        cap = {l: None for l in range(nb)}
        pre = {l: None for l in range(nb)} | {"final": None}
        with torch.no_grad():
            H = reduce_states(rt, read_all_states(model, rt, [toks[j] for j in b], pad, dev, grad=False,
                                                  soft_attn=soft_attn, capture=cap, capture_prefix=pre))
            head[b] = rt.logit(H).float().cpu().numpy()
        r = torch.stack([cap[l] for l in range(nb)] + [H], 1)
        s = torch.stack([pre[l] for l in range(nb)] + [pre["final"]], 1)
        if not (torch.isfinite(r).all() and torch.isfinite(s).all()):
            raise SystemExit("non-finite state")
        RD[b], LS[b] = r.cpu().numpy(), s.cpu().numpy()
    RD.flush()
    LS.flush()
    print(f"[bundle] {a.ref.name} part {i}: {N} items, Qwen3 pass {time.perf_counter()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache() if dev.type == "cuda" else None

    drv = load_drv(a.drv_cell, d, dev)
    rows = [int(x) for x in a.drv_blocks.split(",")]
    Z, Lg = embed(drv, np.concatenate([RD[:, rows], RD[:, nb:]], 1), dev)
    np.save(out / "drv_z.npy", Z.astype(np.float16))
    np.save(out / "drv_logit.npy", Lg)
    np.save(out / "head_logit.npy", head)
    del drv

    pm, ptok = load_model(a.prm_model, dev)
    ppad = ptok.pad_token_id if ptok.pad_token_id is not None else ptok.eos_token_id
    ptoks = [tokenize(ptok, it) for it in items]
    pnb, pd = len(pm.model.layers), pm.config.hidden_size
    PR = np.lib.format.open_memmap(out / "prm.npy", "w+", np.float16, (N, pnb + 1, pd))
    for b in batches([len(t) for t in ptoks], a.max_tokens, None):
        s = last_token_states(pm, [ptoks[j] for j in b], ppad, dev)
        if not torch.isfinite(s).all():
            raise SystemExit("non-finite PRM state")
        PR[b] = s.cpu().numpy()
    PR.flush()
    np.save(out / "y.npy", y_all[mine].astype(np.int8))
    np.save(out / "index.npy", mine)
    with (out / "meta.jsonl").open("w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"[bundle] {a.ref.name} part {i}: done in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
