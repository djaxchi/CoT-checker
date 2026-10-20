#!/usr/bin/env python3
"""A read token's final states along each item's trajectory (read_token_v1 Stage F-T).

For every item of an rt_layers split (scripts/read_token_store.py), the store
rt_traj holds the read token's final post-norm state at the current step and at
every previous step of the same trajectory, each read under its own prefix
exactly as that step would be read as an item:

  rt_traj/<split>/<unit>/   token_seq; per item [current] + [current, step n-1,
      ..., step 0]. Row 0 is the boundary slot the sequence learners skip, so
      `step_tokens` over this store sees the current step first and then its
      predecessors, most recent first: position p is always p steps back, whatever
      the trajectory's length.

Previous steps come from the item's `prior_steps` (ProcessBench: the trace's own
steps; PRM800K: the prefix split on blank lines, so a step containing a blank
line becomes two rows). A previous step whose text is itself an item of the split
reuses that item's final row from rt_layers; the others are encoded:

    python scripts/read_token_traj.py nodes --ckpt tokens/dq_q/read_token.pt \
        --layers <L>/prm/rt_layers/probe_train_full \
        --source prm800k:<data>/prm800k_probe_train_full.jsonl --out <L>/prm --shards 0/4
    python scripts/read_token_traj.py assemble --layers <L>/prm/rt_layers/probe_train_full \
        --source prm800k:<data>/prm800k_probe_train_full.jsonl --out <L>/prm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.read_token_probe import batches, read_all_states, reduce_states, tokenize  # noqa: E402
from scripts.read_token_store import (  # noqa: E402
    load_model, load_read_token, match, source_index, write_store)
from src.repstore.store import TOKEN_SEQ, RepSpec, RepSplit  # noqa: E402


def key(it: dict) -> tuple[str, str, str]:
    return it["problem"], it["prefix"], it["step"]


def prior_nodes(it: dict) -> list[dict]:
    """The previous steps of `it`, each as the item that would read it, oldest first."""
    steps = it["prior_steps"]
    return [{"problem": it["problem"], "prefix": "\n\n".join(steps[:j]), "step": steps[j]}
            for j in range(len(steps))]


def units(layers: Path) -> list[Path]:
    out = sorted(x for x in layers.iterdir() if x.is_dir() and x.name.startswith("shard_"))
    if not out:
        raise SystemExit(f"no shard_* under {layers}")
    return out


def unit_items(kind: str, index: dict, layers: Path) -> list[tuple[Path, list[dict]]]:
    return [(u, match(kind, index, RepSplit(u).meta(), RepSplit(u).y)) for u in units(layers)]


def to_encode(per_unit: list[tuple[Path, list[dict]]]) -> list[tuple[str, str, str]]:
    """Previous steps that are not items of the split, deduplicated, in a fixed order."""
    have = {key(it) for _, items in per_unit for it in items}
    need = {key(n) for _, items in per_unit for it in items for n in prior_nodes(it)}
    return sorted(need - have)


def final_rows(rs: RepSplit) -> np.ndarray:
    """The last row of every item: the read token's final post-norm state."""
    return np.asarray(rs.h[rs.offsets[1:] - 1])


def cmd_nodes(a, kind, index) -> None:
    per_unit = unit_items(kind, index, a.layers)
    nodes = to_encode(per_unit)
    split = a.layers.name
    out = a.out / "rt_traj_nodes" / split
    out.mkdir(parents=True, exist_ok=True)
    n_chunks = -(-len(nodes) // a.chunk)
    i, n = map(int, a.shards.split("/"))
    mine = [c for c in range(n_chunks) if c % n == i]
    print(f"[nodes] {split}: {len(nodes)} previous steps to encode, {n_chunks} chunks, "
          f"this worker {mine}", flush=True)
    if not mine:
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rt, ck = load_read_token(a.ckpt, model.config.hidden_size, dev)
    soft_attn = a.soft_attn or ck.get("soft_attn")
    for c in mine:
        t0 = time.perf_counter()
        chunk = nodes[c * a.chunk:(c + 1) * a.chunk]
        toks = [tokenize(tok, {"problem": p, "prefix": q, "step": s}) for p, q, s in chunk]
        if max(len(t) for t in toks) > a.max_seq_len:
            raise SystemExit(f"a previous step exceeds --max_seq_len {a.max_seq_len}; raise it")
        H = np.zeros((len(chunk), rt.soft.shape[1]), np.float16)
        for b in batches([len(t) for t in toks], a.max_tokens, None):
            with torch.no_grad():
                h = reduce_states(rt, read_all_states(model, rt, [toks[j] for j in b], pad, dev,
                                                      grad=False, soft_attn=soft_attn))
            if not torch.isfinite(h).all():
                raise SystemExit("non-finite read-token state")
            H[b] = h.cpu().numpy()
        tmp = out / f"part_{c:03d}.npy.tmp"
        with tmp.open("wb") as f:
            np.save(f, H)
        tmp.rename(out / f"part_{c:03d}.npy")
        print(f"[nodes] {split}/part_{c:03d}: {len(chunk)} in {time.perf_counter()-t0:.0f}s",
              flush=True)


def cmd_assemble(a, kind, index) -> None:
    per_unit = unit_items(kind, index, a.layers)
    nodes = to_encode(per_unit)
    split = a.layers.name
    state: dict[tuple, np.ndarray] = {}
    for u, items in per_unit:
        for it, h in zip(items, final_rows(RepSplit(u))):
            state.setdefault(key(it), h)
    parts = a.out / "rt_traj_nodes" / split
    for c in range(-(-len(nodes) // a.chunk)):
        H = np.load(parts / f"part_{c:03d}.npy")
        chunk = nodes[c * a.chunk:(c + 1) * a.chunk]
        if len(H) != len(chunk):
            raise SystemExit(f"{parts}/part_{c:03d}.npy has {len(H)} rows, expected {len(chunk)}")
        state.update(zip(chunk, H))
    for u, items in per_unit:
        rs = RepSplit(u)
        seqs = []
        for it in items:
            rows = [state[key(it)]] + [state[key(n)] for n in prior_nodes(it)[::-1]]
            seqs.append(np.stack([rows[0]] + rows))
        lengths = np.array([len(s) for s in seqs])
        meta = rs.meta()
        for m, n in zip(meta, lengths):
            m.update({"n_tokens": int(n), "step_start_idx": 1, "pre_step_boundary_idx": 0,
                      "n_prev_steps": int(n) - 2})
        spec = RepSpec(name="rt_traj", kind=TOKEN_SEQ, dim=rs.spec.dim, layer=-1,
                       backbone=rs.spec.backbone, source_split=split, prompt_style="verifier",
                       readout="read_token final, current then previous steps most recent first",
                       git_commit=rs.spec.git_commit)
        write_store(a.out / "rt_traj" / split / u.name, np.concatenate(seqs).astype(np.float16),
                    lengths, rs.y, meta, spec)
        print(f"[assemble] {split}/{u.name}: {len(items)} items, {int(lengths.sum())} rows, "
              f"mean previous steps {lengths.mean() - 2:.2f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["nodes", "assemble"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--layers", type=Path, required=True, help="rt_layers split dir")
    p.add_argument("--source", required=True, help="prm800k:PATH or pb:SUBSET:PATH")
    p.add_argument("--out", type=Path, required=True, help="writes rt_traj_nodes/ and rt_traj/ here")
    p.add_argument("--shards", default="0/1", help="nodes: i/n, this worker encodes chunks %% n == i")
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--soft_attn", choices=["no_cudnn"], default=None)
    a = p.parse_args()
    kind, index = source_index(a.source)
    (cmd_nodes if a.cmd == "nodes" else cmd_assemble)(a, kind, index)


if __name__ == "__main__":
    main()
