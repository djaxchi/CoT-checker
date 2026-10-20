#!/usr/bin/env python3
"""A trained read token's states as leaderboard stores (read_token_v1 Stage F).

The leaderboard cells (scripts/train_rep_learner_cell.py) read repstore splits.
This writes a read token's states for every item of a reference store split, in
that store's own shard layout, order, meta and labels, so a cell trained on them
sees exactly the items the Qwen3-8B Instruct leaderboard rows saw:

  rt_layers/<split>/<shard>/   token_seq; per item [final] + [block outputs at
      --blocks, last read token, pre-norm, offsets included] + [final]. Row 0 is
      the boundary slot that the sequence learners skip, and the last row is the
      post-norm state the read token's own head reads, so the `last_token`
      readout over this store is that state (Stage F-A) and `step_tokens` is the
      read token's path through the layers (Stage F-B).
  rt_append/<split>/<shard>/   vector; block --append_block's output at the last
      read token. The default 34 is hidden_states[35], the layer the step-span
      store holds, so the appended row lives in the same space as the step tokens
      it is appended to (`--append_store` in train_rep_learner_cell.py, Stage F-C).

Each reference shard is written as chunks of at most --chunk items
(shard_03_000, shard_03_001, ...), which ShardedRepSplit reads back in the same
global order, so the work splits evenly over GPUs however the reference is
sharded. Items are matched to their text by uid (PRM800K) or by (trace id, step index)
(ProcessBench). One unmatched item stops the run, and so does a PRM800K label
that disagrees with the reference store. --check runs the matching only.

    python scripts/read_token_store.py --ckpt tokens/dq_q/read_token.pt \
        --ref <repstore>/step_spans/probe_train_full \
        --source prm800k:<data>/prm800k_probe_train_full.jsonl --out_root <out> --shards 0/4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.encode_pooled_multilayer import pb_items, prm800k_items  # noqa: E402
from scripts.read_token_probe import (  # noqa: E402
    ReadToken, batches, read_all_states, reduce_states, tokenize)
from src.repstore.store import TOKEN_SEQ, VECTOR, RepSpec, RepSplit  # noqa: E402

BLOCKS = (3, 7, 11, 15, 19, 23, 27, 31, 34)


def source_index(spec: str) -> tuple[str, dict]:
    """'prm800k:PATH' -> uid -> item; 'pb:SUBSET:PATH' -> (id, step_idx) -> item."""
    kind, rest = spec.split(":", 1)
    if kind == "prm800k":
        return kind, {it["uid"]: it for it in prm800k_items(Path(rest), 0, 0)}
    if kind == "pb":
        sub, path = rest.split(":", 1)
        return kind, {(it["id"], int(it["step_idx"])): it for it in pb_items(Path(path), sub)}
    raise SystemExit(f"--source must be prm800k:PATH or pb:SUBSET:PATH, not {spec!r}")


def match(kind: str, index: dict, meta: list[dict], y: np.ndarray) -> list[dict]:
    """The source item of every reference meta row, in reference order."""
    out, missing = [], []
    for k, m in enumerate(meta):
        key = m["uid"] if kind == "prm800k" else (m["id"], int(m["step_idx"]))
        it = index.get(key)
        if it is None:
            missing.append(key)
            continue
        if kind == "prm800k" and int(it["label"]) != int(y[k]):
            raise SystemExit(f"label mismatch for {key!r}: source {it['label']} store {y[k]}")
        out.append(it)
    if missing:
        raise SystemExit(f"{len(missing)}/{len(meta)} reference items have no source item, "
                         f"e.g. {missing[:3]}; first meta row: {meta[0]}")
    return out


def write_store(out: Path, h: np.ndarray, lengths: np.ndarray, y: np.ndarray, meta: list[dict],
                spec: RepSpec) -> None:
    """The files of src/repstore/store.py's write_split, from a preallocated h."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "h.npy.tmp"
    with tmp.open("wb") as f:
        np.save(f, h)
    np.save(out / "lengths.npy", lengths.astype(np.int32))
    np.save(out / "y.npy", np.asarray(y, dtype=np.int8))
    with (out / "meta.jsonl").open("w") as f:
        for row in meta:
            f.write(json.dumps(row) + "\n")
    (out / "spec.json").write_text(spec.to_json())
    tmp.rename(out / "h.npy")


def shard_meta(meta: list[dict], n_rows: int) -> list[dict]:
    out = []
    for m in meta:
        row = dict(m)
        row.update({"span_n_tokens": m.get("n_tokens"), "n_tokens": n_rows,
                    "step_start_idx": 1 if n_rows > 1 else 0, "pre_step_boundary_idx": 0})
        out.append(row)
    return out


def load_model(name: str, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, local_files_only=True)
    model = model.to(dev).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model, tok


def load_read_token(ckpt: Path, d: int, dev) -> tuple[ReadToken, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    deep = ck.get("deep")
    rt = ReadToken(ck["m"], d, ck["soft"], read=ck.get("read", "last"),
                   head=ck.get("head_kind", "linear"),
                   n_layers=0 if deep is None else deep.shape[0]).to(dev)
    rt.head.load_state_dict(ck["head"])
    if deep is not None:
        rt.deep.data.copy_(deep.to(dev))
    return rt.eval(), ck


def encode_shard(model, tok, rt, items, blocks, append_block, pad, dev, max_tokens, max_seq_len,
                 soft_attn):
    """(layers (n, R, d) float16, append (n, d) float16) with R = len(blocks) + 2."""
    toks = [tokenize(tok, it) for it in items]
    long = [i for i, t in enumerate(toks) if len(t) > max_seq_len]
    if long:
        raise SystemExit(f"{len(long)} items exceed --max_seq_len {max_seq_len}; raise it")
    d = rt.soft.shape[1]
    L = np.zeros((len(items), len(blocks) + 2, d), np.float16)
    A = np.zeros((len(items), d), np.float16)
    want = sorted(set(blocks) | {append_block})
    for b in batches([len(t) for t in toks], max_tokens, None):
        cap = {l: None for l in want}
        with torch.no_grad():
            h = reduce_states(rt, read_all_states(model, rt, [toks[i] for i in b], pad, dev,
                                                  grad=False, soft_attn=soft_attn, capture=cap))
        rows = torch.stack([h] + [cap[l] for l in blocks] + [h], 1)
        if not torch.isfinite(rows).all():
            raise SystemExit("non-finite read-token state")
        L[b] = rows.cpu().numpy()
        A[b] = cap[append_block].cpu().numpy()
    return L, A


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--ref", type=Path, required=True, help="reference split dir with shard_*/")
    p.add_argument("--source", required=True, help="prm800k:PATH or pb:SUBSET:PATH")
    p.add_argument("--out_root", type=Path)
    p.add_argument("--blocks", default=",".join(map(str, BLOCKS)))
    p.add_argument("--append_block", type=int, default=34)
    p.add_argument("--shards", default="0/1", help="i/n: this process writes chunks with index %% n == i")
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--soft_attn", choices=["no_cudnn"], default=None)
    p.add_argument("--check", action="store_true", help="match items only, no model")
    a = p.parse_args()

    kind, index = source_index(a.source)
    shard_dirs = sorted(x for x in a.ref.iterdir() if x.is_dir() and x.name.startswith("shard_"))
    if not shard_dirs:
        raise SystemExit(f"no shard_* under {a.ref}")
    i, n = map(int, a.shards.split("/"))
    units = [(sd, c) for sd in shard_dirs
             for c in range(-(-len(np.load(sd / "lengths.npy", mmap_mode="r")) // a.chunk))]
    mine = [u for j, u in enumerate(units) if j % n == i]
    if a.check:
        total = sum(len(match(kind, index, RepSplit(sd).meta(), RepSplit(sd).y)) for sd in shard_dirs)
        print(f"[check] {a.ref.name}: {total} items matched over {len(shard_dirs)} shards, "
              f"{len(units)} chunks")
        return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rt, ck = load_read_token(a.ckpt, model.config.hidden_size, dev)
    soft_attn = a.soft_attn or ck.get("soft_attn")
    blocks = [int(x) for x in a.blocks.split(",")]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    split = a.ref.name
    for sd, c in mine:
        t0 = time.perf_counter()
        rs = RepSplit(sd)
        if rs.spec.prompt_style != "verifier":
            raise SystemExit(f"{sd} was encoded under {rs.spec.prompt_style!r}; "
                             f"read tokens use the verifier template")
        lo, hi = c * a.chunk, min((c + 1) * a.chunk, len(rs))
        meta, y = rs.meta()[lo:hi], np.asarray(rs.y[lo:hi])
        items = match(kind, index, meta, y)
        L, A = encode_shard(model, tok, rt, items, blocks, a.append_block, pad, dev,
                            a.max_tokens, a.max_seq_len, soft_attn)
        common = dict(dim=L.shape[2], backbone=rs.spec.backbone, source_split=split,
                      prompt_style="verifier", git_commit=commit)
        R, name = L.shape[1], f"{sd.name}_{c:03d}"
        write_store(a.out_root / "rt_layers" / split / name, L.reshape(-1, L.shape[2]),
                    np.full(len(items), R), y, shard_meta(meta, R),
                    RepSpec(name="rt_layers", kind=TOKEN_SEQ, layer=-1,
                            readout=f"read_token final+blocks{blocks}+final", **common))
        write_store(a.out_root / "rt_append" / split / name, A, np.ones(len(items)), y,
                    shard_meta(meta, 1),
                    RepSpec(name="rt_append", kind=VECTOR, layer=a.append_block + 1,
                            readout=f"read_token block{a.append_block}", **common))
        print(f"[store] {split}/{name}: {len(items)} items in {time.perf_counter()-t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
