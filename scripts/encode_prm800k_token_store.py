#!/usr/bin/env python3
"""Encode ALL last-layer token states of every PRM800K step into the repstore.

This is the "store everything once" producer: for each step it saves the full
last-layer hidden states of the whole input sequence (question + prior steps +
candidate step) as a token_seq item, plus the indexing offsets needed to derive
any representation offline without re-encoding:

    step_start_idx        first token of the candidate step  (step span start)
    pre_step_boundary_idx step_start_idx - 1                 (S_{t-1} for delta)
    n_tokens              full sequence length               (step span end = n_tokens-1)

From the store, offline: dense_last = row n_tokens-1; step tokens = rows
[step_start_idx : n_tokens]; delta = row (n_tokens-1) - row (step_start_idx-1);
pooled = mean/max over the step span. (Trajectory needs prior-step boundaries,
a small builder change; the context tokens are already stored here.)

Span-only mode (`--span_only`) writes just the rows any representation reads:
the pre-step boundary state followed by the step's own tokens. That is a 7x
saving (a step spans 38.8 tokens against 283 for the full sequence), and it is
what makes a backbone swap affordable: the full-sequence store for an 8B model
at hidden 4096 is ~1.1 TB, which does not fit, while the span store is ~157 GiB.
The output is byte-identical to encoding in full and then running
`scripts/build_step_span_store.py`, which was verified over all 513,810 training
items before the previous master store was deleted. Offsets are rewritten the
same way (`pre_step_boundary_idx` 0, `step_start_idx` 1), so every reader works
unchanged. What it gives up is the question and prior-step token states, which
no current representation reads.

Memory-safe two-pass per shard: tokenize first to size an on-disk memmap, then
stream the forward pass into it. Whole-node H100:4 -> 4 shards via
CUDA_VISIBLE_DEVICES; read back with repstore.ShardedRepSplit (no merge copy).
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

from scripts.encode_prm800k_hidden_states import build_prompt_prefix  # noqa: E402
from src.repstore.store import TOKEN_SEQ, RepSpec  # noqa: E402


def tokenize_with_offsets(tokenizer, ex: dict, max_seq_len: int) -> tuple[list[int], int]:
    """Return (input_ids, step_start_idx). step_start_idx = len(prefix_ids)."""
    prefix_ids = tokenizer(
        build_prompt_prefix(ex["problem"], ex["prefix"]),
        add_special_tokens=True, truncation=False,
    )["input_ids"]
    cand_ids = tokenizer(ex["candidate_step"], add_special_tokens=False, truncation=False)["input_ids"]
    if not cand_ids:
        raise ValueError("empty candidate step")
    ids = prefix_ids + cand_ids
    if len(ids) > max_seq_len:
        raise ValueError(f"len {len(ids)} > max_seq_len {max_seq_len}")
    return ids, len(prefix_ids)


def encode_split(
    jsonl_path: Path, rep_root: Path, stem: str, tokenizer, model, device,
    layer: int, max_seq_len: int, batch_size: int, pad_id: int,
    shard_idx: int, num_shards: int, backbone: str, limit: int | None,
    span_only: bool = False,
) -> None:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    for gi, r in enumerate(rows):
        r["global_index"] = gi
    shard = [r for r in rows if r["global_index"] % num_shards == shard_idx]
    if limit is not None:
        shard = shard[:limit]

    # ---- Pass 1: tokenize, size the memmap ----
    toks: list[tuple[list[int], int, dict]] = []
    for ex in shard:
        ids, start = tokenize_with_offsets(tokenizer, ex, max_seq_len)
        toks.append((ids, start, ex))
    if span_only and any(start < 1 for _, start, _ in toks):
        raise ValueError("span_only needs a non-empty prefix (step_start_idx >= 1)")
    lengths = np.array(
        [len(ids) - start + 1 if span_only else len(ids) for ids, start, _ in toks],
        dtype=np.int32)
    total_rows = int(lengths.sum())
    d = model.config.hidden_size

    out_dir = rep_root / stem / f"shard_{shard_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    h_mm = np.lib.format.open_memmap(
        out_dir / "h.npy", mode="w+", dtype=np.float16, shape=(total_rows, d)
    )
    print(f"[tokstore] {stem} shard {shard_idx}/{num_shards}: {len(toks)} items, "
          f"{total_rows:,} rows, {total_rows*d*2/1e9:.1f} GB", flush=True)

    # ---- Pass 2: forward, stream into memmap ----
    y = np.zeros(len(toks), dtype=np.int8)
    meta: list[dict] = []
    cursor = 0
    t0 = time.perf_counter()
    for i in range(0, len(toks), batch_size):
        batch = toks[i:i + batch_size]
        maxlen = max(len(t[0]) for t in batch)
        padded, masks = [], []
        for ids, _, _ in batch:
            padded.append(ids + [pad_id] * (maxlen - len(ids)))
            masks.append([1] * len(ids) + [0] * (maxlen - len(ids)))
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        att = torch.tensor(masks, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp, attention_mask=att, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer]  # (b, maxlen, d)
        del out
        for b, (ids, start, ex) in enumerate(batch):
            nt = len(ids)
            lo = start - 1 if span_only else 0
            keep = nt - lo
            vecs = hs[b, lo:nt, :].detach().to(torch.float16).cpu().numpy()
            if not np.isfinite(vecs).all():
                # float16 storage overflowed on an activation outlier. Silently
                # storing inf would poison every readout derived from this step.
                raise ValueError(
                    f"non-finite float16 activations for {ex['uid']} "
                    f"(max |h| = {np.abs(hs[b, lo:nt, :].float().cpu().numpy()).max():.1f}); "
                    f"the store cannot hold this step at float16")
            h_mm[cursor:cursor + keep] = vecs
            y[i + b] = int(ex["label"])
            row = {
                "uid": ex["uid"], "problem_id": ex["problem_id"],
                "solution_id": ex.get("solution_id"), "step_idx": ex["step_idx"],
                "label": int(ex["label"]), "rating": ex.get("rating"),
                "n_tokens": nt, "step_start_idx": start,
                "pre_step_boundary_idx": start - 1,
                "global_index": ex["global_index"],
            }
            if span_only:
                row.update({
                    "orig_n_tokens": nt, "orig_step_start_idx": start,
                    "orig_pre_step_boundary_idx": start - 1,
                    "n_tokens": keep, "step_start_idx": 1,
                    "pre_step_boundary_idx": 0,
                })
            meta.append(row)
            cursor += keep
        del hs
        if (i // batch_size) % 32 == 0 or i + batch_size >= len(toks):
            print(f"[tokstore] {stem} shard {shard_idx}: {i+len(batch)}/{len(toks)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    h_mm.flush()
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "y.npy", y)
    with (out_dir / "meta.jsonl").open("w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    spec = RepSpec(
        name=rep_root.name, kind=TOKEN_SEQ, dim=d, layer=layer, backbone=backbone,
        readout="step_span_with_boundary" if span_only else "token_all_last_layer",
        source_split=stem)
    (out_dir / "spec.json").write_text(spec.to_json())
    print(f"[tokstore] {stem} shard {shard_idx}: done ({cursor:,} rows written)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--rep_root", required=True, type=Path,
                   help="Representation dir, e.g. <repstore>/tokens_last_layer")
    p.add_argument("--splits", nargs="+", required=True, help="jsonl:stem pairs")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="float16",
                   help="Forward-pass dtype. Prefer the backbone's training dtype "
                        "(bfloat16 for Qwen3) so the forward pass keeps its range; "
                        "the store is float16 either way.")
    p.add_argument("--limit_per_file", type=int, default=None)
    p.add_argument("--span_only", action="store_true",
                   help="Store only the pre-step boundary row plus the step's own "
                        "tokens (~7x smaller, byte-identical to encoding in full "
                        "then compacting).")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.model_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only, torch_dtype=dtype,
    ).to(device).eval()

    for spec in args.splits:
        jsonl_name, stem = spec.split(":")
        encode_split(
            args.data_dir / jsonl_name, args.rep_root, stem, tok, model, device,
            args.layer, args.max_seq_len, args.batch_size, pad_id,
            args.shard_idx, args.num_shards, Path(args.model_name_or_path).name,
            args.limit_per_file, args.span_only,
        )


if __name__ == "__main__":
    main()
